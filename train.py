import glob
import os
from base64 import b64decode
from io import BytesIO
from typing import Any, Optional, Union

import cv2
import datasets
import deepspeed
import fire
import numpy as np
import torch
import yaml
from peft import LoraConfig, get_peft_config, get_peft_model
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset
from transformers import (AutoProcessor, AutoTokenizer, CLIPImageProcessor,
                          Trainer, TrainingArguments)

from git_llm.git_llama import GitLlamaConfig, GitLlamaForCausalLM
from git_llm.git_mpt import GitMptConfig, GitMptForCausalLM
from git_llm.git_opt import GitOPTConfig, GitOPTForCausalLM

GitLLMForCausalLM = Any


# SupervisedDataset
class SupervisedDataset(Dataset):
    """Dataset for supervised learning"""

    def __init__(
        self,
        model_name: str,
        vision_model_name: str,
        loaded_dataset: datasets.GeneratorBasedBuilder,
        max_length: int = 128,
    ):
        super(SupervisedDataset, self).__init__()
        self.loaded_dataset = loaded_dataset
        self.max_length = max_length

        self.processor = AutoProcessor.from_pretrained("microsoft/git-base")
        self.processor.image_processor = CLIPImageProcessor.from_pretrained(vision_model_name)
        self.processor.tokenizer = AutoTokenizer.from_pretrained(
            model_name, padding_side="right", use_fast=True if "mpt" in model_name else False
        )
        if "llama" in model_name:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        elif "mpt" in model_name:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

    def __len__(self) -> int:
        return len(self.loaded_dataset)

    def __getitem__(self, index) -> dict:
        # cf: https://huggingface.co/datasets/MMInstruction/M3IT#data-instances
        row = self.loaded_dataset[index]

        # some of nlvr data were broken
        instruction = row["instruction"]  # str
        question = row["inputs"]  # str
        answer = row["outputs"]  # str
        text = f"##Instruction: {instruction} ##Question: {question} ##Answer: {answer}"

        # imageのロード
        image_base64_str_list = row["image_base64_str"]  # str (base64)
        img = Image.open(BytesIO(b64decode(image_base64_str_list[0]))).convert("RGB")
        img = np.array(img)
        if img.shape[2] != 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        inputs = self.processor(
            text,
            img,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        )
        # batch size 1 -> unbatch
        inputs = {k: v[0] for k, v in inputs.items()}
        inputs["labels"] = inputs["input_ids"]
        return inputs


def load_model(
    model_name: str, vision_model_name: str, num_image_with_embedding: Optional[int]
) -> GitLLMForCausalLM:
    """Loading a GIT-LLM depending on configs"""
    if "opt" in model_name:
        git_config = GitOPTConfig.from_pretrained(model_name)
        git_config.set_vision_configs(
            num_image_with_embedding=num_image_with_embedding, vision_model_name=vision_model_name
        )
        model = GitOPTForCausalLM.from_pretrained(model_name, config=git_config)
    elif "llama" in model_name:
        git_config = GitLlamaConfig.from_pretrained(model_name)
        git_config.set_vision_configs(
            num_image_with_embedding=num_image_with_embedding, vision_model_name=vision_model_name
        )
        model = GitLlamaForCausalLM.from_pretrained(model_name, config=git_config)
    elif "mpt" in model_name:
        git_config = GitMptConfig.from_pretrained(model_name)
        git_config.set_vision_configs(
            num_image_with_embedding=num_image_with_embedding, vision_model_name=vision_model_name
        )
        model = GitMptForCausalLM.from_pretrained(model_name, config=git_config)
    return model


def load_pretrained_weight(model: GitLLMForCausalLM, weight_path: str):
    import glob

    weight = {}
    weight_path = glob.glob(f"{weight_path}/pytorch*.bin")
    for w in weight_path:
        weight_temp = torch.load(w, map_location="cpu")
        weight.update(weight_temp)
    model.load_state_dict(weight, strict=False)


def apply_lora_model(model: GitLLMForCausalLM, model_name: str, config: dict) -> GitLLMForCausalLM:
    """Apply LoRA"""
    peft_config = LoraConfig(**config["lora"])
    # apply lora only to LLM
    if "opt" in model_name:
        model.model.decoder = get_peft_model(model.model.decoder, peft_config)
    elif "llama" in model_name:
        target_modules = []
        for m in peft_config.target_modules:
            target_modules += [
                f"model.layers.{i}.self_attn.{m}" for i in range(len(model.model.layers))
            ]

        peft_config.target_modules = target_modules
        model = get_peft_model(model, peft_config)
        model.base_model.model.lm_head = model.lm_head
        # remove peft wrapper
        model = model.base_model.model
    elif "mpt" in model_name:
        model = get_peft_model(model, peft_config)
        model.base_model.model.lm_head = model.lm_head
        # remove peft wrapper
        model = model.base_model.model
    return model


def set_trainable_params(model: GitLLMForCausalLM, model_name: str, keys_finetune: list) -> None:
    if "mpt" in model_name:
        for name, p in model.transformer.named_parameters():
            if np.any([k in name for k in keys_finetune]):
                p.requires_grad = True
            else:
                p.requires_grad = False
    else:
        for name, p in model.model.named_parameters():
            if np.any([k in name for k in keys_finetune]):
                p.requires_grad = True
            else:
                p.requires_grad = False


def get_dataset(config: dict) -> Union[Dataset, Dataset]:
    if config.get("dataset_type") is not None:
        dataset_list = [
            datasets.load_dataset("MMInstruction/M3IT", i) for i in config["dataset_type"]
        ]
        train_dataset = ConcatDataset([d["train"] for d in dataset_list])
        # some dataset have no validation
        for d in dataset_list:
            val_dataset_list = []
            try:
                val_dataset_list.append(d["validation"])
            except:
                print(f"{d['train']._info.config_name} has no validation set.")
        val_dataset = ConcatDataset(val_dataset_list)
    else:
        coco_datasets = datasets.load_dataset("MMInstruction/M3IT", "coco")
        train_dataset = coco_datasets["train"]
        val_dataset = coco_datasets["validation"]
    return train_dataset, val_dataset


def main(config_file: str):
    # get config
    with open(config_file, "r") as i_:
        config = yaml.safe_load(i_)

    if os.environ["WANDB_NAME"] is not None:
        config["training"]["output_dir"] = os.path.join(
            config["training"]["output_dir"], os.environ["WANDB_NAME"]
        )

    # distributed learning
    deepspeed.init_distributed()

    # model
    model_name = config["settings"]["model_name"]
    vision_model_name = config["settings"]["vision_model_name"]
    num_image_with_embedding = config["settings"]["num_image_with_embedding"]

    # DatasetのLoad
    train_dataset, val_dataset = get_dataset(config)

    # configの割り当て
    max_length = config["settings"]["max_length"]
    keys_finetune = config["settings"]["keys_finetune"]

    # 訓練に関するconfig
    training_args = TrainingArguments(**config["training"])

    # load model
    model = load_model(model_name, vision_model_name, num_image_with_embedding)

    # lora
    if config["use_lora"]:
        keys_finetune.append("lora")
        model = apply_lora_model(model, model_name, config)

    # load pretrained weight
    if config["settings"]["load_pretrained"] is not None:
        load_pretrained_weight(model, config["settings"]["load_pretrained"])
        print(
            f'Successfully loading pretrained weights from {config["settings"]["load_pretrained"]}'
        )

    # Set trainable params
    set_trainable_params(model, model_name, keys_finetune)

    trainer = Trainer(
        model=model,
        train_dataset=SupervisedDataset(model_name, vision_model_name, train_dataset, max_length),
        eval_dataset=SupervisedDataset(model_name, vision_model_name, val_dataset, max_length),
        args=training_args,
    )

    with torch.autocast("cuda"):
        result = trainer.train()

    # Save the finel checkpoint
    # https://github.com/huggingface/transformers/blob/v4.31.0/src/transformers/trainer.py#L2281
    final_save_path = os.path.join(
        config["training"]["output_dir"], os.environ["WANDB_NAME"] + "_final"
    )
    trainer.save_model(final_save_path)
    if "zero3" in config["training"]["deepspeed"]:
        # under zero3 model file itself doesn't get saved since it's bogus! Unless deepspeed
        # config `stage3_gather_16bit_weights_on_model_save` is True
        trainer.model_wrapped.save_checkpoint(final_save_path)


if __name__ == "__main__":
    fire.Fire(main)
