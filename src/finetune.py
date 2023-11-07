import os, sys
import copy
import json
import click
import datetime
import argparse
from typing import Dict, Sequence
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from datasets import load_dataset


import transformers
from transformers import Trainer, TrainingArguments

from model import PerSE

KEY_TYPE = "type"
KEY_INSTANCES = "instances"
ds_config = "configs/ds_config_zero3.json"
do_train = True
IGNORE_INDEX = -100

# max_length = 4096
max_length = 2048

model_name_or_path = "meta-llama/Llama-2-7b-hf"


# deepspeed --num_gpus 4 finetune.py
# deepspeed --master_port 61000 --include "localhost:3,4" finetune.py

def get_options():
    args = argparse.ArgumentParser()
    # data options
    args.add_argument("--data_file", type=str, default="../data/PerMPST.k3.src.json")
    args.add_argument("--save_dir", type=str, default="../checkpoints/")
    args.add_argument("--padding_strategy", type=str, default='right')

    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--batch_size", type=int, default=4)
    args.add_argument("--lr", type=float, default=1e-5)
    args.add_argument('--max_steps', type=int, default=4000)
    args.add_argument('--save_total_limit', type=int, default=3)

    args.add_argument("--local_rank", type=int, default=-1)

    args = args.parse_args()
    return args


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        targets = [
            f"{example['output']}{tokenizer.eos_token}" for example in raw_dataset
        ]
        data_dict = preprocess(raw_dataset["input"], targets, tokenizer)
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

        # print(len(self.input_ids), self.input_ids[0].size(), self.labels[0].size())

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # print(instances)
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


def preprocess(sources, targets, tokenizer):
    # remove pairs where at least one record is None
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [
        _tokenize_fn(strings, tokenizer) for strings in (examples, sources)
    ]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


def _tokenize_fn(
    strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            max_length = max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

if __name__ == '__main__':

    args = get_options()
    print(args)

    input_file =  args.data_file
    output_dir = args.save_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # sanity check over the fields of json file
    with open(input_file) as fin:
        json_data = json.load(fin)
        if KEY_TYPE not in json_data.keys():
            raise ValueError(
                f'"{KEY_TYPE}" field must be specified for data, e.g.'
                "{\n"
                f'   "{KEY_TYPE}: "text2text",\n'
                f'   "{KEY_INSTANCES}": [\n'
                '       { "text": "Sentence 1: This is a sentence." }\n'
                '       { "text": "Sentence 2: This is another sentence." }\n'
                f"   ]\n"
                "}"
            )

    # Load the dataset using the HuggingFace dataset library
    extensions = "json"
    raw_dataset = load_dataset(
        extensions,
        data_files=[input_file],
        field=KEY_INSTANCES,
        split="train",
        token=None,
    )

    print(raw_dataset)
    print(len(raw_dataset))

    perse = PerSE(model_name_or_path=model_name_or_path, padding_side=args.padding_strategy)

    run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    data_module = make_supervised_data_module(tokenizer=perse.tokenizer)
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="no",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0,
        weight_decay=0,
        max_steps=args.max_steps,
        logging_strategy="steps",
        logging_first_step=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=False,
        seed=args.seed,
        run_name=run_name,
        greater_is_better=False,
        deepspeed=ds_config,
        log_on_each_node=False,
        fp16=False,
        bf16=True,
        tf32=False,
    )  # tf32=True -> only for A100

    print("Start the trainer")

    if do_train:
        trainer = Trainer(
            model=perse.model,
            args=training_args,
            train_dataset=data_module["train_dataset"],
            eval_dataset=None,
            tokenizer=perse.tokenizer,
            data_collator=data_module["data_collator"],
            # compute_metrics=data_module.compute_metrics,
            preprocess_logits_for_metrics=None,
        )

        train_result = trainer.train()
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        # trainer.save_state()