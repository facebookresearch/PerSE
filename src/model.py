
import os
import re
import json

from tqdm import tqdm
from typing import Iterable, List

import numpy as np

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


MAX_TARGET_LENGTH = 512

class PerSE():
    def __init__(self, model_name_or_path: str = "meta-llama/Llama-2-7b-hf", padding_side: str = "left", device: str = "cuda"):
        super().__init__()
        self.device = device
        
        print(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            padding_side=padding_side,
            use_fast=False,
        )
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)

        print("Vocab Size: ", len(self.tokenizer))
        print("Loaded in model and tokenizer!")

        self.model.train()

    def inference(self, prompts: List, batch_size: int = 1, **kwargs) -> Iterable[str]:
        self.model.eval()

        results = []
        chunk_num = len(prompts) // batch_size + (len(prompts) % batch_size > 0)
        scope = tqdm(range(chunk_num)) if chunk_num > 10 else range(chunk_num)
        with torch.no_grad():
            for i in scope:
                batch_x = prompts[i*batch_size:(i+1)*batch_size]
                inputs = self.tokenizer(batch_x, return_tensors="pt", padding=True, truncation=True).to(self.device)
                prompt_len = inputs.input_ids.shape[1]
                outputs = self.model.generate(**inputs, **kwargs)
                outputs = outputs[:,prompt_len:]
                res = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                results.extend(res)
        return results