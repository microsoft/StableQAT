# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import json
import logging
import random
from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
import torch


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import json
import logging
import random
from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
import torch

from datasets import load_dataset

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_train_val_dataset(train_path, valid_path=None):
    train_data = load_dataset("json", data_files=train_path, streaming=False)['train']
    if valid_path:
        valid_data = load_dataset("json", data_files=valid_path, streaming=False)['train']
    else:
        split_idx = int(len(train_data) * 0.9)
        valid_data = train_data.shuffle(seed=42).take(split_idx)
        train_data = train_data.skip(split_idx)
    return train_data, valid_data

def group_texts(examples, block_size):
    concatenated = {k: sum(examples[k], []) for k in examples}
    total_length = len(concatenated["input_ids"])

    total_length = (total_length // block_size) * block_size

    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated.items()
    }

    result["labels"] = result["input_ids"].copy()
        
    return result

def pre_tokenize(dataset, tokenizer, block_size=1024):
    raw_data = dataset
    tokenized_datasets = []
    print("Tokenizing dataset...")
    tokenized_datasets = dataset.map(lambda x: tokenizer(x["text"]), batched=True).remove_columns(["text", "meta"])
    print("Finished tokenizing.")
    print("Grouping texts...")
    # grouped_dataset = group_texts(tokenized_datasets, block_size)
    grouped_dataset = tokenized_datasets.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=4,
        remove_columns=tokenized_datasets.column_names,
        fn_kwargs={"block_size": block_size},
    )
    print("Finished grouping.")
    
    return grouped_dataset

class CustomJsonDataset(torch.utils.data.IterableDataset):
    def __init__(self, grouped_dataset):
        self.input_ids = grouped_dataset["input_ids"]
        self.labels = grouped_dataset["labels"]
        self.data = [
            dict(input_ids=self.input_ids[i], labels=self.labels[i])
            for i in range(len(self.input_ids))
        ]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

    def __iter__(self):
        return iter(self.data)

def jload(filename, mode="r"):
    """Load a .json file into a dictionary."""
    with open(filename, mode) as f:
        jdict = json.load(f)
    return jdict

def fix_tokenizer_special_tokens(tokenizer, config, log):
    # Ensure pad_token is set
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            log.info(f"Set pad_token_id to eos_token_id: {tokenizer.eos_token_id}")
        else:
            tokenizer.pad_token_id = 0
            log.warning("Set pad_token_id to 0")
    
    # Critical validation: Check tokenizer vocab_size is compatible with model
    log.info(f"Tokenizer vocab_size: {tokenizer.vocab_size}")
    log.info(f"Model config vocab_size: {config.vocab_size}")
    log.info(f"Model max_position_embeddings: {config.max_position_embeddings}")
    
    if tokenizer.vocab_size > config.vocab_size:
        # CRITICAL: Tokenizer can produce token IDs that exceed model's embedding table
        log.error(
            f"CRITICAL ERROR: Tokenizer vocab_size ({tokenizer.vocab_size}) > "
            f"Model vocab_size ({config.vocab_size})"
        )
        raise ValueError(
            f"Tokenizer can produce token IDs up to {tokenizer.vocab_size - 1} "
            f"but model only has embeddings for IDs 0-{config.vocab_size - 1}. "
            f"This will cause CUDA embedding errors!"
        )
    elif tokenizer.vocab_size < config.vocab_size:
        # Safe: Model has extra embeddings that won't be used
        unused = config.vocab_size - tokenizer.vocab_size
        log.info(
            f"Model has {unused} unused embeddings (tokenizer vocab < model vocab). "
            f"This is safe - tokenizer will only produce IDs in [0, {tokenizer.vocab_size - 1}]"
        )
    else:
        log.info(f"Vocabulary sizes match perfectly: {tokenizer.vocab_size}")
    
    # Validate special tokens are within MODEL's vocabulary bounds
    # Special tokens can be outside tokenizer's base vocab if model vocab is larger
    special_tokens = {
        'pad_token_id': tokenizer.pad_token_id,
        'eos_token_id': tokenizer.eos_token_id,
        'bos_token_id': tokenizer.bos_token_id,
        'unk_token_id': tokenizer.unk_token_id,
    }
    
    # Fix any out-of-bounds special tokens
    for name, token_id in special_tokens.items():
        if token_id is not None and (token_id < 0 or token_id >= config.vocab_size):
            # Validate against MODEL vocab_size (not tokenizer vocab_size)
            log.warning(
                f"Special token {name} ({token_id}) is out of model's vocabulary bounds "
                f"[0, {config.vocab_size - 1}]. Attempting to fix..."
            )
            
            # Fix the problematic token
            if name == 'unk_token_id' and token_id == config.vocab_size:
                # Common off-by-one error in tokenizer configs
                # Set to a safe value within bounds
                tokenizer.unk_token_id = config.vocab_size - 1
                log.info(f"Reset unk_token_id to {tokenizer.unk_token_id}")
                special_tokens[name] = tokenizer.unk_token_id
            else:
                # For other cases, raise an error
                log.error(f"Cannot automatically fix {name}: {token_id}")
                raise ValueError(
                    f"Special token {name} ({token_id}) is out of model's vocabulary bounds "
                    f"[0, {config.vocab_size - 1}] and cannot be automatically fixed!"
                )
    
    log.info(f"Tokenizer configured with model_max_length={tokenizer.model_max_length}, vocab_size={tokenizer.vocab_size}")
    log.info(f"Special tokens: {special_tokens}")
    
    return tokenizer