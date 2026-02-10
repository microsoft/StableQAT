import math
import os  # Added for secure retrieval of HF Hub token
from models.configuration_llama import LlamaConfig
from models.modeling_llama_quant import (
    LlamaForCausalLM as LlamaForCausalLMQuant,
)
import copy
import torch
import transformers
from utils import utils
from utils import datautils

from utils.process_args import process_args
from torch import distributed as dist
from transformers import default_data_collator, Trainer
import argparse, yaml

log = utils.get_logger("clm")


def tokenize(args):
    # If tokenized data exist, exit
    tokenized_data_dir = args.tokenized_data_dir
    if os.path.exists(os.path.join(tokenized_data_dir, "train_data")) and os.path.exists(os.path.join(tokenized_data_dir, "valid_data")):
        log.info("Preprocessed data already exists. Skipping tokenization.")
        return
    
    hf_token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    
    # Load Tokenizer
    log.info("Start to load tokenizer...")
    config = LlamaConfig.from_pretrained(args.model_path)
      # Cap model_max_length to the model's actual max_position_embeddings to prevent indexing errors
    max_model_length = min(args.model_max_length, config.max_position_embeddings)
    if args.model_max_length > config.max_position_embeddings:
        log.warning(
            f"Requested model_max_length ({args.model_max_length}) exceeds "
            f"model's max_position_embeddings ({config.max_position_embeddings}). "
            f"Capping to {max_model_length} to prevent indexing errors."
        )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=args.model_path,
        model_max_length=max_model_length,
        padding_side="right",
        add_bos_token=False,
        add_eos_token=False,
    )
    tokenizer = datautils.fix_tokenizer_special_tokens(tokenizer, config, log)    
    log.info("Complete tokenizer loading...")
    
    # Load train and valid data
    log.info("Start to get_train_val_dataset...")
    train_dataset, valid_dataset = datautils.get_train_val_dataset(
        train_path=os.path.join(args.train_data_dir, args.train_data_file),
        valid_path=None,
    )
    log.info("Finished get_train_val_dataset")
    
    # Tokenize
    log.info("Start to tokenize...")
    train_data = datautils.pre_tokenize(
        train_dataset, tokenizer, block_size=args.model_max_length
    )
    valid_data = datautils.pre_tokenize(
        valid_dataset, tokenizer, block_size=args.model_max_length
    )
    log.info("Finish tokenizing.")
    
    # Save tokenized data
    train_data.save_to_disk(os.path.join(tokenized_data_dir, "train_data"))
    valid_data.save_to_disk(os.path.join(tokenized_data_dir, "valid_data"))
    log.info(f"Tokenized data saved to {tokenized_data_dir}")
    log.info(f"Train dataset size: {len(train_data)}")
    log.info(f"Valid dataset size: {len(valid_data)}")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenize_config_file", default="tokenize_configs/llama_1b.yaml")
    parser.add_argument("--train_data_dir", default="/root/autodl-tmp/hub/datasets")
    parser.add_argument("--train_data_file", default="example_train_0.jsonl")
    parser.add_argument("--tokenized_data_dir", default="dataset/tokenized_data")
    args = parser.parse_args()
    with open(args.tokenize_config_file, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    args.model_path = config['model_path']
    args.model_max_length = config['model_max_length']
    tokenize(args)