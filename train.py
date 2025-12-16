# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
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
import argparse, yaml, os
from utils.datautils import fix_tokenizer_special_tokens, CustomJsonDataset
from models.utils_quant import QuantizeLinear as QLinear
from models.utils_quant_dsq import DSQLinear
from models import modeling_llama_quant
from transformers.trainer_utils import get_last_checkpoint

from utils.utils import create_named_param_groups
from optimizer.quantized_optimizer import QOptimizer
from datasets import load_from_disk


log = utils.get_logger("clm")


def train(args):
    if torch.cuda.device_count() > 1:
        dist.init_process_group(backend="nccl")
    
    # Analyze yaml file
    with open(args.train_config_file, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    config['train']['output_dir'] = args.output_dir
    training_args = transformers.TrainingArguments(**config['train'])
    config = argparse.Namespace(**config)
    model_args, data_args = config, config
    
    # Load Model
    log.info("Start to load model...")
    dtype = torch.bfloat16 if training_args.bf16 else torch.float
    config = LlamaConfig.from_pretrained(model_args.model_path)
    config.w_bits = model_args.w_bits
    config.sine_soft_q = model_args.sine_soft_q
    modeling_llama_quant.QuantizeLinear = QLinear if model_args.dsq==False else DSQLinear
    model = LlamaForCausalLMQuant.from_pretrained(
        pretrained_model_name_or_path=model_args.model_path,
        config=config,
        cache_dir=model_args.cache_dir,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map='cpu',
    )
    if not model_args.contain_weight_clip_val:
        for name, param in model.named_parameters():
            if "weight_clip_val" in name:
                weight_name = name.replace("weight_clip_val", "weight")
                weight_param = dict(model.named_parameters()).get(weight_name, None)

                with torch.no_grad():
                    if model_args.w_bits == 1:
                        scale = torch.mean(weight_param.abs(), dim=-1, keepdim=True)
                    elif model_args.w_bits == 0 or model_args.w_bits == 2:
                        scale, _ = torch.max(torch.abs(weight_param), dim=-1, keepdim=True)
                    elif model_args.w_bits == 3 or model_args.w_bits == 4:
                        xmax, _ = torch.max(torch.abs(weight_param), dim=-1, keepdim=True)
                        maxq = 2 ** (model_args.w_bits - 1) - 1
                        scale = xmax / maxq
                    else:
                        raise NotImplementedError

                    param.data.copy_(scale)
            if "uW" in name:
                param.data.copy_(torch.tensor([2 **31 - 1]).float())
            if "lW" in name:
                param.data.copy_(torch.tensor([(-1) * (2**32)]).float())
            if "alpha_dsq" in name:
                param.data.copy_(torch.tensor([0.2]))
    model.cuda()
    torch.cuda.empty_cache()
    model.config.use_cache = False
    log.info("Complete model loading...")

    # Load Tokenizer
    log.info("Start to load tokenizer...")
      # Cap model_max_length to the model's actual max_position_embeddings to prevent indexing errors
    max_model_length = min(data_args.model_max_length, config.max_position_embeddings)
    if data_args.model_max_length > config.max_position_embeddings:
        log.warning(
            f"Requested model_max_length ({data_args.model_max_length}) exceeds "
            f"model's max_position_embeddings ({config.max_position_embeddings}). "
            f"Capping to {max_model_length} to prevent indexing errors."
        )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_args.model_path,
        cache_dir=model_args.cache_dir,
        model_max_length=max_model_length,
        padding_side="right",
        add_bos_token=False,
        add_eos_token=False,
    )
    tokenizer = fix_tokenizer_special_tokens(tokenizer, config, log)
    log.info("Complete tokenizer loading...")
    
    # Load Pre-tokenized Data
    log.info(f"Loading pre-tokenized data...")
    tokenized_data_dir = os.path.join(args.tokenized_data_dir, args.tokenized_data_name)
    train_data = load_from_disk(os.path.join(tokenized_data_dir, "train_data"))
    log.info(f"Loaded train dataset size: {len(train_data)}")
    valid_data = None
    try:
        valid_data = load_from_disk(os.path.join(tokenized_data_dir, "valid_data"))
        log.info(f"Loaded valid dataset size: {len(valid_data)}")
    except:
        log.warning(f"No valid dataset found.")

    # ============== Custom Optimizer Setup ==============
    param_groups = create_named_param_groups(model)
    
    # Option 1: Use the QOptimizer from your codebase
    optimizer = QOptimizer(param_groups, 
                        variant="adamw", 
                        lr=training_args.learning_rate, 
                        lr_quant=training_args.learning_rate, 
                        first_momentum=0.9,
                        second_momentum=0.95,
                        weight_decay=1e-4,
                    )

    # Create learning rate scheduler
    from transformers import get_scheduler
    num_training_steps = (
        len(train_data) // training_args.per_device_train_batch_size 
        // training_args.gradient_accumulation_steps 
        * training_args.num_train_epochs
    )

    lr_scheduler = get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=training_args.warmup_steps,
        num_training_steps=num_training_steps,
    )
    # ============== End Custom Optimizer Setup ==============
    
    # Initialize Trainer
      # Custom trainer to clear cache and prevent memory accumulation
    class MemoryEfficientTrainer(Trainer):
        def training_step(self, model, inputs, num_items_in_batch=None):
            loss = super().training_step(model, inputs, num_items_in_batch)
            # Clear CUDA cache periodically to prevent fragmentation
            if self.state.global_step % 10 == 0:
                torch.cuda.empty_cache()
            return loss

    myTrainer = MemoryEfficientTrainer
    trainer = myTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_data if training_args.do_train else None,
        eval_dataset=valid_data if training_args.do_eval else None,
        data_collator=default_data_collator,
        # optimizers=(optimizer, lr_scheduler)
    )
    
    # Save model checkpoint before training
    log.info("Save model checkpoint before training...")
    if not dist.is_initialized() or dist.get_rank() == 0:
        utils.safe_save_model_for_hf_trainer(trainer, os.path.join(training_args.output_dir, "before_training_checkpoint"))
    
    # Set resume from checkpoint
    last_ckpt = get_last_checkpoint(training_args.output_dir)
    if last_ckpt is None:
        log.warning("No checkpoint found, fall back to train from scratch.")
        resume_from_checkpoint = False
    else:
        resume_from_checkpoint = True
    
    # Train
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        if not dist.is_initialized() or dist.get_rank() == 0:
            trainer.save_state()
            utils.safe_save_model_for_hf_trainer(trainer, os.path.join(training_args.output_dir, "last_checkpoint"))
            # rename the log file
            log_dir = training_args.logging_dir
            old_log_name = os.listdir(log_dir)[0]
            new_log_name = args.train_config_file.replace('/','-').replace('yaml','') + 'tfevents.node-0'
            os.rename(os.path.join(log_dir, old_log_name), os.path.join(log_dir, new_log_name))

    # Evaluation
    if training_args.do_eval:
        model.to("cuda")
        metrics = trainer.evaluate()
        max_eval_samples = len(valid_data)
        metrics["eval_samples"] = min(max_eval_samples, len(valid_data))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    torch.distributed.barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_config_file", default="train_configs/epoch_1/llama_1B_2bits_sine_soft_q.yaml")
    parser.add_argument("--tokenized_data_dir", default="dataset")
    parser.add_argument("--tokenized_data_name", default="tokenized_data")
    parser.add_argument("--output_dir", default="outputs/llama-1B/2bits/baseline")
    args = parser.parse_args()
    train(args)
    
# Code needed when using AutoDL
# from modelscope import snapshot_download
# model_dir = snapshot_download('LLM-Research/Llama-3.2-1B',cache_dir='/root/autodl-tmp')
# print(f"Model downloaded to: {model_dir}")
# First step: sample the data: python sample_slim_pajama_10B.py
