import math
from models.configuration_llama import LlamaConfig
from models.configuration_qwen3 import Qwen3Config
from models.modeling_llama_quant import (
    LlamaForCausalLM as LlamaForCausalLMQuant,
)
from models.modeling_qwen3_quant import (
    Qwen3ForCausalLM as Qwen3ForCausalLMQuant,
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
from models import modeling_llama_quant, modeling_qwen3_quant
from transformers.trainer_utils import get_last_checkpoint

from utils.utils import create_named_param_groups, get_second_last_checkpoint
from optimizer.quantized_optimizer import QOptimizer
from datasets import load_from_disk, concatenate_datasets
from transformers import DataCollatorWithPadding


log = utils.get_logger("clm")

def prepare_train_data_mixed(
    max_train_tokens,
    fineweb_edu_data_args,
    slimpajama_data_args,
    train_data_dir
):

    slimpajama_train_data = load_from_disk(os.path.join(train_data_dir, slimpajama_data_args['data_path']))
    fineweb_edu_train_data = load_from_disk(os.path.join(train_data_dir, fineweb_edu_data_args['data_path']))

    print(f"Slimpajama train dataset size: {len(slimpajama_train_data)}")
    print(f"Fineweb edu train dataset size: {len(fineweb_edu_train_data)}")

    ratio_slimpajama, ratio_fineweb_edu = slimpajama_data_args['mix_ratio'], fineweb_edu_data_args['mix_ratio']

    seq_length = len(slimpajama_train_data[0]["input_ids"]) 

    print(f"Sequence length: {seq_length}")

    num_slimpajama = min(int((max_train_tokens * ratio_slimpajama) // seq_length), len(slimpajama_train_data))
    num_fineweb_edu  = min(int((max_train_tokens * ratio_fineweb_edu ) // seq_length), len(fineweb_edu_train_data))

    print("Number of samples from Slimpajama:", num_slimpajama)
    print("Number of samples from Fineweb edu:", num_fineweb_edu)
    # slimpajama_train_data_selected = slimpajama_train_data.shuffle(seed=42).select(range(num_slimpajama))
    # fineweb_edu_train_data_selected  = fineweb_edu_train_data.shuffle(seed=42).select(range(num_fineweb_edu))
    slimpajama_train_data_selected = slimpajama_train_data.select(range(num_slimpajama))
    fineweb_edu_train_data_selected  = fineweb_edu_train_data.select(range(num_fineweb_edu))

    # mixed_train_data = concatenate_datasets([slimpajama_train_data_selected, fineweb_edu_train_data_selected]).shuffle(seed=42)
    mixed_train_data = concatenate_datasets([slimpajama_train_data_selected, fineweb_edu_train_data_selected])
    return mixed_train_data, None


def compute_num_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Total parameters: {total_params}")
    log.info(f"Trainable parameters: {trainable_params}")
    return total_params, trainable_params

class TruncatedDataCollator(DataCollatorWithPadding):
    def __init__(self, tokenizer, max_length, **kwargs):
        super().__init__(tokenizer=tokenizer, **kwargs)
        self.max_length = max_length
    
    def __call__(self, features):
        for feature in features:
            for key in feature:
                feature[key] = feature[key][:self.max_length]
                
        return super().__call__(features)
    
def train(args):
    if torch.cuda.device_count() > 1:
        dist.init_process_group(backend="nccl")
    
    # Analyze yaml file
    with open(args.train_config_file, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    config['train']['output_dir'] = args.output_dir
    try:
        training_args = transformers.TrainingArguments(**config['train'])
    except:
        config['train']['eval_strategy'] = config['train'].pop('evaluation_strategy', None)
        training_args = transformers.TrainingArguments(**config['train'])
    config = argparse.Namespace(**config)
    model_args, data_args = config, config
    
    # Load Model
    log.info("Start to load model...")
    dtype = torch.bfloat16 if training_args.bf16 else torch.float
    if 'llama' in model_args.model_path.lower():
        quant_class = modeling_llama_quant
        model_class = LlamaForCausalLMQuant
        config_class = LlamaConfig
    elif 'qwen' in model_args.model_path.lower():
        quant_class = modeling_qwen3_quant
        model_class = Qwen3ForCausalLMQuant
        config_class = Qwen3Config
    else:
        raise NotImplementedError
    config = config_class.from_pretrained(model_args.model_path)
    config.w_bits = model_args.w_bits
    config.sine_soft_q = model_args.sine_soft_q
    config.efficient = model_args.efficient
    log.info("model_args.sine_soft_q:", model_args.sine_soft_q, type(model_args.sine_soft_q))
    quant_class.QuantizeLinear = QLinear if model_args.dsq==False else DSQLinear
    model = model_class.from_pretrained(
        pretrained_model_name_or_path=model_args.model_path,
        config=config,
        cache_dir=model_args.cache_dir,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        # device_map='cpu',
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
    # model.cuda()
    torch.cuda.empty_cache()
    # Do not save cache when using gradient checkpointing
    # Recompute activation during backward pass to save memory
    model.config.use_cache = False
    total_params, _ = compute_num_parameters(model)
    if total_params > 3e9:
        model.gradient_checkpointing_enable()
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
    if "llama" in model_args.model_path.lower():
        tokenizer = fix_tokenizer_special_tokens(tokenizer, config, log)

    log.info("Complete tokenizer loading...")
    
    train_data, valid_data = prepare_train_data_mixed(
        data_args.max_train_tokens,
        data_args.fineweb_edu_data,
        data_args.slimpajama_data,
        args.train_data_dir,
    )
    
    # truncate to target length
    log.info(f"Truncating to target length...")
    if hasattr(data_args, 'truncate') and data_args.truncate['enable']:
        target_length = data_args.truncate['length']
        def truncate_sequence(example):
            example['input_ids'] = example['input_ids'][:target_length]
            example['attention_mask'] = example['attention_mask'][:target_length]
            example['labels'] = example['labels'][:target_length]
            return example
        train_data = train_data.map(truncate_sequence)
    log.info(f"Complete truncating.")
    
    # # Load Pre-tokenized Data
    # log.info(f"Loading pre-tokenized data...")
    # tokenized_data_dir = os.path.join(args.train_data_dir, args.tokenized_data_name)
    # train_data = load_from_disk(os.path.join(tokenized_data_dir, "train_data"))
    # log.info(f"Loaded train dataset size: {len(train_data)}")
    # valid_data = None
    # try:
    #     valid_data = load_from_disk(os.path.join(tokenized_data_dir, "valid_data"))
    #     log.info(f"Loaded valid dataset size: {len(valid_data)}")
    # except:
    #     log.warning(f"No valid dataset found.")
    
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
    )
    
    # Save model checkpoint before training
    log.info("Save model checkpoint before training...")
    if not dist.is_initialized() or dist.get_rank() == 0:
        utils.safe_save_model_for_hf_trainer(trainer, os.path.join(training_args.output_dir, "before_training_checkpoint"))
    
    # Set resume from checkpoint
    """
    Can hook-up with resume_path
    """
    if args.resume_ckpt in [None, "None", ""]:
        try:
            last_ckpt = get_last_checkpoint(training_args.output_dir)
        except:
            log.info("Last checkpoint is corrupted, try to get second last checkpoint...")
            last_ckpt = get_second_last_checkpoint(training_args.output_dir)
            log.info(f"Second last checkpoint: {last_ckpt}")
    else:
        last_ckpt = os.path.join(training_args.output_dir, args.resume_ckpt)
    if last_ckpt is None:
        log.warning("No checkpoint found, fall back to train from scratch.")
        resume_from_checkpoint = False
    else:
        resume_from_checkpoint = True
        log.info(f"Resume from checkpoint: {resume_from_checkpoint}, {last_ckpt if resume_from_checkpoint else ''}")
    
    # Train
    if training_args.do_train:
        # train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        train_result = trainer.train(resume_from_checkpoint=last_ckpt if resume_from_checkpoint else None)
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
    parser.add_argument("--train_data_dir", default="dataset")
    parser.add_argument("--tokenized_data_name", default="tokenized_data")
    parser.add_argument("--output_dir", default="outputs/llama-1B/2bits/baseline")
    parser.add_argument("--resume_ckpt", default=None, help="Path to resume training from a checkpoint.")
    args = parser.parse_args()
    train(args)
