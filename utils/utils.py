import logging
import os
from typing import Dict

import torch
import transformers


# Define a utility method for setting the logging parameters of a logger
def get_logger(logger_name):
    # Get the logger with the specified name
    logger = logging.getLogger(logger_name)

    # Set the logging level of the logger to INFO
    logger.setLevel(logging.INFO)

    # Define a formatter for the log messages
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create a console handler for outputting log messages to the console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Add the console handler to the logger
    logger.addHandler(console_handler)

    return logger


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {}
        for key in state_dict.keys():
            if "teacher" in key:
                continue
            cpu_state_dict[key] = state_dict[key].cpu()
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def get_local_rank():
    if os.environ["LOCAL_RANK"]:
        return int(os.environ["LOCAL_RANK"])
    else:
        logging.warning(
            "LOCAL_RANK from os.environ is None, fall back to get rank from torch distributed"
        )
        return torch.distributed.get_rank()


def create_named_param_groups(model): 
    param_groups = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        group = {
            'params': [param],
            'p_names': [name]
        }
        param_groups.append(group)

    return param_groups


def get_second_last_checkpoint(output_dir: str) -> str:
    """
    When the last checkpoint is corrupted, get the second last checkpoint
    """
    ckpts = [
        d for d in os.listdir(output_dir)
        if d.startswith("checkpoint-")
    ]
    ckpts = sorted(ckpts, key=lambda x: int(x.split("-")[-1]))
    
    if len(ckpts) < 2:
        print("No second last checkpoint found. avalilable checkpoints:", ckpts)
        return None
    
    return os.path.join(output_dir, ckpts[-2])
