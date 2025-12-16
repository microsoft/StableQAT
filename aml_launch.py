"""Azure ML launch script for LLM-QAT training + evaluation."""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
import time
from pathlib import Path
from typing import Dict

from azure.ai.ml import MLClient, command, Input, Output
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.identity import (
    DefaultAzureCredential
)
import webbrowser


def build_command(
    args: argparse.Namespace,
    node_count: int,
    gpu_count: int,
    gpus_per_node: int
) -> str:
    port = 29500

    # create a hostfile
    hostfile_path = os.path.join("./", "hostfile")
    # delete the hostfile if it exists
    if os.path.exists(hostfile_path):
        os.remove(hostfile_path)
    with open(hostfile_path, "w", encoding="utf-8") as hostfile:
        for i in range(node_count):
            hostfile.write(f"node-{i} slots={gpus_per_node}\n")

    bash_cmd = textwrap.dedent(
        f"""
        echo 'node count: {node_count}, gpu count: {gpu_count}, gpu count per node: {gpus_per_node}';
        echo "--- Environment Variables ---"
        echo "RANK: $RANK"
        echo "LOCAL_RANK: $LOCAL_RANK"
        echo "NODE_RANK: $NODE_RANK"
        echo "WORLD_SIZE: $WORLD_SIZE"
        echo "MASTER_ADDR: $MASTER_ADDR"
        echo "MASTER_PORT: $MASTER_PORT"
        echo "host list: $AZ_BATCH_HOST_LIST"
        echo "--------------------------"

        LOCAL_DIR='./outputs/'
        CACHE_DIR=$LOCAL_DIR/hf_cache

        mkdir -p $LOCAL_DIR/logs
        mkdir -p $CACHE_DIR

        export TRANSFORMERS_CACHE=$CACHE_DIR
        export HF_HOME=$CACHE_DIR
        
        accelerate launch --config_file accelerate_configs/deepspeed_zero3_manifold.yaml --multi_gpu --num_machines={node_count} --num_processes={gpu_count} \
            --main_process_ip $MASTER_ADDR --main_process_port {port} --machine_rank $NODE_RANK \
            --num_processes {gpu_count} --num_machines {node_count} \
            --deepspeed_hostfile {hostfile_path} \
            train.py \
            --train_config_file {args.train_config_file} \
            --tokenized_data_dir {{inputs.tokenized_data_dir}} \
            --tokenized_data_name {args.tokenized_data_name} \
            --output_dir {{outputs.output_dir}}

        CUDA_VISIBLE_DEVICES=0 python -m lm_eval \
        --model hf \
        --tasks piqa,winogrande,arc_challenge,hellaswag,arc_easy,sciq,openbookqa,boolq \
        --batch_size auto \
        --device cuda \
        --model_args pretrained={{outputs.output_dir}}/last_checkpoint \
        --output_path {{outputs.output_dir}}/evaluation_results
        """
    ).strip()

    return bash_cmd


def submit_to_aml(args: argparse.Namespace) -> None:
    """Submit the LLM-QAT training job to Azure Machine Learning.

    Enhancements:
    - Optional workspace profile selection (manifold | playground | dedicated) to auto-populate
      resource group, workspace name, default compute, gpus/node, and env vars.
    - Automatic default output / evaluation output URIs when not provided.
    """
    
    # Credential / client
    if args.workspace_profile == "manifold":
        args.resource_group = "manifold2-rg"
        args.workspace_name = "as-manifold2-sa-ws"
    elif args.workspace_profile == "playground":
        args.resource_group = "playground-rg"
        args.workspace_name = "as-playground-w3-ws"
    elif args.workspace_profile == "dedicated":
        args.resource_group = "dedicated-rg"
        args.workspace_name = "as-dedicated-w3-ws"

    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id="5c9e4789-4852-4ffe-8551-d682affcbd74",
        resource_group_name=args.resource_group,
        workspace_name=args.workspace_name,
    )

    # Workspace profile overrides (if provided); gpus_per_node derived only from profile
    profile_cfg = None
    gpus_per_node = None
    if args.workspace_profile:
        workspace = ml_client.workspaces.get()
        
        profile_cfgs = {
            "manifold": {
                "resource_group": "manifold2-rg",
                "workspace_name": "as-manifold2-sa-ws",
                "default_compute": "/subscriptions/5c9e4789-4852-4ffe-8551-d682affcbd74/resourceGroups/manifold2-rg/providers/Microsoft.MachineLearningServices/virtualclusters/as-manifold2-sa-vc",
                "gpus_per_node": 8,
                "environment_variables": {
                    "_AZUREML_SINGULARITY_JOB_UAI": workspace.identity.user_assigned_identities[0].resource_id,
                    "NCCL_DEBUG": "INFO",
                    "NCCL_IB_DISABLE": "0",
                    "NCCL_SOCKET_IFNAME": "^docker,lo",
                    "UCX_NET_DEVICES": "all",
                },
            },
            "playground": {
                "resource_group": "playground-rg",
                "workspace_name": "as-playground-w3-ws",
                "default_compute": "a100x4",
                "gpus_per_node": 4,
                "environment_variables": {"AZUREML_COMMON_RUNTIME_USE_SBOM_CAPABILITY": "true"},
            },
            "dedicated": {
                "resource_group": "dedicated-rg",
                "workspace_name": "as-dedicated-w3-ws",
                "default_compute": "a100x4",
                "gpus_per_node": 4,
                "environment_variables": {"AZUREML_COMMON_RUNTIME_USE_SBOM_CAPABILITY": "true"},
            },
        }
        if args.workspace_profile not in profile_cfgs:
            raise ValueError(f"Unknown workspace profile: {args.workspace_profile}")
        profile_cfg = profile_cfgs[args.workspace_profile]
        args.resource_group = profile_cfg["resource_group"]
        args.workspace_name = profile_cfg["workspace_name"]
        target_compute = profile_cfg["default_compute"]
        gpus_per_node = profile_cfg["gpus_per_node"]

    if args.workspace_profile == "manifold":
        print("Using Manifold profile.")
        vc_config = {
            "instance_type": "Singularity.ND96r_H100_v5",
            "instance_count": args.instance_count,
            "properties": {
                "AISuperComputer": {
                    # "slaTier": "Premium",
                    "slaTier": "Standard",
                    # "priority": "Medium" # Options: Low, Medium, High
                    "priority": "Medium" 
                }
            }
        }
    else:
        vc_config = None

    if args.instance_count < 1:
        raise ValueError("instance_count must be >= 1")

    if gpus_per_node is None:
        # Fallback if somehow no profile was applied (should not happen since default provided)
        gpus_per_node = 1
    if gpus_per_node < 1:
        raise ValueError("Derived gpus_per_node must be >= 1")
    args._derived_gpus_per_node = gpus_per_node  # stash for later use

    inputs: Dict[str, Input] = {
        "train_data_dir": Input(
            type=AssetTypes.URI_FOLDER,
            path=args.train_data_uri,
            mode=InputOutputModes.RO_MOUNT,
        ),
        "tokenized_data_dir": Input(
            type=AssetTypes.URI_FOLDER,
            path=args.tokenized_data_uri,
            mode=InputOutputModes.RO_MOUNT,
        ),
    }

    outputs: Dict[str, Output] = {
        "output_dir": Output(
            type=AssetTypes.URI_FOLDER,
            path=os.path.join(args.output_uri, args.display_name),
            mode=InputOutputModes.RW_MOUNT
        )
    }
    
    # Derive counts for command builder using derived gpus_per_node
    node_count = args.instance_count
    gpu_count = node_count * gpus_per_node
    command_line = build_command(
        args,
        node_count=node_count,
        gpu_count=gpu_count,
        gpus_per_node=gpus_per_node,
    )

    environment_variables = profile_cfg["environment_variables"]

    # MLflow env variables
    environment_variables["MLFLOW_MAX_LOG_PARAMS"] = "200"  # Fix for MLflow max param length issue in AML
    environment_variables["MLFLOW_NESTED_RUN"] = "True"  # Fix for MLflow nested runs issue in AML
    environment_variables["NCCL_DEBUG"] = "INFO"
    environment_variables["_AZUREML_SINGULARITY_JOB_UAI"] = ml_client.workspaces.get().identity.user_assigned_identities[0].resource_id
    environment_variables["NCCL_IB_DISABLE"] = "0"  # Enable Infiniband

    # Add HF token if provided
    environment_variables["HF_TOKEN"] = args.hf_token

    environment_variables["CUDA_LAUNCH_BLOCKING"] = "1"

    print("environment_variables:", environment_variables)

    env = ml_client.environments.get(
        "llm_qat_env", 
        version="11"
    )

    job = command(
        code="./",
        command=command_line,
        environment=env,
        inputs=inputs,
        outputs=outputs,
        compute=target_compute,
        environment_variables=environment_variables,
        experiment_name=args.experiment_name,
        display_name=args.display_name,
        instance_count=args.instance_count,
        resources=vc_config,
        distribution={"type": "pytorch", "process_count_per_instance": 1},
    )

    submitted_job = ml_client.jobs.create_or_update(job)
    print("Job successfully submitted.")
    print("Name:", submitted_job.name)
    print("Status:", submitted_job.status)
    print("Studio URL:", submitted_job.studio_url)
        
    try:
        webbrowser.open(submitted_job.studio_url)
        print("AML Studio url: ", submitted_job.studio_url)
    except:
        print("Web browser failed to open.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit LLM-QAT training and evaluation to Azure Machine Learning."
    )
    parser.add_argument("--resource-group", required=False, default=None, help="Azure resource group name (overridden by --workspace-profile if provided)")
    parser.add_argument("--workspace-name", required=False, default=None, help="Azure ML workspace name (overridden by --workspace-profile if provided)")
    parser.add_argument(
        "--workspace-profile",
        choices=["manifold", "playground", "dedicated"],
        default="manifold",
        help="Workspace profile preset (default: manifold) sets RG, workspace, default compute, gpus per node, env vars.",
    )
    # parser.add_argument(
    #     "--tokenized-data-uri",
    #     default="azureml://subscriptions/5c9e4789-4852-4ffe-8551-d682affcbd74/resourcegroups/manifold2-rg/workspaces/as-manifold2-sa-ws/datastores/slimpajama/paths/llama_tokenized_data_slimpajama_10b_v1/",
    #     required=False,
    #     help="Azure ML data asset or cloud URI providing training data (mapped to train_data_local_path).",
    # )
    parser.add_argument(
        "--tokenized-data-uri",
        default="azureml://subscriptions/5c9e4789-4852-4ffe-8551-d682affcbd74/resourcegroups/manifold2-rg/workspaces/as-manifold2-sa-ws/datastores/sa_qat/paths/",
        required=False,
        help="Azure ML data asset or cloud URI providing training data (mapped to train_data_local_path).",
    )
    parser.add_argument(
        "--tokenized-data-name",
        default="llama_tokenized_data_slimpajama_10b_v1",
        required=False,
        help="Azure ML data asset or cloud URI providing training data (mapped to train_data_local_path).",
    )
    parser.add_argument(
        "--train-data-uri",
        default="azureml://subscriptions/5c9e4789-4852-4ffe-8551-d682affcbd74/resourcegroups/manifold2-rg/workspaces/as-manifold2-sa-ws/datastores/slimpajama/paths/text_data_mixed/",
        required=False,
        help="Azure ML data asset or cloud URI providing training data (mapped to train_data_local_path).",
    )
    parser.add_argument(
        "--train-data-file",
        default="slimpajama10b_v1.jsonl",
        required=False,
        help="Azure ML data asset or cloud URI providing training data (mapped to train_data_local_path).",
    )
    parser.add_argument(
        "--output-uri",
        required=False,
        default="azureml://subscriptions/5c9e4789-4852-4ffe-8551-d682affcbd74/resourcegroups/manifold2-rg/workspaces/as-manifold2-sa-ws/datastores/llmqatcheckpoints/paths/",
        help="Azure ML URI folder where training outputs (local_dir) will be stored. If omitted, an auto path based on experiment/timestamp is used.",
    )
    parser.add_argument(
        "--display-name",
        default=time.strftime("llm-qat-%Y%m%d-%H%M%S"),
        help="Display name for the Azure ML job.",
    )
    parser.add_argument(
        "--experiment-name",
        default="llm-qat-training",
        help="Azure ML experiment name.",
    )
    parser.add_argument(
        "--instance-count",
        type=int,
        default=1,
        help="Number of nodes to allocate for the run.",
    )
    # Removed --gpus-per-node: now derived strictly from workspace profile.
    parser.add_argument(
        "--tokenize-config-file",
        default="tokenize_configs/llama_1b.yaml",
        help="Config yaml file for model tokenizing",
    )
    parser.add_argument(
        "--train-config-file",
        default=None,
        help="Config yaml file for model training",
    )
    parser.add_argument(
        "--eval-config-file",
        default=None,
        help="Config yaml file for model evaluation",
    )
    parser.add_argument(
        "--hf-token",
        default='',
        help="Hugging Face Hub token for accessing gated models (e.g., Llama). Will be set as HF_TOKEN environment variable.",
    )
    args = parser.parse_args()

    return args


def main() -> None:
    args = parse_args()
    args.display_name = args.display_name + "-" + args.tokenized_data_name + "-" + args.train_config_file.replace('.yaml','').replace('train_configs/','').replace('/','-')
    submit_to_aml(args)


if __name__ == "__main__":
    main()
