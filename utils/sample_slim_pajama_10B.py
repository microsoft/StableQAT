#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Uniformly sample 10B tokens from the train/ directory of HF hub 
cerebras/SlimPajama-627B → local slim_pajama_10B.jsonl

Usage:
    python sample_10B.py
"""
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
#os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"

import json, math, os, functools, io, math, random
from typing import Dict, List, Iterable
import huggingface_hub as hf_hub
import zstandard as zstd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

# ========= 1.Configs =========
CONFIG = {
    "repo_id"      : "cerebras/SlimPajama-627B",
    "revision"     : "main",               # can be fixed to a certain commit
    "target_tokens": 10_000_000_000,        # 10B
    "cache_dir"    : "tmp_slim_pajama",  # download cache dir
    "out_file"     : "slim_pajama_10B.jsonl",
    "workers"      : 8,                    # Parallel download/parsing
    "seed"         : 42,
}
random.seed(CONFIG["seed"])

# ========= 2. scan the files =========
def list_jsonl_zst_files(repo_id: str, subfolder: str = "train") -> List[str]:
    """
    Return the repo-relative paths of all *.jsonl.zst under 'train/' in the repo
    """
    api = hf_hub.HfApi()
    all_files = api.list_repo_files(repo_id, repo_type="dataset")
    prefix = subfolder.rstrip("/") + "/"
    jsonl_files = [f for f in all_files if f.startswith(prefix) and f.endswith(".jsonl.zst")]
    return [{f'chunk{i}':[f for f in jsonl_files if f'chunk{i}' in f]} for i in range(1, 11)]

# ========= 3. Calculate how many files to sample for each chunk =========
def compute_files_per_chunk(paths: List[str], total_tokens: int) -> Dict[str, int]:
    """
    Return dict {chunk: the number of files to sample}
    """
    return {chunk: math.ceil(len(paths[chunk]) * CONFIG['target_tokens'] / 626e+9)  for chunk in paths}

# ========= 4. Streamn processing each file =========
def sample_file(repo_path: str, repo_id: str, revision: str, cache_dir: str) -> List[dict]:
    """
    Download → Extract → Parse
    Return the collected samples (List[dict])
    """
    local_zst = os.path.join(cache_dir, repo_path.replace("/", os.sep))
    os.makedirs(os.path.dirname(local_zst), exist_ok=True)

    # 4.1 Download (Skip if it already exists locally)
    if not os.path.exists(local_zst):
        hf_hub.hf_hub_download(
            repo_id=repo_id,
            filename=repo_path,
            repo_type="dataset",
            revision=revision,
            local_dir=cache_dir,
            #local_dir_use_symlinks=False,
            #show_progress=False,
        )
    # 4.2 Stream extracting + sampling
    samples: List[dict] = []
    with open(local_zst, "rb") as f_zst:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(f_zst) as reader:
            # Key point: Binary stream → Text buffering
            text_io = io.TextIOWrapper(reader, encoding="utf-8", errors="ignore")
            for line in text_io:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                samples.append(obj)
                
    # 4.3 remove the downloaded temp file
    os.remove(local_zst)
    return samples

# ========= 5. parallel sample =========
def parallel_sample(all_files, quotas: Dict[str, int]) -> Iterable[List[dict]]:
    """
    Use multiprocessing.Pool to parallely process multiple files
    Return the iterable List[dict], which can be written to disk sequentially
    """
    worker = functools.partial(
        sample_file,
        repo_id=CONFIG["repo_id"],
        revision=CONFIG["revision"],
        cache_dir=CONFIG["cache_dir"],
    )

    sample_files = []
    for chunk, quota in quotas.items():
        sample_files.extend(random.sample(list(all_files[chunk]), int(quota)))
    
    with Pool(CONFIG["workers"]) as pool:
        for samples in tqdm(pool.imap(worker, sample_files),
                            total=len(sample_files),
                            desc="Sampling files"):
            yield samples

# ========= 5. Serial sample =========
def serial_sample(all_files, quotas: Dict[str, int]) -> Iterable[List[dict]]:
    # Calculate the number of files in each chunk
    chunk_files_count = {chunk: int(quotas[chunk]) for chunk in quotas}
    
    # List and save sample_files
    sample_files_dic = {}
    for chunk in quotas:
        sample_files = (random.sample(list(all_files[chunk]), chunk_files_count[chunk]))
        sample_files_dic.update({chunk:sample_files})
    with open(f'./dataset/sample_files_seed_{CONFIG["seed"]}.json', 'w', encoding='utf-8') as f:
        json.dump(sample_files_dic, f, ensure_ascii=False, indent=4)
    
    # Process each file in each chunk
    for chunk in sample_files_dic: 
        sample_files = sample_files_dic[chunk]
        progress_desc = f"Chunk {chunk} ({chunk_files_count[chunk]} files)"
        for file_name in tqdm(sample_files, desc=progress_desc, unit="file"):
            samples = sample_file(
                repo_path=file_name, 
                repo_id=CONFIG["repo_id"], 
                revision=CONFIG["revision"], 
                cache_dir=CONFIG["cache_dir"]
            )
            yield samples

# ========= 6. Save =========
def write_jsonl(samples_iter: Iterable[List[dict]], out_file: str):
    """
    Write incrementally to avoid eating up memory at once
    """
    total_lines = 0
    with open(out_file, "w", encoding="utf-8") as fw:
        for batch in samples_iter:
            for obj in batch:
                fw.write(json.dumps(obj, ensure_ascii=False) + "\n")
                total_lines += len(batch)
    
    print(f"\n✅ All files are processed! The {total_lines} rows are saved to train.jsonl")

# ========= 7. Main function =========
def main():
    os.makedirs(CONFIG["cache_dir"], exist_ok=True)

    # 7.1 Scan
    all_files_path = os.path.join('dataset', 'all_files.json')
    if not os.path.exists(all_files_path):
        all_files = list_jsonl_zst_files(CONFIG["repo_id"])
        with open(all_files_path, 'w', encoding='utf-8') as f:
            json.dump(all_files, f, ensure_ascii=False, indent=4)
    else:
        with open(all_files_path, 'r', encoding='utf-8') as f:
            all_files = json.load(f)
    
    # 7.2 Quota
    quotas = compute_files_per_chunk(all_files, CONFIG["target_tokens"])
    print("File quotas computed.")
    print(quotas)

    # 7.3 Sample + Save
    write_jsonl(serial_sample(all_files, quotas), CONFIG["out_file"])
    print(f"Done. {CONFIG['target_tokens']/1e-9}B-token sample written to {CONFIG['out_file']}")

if __name__ == "__main__":
    main()