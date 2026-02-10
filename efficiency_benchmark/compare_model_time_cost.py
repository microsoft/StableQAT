#!/usr/bin/env python3
"""
Benchmark of three quantized Llama-3.2-1B implementations:
  A) ParetoQ
  B) DSQ
  C) Sine-Soft-Q
Print forward / backward latency & memory.
"""
import gc, time, statistics, torch, argparse
from contextlib import contextmanager
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from typing import Tuple
import sys

sys.path.append("./")
from models.utils_quant import QuantizeLinear as QLinear
from models.utils_quant_dsq import DSQLinear
from models import modeling_llama_quant
from models.configuration_llama import LlamaConfig
from models.modeling_llama_quant import (
    LlamaForCausalLM as LlamaForCausalLMQuant,
)

DEVICE = "cuda:0"
WARMUP, REPEAT = 5, 30
TORCH_BACKENDS = {
    "cudnn.benchmark": False,
    "cudnn.deterministic": True,
    "allow_tf32": False,
}

# ------------- Timer -------------
@contextmanager
def cuda_timer():
    torch.cuda.synchronize(DEVICE)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    yield
    end.record()
    torch.cuda.synchronize(DEVICE)
    cuda_timer.elapsed = start.elapsed_time(end)  # ms

# ------------- buil model -------------
def build_model(args):
    dtype = torch.bfloat16 if args.bf16 else torch.float
    config = LlamaConfig.from_pretrained(args.model_path)
    config.w_bits = args.w_bits
    config.sine_soft_q = args.sine_soft_q
    config.efficient = args.efficient
    modeling_llama_quant.QuantizeLinear = QLinear if args.dsq==False else DSQLinear
    model = LlamaForCausalLMQuant.from_pretrained(
        pretrained_model_name_or_path=args.model_path,
        config=config,
        # cache_dir=args.cache_dir,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=DEVICE,
    )
    for name, param in model.named_parameters():
        if "weight_clip_val" in name:
            weight_name = name.replace("weight_clip_val", "weight")
            weight_param = dict(model.named_parameters()).get(weight_name, None)

            with torch.no_grad():
                if args.w_bits == 1:
                    scale = torch.mean(weight_param.abs(), dim=-1, keepdim=True)
                elif args.w_bits == 0 or args.w_bits == 2:
                    scale, _ = torch.max(torch.abs(weight_param), dim=-1, keepdim=True)
                elif args.w_bits == 3 or args.w_bits == 4:
                    xmax, _ = torch.max(torch.abs(weight_param), dim=-1, keepdim=True)
                    maxq = 2 ** (args.w_bits - 1) - 1
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
    return model

# -------------  benchmark  -------------
@torch.no_grad()
def benchmark_fwd(model, input_ids, warmup=WARMUP, repeat=REPEAT) -> Tuple[float, float]:
    model.eval()
    for _ in range(warmup):
        _ = model(input_ids)
    times = []
    for _ in range(repeat):
        with cuda_timer():
            _ = model(input_ids)
        times.append(cuda_timer.elapsed)

    # times = times[int(0.05 * repeat) : int(0.95 * repeat)]
    # return statistics.median(times), statistics.stdev(times)
    return statistics.mean(times), statistics.stdev(times)

def benchmark_bwd(model, input_ids, warmup=WARMUP, repeat=REPEAT) -> Tuple[float, float]:
    model.train()
    for _ in range(warmup):
        out = model(input_ids, labels=input_ids)
        loss = out.loss
        loss.backward()
        model.zero_grad()
    times = []
    for _ in range(repeat):
        out = model(input_ids, labels=input_ids)
        loss = out.loss
        with cuda_timer():
            loss.backward()
        model.zero_grad()
        times.append(cuda_timer.elapsed)
    # times = times[int(0.05 * repeat) : int(0.95 * repeat)]
    # return statistics.median(times), statistics.stdev(times)
    return statistics.mean(times), statistics.stdev(times)

def max_memory_mb() -> float:
    return torch.cuda.max_memory_allocated(DEVICE) / 1024**2

# -------------  main  -------------
def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model_path", type=str, default="/root/autodl-tmp/hub/models/Llama-3.2-1B")
    parser.add_argument("--model_path", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seqlen", type=int, default=128)
    parser.add_argument("--w_bits", type=int, default=4)
    parser.add_argument("--bf16", type=bool, default=True)
    parser.add_argument("--efficient", type=bool, default=True)

    args = parser.parse_args()
    
    # for k, v in TORCH_BACKENDS.items():
    #     torch.backends.__dict__[k.split(".")[0]].__dict__[k.split(".")[1]] = v
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.allow_tf32 = False
    
    torch.manual_seed(42)
    torch.cuda.reset_peak_memory_stats()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    x = torch.randint(0, tokenizer.vocab_size, (args.batch, args.seqlen), device=DEVICE)

    results = []
    for tag in ['baseline', 'sine_soft_q', 'dsq']:  #in [("transformers", build_modelA), ("my-impl", build_modelB)]:
        # args.dsq, args.sine_soft_q = (tag == 'dsq'), (tag == 'sine_soft_q')
        if tag == 'dsq':
            args.dsq = True
            args.sine_soft_q = {'enable': False}
        elif tag == 'sine_soft_q':
            args.dsq = False
            args.sine_soft_q = {'enable': True, 'amplitude': [0.21]}
        else:
            args.dsq = False
            args.sine_soft_q = {'enable': False}
        print(f"\n>>> Testing {tag} ...")
        torch.cuda.empty_cache(); gc.collect()
        # torch.cuda.reset_peak_memory_stats()
        model = build_model(args)
        torch.cuda.reset_peak_memory_stats()

        fwd_med, fwd_std = benchmark_fwd(model, x)
        mem_fwd = max_memory_mb()

        torch.cuda.empty_cache(); gc.collect()
        torch.cuda.reset_peak_memory_stats()
        bwd_med, bwd_std = benchmark_bwd(model, x)
        mem_bwd = max_memory_mb()

        results.append({
            "impl": tag,
            "fwd_ms": f"{fwd_med:.2f}±{fwd_std:.2f}",
            "fwd_mem_MB": f"{mem_fwd:.0f}",
            "bwd_ms": f"{bwd_med:.2f}±{bwd_std:.2f}",
            "bwd_mem_MB": f"{mem_bwd:.0f}",
        })

    print(results)
    print("\n### Result Summary")
    print(f"| {'Implementation':14} | {'forward (ms)':13} | {'forward (MB)':13} | {'backward (ms)':13} | {'backward (MB)':13} |")
    print("| "+ "-"*14 + " | " + "-"*13 + " | " + "-"*13 + " | " + "-"*13 + " | " + "-"*13 + " |")
    for r in results:
        print(f"| {r['impl']:14} | {r['fwd_ms']:13} | {r['fwd_mem_MB']:13} | {r['bwd_ms']:13} | {r['bwd_mem_MB']:13} |")

if __name__ == "__main__":
    main()