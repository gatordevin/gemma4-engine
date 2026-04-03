"""
Benchmark suite for Gemma 4 E4B custom engine.
Compares HuggingFace baseline vs custom engine vs compiled vs TurboQuant.

Usage:
    python benchmark.py          # Full benchmark (all configurations)
    python benchmark.py --quick  # Quick benchmark (skip HF baseline + compile)
"""

import torch
import time
import sys
import gc


def free_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(1)


def bench_generate(engine, prompt, max_tokens, temperature, runs=3, use_compile=False):
    results = []
    text = ""
    for _ in range(runs):
        text, n, t = engine.generate(prompt, max_new_tokens=max_tokens,
                                      temperature=temperature, use_compile=use_compile)
        results.append(n / t)
    return results, text


def print_gpu():
    print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB "
          f"/ {torch.cuda.get_device_properties(0).total_mem/1e9:.1f} GB")


def main():
    quick = "--quick" in sys.argv
    prompt_short = "What are you? Respond in 2 sentences."
    prompt_long = "Explain the history of computing from the abacus to modern AI."

    print("=" * 70)
    print("  GEMMA 4 E4B — INFERENCE BENCHMARK")
    print("=" * 70)
    print()

    # System info
    print("System:")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem/1e9:.1f} GB")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA: {torch.version.cuda}")
    print()

    results_summary = {}

    # ── 1. HuggingFace Baseline ──
    if not quick:
        print("-" * 70)
        print("1. HuggingFace Baseline (transformers, bf16, no optimizations)")
        print("-" * 70)
        from transformers import AutoProcessor, AutoModelForCausalLM

        proc = AutoProcessor.from_pretrained("google/gemma-4-E4B-it")
        model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-4-E4B-it", dtype=torch.bfloat16, device_map="auto"
        )
        print_gpu()

        messages = [{"role": "user", "content": prompt_short}]
        text = proc.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
        inputs = proc(text=text, return_tensors="pt").to("cuda")

        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=16)

        runs = []
        for i in range(3):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=128)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            ntoks = out.shape[-1] - inputs["input_ids"].shape[-1]
            runs.append(ntoks / dt)

        avg = sum(runs) / len(runs)
        results_summary["HF Baseline"] = avg
        print(f"  128 tokens: {avg:.1f} tok/s  (runs: {[f'{r:.1f}' for r in runs]})")

        del model, proc, inputs, out
        free_gpu()
        print()

    # ── 2. Custom Engine (uncompiled) ──
    print("-" * 70)
    print("2. Custom Engine (bare-metal, fused weights, static KV cache)")
    print("-" * 70)
    from engine import Gemma4Engine

    engine = Gemma4Engine()
    print_gpu()

    runs_128, _ = bench_generate(engine, prompt_short, 128, 0.0)
    runs_256, _ = bench_generate(engine, prompt_long, 256, 0.0)

    avg_128 = sum(runs_128) / len(runs_128)
    avg_256 = sum(runs_256) / len(runs_256)
    results_summary["Custom Engine"] = avg_128
    print(f"  128 tokens: {avg_128:.1f} tok/s  (runs: {[f'{r:.1f}' for r in runs_128]})")
    print(f"  256 tokens: {avg_256:.1f} tok/s  (runs: {[f'{r:.1f}' for r in runs_256]})")

    # ── 3. Custom Engine (compiled) ──
    if not quick:
        print()
        print("-" * 70)
        print("3. Custom Engine + torch.compile")
        print("-" * 70)

        print("  Compiling (one-time cost)...")
        t0 = time.time()
        engine.generate("Hi", max_new_tokens=32, use_compile=True)
        engine.generate("Hey", max_new_tokens=32, use_compile=True)
        compile_time = time.time() - t0
        print(f"  Compile time: {compile_time:.0f}s")

        runs_128c, _ = bench_generate(engine, prompt_short, 128, 0.0, use_compile=True)
        runs_256c, text_256 = bench_generate(engine, prompt_long, 256, 0.0, use_compile=True)

        avg_128c = sum(runs_128c) / len(runs_128c)
        avg_256c = sum(runs_256c) / len(runs_256c)
        results_summary["Compiled"] = avg_128c
        print(f"  128 tokens: {avg_128c:.1f} tok/s  (runs: {[f'{r:.1f}' for r in runs_128c]})")
        print(f"  256 tokens: {avg_256c:.1f} tok/s  (runs: {[f'{r:.1f}' for r in runs_256c]})")
        print(f"  Sample output: {text_256[:120]}...")

    del engine
    free_gpu()
    print()

    # ── 4. TurboQuant Engine ──
    print("-" * 70)
    print("4. TurboQuant Engine (compressed KV cache, 16K context)")
    print("-" * 70)
    from engine_turbo import Gemma4TurboEngine

    turbo = Gemma4TurboEngine(max_seq_len=16384)
    print_gpu()

    runs_128t, _ = bench_generate(turbo, prompt_short, 128, 0.0)
    runs_512t, _ = bench_generate(turbo, prompt_long, 512, 0.7)

    avg_128t = sum(runs_128t) / len(runs_128t)
    avg_512t = sum(runs_512t) / len(runs_512t)
    results_summary["TurboQuant 16K"] = avg_128t
    print(f"  128 tokens: {avg_128t:.1f} tok/s")
    print(f"  512 tokens: {avg_512t:.1f} tok/s")
    print(f"  Max context: 16,384 tokens (4x baseline)")

    del turbo
    free_gpu()
    print()

    # ── Summary ──
    print("=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)
    baseline = results_summary.get("HF Baseline", 17.5)
    for name, speed in results_summary.items():
        bar = "#" * int(speed / 2)
        print(f"  {name:20s}  {speed:6.1f} tok/s  ({speed/baseline:.1f}x)  {bar}")
    print(f"  {'Theoretical max':20s}  {'101.0':>6s} tok/s  (bandwidth limit)")
    print()
    if "Compiled" in results_summary:
        print(f"  Peak speedup: {results_summary['Compiled']/baseline:.1f}x over HuggingFace")
        print(f"  Bandwidth utilization: {results_summary['Compiled']/101*100:.0f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
