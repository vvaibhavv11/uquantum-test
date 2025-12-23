"""
Evaluate SafeRLTranspiler (new RL model) + small parallel vs static on Benchpress.

Output: CSV with static, parallel, and safe-RL results.

Example:
  python3 new_benchpress_eval.py \
    --benchpress-dir ./benchpress/benchpress \
    --subset qasmbench-small \
    --limit 10 \
    --model ./models/new_rl_transpiler.zip \
    --out new_benchpress_results.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import List, Tuple

from qiskit import QuantumCircuit, transpile
from qiskit.providers.fake_provider import GenericBackendV2

from new_rl_transpiler import (
    SafeRLTranspiler,
    SmallParallelTranspiler,
    count_twoq,
    load_benchpress_qasm,
    baseline_static_metrics,
    CircuitEntry,
)


def evaluate_circuit(
    circuit: QuantumCircuit,
    backend: GenericBackendV2,
    rl_runner: SafeRLTranspiler,
    parallel_runner: SmallParallelTranspiler,
):
    # Static baseline
    start = time.time()
    static = transpile(circuit, backend=backend, optimization_level=3, seed_transpiler=42)
    t_static = time.time() - start
    static_twoq = count_twoq(static)
    static_depth = static.depth()

    # Parallel small race
    start = time.time()
    pbest = parallel_runner.run(circuit)
    t_parallel = time.time() - start + pbest["time"]  # include transpile time (already measured)

    # Safe RL with fallback to parallel/static built in
    rl_start = time.time()
    choice = rl_runner.transpile_safe(circuit, (static_depth, static_twoq), parallel_runner)
    t_rl = time.time() - rl_start

    return {
        "orig_depth": circuit.depth(),
        "orig_twoq": count_twoq(circuit),
        "static_depth": static_depth,
        "static_twoq": static_twoq,
        "static_time": t_static,
        "parallel_depth": pbest["depth"],
        "parallel_twoq": pbest["twoq"],
        "parallel_time": t_parallel,
        "parallel_strategy": pbest["strategy"],
        "rl_choice": choice["choice"],
        "rl_depth": choice["depth"],
        "rl_twoq": choice["twoq"],
        "rl_time": t_rl,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchpress-dir", type=str, default="./benchpress/benchpress")
    parser.add_argument("--subset", type=str, default="qasmbench-small")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--model", type=str, default="./models/new_rl_transpiler.zip")
    parser.add_argument("--out", type=str, default="new_benchpress_results.csv")
    parser.add_argument("--backend-qubits", type=int, default=27)
    args = parser.parse_args()

    bench_dir = Path(args.benchpress_dir)
    circuits_raw = load_benchpress_qasm(bench_dir, [args.subset], args.limit)
    backend = GenericBackendV2(num_qubits=args.backend_qubits)

    # Precompute baselines for env construction
    entries: List[CircuitEntry] = []
    for c in circuits_raw:
        b_depth, b_twoq = baseline_static_metrics(c, backend)
        entries.append(CircuitEntry(c, b_depth, b_twoq))

    # Setup runners
    rl_runner = SafeRLTranspiler(
        backend,
        circuits=entries,
        model_dir=Path(args.model).parent,
        log_dir=Path("./logs"),
    )
    rl_runner.load(Path(args.model))
    parallel_runner = SmallParallelTranspiler(backend)

    results = []
    for idx, circ in enumerate(circuits_raw, start=1):
        print(f"\n=== {args.subset} circuit {idx}/{len(circuits_raw)} ===")
        res = evaluate_circuit(circ, backend, rl_runner, parallel_runner)
        results.append(res)
        print(
            f"static depth/twoq/time: {res['static_depth']}/{res['static_twoq']}/{res['static_time']:.3f}s"
        )
        print(
            f"parallel depth/twoq/time: {res['parallel_depth']}/{res['parallel_twoq']}/{res['parallel_time']:.3f}s"
        )
        print(
            f"rl({res['rl_choice']}) depth/twoq/time: {res['rl_depth']}/{res['rl_twoq']}/{res['rl_time']:.3f}s"
        )

    # Write CSV
    out_path = Path(args.out)
    fieldnames = list(results[0].keys())
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()

