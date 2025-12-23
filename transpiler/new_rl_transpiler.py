"""
Safety-focused RL transpiler with Benchpress training data and hybrid fallback.

Key ideas implemented:
- Train on Benchpress QASM circuits (target distribution).
- Reward prioritizes two-qubit reduction; hard penalties for regressions vs static baseline.
- Early termination when exceeding baseline by a threshold (default 1.05x).
- Safety filter at inference: accept RL only if it matches/improves static two-qubit;
  otherwise fall back to parallel/static.
- Hybrid option: compare RL output against a small parallel race and pick best.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from qiskit import QuantumCircuit, transpile
from qiskit.providers.fake_provider import GenericBackendV2
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def count_twoq(circ: QuantumCircuit) -> int:
    return sum(1 for g in circ.data if len(g.qubits) == 2)


def load_benchpress_qasm(
    benchpress_dir: Path, subsets: List[str], limit_per_subset: int | None = 20
) -> List[QuantumCircuit]:
    """Load QASM circuits from Benchpress repo subsets."""
    circuits: List[QuantumCircuit] = []
    for subset in subsets:
        folder = benchpress_dir / "qasm" / subset
        if not folder.exists():
            raise FileNotFoundError(f"Benchpress subset not found: {folder}")
        files = sorted(folder.glob("**/*.qasm"))
        if limit_per_subset:
            files = files[:limit_per_subset]
        for f in files:
            try:
                circuits.append(QuantumCircuit.from_qasm_file(f))
            except Exception as exc:
                print(f"⚠️  Skipping {f}: {exc}", file=sys.stderr)
    if not circuits:
        raise RuntimeError("No circuits loaded from Benchpress.")
    return circuits


def baseline_static_metrics(
    circuit: QuantumCircuit, backend: GenericBackendV2
) -> Tuple[int, int]:
    """Compute static opt3 baseline depth and two-qubit count."""
    static = transpile(circuit, backend=backend, optimization_level=3, seed_transpiler=42)
    return static.depth(), count_twoq(static)


# --------------------------------------------------------------------------------------
# Parallel transpiler (small, reliable race)
# --------------------------------------------------------------------------------------


class SmallParallelTranspiler:
    """Lightweight parallel race of a few strong strategies."""

    def __init__(self, backend: GenericBackendV2, max_workers: int = 4):
        self.backend = backend
        self.strategies = [
            {"layout": "sabre", "routing": "sabre", "opt_level": 2},
            {"layout": "dense", "routing": "sabre", "opt_level": 2},
            {"layout": None, "routing": "sabre", "opt_level": 2},
        ]

    def run(self, circuit: QuantumCircuit):
        best = None
        for strat in self.strategies:
            start = time.time()
            out = transpile(
                circuit,
                backend=self.backend,
                optimization_level=strat["opt_level"],
                layout_method=strat["layout"],
                routing_method=strat["routing"],
                seed_transpiler=42,
            )
            elapsed = time.time() - start
            info = {
                "circuit": out,
                "depth": out.depth(),
                "twoq": count_twoq(out),
                "time": elapsed,
                "strategy": strat,
            }
            if best is None or (info["twoq"], info["depth"]) < (
                best["twoq"],
                best["depth"],
            ):
                best = info
        return best


# --------------------------------------------------------------------------------------
# RL Environment with baseline-aware reward and hard penalties
# --------------------------------------------------------------------------------------


@dataclass
class CircuitEntry:
    circuit: QuantumCircuit
    base_depth: int
    base_twoq: int


class BenchpressEnv(gym.Env):
    """Baseline-aware environment focused on two-qubit reliability."""

    def __init__(
        self,
        backend: GenericBackendV2,
        circuits: List[CircuitEntry],
        max_steps: int = 6,
        over_threshold: float = 1.05,
        seed: int = 123,
    ):
        super().__init__()
        self.backend = backend
        self.circuits = circuits
        self.max_steps = max_steps
        self.over_threshold = over_threshold
        self.rng = np.random.default_rng(seed)

        # Bounded but diverse strategy set
        self.strategies: Dict[int, Dict[str, Optional[str] | int]] = {
            0: {"layout": "sabre", "routing": "sabre", "opt_level": 2},
            1: {"layout": "sabre", "routing": "stochastic", "opt_level": 2},
            2: {"layout": "dense", "routing": "sabre", "opt_level": 2},
            3: {"layout": "dense", "routing": "stochastic", "opt_level": 2},
            4: {"layout": None, "routing": "sabre", "opt_level": 2},
            5: {"layout": None, "routing": "stochastic", "opt_level": 2},
        }
        self.action_space = spaces.Discrete(len(self.strategies))
        # obs: depth_ratio vs base, twoq_ratio vs base, qubit_frac, step_frac
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0], dtype=np.float32),
            high=np.array([5, 5, 1, 1], dtype=np.float32),
            dtype=np.float32,
        )

        # State
        self.entry: CircuitEntry | None = None
        self.current: QuantumCircuit | None = None
        self.best_depth = np.inf
        self.best_twoq = np.inf
        self.step_idx = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.entry = self.rng.choice(self.circuits)
        self.current = self.entry.circuit.copy()
        self.best_depth = self.entry.base_depth
        self.best_twoq = self.entry.base_twoq
        self.step_idx = 0
        return self._obs(), {}

    def _obs(self) -> np.ndarray:
        depth = self.current.depth()
        twoq = count_twoq(self.current)
        depth_ratio = depth / max(self.entry.base_depth, 1)
        twoq_ratio = twoq / max(self.entry.base_twoq, 1)
        qubit_frac = self.current.num_qubits / self.backend.num_qubits
        step_frac = self.step_idx / max(self.max_steps, 1)
        obs = np.array([depth_ratio, twoq_ratio, qubit_frac, step_frac], dtype=np.float32)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def step(self, action: int):
        self.step_idx += 1
        strat = self.strategies[int(action)]
        prev_depth = self.current.depth()
        prev_twoq = count_twoq(self.current)

        try:
            cand = transpile(
                self.current,
                backend=self.backend,
                optimization_level=strat["opt_level"],
                layout_method=strat["layout"],
                routing_method=strat["routing"],
                seed_transpiler=42 + self.step_idx,  # slight seed variation
            )
            self.current = cand
            success = True
        except Exception:
            success = False

        depth = self.current.depth()
        twoq = count_twoq(self.current)

        # Hard constraint: if over threshold of baseline, big penalty and terminate
        over_baseline = twoq > self.entry.base_twoq * self.over_threshold
        reward = 0.0
        if over_baseline:
            reward = -50.0
            terminated = True
        else:
            # Primary: two-qubit improvement vs baseline and best-so-far
            best_gain = (self.best_twoq - twoq) / max(self.entry.base_twoq, 1)
            base_gain = (self.entry.base_twoq - twoq) / max(self.entry.base_twoq, 1)
            depth_gain = (self.entry.base_depth - depth) / max(self.entry.base_depth, 1)

            reward += 40.0 * base_gain + 10.0 * best_gain + 3.0 * depth_gain

            # Penalties for regressions vs previous step
            if twoq > prev_twoq:
                reward -= 8.0
            if depth > prev_depth:
                reward -= 2.0

            # Update bests
            if twoq < self.best_twoq or (twoq == self.best_twoq and depth < self.best_depth):
                self.best_twoq = twoq
                self.best_depth = depth

            terminated = self.step_idx >= self.max_steps

        obs = self._obs()
        info = {"success": success, "depth": depth, "twoq": twoq}
        reward = float(np.clip(reward, -60.0, 120.0))
        return obs, reward, terminated, False, info


# --------------------------------------------------------------------------------------
# RL wrapper with safety filter
# --------------------------------------------------------------------------------------


class SafeRLTranspiler:
    def __init__(
        self,
        backend: GenericBackendV2,
        circuits: List[CircuitEntry],
        model_dir: Path,
        log_dir: Path,
    ):
        ensure_dir(model_dir)
        ensure_dir(log_dir)
        self.backend = backend
        self.env = BenchpressEnv(backend, circuits)
        self.model_dir = model_dir
        self.log_dir = log_dir
        self.model: Optional[PPO] = None

    def train(self, total_timesteps: int = 30_000, eval_freq: int = 5_000) -> Path:
        env = Monitor(self.env, filename=str(self.log_dir / "train_monitor"))
        eval_env = Monitor(BenchpressEnv(self.backend, self.env.circuits, seed=999))

        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=str(self.model_dir),
            log_path=str(self.log_dir),
            eval_freq=eval_freq,
            deterministic=True,
            render=False,
        )

        policy_kwargs = dict(
            net_arch=[dict(pi=[256, 128], vf=[256, 128])],
            activation_fn=torch.nn.SiLU,
        )

        self.model = PPO(
            "MlpPolicy",
            env,
            learning_rate=2e-4,
            n_steps=1024,
            batch_size=128,
            n_epochs=10,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.6,
            verbose=1,
            policy_kwargs=policy_kwargs,
        )

        self.model.learn(total_timesteps=total_timesteps, callback=eval_cb)
        out_path = self.model_dir / "new_rl_transpiler"
        self.model.save(out_path)
        print(f"✓ Saved model to {out_path}")
        return out_path

    def load(self, model_path: Path):
        # Handle numpy rename issue for older pickles
        import numpy as _np

        sys.modules.setdefault("numpy._core", _np.core)
        sys.modules.setdefault("numpy._core.numeric", _np.core.numeric)
        self.model = PPO.load(model_path, env=self.env)
        print(f"✓ Loaded model from {model_path}")

    def transpile_safe(
        self,
        circuit: QuantumCircuit,
        baseline_static: Tuple[int, int],
        parallel_runner: SmallParallelTranspiler | None = None,
    ):
        """
        Run RL; if RL worsens two-qubit vs static, fall back to parallel/static best.
        """
        if self.model is None:
            raise ValueError("Model not loaded")

        base_depth, base_twoq = baseline_static

        # Run RL episode with early stopping
        env = BenchpressEnv(self.backend, [], seed=777)
        env.entry = CircuitEntry(circuit.copy(), base_depth, base_twoq)
        env.current = circuit.copy()
        env.best_depth = base_depth
        env.best_twoq = base_twoq
        env.step_idx = 0
        obs = env._obs()

        best_circ = circuit
        best_twoq = base_twoq
        best_depth = base_depth

        for _ in range(env.max_steps):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, done, _, info = env.step(int(action))
            twoq = info["twoq"]
            depth = info["depth"]
            if twoq < best_twoq or (twoq == best_twoq and depth < best_depth):
                best_twoq = twoq
                best_depth = depth
                best_circ = env.current.copy()
            if done:
                break

        # Safety filter: compare against static baseline (and optional parallel)
        candidates = [
            ("static", base_twoq, base_depth, circuit, 0.0),
            ("rl", best_twoq, best_depth, best_circ, 0.0),
        ]
        if parallel_runner:
            start = time.time()
            pbest = parallel_runner.run(circuit)
            candidates.append(
                (
                    "parallel",
                    pbest["twoq"],
                    pbest["depth"],
                    pbest["circuit"],
                    pbest["time"],
                )
            )

        candidates.sort(key=lambda x: (x[1], x[2]))  # prioritize twoq then depth
        chosen = candidates[0]
        return {
            "choice": chosen[0],
            "twoq": chosen[1],
            "depth": chosen[2],
            "circuit": chosen[3],
        }


# --------------------------------------------------------------------------------------
# Main: train on Benchpress subsets and save model
# --------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchpress-dir", type=str, default="./benchpress/benchpress")
    parser.add_argument(
        "--subsets",
        type=str,
        default="qasmbench-small",
        help="Comma-separated subsets under benchpress/qasm/",
    )
    parser.add_argument("--limit", type=int, default=20, help="Limit per subset")
    parser.add_argument("--timesteps", type=int, default=30_000)
    parser.add_argument("--model-out", type=str, default="./models/new_rl_transpiler")
    parser.add_argument("--log-dir", type=str, default="./logs")
    parser.add_argument("--backend-qubits", type=int, default=27)
    args = parser.parse_args()

    bench_dir = Path(args.benchpress_dir)
    subsets = [s.strip() for s in args.subsets.split(",") if s.strip()]

    # Load circuits and precompute baselines
    raw_circuits = load_benchpress_qasm(bench_dir, subsets, args.limit)
    backend = GenericBackendV2(num_qubits=args.backend_qubits)
    entries: List[CircuitEntry] = []
    for circ in raw_circuits:
        b_depth, b_twoq = baseline_static_metrics(circ, backend)
        entries.append(CircuitEntry(circ, b_depth, b_twoq))

    model_dir = Path(args.model_out).parent
    log_dir = Path(args.log_dir)
    ensure_dir(model_dir)
    ensure_dir(log_dir)

    agent = SafeRLTranspiler(backend, entries, model_dir, log_dir)
    agent.train(total_timesteps=args.timesteps, eval_freq=5_000)
    print("Training complete.")


if __name__ == "__main__":
    main()

