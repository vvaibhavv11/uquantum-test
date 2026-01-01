import time
import logging
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
from fastapi import HTTPException, status
from qiskit import QuantumCircuit, transpile
from qiskit.providers.fake_provider import GenericBackendV2
from settings import settings

try:
    from qiskit.providers.aer import Aer
except ImportError:  # pragma: no cover
    try:
        from qiskit_aer import Aer  # type: ignore
    except ImportError:  # pragma: no cover
        Aer = None  # sentinel for missing Aer

try:
    # Prefer qiskit.providers.aer.noise if available
    from qiskit.providers.aer import noise  # type: ignore
except ImportError:  # pragma: no cover
    try:
        from qiskit_aer import noise  # type: ignore
    except ImportError:  # pragma: no cover
        noise = None  # type: ignore[assignment]

from transpiler.new_rl_transpiler import (
    SafeRLTranspiler,
    SmallParallelTranspiler,
    count_twoq,
)

logger = logging.getLogger("transpile_service")
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


class TranspileService:
    def __init__(self):
        backend_qubits = settings.TRANSPILER_BACKEND_QUBITS
        self.backend = GenericBackendV2(num_qubits=backend_qubits)
        # Default model path: inside backend/transpiler/new_rl_transpiler.zip
        default_model_path = Path(__file__).resolve().parent.parent / "transpiler" / "new_rl_transpiler.zip"
        model_path_str = settings.TRANSPILER_MODEL_PATH or str(default_model_path)
        self.model_path = Path(model_path_str)
        self.model_dir = self.model_path.parent
        self.log_dir = Path(settings.TRANSPILER_LOG_DIR)

        self.parallel_runner = SmallParallelTranspiler(self.backend)
        self.safe_rl: Optional[SafeRLTranspiler] = None

    def _ensure_safe_rl_loaded(self) -> SafeRLTranspiler:
        if self.safe_rl is None:
            if not self.model_path.exists():
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"RL model not found at {self.model_path}",
                )
            logger.info("Loading SafeRLTranspiler model from %s", self.model_path)
            self.safe_rl = SafeRLTranspiler(
                backend=self.backend,
                circuits=[],  # not needed for inference
                model_dir=self.model_dir,
                log_dir=self.log_dir,
            )
            self.safe_rl.load(self.model_path)
        return self.safe_rl

    def _parse_circuit(self, qasm: str) -> QuantumCircuit:
        try:
            return QuantumCircuit.from_qasm_str(qasm)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to parse QASM: {exc}",
            ) from exc

    def _static_transpile(self, circuit: QuantumCircuit) -> Dict:
        start = time.time()
        static = transpile(
            circuit,
            backend=self.backend,
            optimization_level=3,
            seed_transpiler=42,
        )
        elapsed = time.time() - start
        return {
            "choice": "static",
            "circuit": static,
            "depth": static.depth(),
            "two_qubit_count": count_twoq(static),
            "wall_time_s": elapsed,
            "strategy": {"optimization_level": 3, "seed_transpiler": 42},
            "notes": None,
        }

    def _transpile_with_mode(self, circuit: QuantumCircuit, mode: str) -> Dict:
        # Static baseline
        static_result = self._static_transpile(circuit)
        static_depth = static_result["depth"]
        static_twoq = static_result["two_qubit_count"]

        # Parallel race
        p_start = time.time()
        parallel_best = self.parallel_runner.run(circuit)
        parallel_elapsed = time.time() - p_start + parallel_best["time"]
        parallel_result = {
            "choice": "parallel",
            "circuit": parallel_best["circuit"],
            "depth": parallel_best["depth"],
            "two_qubit_count": parallel_best["twoq"],
            "wall_time_s": parallel_elapsed,
            "strategy": parallel_best["strategy"],
            "notes": None,
        }

        if mode == "static":
            return static_result
        if mode == "parallel":
            return parallel_result

        # Safe RL path
        rl = self._ensure_safe_rl_loaded()
        rl_start = time.time()
        choice = rl.transpile_safe(
            circuit=circuit,
            baseline_static=(static_depth, static_twoq),
            parallel_runner=self.parallel_runner,
        )
        rl_elapsed = time.time() - rl_start
        rl_result = {
            "choice": choice["choice"],
            "circuit": choice["circuit"],
            "depth": choice["depth"],
            "two_qubit_count": choice["twoq"],
            "wall_time_s": rl_elapsed,
            "strategy": choice.get("strategy"),
            "notes": None,
            # Attach comparisons so the frontend can display both baselines and RL.
            "baseline_static": self._strip_circuit(static_result),
            "baseline_parallel": self._strip_circuit(parallel_result),
        }

        if choice["choice"] in {"parallel", "static"}:
            rl_result["notes"] = f"rl fell back to {choice['choice']}"

        return rl_result

    def transpile_circuit(self, qasm: str, mode: str, timeout_s: float | None = None):
        circuit = self._parse_circuit(qasm)
        result = self._transpile_with_mode(circuit, mode)
        return self._strip_circuit(result)

    def _build_noise_model(self, strength: float = 0.01):
        """
        Build a simple depolarizing noise model.

        This is intentionally conservative and parameterized only by a single
        strength value to keep the API simple for the UI.
        """
        if noise is None:
            return None

        try:
            nm = noise.NoiseModel()
            # 1-qubit depolarizing error
            dep1 = noise.depolarizing_error(strength, 1)
            # 2-qubit depolarizing error slightly stronger
            dep2 = noise.depolarizing_error(min(strength * 1.5, 1.0), 2)
            nm.add_all_qubit_quantum_error(dep1, ["x", "y", "z", "h"])
            nm.add_all_qubit_quantum_error(dep2, ["cx", "cz"])
            return nm
        except Exception:
            # If anything goes wrong building the model, fall back gracefully
            return None

    @staticmethod
    def _bloch_vector_from_statevector(statevector, qubit_index: int = 0) -> List[float]:
        """
        Compute Bloch vector (⟨X⟩, ⟨Y⟩, ⟨Z⟩) for a single qubit from a statevector.

        For multi-qubit systems, we trace out all but the selected qubit.
        """
        vec = np.asarray(statevector, dtype=complex)
        dim = vec.size
        if dim == 0 or dim & (dim - 1) != 0:
            # Not a power-of-two dimension; return default |0> Bloch vector
            return [0.0, 0.0, 1.0]

        num_qubits = int(np.log2(dim))
        if qubit_index < 0 or qubit_index >= num_qubits:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid qubit index {qubit_index} for {num_qubits}-qubit state.",
            )

        # Full density matrix
        rho = np.outer(vec, np.conjugate(vec))

        # Reshape for partial trace; axes: q0..qn-1, q0'..qn-1'
        rho = rho.reshape([2] * (2 * num_qubits))

        # Trace out all qubits except target
        remaining = [q for q in range(num_qubits) if q != qubit_index]
        for q in reversed(remaining):
            rho = np.trace(rho, axis1=q, axis2=q + num_qubits)

        rho = rho.reshape((2, 2))

        sx = np.array([[0, 1], [1, 0]], dtype=complex)
        sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sz = np.array([[1, 0], [0, -1]], dtype=complex)

        def exp(pauli):
            return float(np.real(np.trace(rho @ pauli)))

        return [exp(sx), exp(sy), exp(sz)]

    @staticmethod
    def _all_bloch_vectors(statevector) -> List[List[float]]:
        """
        Compute Bloch vectors for all qubits in the statevector.
        """
        vec = np.asarray(statevector, dtype=complex)
        dim = vec.size
        if dim == 0 or dim & (dim - 1) != 0:
            return []
        num_qubits = int(np.log2(dim))
        return [
            TranspileService._bloch_vector_from_statevector(vec, qubit_index=i)
            for i in range(num_qubits)
        ]

    @staticmethod
    def _distribution_from_statevector(statevector) -> Dict[str, float]:
        """Return ideal probabilities over bitstrings from a statevector."""
        vec = np.asarray(statevector, dtype=complex)
        dim = vec.size
        if dim == 0 or dim & (dim - 1) != 0:
            return {}
        num_qubits = int(np.log2(dim))
        probs: Dict[str, float] = {}
        for i, amp in enumerate(vec):
            p = float(np.abs(amp) ** 2)
            if p <= 0.0:
                continue
            bitstring = format(i, f"0{num_qubits}b")
            probs[bitstring] = p
        return probs

    @staticmethod
    def _normalize_counts(counts: Dict[str, int]) -> Dict[str, float]:
        total = sum(int(v) for v in counts.values())
        if total <= 0:
            return {k: 0.0 for k in counts}
        return {k: int(v) / total for k, v in counts.items()}

    @staticmethod
    def _kl_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
        """KL(p || q) with small epsilon regularization."""
        eps = 1e-12
        support = set(p.keys()) | set(q.keys())
        val = 0.0
        for k in support:
            pk = max(p.get(k, 0.0), eps)
            qk = max(q.get(k, 0.0), eps)
            val += pk * np.log(pk / qk)
        return float(val)

    @staticmethod
    def _total_variation_distance(p: Dict[str, float], q: Dict[str, float]) -> float:
        support = set(p.keys()) | set(q.keys())
        return 0.5 * float(sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in support))

    @staticmethod
    def _fidelity(p: Dict[str, float], q: Dict[str, float]) -> float:
        """Classical fidelity between two distributions."""
        support = set(p.keys()) | set(q.keys())
        val = sum(np.sqrt(p.get(k, 0.0) * q.get(k, 0.0)) for k in support)
        return float(val**2)

    def transpile_and_execute(
        self,
        qasm: str,
        mode: str,
        shots: int = 1024,
        timeout_s: float | None = None,
        noise_enabled: bool = False,
        noise_strength: float | None = None,
        noise_metrics: Optional[List[str]] = None,
    ):
        circuit = self._parse_circuit(qasm)

        if shots <= 0 or shots > 16384:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Shots must be between 1 and 16384.",
            )

        if circuit.num_qubits > self.backend.num_qubits:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Circuit has {circuit.num_qubits} qubits, backend supports {self.backend.num_qubits}.",
            )

        if len(circuit.data) > 4000:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Circuit too large (operation count > 4000).",
            )

        # Ideal statevector (without final measurements) for Bloch + ideal distribution.
        # IMPORTANT: we use the *logical* circuit (before backend-specific transpilation)
        # so that ideal behaviour is the same for all modes (static / parallel / safe-rl).
        bloch_vectors: List[List[float]] = []
        ideal_probs: Dict[str, float] = {}
        if Aer is not None:
            try:
                sv_backend = Aer.get_backend("aer_simulator_statevector")
                logical_circ = circuit.remove_final_measurements(inplace=False)
                logical_circ.save_statevector()
                sv_job = sv_backend.run(logical_circ)
                sv_result = sv_job.result()
                statevector = sv_result.get_statevector()
                bloch_vectors = self._all_bloch_vectors(statevector)
                ideal_probs = self._distribution_from_statevector(statevector)
            except Exception as exc:
                logger.warning("Failed to compute statevector / Bloch vector: %s", exc)

        # Now transpile using the selected mode (static / parallel / safe-rl) and execute with (optional) noise.
        transpiled = self._transpile_with_mode(circuit, mode)
        transpiled_circ: QuantumCircuit = transpiled["circuit"]

        # Execute (optionally with noise) to obtain counts.
        exec_start = time.time()
        backend = self._sim_backend()
        run_kwargs = {"shots": shots}

        active_noise_model = None
        effective_noise_strength = float(noise_strength) if noise_strength is not None else 0.01
        if noise_enabled:
            active_noise_model = self._build_noise_model(effective_noise_strength)
            if active_noise_model is not None:
                run_kwargs["noise_model"] = active_noise_model

        job = backend.run(transpiled_circ, **run_kwargs)
        result = job.result()
        counts = result.get_counts()
        exec_elapsed = time.time() - exec_start

        out = self._strip_circuit(transpiled)
        out.update(
            {
                "counts": counts,
                "shots": shots,
                "exec_time_s": exec_elapsed,
            }
        )

        # Attach state information (for Bloch sphere and analysis).
        if bloch_vectors:
            out["state"] = {
                "bloch_vectors": bloch_vectors,
                "num_qubits": transpiled_circ.num_qubits,
            }

        # Attach noise analysis if requested.
        if noise_enabled and counts:
            noisy_probs = self._normalize_counts(counts)
            requested = set(noise_metrics or [])
            metrics_out: Dict[str, float] = {}

            # Always compute basic metrics from the noisy distribution alone.
            if noisy_probs:
                # Shannon entropy (base 2)
                probs_arr = np.array(list(noisy_probs.values()), dtype=float)
                probs_arr = probs_arr[probs_arr > 0]  # avoid log(0)
                if probs_arr.size > 0:
                    entropy = float(-np.sum(probs_arr * np.log2(probs_arr)))
                    metrics_out["noisy_entropy"] = entropy
                metrics_out["noisy_max_probability"] = float(max(noisy_probs.values()))

            # If we have an ideal distribution, compute comparison metrics.
            if ideal_probs:
                tvd = self._total_variation_distance(ideal_probs, noisy_probs)
                kl = self._kl_divergence(ideal_probs, noisy_probs)
                fid = self._fidelity(ideal_probs, noisy_probs)

                metrics_out["total_variation_distance"] = tvd
                metrics_out["kl_divergence"] = kl
                metrics_out["fidelity"] = fid

                # Simple "error rate" proxy: 1 - probability of the ideal most-likely outcome.
                optimal_bit = max(ideal_probs.items(), key=lambda kv: kv[1])[0]
                err_rate = 1.0 - noisy_probs.get(optimal_bit, 0.0)
                metrics_out["dominant_state_error_rate"] = err_rate

            # If the UI requested specific metrics, try to honor that, but never
            # return an empty metrics dict – fall back to all available metrics.
            if requested:
                filtered = {k: v for k, v in metrics_out.items() if k in requested}
                if filtered:
                    metrics_out = filtered

            out["noise"] = {
                "enabled": True,
                "model": "depolarizing",
                "strength": effective_noise_strength,
                "metrics": metrics_out,
            }
        else:
            out["noise"] = {
                "enabled": bool(noise_enabled),
                "model": "depolarizing" if noise_enabled else None,
                "strength": float(noise_strength) if noise_strength is not None else None,
                "metrics": {},
            }

        return out

    @staticmethod
    def _strip_circuit(result: Dict) -> Dict:
        """Remove heavy circuit object; return metadata only."""
        base = {
            "choice": result["choice"],
            "depth": result["depth"],
            "two_qubit_count": result["two_qubit_count"],
            "wall_time_s": result["wall_time_s"],
            "strategy": result.get("strategy"),
            "notes": result.get("notes"),
        }
        # Preserve any comparison metadata (no circuit objects inside).
        for key in ("baseline_static", "baseline_parallel"):
            if key in result:
                base[key] = result[key]
        return base

    def _sim_backend(self):
        if Aer is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="qiskit-aer is not installed. Please install qiskit-aer.",
            )
        return Aer.get_backend("aer_simulator")


transpile_service = TranspileService()
