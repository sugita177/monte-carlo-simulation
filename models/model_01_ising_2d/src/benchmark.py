import sys
import time
from pathlib import Path

# プロジェクトルートを sys.path に追加（直接実行対応）
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import mc_core
from models.model_01_ising_2d.src.ising import Ising2D


def run_benchmark(L: int = 50, mcs: int = 1000, temperature: float = 2.27) -> None:
    print("=" * 60)
    print("Ising 2D Engine Benchmark: Python vs Rust")
    print(f"Lattice size: {L} x {L} (N = {L*L})")
    print(f"Temperature: {temperature} (near Tc)")
    print(f"Steps: {mcs} MCS (Total {mcs * L * L:,} spin flip trials)")
    print("=" * 60)

    # ----------------------------------------------------
    # 1. Python Engine
    # ----------------------------------------------------
    print("\n[1/2] Running Python Engine ...")
    py_engine = Ising2D(L=L, temperature=temperature, seed=42)
    py_engine.initialize_spins(method="random")

    start_py = time.perf_counter()
    for _ in range(mcs):
        py_engine.step_metropolis()
    time_py = time.perf_counter() - start_py
    print(f"Python Engine: {time_py:.3f} seconds ({mcs / time_py:.1f} MCS/s)")

    # ----------------------------------------------------
    # 2. Rust Engine
    # ----------------------------------------------------
    print("\n[2/2] Running Rust Engine ...")
    rust_engine = mc_core.Ising2DRust(L, temperature, seed=42)
    rust_engine.initialize_spins("random")

    start_rust = time.perf_counter()
    for _ in range(mcs):
        rust_engine.step_metropolis()
    time_rust = time.perf_counter() - start_rust
    print(f"Rust Engine:   {time_rust:.4f} seconds ({mcs / time_rust:.1f} MCS/s)")

    # ----------------------------------------------------
    # 3. 速度比較サマリー
    # ----------------------------------------------------
    speedup = time_py / time_rust
    print("\n" + "=" * 60)
    print(f"★ Speedup Result: Rust is {speedup:.1f}x FASTER than pure Python!")
    print("=" * 60)


if __name__ == "__main__":
    run_benchmark()
