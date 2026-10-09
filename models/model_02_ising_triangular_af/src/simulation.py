"""三角格子反強磁性イジング模型のシミュレーション実行スクリプト."""

from __future__ import annotations

import sys
from pathlib import Path

# プロジェクトルートを sys.path に追加（直接実行対応）
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
from mc_core import IsingTriangularRust
from tqdm import tqdm

from models.model_02_ising_triangular_af.src.observables import (
    ObservableAccumulator,
    SimulationResult,
)


def run_single_temperature(
    lattice_size: int,
    temperature: float,
    j_af: float = 1.0,
    mcs_thermalize: int = 2000,
    mcs_measure: int = 5000,
    sample_interval: int = 2,
    initial_spins: np.ndarray | None = None,
    seed: int | None = None,
) -> tuple[SimulationResult, np.ndarray]:
    """指定された温度で熱平衡化とサンプリング計測を実行する.

    Returns:
        (SimulationResult, 最終スピン配位)
    """
    sim = IsingTriangularRust(
        l=lattice_size, temperature=temperature, j_af=j_af, seed=seed
    )

    # 前の温度のスピン配位を引き継ぐ場合、初期化を行う
    if initial_spins is not None:
        spins_c = np.ascontiguousarray(initial_spins, dtype=np.int8)
        sim.set_spins(spins_c)

    # 1. 熱平衡化 (thermalization)
    for _ in range(mcs_thermalize):
        sim.step_metropolis()

    # 2. サンプリング計測
    accumulator = ObservableAccumulator(L=lattice_size, temperature=temperature)
    for step in range(mcs_measure):
        sim.step_metropolis()
        if step % sample_interval == 0:
            accumulator.record(
                energy=sim.total_energy(), magnetization=sim.magnetization()
            )

    return accumulator.compute_results(), sim.get_spins()


def run_temperature_sweep(
    lattice_size: int,
    temperatures: np.ndarray,
    j_af: float = 1.0,
    mcs_thermalize: int = 2000,
    mcs_measure: int = 5000,
    sample_interval: int = 2,
    seed: int | None = None,
) -> list[SimulationResult]:
    """温度グリッドに沿ってスイープ計算を実行する."""

    # 高温から低温にソート（徐冷アニーリング）
    temps_descending = np.sort(temperatures)[::-1]

    # 高温から低温への徐冷 (Cooling) ループを実装
    results_dict: dict[float, SimulationResult] = {}
    sim = IsingTriangularRust(
        l=lattice_size, temperature=temps_descending[0], j_af=j_af, seed=seed
    )

    iterator = (
        tqdm(temps_descending, desc=f"Cooling sweep (L={lattice_size})")
    )

    for T in iterator:
        T_val = float(T)
        sim.set_temperature(T_val)

        # 熱平衡化
        for _ in range(mcs_thermalize):
            sim.step_metropolis()

        # サンプリング計測
        accumulator = ObservableAccumulator(L=lattice_size, temperature=T_val)
        for step in range(mcs_measure):
            sim.step_metropolis()
            if step % sample_interval == 0:
                accumulator.record(
                    energy=sim.total_energy(), magnetization=sim.magnetization()
                )
        results_dict[T_val] = accumulator.compute_results()

    # 解析やプロットが扱いやすいよう、元の昇順 (低温 -> 高温) で結果を整列して返却
    return [results_dict[float(T)] for T in np.sort(temperatures)]
