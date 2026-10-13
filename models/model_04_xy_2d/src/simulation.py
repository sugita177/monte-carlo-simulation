"""2次元XY模型のシミュレーション実行制御モジュール (Rust コアエンジン連携)."""

from __future__ import annotations

import sys
from pathlib import Path

# プロジェクトルートのパス解決
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import Literal
import numpy as np
from tqdm import tqdm

from mc_core import XY2DRust
from models.model_04_xy_2d.src.observables import ObservableAccumulator, SimulationResult


def run_single_temperature(
    lattice_size: int,
    temperature: float,
    mcs_thermalize: int = 5000,
    mcs_measure: int = 10000,
    sample_interval: int = 5,
    algorithm: Literal["wolff", "metropolis"] = "wolff",
    j: float = 1.0,
    delta_theta: float = 1.0,
    initial_thetas: np.ndarray | None = None,
    seed: int | None = None,
) -> tuple[SimulationResult, np.ndarray]:
    """指定された温度で熱平衡化とサンプリング計測を実行する.

    Args:
        lattice_size: 格子の一辺の長さ L
        temperature: 温度 T
        mcs_thermalize: 平衡化のためのモンテカルロステップ数
        mcs_measure: サンプリング計測を行うモンテカルロステップ数
        sample_interval: 観測量を記録する間隔 (自己相関を低減するため)
        algorithm: "wolff" または "metropolis"
        j: 強磁性相互作用定数 J
        delta_theta: Metropolis 提案変位の最大幅
        initial_thetas: 初期スピン角度配位 (アニーリング用)
        seed: 乱数シード

    Returns:
        (SimulationResult, 最終スピン角度配位)
    """
    sim = XY2DRust(
        l=lattice_size,
        temperature=temperature,
        j=j,
        delta_theta=delta_theta,
        seed=seed,
    )

    if initial_thetas is not None:
        thetas_c = np.ascontiguousarray(initial_thetas, dtype=np.float64)
        sim.set_thetas(thetas_c)

    n_spins = lattice_size * lattice_size

    def step_one_mcs() -> None:
        """1 モンテカルロステップ (MCS) を進める.

        - algorithm が "metropolis" の場合:
          sim.step_metropolis() を 1 回呼ぶ (内部で N 回の局所更新試行が行われます)。

        - algorithm が "wolff" の場合:
          Wolff クラスター法は 1 回のクラスターステップで反転するスピン数が変動する。
          1 MCS (約 N 個のスピン更新) に相当させるため、
          反転したスピン数 (sim.step_wolff() の戻り値) の合計が n_spins 以上になるまで
          sim.step_wolff() を繰り返し呼び出す。
        """
        if algorithm == "metropolis":
            sim.step_metropolis()
        elif algorithm == "wolff":
            sim.step_wolff_mcs()
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    # 1. 熱平衡化 (Thermalization)
    for _ in range(mcs_thermalize):
        step_one_mcs()

    # 2. サンプリング計測 (Measurement)
    accumulator = ObservableAccumulator(L=lattice_size, temperature=temperature)
    for step in range(mcs_measure):
        step_one_mcs()

        # sample_interval ごとに観測量を計測して accumulator に記録する。
        # 記録する値:
        #   - energy: sim.total_energy()
        #   - magnetization: sim.magnetization()
        #   - vortex_density: sim.vortex_density()
        if step % sample_interval == 0:
            accumulator.record(
                energy=sim.total_energy(),
                magnetization=sim.magnetization(),
                vortex_density=sim.vortex_density(),
            )

    return accumulator.compute_results(), sim.get_thetas()


def run_temperature_sweep(
    lattice_size: int,
    temperatures: np.ndarray,
    mcs_thermalize: int = 5000,
    mcs_measure: int = 10000,
    sample_interval: int = 5,
    algorithm: Literal["wolff", "metropolis"] = "wolff",
    j: float = 1.0,
    delta_theta: float = 1.0,
    seed: int | None = 42,
) -> list[SimulationResult]:
    """高温から低温への徐冷 (Cooling Sweep) により温度依存性をシミュレーションする."""
    # 高温から低温へとソート (徐冷アニーリング)
    temps_descending = np.sort(temperatures)[::-1]
    results: list[SimulationResult] = []

    last_thetas: np.ndarray | None = None
    rng = np.random.default_rng(seed)

    for T in tqdm(temps_descending, desc=f"XY Sweep (L={lattice_size}, {algorithm})"):
        step_seed = int(rng.integers(0, 2**31 - 1)) if seed is not None else None
        res, last_thetas = run_single_temperature(
            lattice_size=lattice_size,
            temperature=float(T),
            mcs_thermalize=mcs_thermalize,
            mcs_measure=mcs_measure,
            sample_interval=sample_interval,
            algorithm=algorithm,
            j=j,
            delta_theta=delta_theta,
            initial_thetas=last_thetas,
            seed=step_seed,
        )
        results.append(res)

    # 低温順 (昇順) に戻して返す
    results.sort(key=lambda r: r.temperature)
    return results
