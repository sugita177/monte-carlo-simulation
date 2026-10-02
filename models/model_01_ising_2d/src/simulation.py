"""Simulation runner orchestrating Ising2D engine and ObservableAccumulator."""

from __future__ import annotations

from typing import Literal

from models.model_01_ising_2d.src.ising import Ising2D
from models.model_01_ising_2d.src.observables import ObservableAccumulator, SimulationResult


def run_simulation(
    L: int,
    temperature: float,
    mcs_thermalize: int = 1000,
    mcs_measure: int = 5000,
    sample_interval: int = 5,
    J: float = 1.0,
    h: float = 0.0,
    init_method: Literal["random", "all_up", "all_down"] = "random",
    engine_type: Literal["rust", "python"] = "rust",
    seed: int | None = None,
) -> SimulationResult:
    """
    指定されたパラメータで 2次元イジング模型の MCMC シミュレーションを実行する.

    Parameters
    ----------
    L : int
        格子の一辺 (全スピン数 N = L * L)
    temperature : float
        系の温度 T (k_B = 1)
    mcs_thermalize : int
        熱平衡化（Burn-in）のステップ数（データ破棄）
    mcs_measure : int
        測定を行うステップ数
    sample_interval : int
        自己相関を低減するためのサンプリング間隔 (MCS)
    J : float
        交換相互作用定数
    h : float
        外部磁場
    init_method : str
        初期配位 ("random", "all_up", "all_down")
    seed : int | None
        乱数シード

    Returns
    -------
    SimulationResult
        統計集計された物理量結果
    """
    # 1. エンジンとアキュムレータの初期化
    if engine_type == "rust":
        import mc_core
        ising = mc_core.Ising2DRust(L, temperature, j=J, h=h, seed=seed)
    elif engine_type == "python":
        ising = Ising2D(L=L, temperature=temperature, J=J, h=h, seed=seed)
    else:
        raise ValueError(f"Unknown engine_type: {engine_type}")

    ising.initialize_spins(method=init_method if engine_type == "python" else init_method)
    accumulator = ObservableAccumulator(L=L, temperature=temperature)

    # 2. 初期熱平衡化（Burn-in: 計測は行わない）
    for _ in range(mcs_thermalize):
        ising.step_metropolis()

    # 3. 本測定（サンプリング間隔ごとに物理量を記録）
    for mcs in range(mcs_measure):
        ising.step_metropolis()
        if mcs % sample_interval == 0:
            E = ising.total_energy()
            M = ising.total_magnetization()
            accumulator.record(energy=E, magnetization=M)

    # 4. 統計結果の算出
    return accumulator.compute_results()
