"""2次元 q状態クロック模型のシミュレーション制御モジュール (Rust コアエンジン連携)."""

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

from mc_core import Clock2DRust
from models.model_05_clock_model.src.observables import (
    ClockObservables,
    ClockSimulationResult,
)


class ClockSimulation:
    """クロック模型の MCMC シミュレーション制御クラス."""

    def __init__(
        self,
        lattice_size: int,
        q: int,
        temperature: float,
        j: float = 1.0,
        mode: Literal["random", "adjacent"] = "adjacent",
        seed: int | None = None,
        initial: Literal["random", "ground"] = "random",
    ) -> None:
        self.L = lattice_size
        self.q = q
        self.temperature = temperature
        self.beta = 1.0 / temperature if temperature > 0 else 1e9
        self.mode = mode

        # Rust コアエンジンのインスタンス化
        self.core = Clock2DRust(
            l=lattice_size,
            q=q,
            j=j,
            seed=seed,
            initial=initial,
        )

    def thermalize(self, steps: int) -> None:
        """指定ステップ数だけ熱平衡化を実行する."""
        for _ in range(steps):
            self.core.step_metropolis(self.beta, self.mode)

    def run(
        self,
        mcs_measure: int,
        sample_interval: int = 2,
    ) -> ClockSimulationResult:
        """サンプリング計測ループを実行する.

        【学習用 TODO】:
        以下の手順でサンプリングを行い、ClockSimulationResult を返してください。

        手順:
        1. ClockObservables(self.L, self.q, self.temperature) を初期化
        2. mcs_measure ステップのループを回す:
           - self.core.step_metropolis(self.beta, self.mode) を実行し、受託数 accepted を取得
           - ステップが sample_interval の倍数であれば:
             - energy = self.core.total_energy()
             - mx, my, norm_m = self.core.magnetization()
             - obs.record(energy, mx, my, norm_m, accepted) で記録
        3. obs.compute_results() を呼び出して結果を返す
        """
        obs = ClockObservables(self.L, self.q, self.temperature)

        for step in range(mcs_measure):
            accepted = self.core.step_metropolis(self.beta, self.mode)
            if step % sample_interval == 0:
                energy = self.core.total_energy()
                mx, my, norm_m = self.core.magnetization()
                obs.record(energy, mx, my, norm_m, accepted)

        return obs.compute_results()


def run_single_temperature(
    lattice_size: int,
    q: int,
    temperature: float,
    mcs_thermalize: int = 2000,
    mcs_measure: int = 5000,
    sample_interval: int = 2,
    mode: Literal["random", "adjacent"] = "adjacent",
    j: float = 1.0,
    seed: int | None = None,
) -> ClockSimulationResult:
    """単一温度における熱平衡化とサンプリング計測のヘルパー関数."""
    sim = ClockSimulation(
        lattice_size=lattice_size,
        q=q,
        temperature=temperature,
        j=j,
        mode=mode,
        seed=seed,
    )
    sim.thermalize(mcs_thermalize)
    return sim.run(mcs_measure=mcs_measure, sample_interval=sample_interval)
