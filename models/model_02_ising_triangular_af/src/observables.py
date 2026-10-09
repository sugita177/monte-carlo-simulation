"""三角格子反強磁性イジング模型の観測量計測および統計集計アキュムレータ."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SimulationResult:
    """物理量の集計結果を保持するデータクラス."""

    temperature: float
    L: int
    N: int
    mean_energy_per_spin: float  # <e> = <E> / N
    mean_magnetization_per_spin: float  # <|m|> = <|M|> / N
    specific_heat: float  # C_V = (<E^2> - <E>^2) / (N * T^2)
    susceptibility: float  # chi = (<M^2> - <|M|>^2) / (N * T)
    num_samples: int


class ObservableAccumulator:
    """モンテカルロシミュレーションの観測量を記録・集計するアキュムレータ."""

    def __init__(self, L: int, temperature: float) -> None:
        self.L = L
        self.N = L * L
        self.temperature = temperature

        # 時系列記録用リスト
        self.energies: list[float] = []
        self.magnetizations: list[float] = []

    def record(self, energy: float, magnetization: float) -> None:
        """1 回のサンプリング測定値を記録する."""
        self.energies.append(energy)
        self.magnetizations.append(magnetization)

    @property
    def sample_count(self) -> int:
        return len(self.energies)

    def compute_results(self) -> SimulationResult:
        """蓄積された時系列データから熱力学的平均量を算出する."""
        if self.sample_count == 0:
            raise ValueError("サンプルが記録されていません。")

        E = np.array(self.energies, dtype=np.float64)
        M = np.array(self.magnetizations, dtype=np.float64)
        N = float(self.N)
        T = self.temperature

        # 1サイトあたりの平均エネルギー <e>
        mean_E = float(np.mean(E))
        mean_e = mean_E / N

        # 1サイトあたりの平均絶対磁化 <|m|>
        abs_M = np.abs(M)
        mean_abs_m = float(np.mean(abs_M)) / N

        # 比熱 C_V = (<E^2> - <E>^2) / (N * T^2)
        var_E = float(np.var(E, ddof=0))
        c_v = var_E / (N * (T**2)) if T > 0.0 else 0.0

        # 帯磁率 chi = (<M^2> - <|M|>^2) / (N * T)
        var_abs_M = float(np.var(abs_M, ddof=0))
        chi = var_abs_M / (N * T) if T > 0.0 else 0.0

        return SimulationResult(
            temperature=T,
            L=self.L,
            N=self.N,
            mean_energy_per_spin=mean_e,
            mean_magnetization_per_spin=mean_abs_m,
            specific_heat=c_v,
            susceptibility=chi,
            num_samples=self.sample_count,
        )
