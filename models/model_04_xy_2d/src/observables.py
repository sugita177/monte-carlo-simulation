"""2次元XY模型の観測量計測および統計集計アキュムレータ."""

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
    mean_magnetization: float  # <|m|>
    specific_heat: float  # C_V = (<E^2> - <E>^2) / (N * T^2)
    vortex_density: float  # <rho_v> = <(Σ |q_p|) / N>
    num_samples: int


class ObservableAccumulator:
    """モンテカルロシミュレーションの観測量を記録・集計するアキュムレータ."""

    def __init__(self, L: int, temperature: float) -> None:
        self.L = L
        self.N = L * L
        self.temperature = temperature

        # 時系列サンプリング記録用リスト
        self.energies: list[float] = []
        self.magnetizations: list[float] = []
        self.vortex_densities: list[float] = []

    def record(self, energy: float, magnetization: float, vortex_density: float) -> None:
        """1 回のサンプリング測定値を記録する."""
        self.energies.append(energy)
        self.magnetizations.append(magnetization)
        self.vortex_densities.append(vortex_density)

    @property
    def sample_count(self) -> int:
        return len(self.energies)

    def compute_results(self) -> SimulationResult:
        """蓄積された時系列データから熱力学的平均量を算出する.

        以下の統計量を計算して SimulationResult を返す。

        1. 1サイトあたりの平均エネルギー <e> = mean(E) / N
        2. 平均磁化ノルム <|m|> = mean(M)
        3. 比熱 C_V:
           エネルギーのゆらぎ公式:
               C_V = (<E^2> - <E>^2) / (N * T^2) = Var(E) / (N * T^2)
           ※ T = 0 の場合はゼロ割りを防ぐため 0.0 とする。
        4. 平均渦度密度 <rho_v> = mean(vortex_densities)
        """
        if self.sample_count == 0:
            raise ValueError("サンプルが記録されていません。")

        E = np.array(self.energies, dtype=np.float64)
        M = np.array(self.magnetizations, dtype=np.float64)
        V = np.array(self.vortex_densities, dtype=np.float64)
        N = float(self.N)
        T = self.temperature

        mean_e = np.mean(E) / N      # 1サイトあたり平均エネルギー
        mean_m = np.mean(M)          # 平均磁化
        c_v = np.var(E) / (N * T**2) # 比熱 C_V
        mean_rho_v = np.mean(V)      # 平均渦度密度

        return SimulationResult(
            temperature=T,
            L=self.L,
            N=self.N,
            mean_energy_per_spin=mean_e,
            mean_magnetization=mean_m,
            specific_heat=c_v,
            vortex_density=mean_rho_v,
            num_samples=self.sample_count,
        )
