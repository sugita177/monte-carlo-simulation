"""Observables measurement and statistical accumulation for 2D Ising model."""

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
    binder_cumulant: float  # U_4 = 1 - <M^4> / (3 * <M^2>^2)
    num_samples: int


class ObservableAccumulator:
    """モンテカルロシミュレーションの観測量を記録・集計するアキュムレータ."""

    def __init__(self, L: int, temperature: float) -> None:
        self.L = L
        self.N = L * L
        self.temperature = temperature

        # 時系列記録用リスト
        self.energies: list[float] = []
        self.magnetizations: list[int] = []

    def record(self, energy: float, magnetization: int) -> None:
        """1 回のサンプリング測定値を記録する."""
        self.energies.append(energy)
        self.magnetizations.append(magnetization)

    @property
    def sample_count(self) -> int:
        return len(self.energies)

    def compute_results(self) -> SimulationResult:
        """蓄積された時系列データから統計量（平均・比熱・帯磁率・Binder比）を算出する."""
        if self.sample_count == 0:
            raise ValueError("No samples have been recorded yet.")

        E = np.array(self.energies, dtype=np.float64)
        M = np.array(self.magnetizations, dtype=np.float64)
        abs_M = np.abs(M)
        T = self.temperature
        N = self.N

        # 各統計量を NumPy の mean 等を用いて計算する
        # 以下の計算のためにモーメントを求めておく
        mean_E = np.mean(E)
        mean_E2 = np.mean(E**2)
        mean_abs_M = np.mean(abs_M)
        mean_M2 = np.mean(M**2)
        mean_M4 = np.mean(M**4)
        
        # 1. 1スピンあたりの平均エネルギー: mean_e = <E> / N
        mean_e = mean_E / N

        # 2. 1スピンあたりの平均絶対値磁化: mean_m = <|M|> / N
        mean_m = mean_abs_M / N

        # 3. 比熱: specific_heat = (<E^2> - <E>^2) / (N * T^2)
        specific_heat = (mean_E2 - mean_E**2) / (N * T**2)

        # 4. 帯磁率: susceptibility = (<M^2> - <|M|>^2) / (N * T)
        susceptibility = (mean_M2 - mean_abs_M**2) / (N * T)

        # 5. ビンダー比: binder_cumulant = 1 - <M^4> / (3 * <M^2>^2)
        eps = 1e-15 # ゼロ除算対策
        binder_cumulant = 1.0 - mean_M4 / (3.0 * (mean_M2**2) + eps)

        return SimulationResult(
            temperature=T,
            L=self.L,
            N=N,
            mean_energy_per_spin=mean_e,
            mean_magnetization_per_spin=mean_m,
            specific_heat=specific_heat,
            susceptibility=susceptibility,
            binder_cumulant=binder_cumulant,
            num_samples=self.sample_count,
        )
