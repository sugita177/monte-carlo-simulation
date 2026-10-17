"""2次元 q状態クロック模型の観測量計測および統計集計アキュムレータ."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class ClockSimulationResult:
    """クロック模型の物理量集計結果を保持するデータクラス."""

    temperature: float
    L: int
    N: int
    q: int
    mean_energy_per_spin: float  # <e> = <E> / N
    mean_magnetization: float  # <|m|>
    specific_heat: float  # C_v = Var(E) / (N * T^2)
    susceptibility: float  # χ = (N / T) * Var(|m|)
    clock_orientation: float  # Φ_q = <cos(q * Ψ)> (Ψ = atan2(my, mx))
    acceptance_rate: float  # メトロポリス受託率
    num_samples: int


class ClockObservables:
    """クロック模型モンテカルロシミュレーションの観測量集計クラス."""

    def __init__(self, L: int, q: int, temperature: float) -> None:
        self.L = L
        self.N = L * L
        self.q = q
        self.temperature = temperature

        # 時系列サンプリング記録用リスト
        self.energies: list[float] = []
        self.mx_list: list[float] = []
        self.my_list: list[float] = []
        self.magnetizations: list[float] = []
        self.accepted_counts: list[int] = []

    def record(
        self,
        energy: float,
        mx: float,
        my: float,
        norm_m: float,
        accepted: int,
    ) -> None:
        """1 回のサンプリング測定値を記録する."""
        self.energies.append(energy)
        self.mx_list.append(mx)
        self.my_list.append(my)
        self.magnetizations.append(norm_m)
        self.accepted_counts.append(accepted)

    @property
    def sample_count(self) -> int:
        return len(self.energies)

    def compute_results(self) -> ClockSimulationResult:
        """蓄積された時系列データから熱力学的平均量を算出する.

        以下の統計量を算出して ClockSimulationResult を返すロジックを実装する。

        1. 1サイトあたりの平均エネルギー: mean_energy_per_spin = mean(energies) / N
        2. 平均磁化ノルム: mean_m = mean(magnetizations)
        3. 比熱 C_v:
           ゆらぎの公式: C_v = Var(energies) / (N * T^2)
           ※ T <= 0 の場合はゼロ割りを防ぐため 0.0 とする。
        4. 帯磁率 χ:
           ゆらぎの公式: χ = (N / T) * Var(magnetizations)
           ※ T <= 0 の場合は 0.0 とする。
        5. クロック配向秩序変数 Φ_q:
           各サンプルの磁化ベクトル (mx, my) の配向角 Ψ = arctan2(my, mx) に対し、
           cos(q * Ψ) の平均値:
           Φ_q = mean(cos(q * arctan2(my, mx)))
           - 低温強磁性相 (T < T_c1): 特定の離散軸に固定されるため Φ_q ≈ 1
           - 中間 BKT 相 (T_c1 < T < T_c2): 擬似連続 U(1) により Ψ が等方にゆらぎ Φ_q ≈ 0
        6. 平均受託率: acceptance_rate = sum(accepted_counts) / (sample_count * N)
        """
        if self.sample_count == 0:
            raise ValueError("No samples have been recorded yet.")

        mean_energy_per_spin = np.mean(self.energies) / self.N
        mean_magnetization = np.mean(self.magnetizations)
        specific_heat = 0.0 if self.temperature <= 0.0 else np.var(self.energies, ddof=0) / (self.N * self.temperature**2)
        susceptibility = 0.0 if self.temperature <= 0.0 else self.N * np.var(self.magnetizations, ddof=0) / self.temperature
        clock_orientation = np.mean(np.cos(self.q * np.arctan2(self.my_list, self.mx_list)))
        acceptance_rate = float(np.mean(self.accepted_counts)) / self.N

        return ClockSimulationResult(
            temperature=self.temperature,
            L=self.L,
            N=self.N,
            q=self.q,
            mean_energy_per_spin=mean_energy_per_spin,
            mean_magnetization=mean_magnetization,
            specific_heat=specific_heat,
            susceptibility=susceptibility,
            clock_orientation=clock_orientation,
            acceptance_rate=acceptance_rate,
            num_samples=self.sample_count,
        )
        
        
