"""2D Square Lattice Ising Model Simulation Engine."""

from __future__ import annotations

from typing import Literal
import numpy as np



class Ising2D:
    """2次元正方格子イジング模型のシミュレーションエンジン."""

    def __init__(
        self,
        L: int,
        temperature: float,
        J: float = 1.0,
        h: float = 0.0,
        seed: int | None = None,
    ) -> None:
        """
        Parameters
        ----------
        L : int
            正方格子の1辺の長さ (全スピン数 N = L * L)
        temperature : float
            系の温度 T (k_B = 1)
        J : float
            交換相互作用定数 (デフォルト 1.0: 強磁性)
        h : float
            外部磁場 (デフォルト 0.0)
        seed : int | None
            乱数シード
        """
        self.L = L
        self.N = L * L
        self.temperature = temperature
        self.beta = 1.0 / temperature if temperature > 0 else float("inf")
        self.J = J
        self.h = h
        self.rng = np.random.default_rng(seed)

        # スピン配位配列 (形状: L x L, 要素: int8)
        self.spins = np.zeros((L, L), dtype=np.int8)

        # 確率ルックアップテーブルの準備（後述）
        self._init_lookup_table()

    def initialize_spins(self, method: Literal["random", "all_up", "all_down"] = "random") -> None:
        """スピン配位を初期化する."""
        if method == "random":
            # 各スピンを +1 または -1 に一様ランダムに初期化
            self.spins = self.rng.choice(np.array([1, -1], dtype=np.int8), size=(self.L, self.L))
        elif method == "all_up":
            # 全スピンを +1 (低温完全強磁性) に初期化
            self.spins.fill(1)
        elif method == "all_down":
            # 全スピンを -1 に初期化
            self.spins.fill(-1)
        else:
            raise ValueError(f"Unknown initialization method: {method}")

    def _init_lookup_table(self) -> None:
        """
        メトロポリス受託確率の事前計算テーブル.
        h = 0 のとき dE / J は {-8, -4, 0, 4, 8} のみを取り得るため、
        指数関数 exp(-beta * dE) を毎回計算せずテーブル化しておく.
        """
        # 必要な dE に対する exp(-beta * dE) を事前計算して辞書などに保持
        self.exp_table: dict[int, float] = {
            de_over_j: np.exp(-self.beta * self.J * de_over_j) for de_over_j in [4, 8]
        }

    def delta_e_over_J(self, x: int, y: int) -> float:
        """
        格子点 (x, y) のスピンを反転させたときの相互作用に関する局所エネルギー変化 dE/J を計算する.
        周期境界条件 (PBC) を考慮すること.
        """
        # 1. サイト (x, y) の現在のスピン s を取得
        # 2. 上・下・左・右の最近接 4 スピンの和 (sum_neighbors) を計算 (PBC 考慮)
        # 3. 理論ドキュメントで導出した dE = 2 * s * (J * sum_neighbors + h) を返す
        s = self.spins[x, y]
        sum_neighbors = (
            self.spins[(x + 1) % self.L, y]  # 右
            + self.spins[(x - 1) % self.L, y]  # 左
            + self.spins[x, (y + 1) % self.L]  # 上
            + self.spins[x, (y - 1) % self.L]  # 下
        )
        return 2 * s * sum_neighbors

    def delta_e(self, x: int, y: int, dE_over_J: float) -> float:
        """
        格子点 (x, y) のスピンを反転させたときの局所エネルギー変化 dE を計算する.
        """
        return dE_over_J * self.J + 2 * self.h * self.spins[x, y]

    def step_metropolis(self) -> tuple[int, int]:
        """
        1 モンテカルロステップ (1 MCS = N 回の局所スピン反転試行) を実行する.

        Returns
        -------
        accepted : int
            受託されたスピン反転の回数
        trials : int
            試行回数 (= N)
        """
        accepted = 0
        # N 回のループ:
        for _ in range(self.N):
            # 1. ランダムな格子点 (x, y) を選ぶ
            x = self.rng.integers(0, self.L)
            y = self.rng.integers(0, self.L)

            # 2. delta_e(x, y) を計算する
            dE_over_J = self.delta_e_over_J(x, y)
            dE = self.delta_e(x, y, dE_over_J)

            # 3. メトロポリス判定 (dE <= 0 または乱数 r < exp(-beta * dE))
            if dE <= 0 or self.rng.random() < self.exp_table.get(dE_over_J, 0.0):
                # 4. 受託されたらスピンを反転 (spins[x, y] = -spins[x, y]) し、accepted += 1
                self.spins[x, y] *= -1
                accepted += 1

        return accepted, self.N

    def total_energy(self) -> float:
        """系全体の全ハミルトニアン H を厳密に計算する (PBC 考慮)."""
        # 右隣との相互作用
        right_spins = np.roll(self.spins, shift=-1, axis=0)
        # 下隣との相互作用
        down_spins = np.roll(self.spins, shift=-1, axis=1)

        # 相互作用項（二重カウントなし）
        interaction_energy = -self.J * (
            np.sum(self.spins * right_spins) + np.sum(self.spins * down_spins)
        )

        # 外部磁場項
        field_energy = -self.h * np.sum(self.spins) if self.h != 0 else 0.0

        return float(interaction_energy + field_energy)
        

    def total_magnetization(self) -> int:
        """系全体の全磁化 M = sum(s_i) を計算する."""
        # 全スピンの総和
        return int(np.sum(self.spins))

    def get_spins(self) -> np.ndarray:
        """現在のスピン配位 (L, L) を返す (Rust コアエンジンとの互換用)."""
        return self.spins
