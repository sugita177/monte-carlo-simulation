"""Unit tests for 2D Ising model simulation engine."""

import numpy as np
import pytest

from models.model_01_ising_2d.src.ising import Ising2D


def test_initialization():
    """スピン初期化の形状、値、磁化のテスト."""
    L = 10
    ising = Ising2D(L=L, temperature=2.0, seed=42)

    # all_up
    ising.initialize_spins(method="all_up")
    assert ising.spins.shape == (L, L)
    assert np.all(ising.spins == 1)
    assert ising.total_magnetization() == L * L

    # all_down
    ising.initialize_spins(method="all_down")
    assert np.all(ising.spins == -1)
    assert ising.total_magnetization() == -L * L

    # random
    ising.initialize_spins(method="random")
    assert np.all(np.isin(ising.spins, [-1, 1]))


def test_ground_state_energy():
    """基底状態（全スピン平行）における厳密なエネルギーのテスト."""
    L = 8
    J = 1.5
    N = L * L
    ising = Ising2D(L=L, temperature=1.0, J=J, h=0.0)

    # 配位数 z = 4, ボンド数 = 2N, E = -2 * J * N
    ising.initialize_spins(method="all_up")
    expected_energy = -2.0 * J * N
    assert np.isclose(ising.total_energy(), expected_energy)

    ising.initialize_spins(method="all_down")
    assert np.isclose(ising.total_energy(), expected_energy)


def test_delta_e_consistency():
    """
    局所エネルギー変化 delta_e が、全エネルギーの実際の差分と厳密に一致するかを検証.
    PBC や係数 2 の正しさを保証する最も重要なテスト.
    """
    L = 6
    ising = Ising2D(L=L, temperature=2.27, J=1.0, h=0.0, seed=123)

    # ランダムな状態で複数回検証
    for _ in range(20):
        ising.initialize_spins(method="random")
        x = ising.rng.integers(0, L)
        y = ising.rng.integers(0, L)

        # 反転前の全エネルギー
        E_before = ising.total_energy()

        # 局所エネルギー変化量 dE の計算
        dE_over_J = ising.delta_e_over_J(x, y)
        dE = ising.delta_e(x, y, dE_over_J)

        # 実際にスピンを反転
        ising.spins[x, y] *= -1

        # 反転後の全エネルギー
        E_after = ising.total_energy()

        # 差分が dE と完全一致することを確認
        assert np.isclose(E_after - E_before, dE), (
            f"delta_e mismatch: calculated dE={dE}, actual difference={E_after - E_before}"
        )


def test_low_temperature_freezing():
    """極低温 (T -> 0) において、基底状態からの反転が棄却される (凍結) ことを検証."""
    L = 8
    ising = Ising2D(L=L, temperature=0.01, J=1.0, h=0.0, seed=42)
    ising.initialize_spins(method="all_up")

    # 1 MCS 実行
    accepted, trials = ising.step_metropolis()

    # 極低温では基底状態からエネルギーが増大する反転 (dE = 8J) は受託確率 ~0
    assert accepted == 0
    assert np.all(ising.spins == 1)


def test_high_temperature_acceptance():
    """超高温 (T -> inf) において、受託確率がほぼ 1 になりスピンが活発に反転することを検証."""
    L = 8
    ising = Ising2D(L=L, temperature=1000.0, J=1.0, h=0.0, seed=42)
    ising.initialize_spins(method="all_up")

    # 1 MCS 実行
    accepted, trials = ising.step_metropolis()

    # 高温では受託確率 exp(-beta * dE) ~ 1 となるため、ほぼすべての反転が受託される
    # (受託率 > 0.9)
    acceptance_rate = accepted / trials
    assert acceptance_rate > 0.8


def test_rust_engine_ground_state_and_limits():
    """Rust 実装 Ising2DRust の基底状態エネルギー、スピン取得、極限挙動の検証."""
    import mc_core

    L = 8
    J = 1.5
    N = L * L

    # 1. 基底状態エネルギー
    rust_ising = mc_core.Ising2DRust(L, 1.0, j=J, seed=42)
    rust_ising.initialize_spins("all_up")
    assert rust_ising.total_magnetization() == N
    assert np.isclose(rust_ising.total_energy(), -2.0 * J * N)

    # NumPy 配列として取得
    spins_np = rust_ising.get_spins()
    assert spins_np.shape == (L, L)
    assert np.all(spins_np == 1)

    # 2. 極低温 (T -> 0) 凍結
    cold_ising = mc_core.Ising2DRust(L, 0.01, j=1.0, seed=42)
    cold_ising.initialize_spins("all_up")
    accepted, trials = cold_ising.step_metropolis()
    assert accepted == 0
    assert trials == N

    # 3. 超高温 (T -> inf) 受託率
    hot_ising = mc_core.Ising2DRust(L, 1000.0, j=1.0, seed=42)
    hot_ising.initialize_spins("all_up")
    accepted, trials = hot_ising.step_metropolis()
    assert (accepted / trials) > 0.8

