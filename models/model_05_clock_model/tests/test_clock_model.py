"""2次元 q状態クロック模型の Python 統合テスト."""

from __future__ import annotations

import math
import numpy as np
import pytest

from mc_core import Clock2DRust
from models.model_05_clock_model.src.observables import ClockObservables
from models.model_05_clock_model.src.simulation import ClockSimulation, run_single_temperature


def test_rust_core_binding() -> None:
    """Rust コアエンジンの基本動作と NumPy 配列連携を検証."""
    L = 6
    q = 6
    sim = Clock2DRust(L, q, j=1.0, seed=42, initial="random")

    # 初期エネルギーと磁化
    e = sim.total_energy()
    assert isinstance(e, float)
    mx, my, norm_m = sim.magnetization()
    assert 0.0 <= norm_m <= 1.0001

    # NumPy 配列としてのスピン状態取得
    spins = sim.get_spins()
    assert spins.shape == (L, L)
    assert spins.dtype == np.uint8
    assert np.all(spins < q)

    # NumPy 配列としての連続角度取得
    angles = sim.get_angles()
    assert angles.shape == (L, L)
    assert angles.dtype == np.float64
    assert np.all(angles >= 0.0)
    assert np.all(angles < 2.0 * math.pi)


def test_observables_calculation() -> None:
    """ClockObservables の統計量算出ロジックを既知データで検証."""
    L = 4
    N = L * L
    q = 4
    T = 2.0
    obs = ClockObservables(L=L, q=q, temperature=T)

    # 既知のダミーサンプルを記録
    # サンプル1: E = -20.0, mx = 1.0, my = 0.0 (|m| = 1.0), accepted = 10
    # サンプル2: E = -30.0, mx = 0.0, my = 1.0 (|m| = 1.0), accepted = 6
    obs.record(energy=-20.0, mx=1.0, my=0.0, norm_m=1.0, accepted=10)
    obs.record(energy=-30.0, mx=0.0, my=1.0, norm_m=1.0, accepted=6)

    res = obs.compute_results()

    # 1. 平均エネルギー: (-20 + -30) / (2 * 16) = -50 / 32 = -1.5625
    assert math.isclose(res.mean_energy_per_spin, -1.5625, rel_tol=1e-6)

    # 2. 平均磁化: 1.0
    assert math.isclose(res.mean_magnetization, 1.0, rel_tol=1e-6)

    # 3. 比熱: Var(E) = [(-20 - -25)^2 + (-30 - -25)^2] / 2 = 25.0
    # C_v = 25.0 / (16 * 2.0^2) = 25.0 / 64.0 = 0.390625
    assert math.isclose(res.specific_heat, 0.390625, rel_tol=1e-6)

    # 4. 磁化ゆらぎ: Var(|m|) = 0.0 なので χ = 0.0
    assert math.isclose(res.susceptibility, 0.0, abs_tol=1e-6)

    # 5. クロック配向秩序度:
    # サンプル1: Ψ = 0 -> cos(4 * 0) = 1.0
    # サンプル2: Ψ = π/2 -> cos(4 * π/2) = cos(2π) = 1.0
    # 平均: 1.0
    assert math.isclose(res.clock_orientation, 1.0, rel_tol=1e-6)

    # 6. 受託率: (10 + 6) / (2 * 16) = 16 / 32 = 0.5
    assert math.isclose(res.acceptance_rate, 0.5, rel_tol=1e-6)


def test_simulation_run() -> None:
    """ClockSimulation のサンプリング実行を検証."""
    L = 6
    q = 6
    sim = ClockSimulation(lattice_size=L, q=q, temperature=1.0, seed=123)
    sim.thermalize(steps=20)
    res = sim.run(mcs_measure=50, sample_interval=2)

    assert res.num_samples == 25
    assert res.L == L
    assert res.q == q
    assert res.temperature == 1.0
    assert -2.0 <= res.mean_energy_per_spin <= 2.0
    assert 0.0 <= res.mean_magnetization <= 1.0
    assert res.specific_heat >= 0.0
    assert 0.0 <= res.acceptance_rate <= 1.0


def test_low_temperature_ordered_state() -> None:
    """低温極限で強磁性秩序 (基底状態) に達することを検証."""
    res = run_single_temperature(
        lattice_size=6,
        q=6,
        temperature=0.1,  # 極低温
        mcs_thermalize=200,
        mcs_measure=200,
        sample_interval=2,
        mode="adjacent",
        seed=42,
    )

    # 基底状態近傍: エネルギー e ≈ -2.0, 磁化 |m| ≈ 1.0, クロック秩序度 Φ_q ≈ 1.0
    assert res.mean_energy_per_spin < -1.8
    assert res.mean_magnetization > 0.9
    assert res.clock_orientation > 0.8


def test_set_spins() -> None:
    """Rust コアへの set_spins による任意スピン配位の設定を検証."""
    L = 4
    q = 6
    sim = Clock2DRust(L, q, j=1.0, seed=42)

    # 全てスピン 2 の配位を作成
    custom_spins = np.full((L, L), 2, dtype=np.uint8)
    sim.set_spins(custom_spins)

    # 取得して一致を確認
    retrieved = sim.get_spins()
    assert np.array_equal(retrieved, custom_spins)

    # 全て同じスピンなので基底状態エネルギー E = -2N = -32.0
    assert math.isclose(sim.total_energy(), -32.0, rel_tol=1e-6)


def test_temperature_sweep_annealing() -> None:
    """徐冷アニーリング (run_temperature_sweep) の実行と温度順序を検証."""
    from models.model_05_clock_model.src.simulation import run_temperature_sweep

    L = 4
    q = 6
    temps = [0.4, 0.8, 1.2]
    results = run_temperature_sweep(
        lattice_size=L,
        q=q,
        temperatures=temps,
        mcs_thermalize=50,
        mcs_measure=100,
        sample_interval=2,
        mode="adjacent",
        seed=42,
    )

    assert len(results) == 3
    # 戻り値は温度の昇順にソートされていること
    assert results[0].temperature == 0.4
    assert results[1].temperature == 0.8
    assert results[2].temperature == 1.2

    # 低温 (T=0.4) の方が高温 (T=1.2) よりエネルギーが低いこと
    assert results[0].mean_energy_per_spin < results[2].mean_energy_per_spin
