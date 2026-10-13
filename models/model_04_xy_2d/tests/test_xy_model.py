"""2次元XY模型の Python 側統合テスト."""

from __future__ import annotations

import numpy as np
import pytest

from mc_core import XY2DRust
from models.model_04_xy_2d.src.observables import ObservableAccumulator, SimulationResult
from models.model_04_xy_2d.src.simulation import run_single_temperature


def test_accumulator_computations() -> None:
    """ObservableAccumulator の統計集計ロジックを既知のデータで検証."""
    L = 4
    N = L * L  # 16
    T = 2.0
    acc = ObservableAccumulator(L=L, temperature=T)

    # 意図的なダミーデータを記録
    # E = [-10.0, -20.0, -30.0] -> mean(E) = -20.0, Var(E) = (100 + 0 + 100) / 3 = 200 / 3
    # M = [0.2, 0.4, 0.6] -> mean(M) = 0.4
    # V = [0.1, 0.2, 0.3] -> mean(V) = 0.2
    acc.record(energy=-10.0, magnetization=0.2, vortex_density=0.1)
    acc.record(energy=-20.0, magnetization=0.4, vortex_density=0.2)
    acc.record(energy=-30.0, magnetization=0.6, vortex_density=0.3)

    res = acc.compute_results()

    # 1. 平均エネルギー <e> = -20.0 / 16 = -1.25
    assert np.isclose(res.mean_energy_per_spin, -1.25)

    # 2. 平均磁化 <|m|> = 0.4
    assert np.isclose(res.mean_magnetization, 0.4)

    # 3. 比熱 C_V = Var(E) / (N * T^2) = (200 / 3) / (16 * 4) = 200 / 192 = 25 / 24
    expected_cv = (200.0 / 3.0) / (16.0 * 4.0)
    assert np.isclose(res.specific_heat, expected_cv)

    # 4. 平均渦度密度 <rho_v> = 0.2
    assert np.isclose(res.vortex_density, 0.2)
    assert res.num_samples == 3


def test_ground_state_energy_via_python() -> None:
    """Python 側から初期化した基底状態のエネルギーが e = -2.0 となることを検証."""
    L = 6
    sim = XY2DRust(l=L, temperature=1.0)
    sim.initialize_spins("uniform")

    assert np.isclose(sim.energy_per_spin(), -2.0)
    assert np.isclose(sim.magnetization(), 1.0)
    assert np.isclose(sim.vortex_density(), 0.0)


def test_run_single_temperature_metropolis() -> None:
    """Metropolis アルゴリズムでの単一温度シミュレーション実行テスト."""
    L = 4
    T = 1.0
    res, thetas = run_single_temperature(
        lattice_size=L,
        temperature=T,
        mcs_thermalize=20,
        mcs_measure=40,
        sample_interval=2,
        algorithm="metropolis",
        seed=42,
    )

    assert isinstance(res, SimulationResult)
    assert thetas.shape == (L, L)
    # エネルギーは -2.0 から +2.0 の物理的範囲内
    assert -2.0 <= res.mean_energy_per_spin <= 2.0
    # 磁化は 0 から 1 の範囲内
    assert 0.0 <= res.mean_magnetization <= 1.0
    # 比熱は非負
    assert res.specific_heat >= 0.0
    # 渦密度は非負
    assert res.vortex_density >= 0.0


def test_run_single_temperature_wolff() -> None:
    """Wolff クラスターアルゴリズムでの単一温度シミュレーション実行テスト."""
    L = 4
    T = 0.5
    res, thetas = run_single_temperature(
        lattice_size=L,
        temperature=T,
        mcs_thermalize=20,
        mcs_measure=40,
        sample_interval=2,
        algorithm="wolff",
        seed=123,
    )

    assert isinstance(res, SimulationResult)
    assert thetas.shape == (L, L)
    assert -2.0 <= res.mean_energy_per_spin <= 2.0
    assert 0.0 <= res.mean_magnetization <= 1.0
    assert res.specific_heat >= 0.0
    assert res.vortex_density >= 0.0
