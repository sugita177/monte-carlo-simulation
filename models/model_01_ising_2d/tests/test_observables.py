"""Tests for observable accumulation and simulation runner."""

import pytest

from models.model_01_ising_2d.src.observables import ObservableAccumulator
from models.model_01_ising_2d.src.simulation import run_simulation


def test_observable_accumulator_computation():
    """アキュムレータの統計量計算のテスト."""
    L = 4
    accumulator = ObservableAccumulator(L=L, temperature=2.0)

    # 意図的なサンプルを記録
    # 完全に整列した状態 (E = -32, M = 16) を 10 回
    for _ in range(10):
        accumulator.record(energy=-32.0, magnetization=16)

    res = accumulator.compute_results()
    assert res.mean_energy_per_spin == -2.0  # -32 / 16
    assert res.mean_magnetization_per_spin == 1.0  # 16 / 16
    assert res.specific_heat == 0.0  # ゆらぎなし -> 比熱 0
    assert res.susceptibility == 0.0  # ゆらぎなし -> 帯磁率 0
    assert pytest.approx(res.binder_cumulant, abs=1e-5) == 2.0 / 3.0  # 完全秩序相: U_4 = 2/3


def test_simulation_low_temperature():
    """低温相 (T = 1.0 < Tc) において、自発磁化が 1 に近く、エネルギーが基底状態に近いことを検証."""
    res = run_simulation(
        L=8,
        temperature=1.0,
        mcs_thermalize=200,
        mcs_measure=400,
        sample_interval=2,
        init_method="all_up",
        seed=42,
    )
    # T = 1.0 では強い強磁性秩序を示す
    assert res.mean_magnetization_per_spin > 0.95
    assert res.mean_energy_per_spin < -1.8


def test_simulation_high_temperature():
    """高温相 (T = 5.0 > Tc) において、平均自発磁化が小さく、Binder 比が 0 に近づくことを検証."""
    res = run_simulation(
        L=8,
        temperature=5.0,
        mcs_thermalize=200,
        mcs_measure=400,
        sample_interval=2,
        init_method="random",
        seed=42,
    )
    # T = 5.0 では無秩序相
    assert res.mean_magnetization_per_spin < 0.4
    assert res.binder_cumulant < 0.3
