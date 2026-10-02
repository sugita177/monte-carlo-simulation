"""Integration tests verifying Rust core engine (mc_core.Ising2DRust) and Python bindings."""

import numpy as np
import pytest

import mc_core
from models.model_01_ising_2d.src.ising import Ising2D
from models.model_01_ising_2d.src.simulation import run_simulation


def test_rust_initialization_methods():
    """Rust エンジンの初期化手法、NumPy ゼロコピー配列形状、無効入力時の例外ハンドリング."""
    L = 8
    ising_rust = mc_core.Ising2DRust(L, temperature=2.27, j=1.0, h=0.0, seed=42)

    # 1. all_up
    ising_rust.initialize_spins("all_up")
    spins = ising_rust.get_spins()
    assert spins.shape == (L, L)
    assert spins.dtype == np.int8
    assert np.all(spins == 1)
    assert ising_rust.total_magnetization() == L * L

    # 2. all_down
    ising_rust.initialize_spins("all_down")
    assert np.all(ising_rust.get_spins() == -1)
    assert ising_rust.total_magnetization() == -L * L

    # 3. random
    ising_rust.initialize_spins("random")
    spins_rand = ising_rust.get_spins()
    assert np.all(np.isin(spins_rand, [-1, 1]))

    # 4. 不正な初期化名で ValueError が送出されること (PyResult の検証)
    with pytest.raises(ValueError, match="Unknown method"):
        ising_rust.initialize_spins("invalid_mode")


def test_rust_vs_python_energy_consistency():
    """
    同一のスピン配位に対して、Rust と Python のエネルギー計算値が
    完全に一致することを検証 (ゼロ磁場および外場あり).
    """
    L = 6
    J = 1.2
    h = 0.5
    N = L * L

    rust_ising = mc_core.Ising2DRust(L, temperature=2.0, j=J, h=h, seed=123)
    py_ising = Ising2D(L=L, temperature=2.0, J=J, h=h, seed=123)

    # 1. 基底状態 (all_up): E = -2*J*N - h*N
    rust_ising.initialize_spins("all_up")
    py_ising.initialize_spins("all_up")
    expected_e = -2.0 * J * N - h * N
    assert np.isclose(rust_ising.total_energy(), expected_e)
    assert np.isclose(py_ising.total_energy(), expected_e)
    assert np.isclose(rust_ising.total_energy(), py_ising.total_energy())

    # 2. ランダム状態でスピンを共有して計算比較
    rust_ising.initialize_spins("random")
    py_ising.spins = rust_ising.get_spins().copy()

    assert rust_ising.total_magnetization() == py_ising.total_magnetization()
    assert np.isclose(rust_ising.total_energy(), py_ising.total_energy())


def test_rust_step_metropolis_contract():
    """Rust の step_metropolis の試行回数および極限状態挙動の検証."""
    L = 8
    N = L * L

    # 極低温 (T -> 0): 基底状態から 1 MCS 実行しても反転数は 0
    cold_ising = mc_core.Ising2DRust(L, temperature=0.01, j=1.0, h=0.0, seed=42)
    cold_ising.initialize_spins("all_up")
    accepted, trials = cold_ising.step_metropolis()
    assert trials == N
    assert accepted == 0

    # 超高温 (T -> inf): ほぼ全てが受託される
    hot_ising = mc_core.Ising2DRust(L, temperature=1000.0, j=1.0, h=0.0, seed=42)
    hot_ising.initialize_spins("all_up")
    accepted, trials = hot_ising.step_metropolis()
    assert trials == N
    assert (accepted / trials) > 0.8


def test_run_simulation_with_both_engines():
    """run_simulation を Python エンジンと Rust エンジンの双方で実行し、同等の物理秩序を得られることを検証."""
    common_params = dict(
        L=8,
        temperature=1.0,
        mcs_thermalize=100,
        mcs_measure=200,
        sample_interval=2,
        init_method="all_up",
        seed=42,
    )

    res_rust = run_simulation(**common_params, engine_type="rust")
    res_py = run_simulation(**common_params, engine_type="python")

    # 低温秩序相におけるエネルギーと磁化がともに理論予測通り高秩序を示す
    assert res_rust.mean_magnetization_per_spin > 0.95
    assert res_py.mean_magnetization_per_spin > 0.95
    assert res_rust.mean_energy_per_spin < -1.8
    assert res_py.mean_energy_per_spin < -1.8
