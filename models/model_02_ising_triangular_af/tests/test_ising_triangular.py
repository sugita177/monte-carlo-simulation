"""三角格子反強磁性イジング模型 (IsingTriangularRust) の単体テスト."""

import numpy as np
import pytest
from mc_core import IsingTriangularRust


def test_sum_neighbors_and_periodic_boundary():
    """全スピンが一様のとき、すべてのサイトで 6近傍和が理論値 (+-6) になることを検証."""
    l = 6
    sim = IsingTriangularRust(l=l, temperature=1.0, j_af=1.0, seed=42)

    # 1. all_up 配位: すべてのサイトで 6近傍和は +6
    sim.initialize_spins("all_up")
    for x in range(l):
        for y in range(l):
            assert (
                sim.sum_neighbors(x, y) == 6
            ), f"Site ({x}, {y}) neighbor sum is not 6"

    # 2. all_down 配位: すべてのサイトで 6近傍和は -6
    sim.initialize_spins("all_down")
    for x in range(l):
        for y in range(l):
            assert (
                sim.sum_neighbors(x, y) == -6
            ), f"Site ({x}, {y}) neighbor sum is not -6"

    # 3. get_spins() の形状・値域確認
    spins = sim.get_spins()
    assert spins.shape == (l, l)
    assert np.all(spins == -1)


def test_energy_scaling_all_aligned():
    """一様配位における全エネルギーの理論値一致テスト (トートロジーを排除した厳密検証).

    三角格子では総ボンド数 = 3 * N 本。
    全スピンが一様 (+1 または -1) の場合、各ボンドは (+1)*(+1) = +1 または (-1)*(-1) = +1。
    したがって、全エネルギーは閉じた数式として H = 3 * N * J_af, 1サイトあたり e = 3.0 * J_af。
    """
    l = 6
    n = l * l
    j_af = 1.5
    sim = IsingTriangularRust(l=l, temperature=1.0, j_af=j_af, seed=123)

    expected_total = 3.0 * n * j_af
    expected_per_spin = 3.0 * j_af

    # 1. all_up の理論値検証
    sim.initialize_spins("all_up")
    assert np.isclose(sim.total_energy(), expected_total)
    assert np.isclose(sim.energy_per_spin(), expected_per_spin)
    assert sim.magnetization() == float(n)

    # 2. all_down の理論値検証
    sim.initialize_spins("all_down")
    assert np.isclose(sim.total_energy(), expected_total)
    assert np.isclose(sim.energy_per_spin(), expected_per_spin)
    assert sim.magnetization() == -float(n)


def test_local_energy_diff_matches_total():
    """局所 dE と全エネルギー再計算差分 E_new - E_old の完全一致テスト.

    ランダム配位において任意サイト (x, y) のスピンを反転させたとき、
    局所計算による dE = -2 * J_af * s_i * S_nn と、
    全エネルギーの再計算差分 E_new - E_old が完全に一致することを検証する。
    """
    l = 8
    sim = IsingTriangularRust(l=l, temperature=1.5, j_af=1.2, seed=999)
    sim.initialize_spins("random")

    # 境界や内部を含む複数サイトで局所 dE と全体差分の一致を検証
    test_sites = [(0, 0), (1, 2), (3, 5), (l - 1, 0), (l - 1, l - 1)]
    for x, y in test_sites:
        local_de = sim.delta_energy(x, y)
        e_old = sim.total_energy()

        # スピンを反転
        sim.flip_spin(x, y)
        e_new = sim.total_energy()

        energy_diff = e_new - e_old
        assert np.isclose(
            energy_diff, local_de
        ), f"Site ({x}, {y}): diff {energy_diff} != local_de {local_de}"


def test_metropolis_step_runs():
    """Metropolis 更新ステップが実行可能で、受託数が [0, N] の範囲に収まることを検証."""
    l = 8
    n = l * l
    sim = IsingTriangularRust(l=l, temperature=2.0, j_af=1.0, seed=42)

    accepted = sim.step_metropolis()
    assert 0 <= accepted <= n

    m = sim.magnetization()
    assert -n <= m <= n

    m_per_spin = sim.magnetization_per_spin()
    assert -1.0 <= m_per_spin <= 1.0


def test_invalid_initialization_method():
    """不正な初期化手法を指定した際に ValueError が送出されることを検証."""
    sim = IsingTriangularRust(l=4, temperature=1.0, j_af=1.0, seed=42)
    with pytest.raises(ValueError, match="Unknown method"):
        sim.initialize_spins("invalid_mode")
