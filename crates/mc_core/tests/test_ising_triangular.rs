use mc_core::ising_triangular::IsingTriangularRust;

#[test]
fn test_sum_neighbors_all_up() {
    let l = 6;
    let mut sim = IsingTriangularRust::new(l, 1.0, 1.0, Some(42));
    // すべてのスピンを +1 に設定
    for s in sim.spins.iter_mut() {
        *s = 1;
    }

    // すべてのサイトで 6 近傍スピン和は 6 になるはず
    for x in 0..l {
        for y in 0..l {
            assert_eq!(
                sim.sum_neighbors(x, y),
                6,
                "サイト ({}, {}) の近傍和が 6 ではありません",
                x,
                y
            );
        }
    }
}

#[test]
fn test_energy_all_up() {
    let l = 6;
    let mut sim = IsingTriangularRust::new(l, 1.0, 1.0, Some(42));
    // すべてのスピンを +1 に設定
    for s in sim.spins.iter_mut() {
        *s = 1;
    }

    // 全ボンド数は 3 * N 本。J_af = 1.0 なので全エネルギーは 3 * N * 1.0
    // 1サイトあたりのエネルギーは 3.0
    let n = (l * l) as f64;
    assert!((sim.total_energy() - 3.0 * n).abs() < 1e-10);
    assert!((sim.energy_per_spin() - 3.0).abs() < 1e-10);
}

#[test]
fn test_local_energy_diff_matches_total() {
    let l = 6;
    let mut sim = IsingTriangularRust::new(l, 2.0, 1.0, Some(42));

    // ランダム配位において、あるサイト (x, y) のスピンを反転させたときの
    // 局所 dE = -2 * J_af * s_i * S_nn と、全エネルギーの再計算差分 E_new - E_old が完全一致するか検証
    let x = 2;
    let y = 3;
    let idx = x * l + y;
    let s_old = sim.spins[idx] as i32;
    let sum_nn = sim.sum_neighbors(x, y);

    let local_de = -2.0 * sim.j_af * (s_old as f64) * (sum_nn as f64);

    let e_old = sim.total_energy();
    sim.spins[idx] = -sim.spins[idx];
    let e_new = sim.total_energy();

    let diff = e_new - e_old;
    assert!(
        (diff - local_de).abs() < 1e-10,
        "局所 dE ({}) と 全エネルギー差分 ({}) が一致しません",
        local_de,
        diff
    );
}

#[test]
fn test_step_metropolis_runs() {
    let l = 8;
    let mut sim = IsingTriangularRust::new(l, 2.0, 1.0, Some(42));
    let accepted = sim.step_metropolis();
    assert!(accepted <= l * l);
}
