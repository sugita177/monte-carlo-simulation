use mc_core::clock_2d::Clock2DRust;

#[test]
fn test_lut_generation() {
    let eps = 1e-12;

    // q = 2 (Ising): [cos(0), cos(π)] = [1.0, -1.0]
    let sim_q2 = Clock2DRust::new(4, 2, 1.0, Some(42), "ground").unwrap();
    assert_eq!(sim_q2.lut_cos.len(), 2);
    assert!((sim_q2.lut_cos[0] - 1.0).abs() < eps);
    assert!((sim_q2.lut_cos[1] - (-1.0)).abs() < eps);

    // q = 4: [cos(0), cos(π/2), cos(π), cos(3π/2)] = [1.0, 0.0, -1.0, 0.0]
    let sim_q4 = Clock2DRust::new(4, 4, 1.0, Some(42), "ground").unwrap();
    assert_eq!(sim_q4.lut_cos.len(), 4);
    assert!((sim_q4.lut_cos[0] - 1.0).abs() < eps);
    assert!(sim_q4.lut_cos[1].abs() < eps);
    assert!((sim_q4.lut_cos[2] - (-1.0)).abs() < eps);
    assert!(sim_q4.lut_cos[3].abs() < eps);

    // q = 6: cos(π/3) = 0.5, cos(2π/3) = -0.5
    let sim_q6 = Clock2DRust::new(4, 6, 1.0, Some(42), "ground").unwrap();
    assert_eq!(sim_q6.lut_cos.len(), 6);
    assert!((sim_q6.lut_cos[0] - 1.0).abs() < eps);
    assert!((sim_q6.lut_cos[1] - 0.5).abs() < eps);
    assert!((sim_q6.lut_cos[2] - (-0.5)).abs() < eps);
    assert!((sim_q6.lut_cos[3] - (-1.0)).abs() < eps);
}

#[test]
fn test_ground_state_energy_and_magnetization() {
    let l = 8;
    let n = l * l;
    let q = 6;
    let sim = Clock2DRust::new(l, q, 1.0, Some(42), "ground").unwrap();

    // 基底状態 (全スピン 0):
    // 2N 本のボンドすべてで cos(0) = 1.0 なので E = -J * 2N = -2 * 64 = -128.0
    let expected_energy = -2.0 * (n as f64);
    assert!(
        (sim.total_energy() - expected_energy).abs() < 1e-10,
        "Total energy should be -2N for ground state"
    );

    // 磁化ベクトル: mx = 1.0, my = 0.0, |m| = 1.0
    let (mx, my, norm_m) = sim.magnetization();
    assert!((mx - 1.0).abs() < 1e-10);
    assert!(my.abs() < 1e-10);
    assert!((norm_m - 1.0).abs() < 1e-10);
}

#[test]
fn test_antiferro_energy_even_q() {
    let l = 6;
    let n = l * l;
    let q = 6;
    let mut sim = Clock2DRust::new(l, q, 1.0, Some(42), "ground").unwrap();

    // チェッカーボード配置: (r + c) が奇数なら q/2 (反平行スピン, 角度 π), 偶数なら 0
    // 最隣接スピン差分は常に π となり、cos(π) = -1.0
    // したがって E = -J * (-1.0) * 2N = +2N
    let opp_spin = (q / 2) as u8;
    for r in 0..l {
        for c in 0..l {
            let s = if (r + c) % 2 == 1 { opp_spin } else { 0 };
            sim.set_spin(r, c, s);
        }
    }

    let expected_energy = 2.0 * (n as f64);
    assert!(
        (sim.total_energy() - expected_energy).abs() < 1e-10,
        "Antiferromagnetic state should have E = +2N"
    );
}

#[test]
fn test_local_energy_diff_matches_total() {
    let l = 6;
    let q = 6;
    let mut sim = Clock2DRust::new(l, q, 1.0, Some(123), "random").unwrap();

    // 代表的なサイト (内部点, 境界点, 四隅) で検証
    let test_sites = [(2, 2), (0, 3), (3, 0), (0, 0), (l - 1, l - 1)];

    for &(r, c) in &test_sites {
        let old_s = sim.get_spin(r, c);
        for offset in 1..q {
            let new_s = ((old_s as usize + offset) % q) as u8;

            let e_before = sim.total_energy();
            let delta_e_local = sim.delta_energy(r, c, new_s);

            // スピンを変更して全エネルギーを再計算
            sim.set_spin(r, c, new_s);
            let e_after = sim.total_energy();

            let diff_total = e_after - e_before;
            assert!(
                (delta_e_local - diff_total).abs() < 1e-10,
                "delta_energy ({}) must match diff_total ({}) for site ({}, {})",
                delta_e_local,
                diff_total,
                r,
                c
            );

            // 元に戻す
            sim.set_spin(r, c, old_s);
        }
    }
}

#[test]
fn test_metropolis_step_bounds_and_decrease() {
    let l = 8;
    let q = 6;
    let mut sim = Clock2DRust::new(l, q, 1.0, Some(999), "random").unwrap();

    // random モードと adjacent モードの両方を数ステップ実行
    for _ in 0..5 {
        sim.step_metropolis(1.0, "random").unwrap();
        sim.step_metropolis(1.0, "adjacent").unwrap();
    }

    // 全サイトのスピンが 0..q の範囲内であることを確認
    for r in 0..l {
        for c in 0..l {
            let s = sim.get_spin(r, c);
            assert!(
                (s as usize) < q,
                "Spin value {} at ({}, {}) exceeds q={}",
                s,
                r,
                c,
                q
            );
        }
    }

    // 低温 (beta = 50.0) で数十ステップ回すとエネルギーが基底状態 (e = -2.0) 付近まで低下する
    for _ in 0..100 {
        sim.step_metropolis(50.0, "adjacent").unwrap();
    }
    let e_per_spin = sim.total_energy() / ((l * l) as f64);
    assert!(
        e_per_spin < -1.8,
        "Low temperature cooling should lower energy per spin close to -2.0, got {}",
        e_per_spin
    );
}
