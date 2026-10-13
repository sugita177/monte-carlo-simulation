use mc_core::xy_2d::{wrap_angle, XY2DRust};
use std::f64::consts::PI;

#[test]
fn test_wrap_angle() {
    let eps = 1e-12;
    assert!((wrap_angle(0.0) - 0.0).abs() < eps);
    assert!((wrap_angle(PI - 0.1) - (PI - 0.1)).abs() < eps);
    assert!((wrap_angle(PI + 0.1) - (-PI + 0.1)).abs() < eps);
    assert!((wrap_angle(-PI - 0.1) - (PI - 0.1)).abs() < eps);
    assert!((wrap_angle(3.0 * PI) - (-PI)).abs() < eps || (wrap_angle(3.0 * PI) - PI).abs() < eps);
}

#[test]
fn test_ground_state_energy() {
    let l = 10;
    let n = l * l;
    let mut sim = XY2DRust::new(l, 1.0, 1.0, 1.0, Some(42));
    sim.initialize_spins("uniform").unwrap();

    // 基底状態 (全スピン θ = 0): E = -J * 2N = -200, e = -2.0
    let expected_e = -2.0 * (n as f64);
    assert!((sim.total_energy() - expected_e).abs() < 1e-10);
    assert!((sim.energy_per_spin() - (-2.0)).abs() < 1e-10);

    // 磁化ベクトル (N, 0), 磁化ノルム 1.0
    assert!((sim.magnetization() - 1.0).abs() < 1e-10);
}

#[test]
fn test_antiferro_energy() {
    let l = 10;
    let n = l * l;
    let mut sim = XY2DRust::new(l, 1.0, 1.0, 1.0, Some(42));

    // チェッカーボード配置: (x + y) が奇数なら π, 偶数なら 0
    // すべての最近接ボンドで cos(π - 0) = -1 となり、E = +2N
    for y in 0..l {
        for x in 0..l {
            let th = if (x + y) % 2 == 1 { PI } else { 0.0 };
            sim.set_spin(x, y, th);
        }
    }

    let expected_e = 2.0 * (n as f64);
    assert!((sim.total_energy() - expected_e).abs() < 1e-10);
    assert!((sim.energy_per_spin() - 2.0).abs() < 1e-10);
}

#[test]
fn test_local_energy_diff_matches_total() {
    let l = 8;
    let mut sim = XY2DRust::new(l, 1.0, 1.0, 1.0, Some(123));

    // 全サイトを累積変更するのではなく、代表的な内部点と境界点について
    // 単一スピン変更前後のエネルギー差を独立に検証する
    let test_points = [
        (2, 3),         // 内部点
        (0, 0),         // 境界点 (左下隅)
        (l - 1, l - 1), // 境界点 (右上隅)
        (0, 3),         // 境界点 (左端)
        (3, l - 1),     // 境界点 (上端)
    ];

    for &(x, y) in &test_points {
        let old_e = sim.total_energy();
        let new_theta = 1.2345;
        let delta_e_local = sim.delta_energy(x, y, new_theta);

        // スピンを更新して全体の差分と比較
        sim.set_spin(x, y, new_theta);
        let new_e = sim.total_energy();

        let delta_e_actual = new_e - old_e;
        assert!(
            (delta_e_local - delta_e_actual).abs() < 1e-10,
            "Mismatch at ({}, {}): local={}, actual={}",
            x,
            y,
            delta_e_local,
            delta_e_actual
        );
    }
}

#[test]
fn test_vortex_detection_elementary() {
    let l = 4;
    let mut sim = XY2DRust::new(l, 1.0, 1.0, 1.0, Some(999));
    sim.initialize_spins("uniform").unwrap();

    // 1. 一様状態では全プラケットの渦度が厳密に 0
    let vorts_uniform = sim.get_vorticities();
    assert!(
        vorts_uniform.iter().all(|&q| q == 0),
        "All vorticities in uniform state must be 0"
    );

    // 2. プラケット (0, 0) に反時計回り渦 (+1) を作成:
    // 頂点配置:
    // (0, 0): 0
    // (1, 0): π/2
    // (1, 1): π
    // (0, 1): -π/2 (3π/2 と同値)
    sim.set_spin(0, 0, 0.0);
    sim.set_spin(1, 0, PI / 2.0);
    sim.set_spin(1, 1, PI);
    sim.set_spin(0, 1, -PI / 2.0);

    let vorts_vortex = sim.get_vorticities();
    // プラケット (0, 0) の渦度が厳密に +1 であることを検証
    assert_eq!(
        vorts_vortex[0], 1,
        "Plaquette (0, 0) must have vorticity +1"
    );
    // 他のプラケットに誤判定がないことを確認 (周囲のプラケットは非ゼロになり得るがインデックス 0 の特定値を確認)
    assert!(sim.vortex_density() > 0.0);

    // 3. プラケット (0, 0) に時計回り反渦 (-1) を作成:
    sim.initialize_spins("uniform").unwrap();
    sim.set_spin(0, 0, 0.0);
    sim.set_spin(1, 0, -PI / 2.0);
    sim.set_spin(1, 1, -PI);
    sim.set_spin(0, 1, PI / 2.0);

    let vorts_antivortex = sim.get_vorticities();
    assert_eq!(
        vorts_antivortex[0], -1,
        "Plaquette (0, 0) must have vorticity -1 (antivortex)"
    );
}

#[test]
fn test_wolff_clustering() {
    let l = 8;
    let n = l * l;
    let mut sim = XY2DRust::new(l, 0.5, 1.0, 1.0, Some(777));
    sim.initialize_spins("uniform").unwrap();

    // 低温 (T=0.5) での一様スピン配位から Wolff ステップを実行
    // 全スピンが平行なので、クラスターは高確率で全系サイズ (N) またはそれに近いサイズになる
    let cluster_size = sim.step_wolff();
    assert!(
        cluster_size >= 1 && cluster_size <= n,
        "Cluster size must be between 1 and N"
    );

    // 反転後も全スピンの角度が [-π, π) に収まっていること
    for y in 0..l {
        for x in 0..l {
            let th = sim.get_spin(x, y);
            assert!(
                th >= -PI && th < PI,
                "Angle {} at ({}, {}) out of bounds",
                th,
                x,
                y
            );
        }
    }
}

#[test]
fn test_step_wolff_mcs() {
    let l = 8;
    let n = l * l;
    let mut sim = XY2DRust::new(l, 1.0, 1.0, 1.0, Some(888));

    // 1 MCS 分の Wolff 更新を実行
    let total_flipped = sim.step_wolff_mcs();

    // 反転スピン総数は必ず N (64) 以上であること
    assert!(
        total_flipped >= n,
        "Total flipped spins ({}) must be >= N ({})",
        total_flipped,
        n
    );
}

