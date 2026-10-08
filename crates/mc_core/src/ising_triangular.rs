use numpy::PyArray2;
use pyo3::prelude::*;
use rand::Rng;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

/// 三角格子反強磁性イジング模型の Rust 高速計算エンジン (学習用スケルトン)
#[pyclass]
pub struct IsingTriangularRust {
    #[pyo3(get)]
    pub l: usize,
    #[pyo3(get)]
    pub n: usize,
    #[pyo3(get)]
    pub temperature: f64,
    pub beta: f64,
    #[pyo3(get)]
    pub j_af: f64,
    // 連続メモリ領域にフラット化されたスピン配列 (x * L + y)
    pub spins: Vec<i8>,
    // dE / J_af の離散値 [0..=12] に対する固定長ルックアップテーブル
    pub exp_table: [f64; 13],
    rng: Xoshiro256PlusPlus,
}

#[pymethods]
impl IsingTriangularRust {
    #[new]
    #[pyo3(signature = (l, temperature, j_af=1.0, seed=None))]
    pub fn new(l: usize, temperature: f64, j_af: f64, seed: Option<u64>) -> Self {
        let n = l * l;
        let beta = if temperature > 0.0 {
            1.0 / temperature
        } else {
            f64::INFINITY
        };

        let mut rng = match seed {
            Some(s) => Xoshiro256PlusPlus::seed_from_u64(s),
            None => Xoshiro256PlusPlus::from_entropy(),
        };

        // ルックアップテーブルの構築 (dE / J_af = 4, 8, 12)
        let mut exp_table = [0.0; 13];
        if temperature > 0.0 {
            exp_table[4] = (-beta * j_af * 4.0).exp();
            exp_table[8] = (-beta * j_af * 8.0).exp();
            exp_table[12] = (-beta * j_af * 12.0).exp();
        }

        // スピンのランダム初期化 (+1 または -1)
        let mut spins = vec![0i8; n];
        for s in spins.iter_mut() {
            *s = if rng.gen_bool(0.5) { 1 } else { -1 };
        }

        Self {
            l,
            n,
            temperature,
            beta,
            j_af,
            spins,
            exp_table,
            rng,
        }
    }

    /// 温度の動的変更 (ルックアップテーブルも再計算)
    pub fn set_temperature(&mut self, temperature: f64) {
        self.temperature = temperature;
        self.beta = if temperature > 0.0 {
            1.0 / temperature
        } else {
            f64::INFINITY
        };
        self.exp_table = [0.0; 13];
        if temperature > 0.0 {
            self.exp_table[4] = (-self.beta * self.j_af * 4.0).exp();
            self.exp_table[8] = (-self.beta * self.j_af * 8.0).exp();
            self.exp_table[12] = (-self.beta * self.j_af * 12.0).exp();
        }
    }

    /// スピン配位の初期化 ("all_up", "all_down", "random")
    pub fn initialize_spins(&mut self, method: &str) -> PyResult<()> {
        match method {
            "all_up" => {
                self.spins.fill(1);
            }
            "all_down" => {
                self.spins.fill(-1);
            }
            "random" => {
                for s in self.spins.iter_mut() {
                    *s = if self.rng.gen_bool(0.5) { 1 } else { -1 };
                }
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown method: {}. Expected 'all_up', 'all_down', or 'random'",
                    method
                )));
            }
        }
        Ok(())
    }

    /// 特定サイト (x, y) のスピン反転に伴う局所エネルギー変化 dE = -2 * J_af * s_i * S_nn
    pub fn delta_energy(&self, x: usize, y: usize) -> f64 {
        let idx = x * self.l + y;
        let s_i = self.spins[idx] as f64;
        let sum_nn = self.sum_neighbors(x, y) as f64;
        -2.0 * self.j_af * s_i * sum_nn
    }

    /// 特定サイト (x, y) のスピンを反転
    pub fn flip_spin(&mut self, x: usize, y: usize) {
        let idx = x * self.l + y;
        self.spins[idx] = -self.spins[idx];
    }

    /// サイト (x, y) の 6 近傍スピンの和を計算 (周期境界条件)
    /// サイト (x, y) の 6 近傍の座標 (フラットインデックス idx = x * L + y):
    /// 1. 右: (x + 1, y)
    /// 2. 左: (x - 1, y)
    /// 3. 上: (x, y + 1)
    /// 4. 下: (x, y - 1)
    /// 5. 右下: (x + 1, y - 1)
    /// 6. 左上: (x - 1, y + 1)
    /// ※ それぞれ L による周期境界条件 (x + 1 == L のときは 0, x == 0 のときは L - 1) を適用。
    #[inline(always)]
    pub fn sum_neighbors(&self, x: usize, y: usize) -> i32 {
        // 6 近傍のスピンを self.spins から取得して合算
        let l = self.l;
        let right_x = if x + 1 == l { 0 } else { x + 1 };
        let left_x = if x == 0 { l - 1 } else { x - 1 };
        let up_y = if y + 1 == l { 0 } else { y + 1 };
        let down_y = if y == 0 { l - 1 } else { y - 1 };

        let s1 = self.spins[right_x * l + y] as i32; // 右 (x+1, y)
        let s2 = self.spins[left_x * l + y] as i32; // 左 (x-1, y)
        let s3 = self.spins[x * l + up_y] as i32; // 上 (x, y+1)
        let s4 = self.spins[x * l + down_y] as i32; // 下 (x, y-1)
        let s5 = self.spins[right_x * l + down_y] as i32; // 右下 (x+1, y-1)
        let s6 = self.spins[left_x * l + up_y] as i32; // 左上 (x-1, y+1)

        s1 + s2 + s3 + s4 + s5 + s6
    }

    /// 1 モンテカルロステップ (Metropolis 法による N 回の局所スピン反転試行)
    /// 1. ランダムなサイト idx を選択 (x = idx / L, y = idx % L)
    /// 2. 現在のスピン s_i と 6近傍和 sum_nn を取得
    /// 3. 局所エネルギー変化: dE / J_af = -2 * s_i * sum_nn
    /// 4. 受託判定:
    ///    - dE <= 0 ならば無条件受託 (true)
    ///    - dE > 0 ならば self.exp_table[dE / J_af] と乱数 self.rng.gen::<f64>() を比較
    /// 5. 受託されたら self.spins[idx] = -self.spins[idx] とし、受託回数をカウント
    pub fn step_metropolis(&mut self) -> usize {
        // N 回の試行ループとメトロポリス受託判定の実装
        let mut accepted = 0;
        let l = self.l;
        for _ in 0..self.n {
            // 1. ランダムなサイト idx を選択 (x = idx / L, y = idx % L)
            let x = self.rng.gen_range(0..l);
            let y = self.rng.gen_range(0..l);
            let idx = x * l + y; // 1次元配列のインデックス
            let s = self.spins[idx]; // 現在のスピン（+1 or -1）
                                     // 2. 現在のスピン s_i と 6近傍和 sum_nn を取得
            let sum_nn = self.sum_neighbors(x, y);
            // 3. 局所エネルギー変化: dE / J_af = -2 * s_i * sum_nn
            let de_over_j = -2 * (s as i32) * sum_nn;
            // 4. 受託判定
            if de_over_j <= 0 || self.rng.gen::<f64>() < self.exp_table[de_over_j as usize] {
                // 受託されたらスピン反転し accepted += 1
                self.spins[idx] = -s;
                accepted += 1;
            }
        }
        accepted
    }

    /// 系全体の全エネルギー H = J_af * Σ <i,j> σ_i σ_j
    /// 各サイト (x, y) から「右」「上」「右下」の 3 本のボンドのみを足し合わせることで、
    /// 全ボンドを重複なく 1 度ずつ集計する。
    pub fn total_energy(&self) -> f64 {
        // 全サイトを走査し、3方向のボンド積の和に self.j_af を掛けて返す
        let mut interaction_sum: i64 = 0;
        let l = self.l;

        for x in 0..l {
            // x方向の隣接サイト
            let right_x = if x + 1 == l { 0 } else { x + 1 };
            for y in 0..l {
                let idx = x * l + y;
                let s = self.spins[idx] as i64;

                // y方向の隣接サイト
                let up_y = if y + 1 == l { 0 } else { y + 1 };
                let down_y = if y == 0 { l - 1 } else { y - 1 };

                // 3方向のボンド積の和を interaction_sum に加算
                interaction_sum += s * self.spins[right_x * l + y] as i64; // 右
                interaction_sum += s * self.spins[x * l + up_y] as i64; // 上
                interaction_sum += s * self.spins[right_x * l + down_y] as i64; // 右下
            }
        }

        // 全ボンドの和に J_af を掛けて返す
        self.j_af * (interaction_sum as f64)
    }

    /// 1 サイトあたりのエネルギー e = E / N
    pub fn energy_per_spin(&self) -> f64 {
        self.total_energy() / (self.n as f64)
    }

    /// 全磁化 M = Σ σ_i
    pub fn magnetization(&self) -> f64 {
        let sum_m: i64 = self.spins.iter().map(|&s| s as i64).sum();
        sum_m as f64
    }

    /// 1 サイトあたりの磁化 m = M / N
    pub fn magnetization_per_spin(&self) -> f64 {
        self.magnetization() / (self.n as f64)
    }

    /// Python 側にスピン配位を (L, L) の NumPy 2次元配列として返す (可視化用)
    pub fn get_spins<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<i8>>> {
        use numpy::PyArrayMethods;
        let arr1d = numpy::PyArray1::from_vec(py, self.spins.clone());
        arr1d.reshape([self.l, self.l])
    }
}
