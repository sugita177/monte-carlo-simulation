use numpy::PyArray2;
use pyo3::prelude::*;
use rand::Rng;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

/// 2次元正方格子イジング模型の Rust 高速計算エンジン
#[pyclass]
pub struct Ising2DRust {
    #[pyo3(get)]
    pub l: usize,
    #[pyo3(get)]
    pub n: usize,
    #[pyo3(get)]
    pub temperature: f64,
    pub beta: f64,
    #[pyo3(get)]
    pub j: f64,
    #[pyo3(get)]
    pub h: f64,
    // 連続メモリ領域にフラット化されたスピン配列 (x * L + y)
    pub spins: Vec<i8>,
    // dE / J の離散値 [0..8] に対する固定長ルックアップテーブル (最速 O(1) 参照)
    pub exp_table: [f64; 9],
    rng: Xoshiro256PlusPlus,
}

#[pymethods]
impl Ising2DRust {
    #[new]
    #[pyo3(signature = (l, temperature, j=1.0, h=0.0, seed=None))]
    pub fn new(l: usize, temperature: f64, j: f64, h: f64, seed: Option<u64>) -> Self {
        let n = l * l;
        let beta = if temperature > 0.0 {
            1.0 / temperature
        } else {
            f64::INFINITY
        };

        // 乱数生成器 (シード指定またはエントロピーから)
        let rng = match seed {
            Some(s) => Xoshiro256PlusPlus::seed_from_u64(s),
            None => Xoshiro256PlusPlus::from_entropy(),
        };

        // 固定長ルックアップテーブルの構築 (dE/J = 4, 8)
        let mut exp_table = [0.0; 9];
        exp_table[4] = (-beta * j * 4.0).exp();
        exp_table[8] = (-beta * j * 8.0).exp();

        Self {
            l,
            n,
            temperature,
            beta,
            j,
            h,
            spins: vec![0; n],
            exp_table,
            rng,
        }
    }

    /// スピン配位の初期化 ("random", "all_up", "all_down")
    pub fn initialize_spins(&mut self, method: &str) -> PyResult<()> {
        match method {
            "all_up" => {
                self.spins.fill(1);
            }
            "all_down" => {
                self.spins.fill(-1);
            }
            "random" => {
                // 各スピンを +1 または -1 にランダム初期化
                for s in self.spins.iter_mut() {
                    *s = if self.rng.gen_bool(0.5) { 1 } else { -1 };
                }
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown method: {}",
                    method
                )))
            }
        }
        Ok(())
    }

    /// 1 モンテカルロステップ (N 回の局所スピン反転試行)
    pub fn step_metropolis(&mut self) -> (usize, usize) {
        let mut accepted = 0;
        let l = self.l;

        // N 回の局所更新ループ:
        for _ in 0..self.n {
            // 1. ランダムなサイト (x, y) を選ぶ
            let x = self.rng.gen_range(0..l);
            let y = self.rng.gen_range(0..l);
            let idx = x * l + y; // 1次元配列のインデックス
            let s = self.spins[idx]; // 現在のスピン（+1 or -1）

            // 2. 周期境界条件 (PBC) を考慮して上下左右の 4 隣接スピン和を求める
            let right = if x + 1 == l { 0 } else { x + 1 };
            let left = if x == 0 { l - 1 } else { x - 1 };
            let up = if y + 1 == l { 0 } else { y + 1 };
            let down = if y == 0 { l - 1 } else { y - 1 };

            // 4つの隣接スピンを足す (PBC: 右, 左, 上, 下)
            let sum_nb = self.spins[right * l + y] as i32
                + self.spins[left * l + y] as i32
                + self.spins[x * l + up] as i32
                + self.spins[x * l + down] as i32;

            // 3. dE_over_j = 2 * s * sum_nb を計算
            let de_over_j = 2 * (s as i32) * sum_nb;

            // 4. メトロポリス判定 (dE_over_j <= 0 または乱数 r < exp_table[dE_over_j as usize])
            if de_over_j <= 0 || self.rng.gen::<f64>() < self.exp_table[de_over_j as usize] {
                // 受託されたらスピン反転し accepted += 1
                self.spins[idx] = -s;
                accepted += 1;
            }
        }

        (accepted, self.n)
    }

    /// 全ハミルトニアン H の厳密計算
    pub fn total_energy(&self) -> f64 {
        let mut interaction_sum: i64 = 0;
        let l = self.l;

        // 各サイトと「右隣」「下隣」のボンド積を足し合わせる (PBC 考慮)
        for x in 0..l {
            let right_x = if x + 1 == l { 0 } else { x + 1 };
            for y in 0..l {
                let down_y = if y + 1 == l { 0 } else { y + 1 };

                let s = self.spins[x * l + y] as i64;
                let s_right = self.spins[right_x * l + y] as i64;
                let s_down = self.spins[x * l + down_y] as i64;
                interaction_sum += s * s_right + s * s_down;
            }
        }

        let interaction_energy = -self.j * (interaction_sum as f64);
        let field_energy = if self.h != 0.0 {
            -self.h * (self.total_magnetization() as f64)
        } else {
            0.0
        };

        interaction_energy + field_energy
    }

    /// 全磁化 M = sum(s_i)
    pub fn total_magnetization(&self) -> i64 {
        self.spins.iter().map(|&s| s as i64).sum()
    }

    /// Python 側にスピン配位を (L, L) の NumPy 2次元配列として返す (可視化用)
    pub fn get_spins<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<i8>>> {
        use numpy::PyArrayMethods;
        let arr1d = numpy::PyArray1::from_vec(py, self.spins.clone());
        arr1d.reshape([self.l, self.l])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ground_state_energy() {
        let l = 8;
        let j = 1.5;
        let n = l * l;
        let mut ising = Ising2DRust::new(l, 1.0, j, 0.0, Some(42));

        // all_up の基底状態
        ising.initialize_spins("all_up").unwrap();
        assert_eq!(ising.total_magnetization(), n as i64);
        assert!((ising.total_energy() - (-2.0 * j * n as f64)).abs() < 1e-10);

        // all_down の基底状態
        ising.initialize_spins("all_down").unwrap();
        assert_eq!(ising.total_magnetization(), -(n as i64));
        assert!((ising.total_energy() - (-2.0 * j * n as f64)).abs() < 1e-10);
    }

    #[test]
    fn test_low_temperature_freezing() {
        let l = 8;
        let mut ising = Ising2DRust::new(l, 0.01, 1.0, 0.0, Some(42));
        ising.initialize_spins("all_up").unwrap();

        // 極低温では反転受託が 0 回で凍結する
        let (accepted, trials) = ising.step_metropolis();
        assert_eq!(accepted, 0);
        assert_eq!(trials, l * l);
    }

    #[test]
    fn test_high_temperature_acceptance() {
        let l = 8;
        let mut ising = Ising2DRust::new(l, 1000.0, 1.0, 0.0, Some(42));
        ising.initialize_spins("all_up").unwrap();

        // 超高温ではほぼ全ての反転が受託される (受託率 > 0.8)
        let (accepted, trials) = ising.step_metropolis();
        let rate = accepted as f64 / trials as f64;
        assert!(rate > 0.8, "Acceptance rate was too low: {}", rate);
    }
}

