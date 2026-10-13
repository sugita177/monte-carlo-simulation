use numpy::{PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;
use rand::Rng;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use std::f64::consts::PI;

const TWO_PI: f64 = 2.0 * PI;

/// 角度を主値区間 [-π, π) に正規化する
#[inline]
pub fn wrap_angle(theta: f64) -> f64 {
    (theta + PI).rem_euclid(TWO_PI) - PI
}

/// 2次元XY模型の高速モンテカルロシミュレーションエンジン (Rust 実装)
#[pyclass]
pub struct XY2DRust {
    l: usize,
    n: usize,
    temperature: f64,
    beta: f64,
    j: f64,
    delta_theta: f64,
    thetas: Vec<f64>,
    rng: Xoshiro256PlusPlus,
}

#[pymethods]
impl XY2DRust {
    /// 新しい 2次元XY模型シミュレータを生成する
    #[new]
    #[pyo3(signature = (l, temperature, j=1.0, delta_theta=1.0, seed=None))]
    pub fn new(l: usize, temperature: f64, j: f64, delta_theta: f64, seed: Option<u64>) -> Self {
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

        // スピン角度を [-π, π) の一様乱数で初期化
        let mut thetas = vec![0.0; n];
        for th in thetas.iter_mut() {
            *th = rng.gen_range(-PI..PI);
        }

        Self {
            l,
            n,
            temperature,
            beta,
            j,
            delta_theta,
            thetas,
            rng,
        }
    }

    /// スピン配位を特定の状態に初期化する ("random", "uniform" / "ferro", "vortex")
    pub fn initialize_spins(&mut self, state: &str) -> PyResult<()> {
        match state {
            "random" => {
                for th in self.thetas.iter_mut() {
                    *th = self.rng.gen_range(-PI..PI);
                }
            }
            "uniform" | "ferro" => {
                for th in self.thetas.iter_mut() {
                    *th = 0.0;
                }
            }
            "vortex" => {
                // 格子中心に単一の渦 (+1) を配置
                let center = (self.l as f64 - 1.0) / 2.0;
                for y in 0..self.l {
                    for x in 0..self.l {
                        let dx = x as f64 - center;
                        let dy = y as f64 - center;
                        self.thetas[y * self.l + x] = dy.atan2(dx);
                    }
                }
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown state: {}. Use 'random', 'uniform', or 'vortex'",
                    state
                )))
            }
        }
        Ok(())
    }

    /// 温度を変更する (逆温度 β も再計算)
    pub fn set_temperature(&mut self, temperature: f64) {
        self.temperature = temperature;
        self.beta = if temperature > 0.0 {
            1.0 / temperature
        } else {
            f64::INFINITY
        };
    }

    /// Metropolis 提案角変位の最大幅 δ を変更する
    pub fn set_delta_theta(&mut self, delta_theta: f64) {
        self.delta_theta = delta_theta;
    }

    /// 1 モンテカルロステップ (MCS: N 回の Metropolis 局所更新試行) を実行する
    pub fn step_metropolis(&mut self) -> usize {
        let mut accepted = 0;
        let l = self.l;
        let n = self.n;
        let j = self.j;
        let delta = self.delta_theta;
        let beta = self.beta;

        for _ in 0..n {
            // ランダムにサイトを選択
            let idx = self.rng.gen_range(0..n);
            let x = idx % l;
            let y = idx / l;
            let current_th = self.thetas[idx];

            // 微小変位を提案: θ' = θ + Δθ
            let d_theta = self.rng.gen_range(-delta..delta);
            let mut proposed_th = current_th + d_theta;
            if proposed_th >= PI {
                proposed_th -= TWO_PI;
            } else if proposed_th < -PI {
                proposed_th += TWO_PI;
            }

            // 4 近傍のインデックス
            let right = (if x + 1 == l { 0 } else { x + 1 }) + y * l;
            let left = (if x == 0 { l - 1 } else { x - 1 }) + y * l;
            let up = x + (if y + 1 == l { 0 } else { y + 1 }) * l;
            let down = x + (if y == 0 { l - 1 } else { y - 1 }) * l;

            let th_r = self.thetas[right];
            let th_l = self.thetas[left];
            let th_u = self.thetas[up];
            let th_d = self.thetas[down];

            // エネルギー変化量 ΔE = -J Σ [cos(θ' - θ_j) - cos(θ - θ_j)]
            let delta_e = -j
                * ((proposed_th - th_r).cos()
                    + (proposed_th - th_l).cos()
                    + (proposed_th - th_u).cos()
                    + (proposed_th - th_d).cos()
                    - (current_th - th_r).cos()
                    - (current_th - th_l).cos()
                    - (current_th - th_u).cos()
                    - (current_th - th_d).cos());

            // Metropolis 受託判定
            if delta_e <= 0.0 || self.rng.gen::<f64>() < (-beta * delta_e).exp() {
                self.thetas[idx] = proposed_th;
                accepted += 1;
            }
        }

        accepted
    }

    /// 埋め込みスピン Wolff クラスター法 (Embedded Wolff Cluster Step)
    ///
    /// 1. ランダムな方向角度 φ ∈ [0, 2π) を選定
    /// 2. 各スピンの射影成分 σ_i = cos(θ_i - φ) を用いてクラスターを成長
    /// 3. クラスター内のスピンを φ 軸に関して鏡映反転: θ'_i = wrap_angle(2φ - θ_i)
    ///
    /// 戻り値: 反転したクラスターのサイズ (サイト数)
    pub fn step_wolff(&mut self) -> usize {
        let l = self.l;
        let n = self.n;
        let beta = self.beta;
        let j = self.j;

        // 1. ランダムな射影角度 φ を選ぶ
        let phi = self.rng.gen_range(0.0..TWO_PI);

        // 2. ランダムなシードサイトを選択
        let seed_idx = self.rng.gen_range(0..n);

        let mut in_cluster = vec![false; n];
        let mut cluster_stack = Vec::with_capacity(n);

        cluster_stack.push(seed_idx);
        in_cluster[seed_idx] = true;
        let mut cluster_size = 0;

        // 幅優先/深さ優先でクラスターを成長
        while let Some(current_idx) = cluster_stack.pop() {
            cluster_size += 1;
            let cx = current_idx % l;
            let cy = current_idx / l;
            let th_c = self.thetas[current_idx];
            let proj_c = (th_c - phi).cos();

            // 4 近傍
            let neighbors = [
                (if cx + 1 == l { 0 } else { cx + 1 }) + cy * l, // 右
                (if cx == 0 { l - 1 } else { cx - 1 }) + cy * l, // 左
                cx + (if cy + 1 == l { 0 } else { cy + 1 }) * l, // 上
                cx + (if cy == 0 { l - 1 } else { cy - 1 }) * l, // 下
            ];

            for &nb_idx in &neighbors {
                if in_cluster[nb_idx] {
                    continue;
                }

                let th_nb = self.thetas[nb_idx];
                let proj_nb = (th_nb - phi).cos();
                let product = proj_c * proj_nb;

                // 射影成分が同符号の場合にボンドを結着
                if product > 0.0 {
                    let p_add = 1.0 - (-2.0 * beta * j * product).exp();
                    if self.rng.gen::<f64>() < p_add {
                        in_cluster[nb_idx] = true;
                        cluster_stack.push(nb_idx);
                    }
                }
            }
        }

        // 3. クラスター内の全スピンを鏡映反転: S' = S - 2(S·r)r => θ' = wrap(π + 2φ - θ)
        for i in 0..n {
            if in_cluster[i] {
                self.thetas[i] = wrap_angle(PI + 2.0 * phi - self.thetas[i]);
            }
        }

        cluster_size
    }

    /// 反転スピン総数が N 以上になるまで Wolff ステップを繰り返し、1 MCS 分の更新を行う
    pub fn step_wolff_mcs(&mut self) -> usize {
        let mut total_flipped = 0;
        while total_flipped < self.n {
            total_flipped += self.step_wolff();
        }
        total_flipped
    }

    /// 重複なし全エネルギー E = -J Σ_<i,j> cos(θ_i - θ_j) を計算する (右と上の 2 方向走査)
    pub fn total_energy(&self) -> f64 {
        let l = self.l;
        let mut cos_sum = 0.0;

        for y in 0..l {
            let up_y = if y + 1 == l { 0 } else { y + 1 };
            for x in 0..l {
                let right_x = if x + 1 == l { 0 } else { x + 1 };
                let th = self.thetas[y * l + x];

                // 右隣のボンド
                cos_sum += (th - self.thetas[y * l + right_x]).cos();
                // 上隣のボンド
                cos_sum += (th - self.thetas[up_y * l + x]).cos();
            }
        }

        -self.j * cos_sum
    }

    /// 1 サイトあたりの平均エネルギー e = E / N
    pub fn energy_per_spin(&self) -> f64 {
        self.total_energy() / (self.n as f64)
    }

    /// 全磁化ベクトル (M_x, M_y)
    pub fn magnetization_vector(&self) -> (f64, f64) {
        let mut mx = 0.0;
        let mut my = 0.0;
        for &th in &self.thetas {
            mx += th.cos();
            my += th.sin();
        }
        (mx, my)
    }

    /// 1 スピンあたりの磁化ノルム |m| = sqrt(Mx^2 + My^2) / N
    pub fn magnetization(&self) -> f64 {
        let (mx, my) = self.magnetization_vector();
        (mx * mx + my * my).sqrt() / (self.n as f64)
    }

    /// 各プラケットの渦度 q ∈ {-1, 0, +1} の 1次元 Vec を計算する (Rust 内部・テスト用)
    pub fn get_vorticities(&self) -> Vec<i8> {
        let l = self.l;
        let mut vorts = vec![0i8; self.n];

        for y in 0..l {
            let next_y = if y + 1 == l { 0 } else { y + 1 };
            for x in 0..l {
                let next_x = if x + 1 == l { 0 } else { x + 1 };

                // 4 頂点角度 (反時計回り)
                let th1 = self.thetas[y * l + x]; // (x, y)
                let th2 = self.thetas[y * l + next_x]; // (x+1, y)
                let th3 = self.thetas[next_y * l + next_x]; // (x+1, y+1)
                let th4 = self.thetas[next_y * l + x]; // (x, y+1)

                // 各辺の位相差を [-π, π) に正規化
                let d21 = wrap_angle(th2 - th1);
                let d32 = wrap_angle(th3 - th2);
                let d43 = wrap_angle(th4 - th3);
                let d14 = wrap_angle(th1 - th4);

                let circulation = d21 + d32 + d43 + d14;
                let q = (circulation / TWO_PI).round() as i8;
                vorts[y * l + x] = q;
            }
        }

        vorts
    }

    /// 各プラケットの渦度 q ∈ {-1, 0, +1} を NumPy 2次元配列 (L, L) として計算する (Python 用)
    pub fn compute_vorticities<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<i8>>> {
        let vorts = self.get_vorticities();
        let arr1d = numpy::PyArray1::from_vec(py, vorts);
        arr1d.reshape([self.l, self.l])
    }

    /// 全プラケットの渦度密度 ρ_v = (Σ |q_p|) / N
    pub fn vortex_density(&self) -> f64 {
        let vorts = self.get_vorticities();
        let total_abs_q: i64 = vorts.iter().map(|&q| q.abs() as i64).sum();
        total_abs_q as f64 / (self.n as f64)
    }

    /// スピン角度配列を (L, L) の NumPy 2次元配列として返す (可視化用)
    pub fn get_thetas<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let arr1d = numpy::PyArray1::from_vec(py, self.thetas.clone());
        arr1d.reshape([self.l, self.l])
    }

    /// NumPy 2次元配列 (L, L) からスピン角度を設定する
    pub fn set_thetas(&mut self, array: &Bound<'_, PyArray2<f64>>) -> PyResult<()> {
        let shape = array.shape();
        if shape != [self.l, self.l] {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Shape mismatch: expected [{}, {}], got {:?}",
                self.l, self.l, shape
            )));
        }
        let readonly = array.readonly();
        let slice = readonly.as_slice()?;
        self.thetas.copy_from_slice(slice);
        Ok(())
    }

    /// 局所サイト (x, y) の角度を変更した際のエネルギー変化 ΔE (単体テスト・検証用)
    pub fn delta_energy(&self, x: usize, y: usize, new_theta: f64) -> f64 {
        let l = self.l;
        let idx = y * l + x;
        let current_th = self.thetas[idx];

        let right = (if x + 1 == l { 0 } else { x + 1 }) + y * l;
        let left = (if x == 0 { l - 1 } else { x - 1 }) + y * l;
        let up = x + (if y + 1 == l { 0 } else { y + 1 }) * l;
        let down = x + (if y == 0 { l - 1 } else { y - 1 }) * l;

        let th_r = self.thetas[right];
        let th_l = self.thetas[left];
        let th_u = self.thetas[up];
        let th_d = self.thetas[down];

        -self.j
            * ((new_theta - th_r).cos()
                + (new_theta - th_l).cos()
                + (new_theta - th_u).cos()
                + (new_theta - th_d).cos()
                - (current_th - th_r).cos()
                - (current_th - th_l).cos()
                - (current_th - th_u).cos()
                - (current_th - th_d).cos())
    }

    /// 局所サイト (x, y) のスピン角度を取得する (テスト・可視化用)
    pub fn get_spin(&self, x: usize, y: usize) -> f64 {
        self.thetas[y * self.l + x]
    }

    /// 局所サイト (x, y) のスピン角度を直接設定する (テスト用)
    pub fn set_spin(&mut self, x: usize, y: usize, theta: f64) {
        self.thetas[y * self.l + x] = wrap_angle(theta);
    }
}
