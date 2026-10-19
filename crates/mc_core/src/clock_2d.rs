//! 2次元正方格子 q 状態クロック模型 (q-state Clock Model) の Rust 実装
//!
//! スピン変数は整数 k_i ∈ {0, 1, ..., q-1} を保持し、
//! 角度は θ_i = 2π k_i / q に対応します。
//! cos/sin の計算は事前生成したルックアップテーブル (LUT) を参照して高速化します。

use numpy::{PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::Rng;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use std::f64::consts::PI;

#[pyclass]
pub struct Clock2DRust {
    #[pyo3(get)]
    pub l: usize,
    #[pyo3(get)]
    pub q: usize,
    #[pyo3(get)]
    pub j: f64,
    pub spins: Vec<u8>,
    pub lut_cos: Vec<f64>,
    pub lut_sin: Vec<f64>,
    rng: Xoshiro256PlusPlus,
}

#[pymethods]
impl Clock2DRust {
    /// 新しい 2D クロック模型インスタンスを生成
    #[new]
    #[pyo3(signature = (l, q, j=1.0, seed=None, initial="random"))]
    pub fn new(l: usize, q: usize, j: f64, seed: Option<u64>, initial: &str) -> PyResult<Self> {
        if q < 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "q must be at least 2",
            ));
        }
        if q > 256 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "q must be at most 256 for u8 representation",
            ));
        }

        let n = l * l;
        let mut rng = match seed {
            Some(s) => Xoshiro256PlusPlus::seed_from_u64(s),
            None => Xoshiro256PlusPlus::from_entropy(),
        };

        // 1. ルックアップテーブル (LUT) の構築: d ∈ [0, q-1]
        let mut lut_cos = Vec::with_capacity(q);
        let mut lut_sin = Vec::with_capacity(q);
        let delta_phi = 2.0 * PI / (q as f64);
        for d in 0..q {
            let phi = (d as f64) * delta_phi;
            lut_cos.push(phi.cos());
            lut_sin.push(phi.sin());
        }

        // 2. 初期スピン配位の生成
        let spins = match initial {
            "ground" => vec![0u8; n],
            "random" => {
                let mut s = Vec::with_capacity(n);
                for _ in 0..n {
                    s.push(rng.gen_range(0..q) as u8);
                }
                s
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "initial must be 'random' or 'ground'",
                ))
            }
        };

        Ok(Self {
            l,
            q,
            j,
            spins,
            lut_cos,
            lut_sin,
            rng,
        })
    }

    /// 系全体の全エネルギー H = -J ∑_<i,j> cos(θ_i - θ_j)
    pub fn total_energy(&self) -> f64 {
        // 全格子点を走査し、周期境界条件のもとで各サイト i の「右隣」と「下隣」との
        // 相互作用エネルギー -J * cos(θ_i - θ_j) を足し合わせるロジックを記述してください。
        //
        // - スピン差分インデックス: d = (s_i + self.q - s_j) % self.q
        // - 相互作用: -self.j * self.lut_cos[d]
        let mut energy = 0.0;
        for r in 0..self.l {
            for c in 0..self.l {
                let i = r * self.l + c;
                let right = if c == self.l - 1 { 0 } else { c + 1 };
                let down = if r == self.l - 1 { 0 } else { r + 1 };
                let right_idx = r * self.l + right;
                let down_idx = down * self.l + c;

                let q_i = self.spins[i] as usize;
                let q_right = self.spins[right_idx] as usize;
                let q_down = self.spins[down_idx] as usize;

                let d_right = (q_i + self.q - q_right) % self.q;
                let d_down = (q_i + self.q - q_down) % self.q;
                energy += -self.j * (self.lut_cos[d_right] + self.lut_cos[d_down]);
            }
        }
        energy
    }

    /// 1 モンテカルロステップ (N = L*L 回のサイト更新試行) のメトロポリス法
    /// mode: "random" (一様ランダム提案) または "adjacent" (±1提案)
    pub fn step_metropolis(&mut self, beta: f64, mode: &str) -> PyResult<usize> {
        // N = L * L 回、ランダムにサイト (r, c) を選択してスピン更新を試行する。
        //
        // 手順:
        // 1. ランダムにサイト index = r * L + c を選択
        // 2. mode に応じて新しいスピン候補 new_s を決定
        //    - "adjacent": 現在のスピン s から ±1 (確率 1/2 ずつ, mod q)
        //    - "random": {0..q} \ {s} からランダムに 1 つ選択
        // 3. 4つの最隣接サイト (上, 下, 左, 右) との相互作用変化量 ΔE を計算:
        //    ΔE = -J * ∑_{nn} (lut_cos[(new_s + q - s_nn) % q] - lut_cos[(s + q - s_nn) % q])
        // 4. メトロポリス受託判定:
        //    ΔE <= 0.0 または乱数 r < exp(-beta * ΔE) ならば self.spins[index] = new_s (受託数をカウント)
        // 5. 受託された試行数を返す
        let mut accepted = 0;
        let l = self.l;
        let n = l * l;

        for _ in 0..n {
            let r = self.rng.gen_range(0..l);
            let c = self.rng.gen_range(0..l);
            let index = r * l + c;
            let old_s = self.spins[index];
            let new_s = match mode {
                "random" => self.rng.gen_range(0..self.q) as u8,
                "adjacent" => {
                    let s = old_s as isize;
                    if self.rng.gen() {
                        ((s + 1) % self.q as isize) as u8
                    } else {
                        ((s - 1 + self.q as isize) % self.q as isize) as u8
                    }
                }
                _ => return Err(PyValueError::new_err("mode must be 'random' or 'adjacent'")),
            };
            let delta_e = self.delta_energy(r, c, new_s);
            if delta_e <= 0.0 || self.rng.gen::<f64>() < ((-beta * delta_e).exp()) {
                self.spins[index] = new_s;
                accepted += 1;
            }
        }
        Ok(accepted)
    }

    /// 磁化ベクトル (mx, my) とそのノルム |m| を計算
    /// m_x = (1/N) ∑ cos(θ_i),  m_y = (1/N) ∑ sin(θ_i)
    pub fn magnetization(&self) -> (f64, f64, f64) {
        // 全サイトを走査し、lut_cos と lut_sin を足し合わせて平均磁化ベクトルと大きさを返します。
        let n = (self.l * self.l) as f64;
        let mut mx = 0.0;
        let mut my = 0.0;
        for s in self.spins.iter() {
            mx += self.lut_cos[*s as usize];
            my += self.lut_sin[*s as usize];
        }
        mx /= n;
        my /= n;
        let m = (mx * mx + my * my).sqrt();
        (mx, my, m)
    }

    /// 特定サイト (r, c) のスピンを設定 (テスト・初期化用)
    pub fn set_spin(&mut self, r: usize, c: usize, val: u8) {
        self.spins[r * self.l + c] = val % (self.q as u8);
    }

    /// 特定サイト (r, c) のスピンを取得
    pub fn get_spin(&self, r: usize, c: usize) -> u8 {
        self.spins[r * self.l + c]
    }

    /// 局所サイト (r, c) のスピンを new_s に変更した際のエネルギー変化 ΔE
    /// 4つの最隣接サイト (上, 下, 左, 右) との相互作用変化:
    /// ΔE = -J * ∑_{nn} [cos(θ_new - θ_nn) - cos(θ_old - θ_nn)]
    pub fn delta_energy(&self, r: usize, c: usize, new_s: u8) -> f64 {
        // サイト (r, c) の 4つの最隣接サイト (up, down, left, right) について、
        // 変更前スピン old_s と変更後スピン new_s の相互作用差分を計算する
        let old_s = self.spins[r * self.l + c];
        let mut delta_energy = 0.0;
        let neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)];
        for (dr, dc) in neighbors {
            let nr = (r as isize + dr + self.l as isize) % self.l as isize;
            let nc = (c as isize + dc + self.l as isize) % self.l as isize;
            let neighbor_spin = self.spins[nr as usize * self.l + nc as usize] as usize;
            delta_energy += -self.j
                * (self.lut_cos[(new_s as usize + self.q - neighbor_spin) % self.q]
                    - self.lut_cos[(old_s as usize + self.q - neighbor_spin) % self.q]);
        }
        delta_energy
    }

    /// スピン状態 (0 .. q-1) を 2次元 NumPy 配列 (L x L) として取得
    pub fn get_spins<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<u8>>> {
        let arr1d = numpy::PyArray1::from_vec(py, self.spins.clone());
        arr1d.reshape([self.l, self.l])
    }

    /// 2次元 NumPy 配列 (L x L) からスピン状態 (0 .. q-1) を設定
    pub fn set_spins(&mut self, array: &Bound<'_, PyArray2<u8>>) -> PyResult<()> {
        let shape = array.shape();
        if shape != [self.l, self.l] {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Shape mismatch: expected [{}, {}], got {:?}",
                self.l, self.l, shape
            )));
        }
        let readonly = array.readonly();
        let slice = readonly.as_slice()?;
        self.spins.copy_from_slice(slice);
        Ok(())
    }

    /// スピンの連続角度 θ_i = 2π k_i / q を 2次元 NumPy 配列 (L x L) として取得
    pub fn get_angles<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let delta_phi = 2.0 * PI / (self.q as f64);
        let angles: Vec<f64> = self.spins.iter().map(|&s| (s as f64) * delta_phi).collect();
        let arr1d = numpy::PyArray1::from_vec(py, angles);
        arr1d.reshape([self.l, self.l])
    }
}
