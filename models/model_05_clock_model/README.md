# Model 05: 2次元正方格子 $q$ 状態クロック模型 ($q$-state Clock Model)

## 1. モデルの定義とハミルトニアン

### 定義
2次元正方格子（$N = L \times L$ サイト、周期境界条件）において、各格子点 $i$ のスピンは単位円上の等間隔な $q$ 個の離散角度のみをとるスピン模型です：

$$\theta_i \in \left\{ \frac{2\pi k}{q} \;\middle|\; k = 0, 1, \dots, q-1 \right\}$$

各サイトのスピンベクトルは単位長 2成分ベクトル $\vec{S}_i = (\cos\theta_i, \sin\theta_i)$ で表されます。
整数インデックス $k_i \in \{0, 1, \dots, q-1\}$ を状態変数として保持します。

### ハミルトニアン
最隣接スピン対 $\langle i, j \rangle$ 間に強磁性的交換相互作用 $J > 0$（以下 $J=1$）が働くとき、系の全エネルギーは次式で与えられます：

$$H = -J \sum_{\langle i, j \rangle} \vec{S}_i \cdot \vec{S}_j = -J \sum_{\langle i, j \rangle} \cos(\theta_i - \theta_j)$$

外場 $\vec{h} = (h_x, h_y)$ が存在する場合の拡張：
$$H = -J \sum_{\langle i, j \rangle} \cos(\theta_i - \theta_j) - \sum_{i} \vec{h} \cdot \vec{S}_i$$

---

## 2. 物理的背景・対称性と特殊極限

クロック模型は、**イジング模型（離散 $\mathbb{Z}_2$ 対称性）** と **XY 模型（連続 $U(1) \cong O(2)$ 対称性）** を架橋する極めて重要な格子模型です。

### 特殊な $q$ における厳密な対応関係
1. **$q = 2$ (2次元イジング模型)**:
   - 角度は $\theta \in \{0, \pi\} \implies \cos(0)=1, \cos(\pi)=-1$。
   - $s_i = \cos\theta_i \in \{-1, +1\}$ と置けば、ハミルトニアンは厳密に $H = -J \sum_{\langle i,j \rangle} s_i s_j$。
   - 転移温度: Onsager 厳密解 $T_c = \frac{2}{\ln(1+\sqrt{2})} \approx 2.269$。
2. **$q = 3$ (3状態強磁性 Potts 模型)**:
   - 角度は $\theta \in \{0, 2\pi/3, 4\pi/3\}$。
   - スピン対の内積は $\cos(0) = 1$、$\cos(\pm 2\pi/3) = -1/2$。
   - Potts 変数 $\sigma_i \in \{0, 1, 2\}$ とクロネッカーのデルタ $\delta_{\sigma_i, \sigma_j}$ を用いると：
     $$\cos(\theta_i - \theta_j) = \frac{3}{2} \delta_{\sigma_i, \sigma_j} - \frac{1}{2}$$
   - 定数項のシフトを除き、3状態 Potts 模型と厳密に等価。転移温度 $T_c = \frac{3}{2 \ln(1+\sqrt{3})} \approx 1.492$（2次相転移）。
3. **$q = 4$ (独立な2つのイジング模型)**:
   - スピンベクトルは $(\pm 1/\sqrt{2}, \pm 1/\sqrt{2})$ の4方向。
   - $\vec{S}_i = \frac{1}{\sqrt{2}}(\sigma_i, \tau_i)$（$\sigma_i, \tau_i \in \{-1, +1\}$）と直交分解すると：
     $$\vec{S}_i \cdot \vec{S}_j = \frac{1}{2}(\sigma_i \sigma_j + \tau_i \tau_j)$$
   - 結合定数 $J/2$ の独立な 2 つのイジング模型に直積分解される。
   - 転移温度: $T_c = T_c^{\mathrm{Ising}} / 2 \approx 1.134$。
4. **$q \to \infty$ (2次元 XY 模型)**:
   - 角度の離散性が消失し、連続 $U(1)$ 対称性を持つ 2次元 XY 模型に厳密に一致。
   - BKT 転移温度 $T_{\mathrm{BKT}} \approx 0.893$。

---

## 3. $q \ge 5$ における 2 段階相転移と中間 BKT 相

$q \ge 5$ において、クロック模型は統計物理学上最も魅惑的な現象の一つである **「2段階相転移」** を示します。

```text
温度 T:  0 ------------- T_c1 ------------- T_c2 ------------- ∞
相:       長距離秩序相      中間臨界相 (BKT相)      高温無秩序相
         (強磁性相)        (準長距離秩序相)        (常磁性相)
対称性:  Z_q 自発的破れ    擬似 U(1) 連続対称性     Z_q 対称性回復
秩序:    <|m|> > 0         <|m|> = 0               <|m|> = 0
相関:    C(r) -> const     C(r) ~ r^(-η(T))        C(r) ~ e^(-r/ξ)
トポロジー: 渦励起なし       渦・反渦対が束縛        渦・反渦が解離 (自由プラズマ)
```

### 繰り込み群（RG）による理論的解釈
連続 XY 模型のハミルトニアンに、$\mathbb{Z}_q$ 異方性を表す摂動項 $V(\theta) = -h_q \cos(q\theta)$ を加えた有効 Sine-Gordon 模型を考えます：

$$\mathcal{H}_{\mathrm{eff}} = \int d^2 r \left[ \frac{K}{2} |\nabla \theta|^2 - h_q \cos(q\theta) \right]$$

この異方性項 $h_q$ のスケーリング次元（RG固有値）は次式で決まります：
$$y_q = 2 - \frac{q^2}{4\pi K}$$

- **低温側 ($T < T_{c1}$)**:
  $K$ が大きく $y_q > 0$ となるため、離散異方性 $h_q$ が **Relevant（増大）** します。スピンは $q$ 個の離散的な谷に落ち込み、$\mathbb{Z}_q$ 対称性が自発的に破れた **有限磁化相（長距離秩序相）** が安定化します。
- **中間相 ($T_{c1} < T < T_{c2}$)**:
  $q \ge 5$ では、異方性が **Irrelevant（減衰してゼロへ流れる）** となる温度領域が有限の幅で存在します。この領域では系は離散性を忘れ、あたかも連続 $U(1)$ 対称性を持つ XY 模型のように振る舞います。
  - スピン波ゆらぎにより有限磁化は消失（$\langle |\vec{m}| \rangle \to 0$）。
  - しかし、渦・反渦対は静電ポテンシャル的に束縛（Bound Pair）されており、相関関数は代数的に減衰（$C(r) \sim r^{-\eta(T)}$）する **準長距離秩序相（BKT臨界相）** を形成します。
- **高温側 ($T > T_{c2}$)**:
  渦の化学ポテンシャルが負となり、渦・反渦対が解離（Unbinding）して自由渦プラズマが生じ、通常の高温常磁性相へと転移します。

### 代表的な転移温度の文献値 ($q=6$)
- $T_{c1} \approx 0.70$（長距離秩序 $\to$ BKT相転移）
- $T_{c2} \approx 0.90$（BKT相 $\to$ 常磁性相転移）

---

## 4. モンテカルロアルゴリズムの設計

### ルックアップテーブル (LUT) による高速化
クロック模型の離散角度差 $\Delta \theta = \frac{2\pi}{q} (k_i - k_j)$ は、整数インデックスの差 $\Delta k = (k_i - k_j) \bmod q$ だけで決まります。
そのため、実行時に重い $\cos$ 関数を呼び出さず、あらかじめ長さ $q$ の配列：
$$\text{LUT}[d] = \cos\left(\frac{2\pi d}{q}\right) \quad (d = 0, 1, \dots, q-1)$$
を用意しておくことで、エネルギー計算や遷移判定を極めて高速（整数の差分アクセスのみ）に実行できます。

### 1. メトロポリス法 (Metropolis Update)
1. サイト $i$（現在のスピン状態 $k_i \in \{0, \dots, q-1\}$）を選択。
2. 提案方法（両方を選択可能）：
   - **一様ランダム提案 (`random`)**: $k_i' \in \{0, \dots, q-1\} \setminus \{k_i\}$ から一様ランダムに選択（高温・小 $q$ で探索効率が高い）。
   - **局所隣接提案 (`adjacent`)**: $k_i' = (k_i \pm 1) \bmod q$（各確率 1/2。低温・大 $q$ で受託率を維持）。
3. エネルギー変化量 $\Delta E$ の計算：
   $$\Delta E = -J \sum_{j \in \mathrm{nn}(i)} \left[ \text{LUT}[(k_i' - k_j + q) \bmod q] - \text{LUT}[(k_i - k_j + q) \bmod q] \right]$$
4. 受託確率：
   $$P_{\mathrm{acc}} = \min\left(1, e^{-\beta \Delta E}\right)$$
   一様乱数 $r \sim U(0, 1) < P_{\mathrm{acc}}$ ならば $k_i \leftarrow k_i'$ と更新。

---

## 5. 観測すべき物理量

1. **内部エネルギー $e$ と比熱 $C_v$**:
   $$e = \frac{\langle H \rangle}{N}, \quad C_v = \frac{N}{T^2} \left( \langle e^2 \rangle - \langle e \rangle^2 \right)$$
   $q \ge 5$ では、$T_{c1}$ 付近と $T_{c2}$ 付近で異なる比熱の振る舞い（肩やブロードなピーク）が観測されます。
2. **磁化ベクトル $\vec{m}$ と磁化ノルム $\langle |\vec{m}| \rangle$**:
   $$\vec{m} = \frac{1}{N} \sum_{i=1}^N (\cos\theta_i, \sin\theta_i), \quad |\vec{m}| = \sqrt{m_x^2 + m_y^2}$$
   - $T < T_{c1}$: $\langle |\vec{m}| \rangle > 0$（非ゼロの自発磁化）
   - $T > T_{c1}$: $\langle |\vec{m}| \rangle \to 0$（熱力学極限で磁化消失）
3. **クロック配向秩序変数 (Clock Orientation Order Parameter)**:
   全磁化ベクトルの配向角 $\Psi = \mathrm{atan2}(m_y, m_x)$ が $q$ 個の好ましい離散軸（$2\pi k / q$）に固定されている度合いを測定：
   $$\Phi_q = \langle \cos(q \Psi) \rangle$$
   - 低温長距離秩序相 ($T < T_{c1}$): 特定の離散軸に自発的に固定されるため $\Phi_q \approx 1$。
   - 中間 BKT 相 ($T_{c1} < T < T_{c2}$): 擬似連続 $U(1)$ 対称性により $\Psi$ は円周上を均等にゆらぎ、$\Phi_q \approx 0$。
4. **プラケット渦度密度 $\rho_v$**:
   隣接スピン間の位相差周回積分（XY模型と同様の定義）により渦度 $q_p = \pm 1$ を検出し、単位格子あたりの渦の密度 $\rho_v = \frac{1}{N}\sum_p |q_p|$ を測定。
   - $T < T_{c2}$: 渦対は強く束縛され低密度。
   - $T > T_{c2}$: 渦対が解離し $\rho_v$ が急激に立ち上がる。

---

## 6. ディレクトリ構成

本モデルは、高速計算コアを Rust で実装し、PyO3 を介して Python から制御・可視化する構成をとります。

```text
monte-carlo-simulation/
├── crates/mc_core/                 # 高速シミュレーションエンジン (Rust)
│   ├── Cargo.toml
│   ├── src/
│   │   ├── lib.rs                  # Python モジュール公開定義
│   │   ├── clock_2d.rs             # 2D クロック模型コア実装 (LUT, Metropolis)
│   │   └── ...
│   └── tests/
│       ├── test_clock_2d.rs        # Rust 単体テスト (q=2 Onsager極限, 詳細釣り合い)
│       └── ...
│
└── models/model_05_clock_model/    # クロック模型解析・可視化 (Python)
    ├── README.md                   # 本ドキュメント（理論・アルゴリズム・実験手順）
    ├── src/
    │   ├── __init__.py
    │   ├── simulation.py           # シミュレーション実行・サンプリング制御
    │   ├── observables.py          # 物理量集計 (エネルギー, 磁化, クロック秩序度, 渦度)
    │   ├── plot_two_step.py        # 2段階相転移の温度スイープ解析プロット
    │   └── animate_clock.py        # 離散スピン配位・渦ダイナミクスのアニメーション
    ├── tests/
    │   ├── __init__.py
    │   └── test_clock_model.py     # Python 統合テスト
    └── outputs/                    # 解析結果プロット・GIF保存先
```
