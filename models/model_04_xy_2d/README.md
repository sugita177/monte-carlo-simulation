# 2次元XY模型 (2D XY Model / $O(2)$ Spin System)
連続対称性、BKT転移（トポロジカル相転移）、渦・反渦の検出

---

## 1. モデルの定義とハミルトニアン

2次元XY模型は、正方格子上の各サイト $i = (x, y)$ に、$xy$ 平面内で 360 度自由に回転できる 2 次元単位ベクトル（古典スピン）が配置された系です。

$$
\vec{S}_i = (\cos \theta_i, \sin \theta_i), \quad \theta_i \in [-\pi, \pi)
$$

### 1.1. ハミルトニアン
最近接格子点対 $\langle i, j \rangle$ 間の強磁性相互作用（$J > 0$）を考慮すると、全エネルギーはスピン同士の内積で定義されます：

$$
H = -J \sum_{\langle i, j \rangle} \vec{S}_i \cdot \vec{S}_j = -J \sum_{\langle i, j \rangle} \cos(\theta_i - \theta_j)
$$

外場 $\vec{h} = (h, 0)$ が印加されている場合：
$$
H = -J \sum_{\langle i, j \rangle} \cos(\theta_i - \theta_j) - h \sum_i \cos \theta_i
$$

---

## 2. 物理的背景と相転移の性質

### 2.1. 連続対称性と Mermin-Wagner の定理
イジング模型の離散対称性（$\mathbb{Z}_2$：反転対称性）と異なり、XY模型のハミルトニアンは全スピンを一斉に任意の角度 $\phi$ だけ回転させても不変です（**$U(1) \cong SO(2)$ 連続大域対称性**）。

> **Mermin-Wagner の定理 (1966)**:  
> 2次元以下の系で、短距離相互作用を持つ連続対称な系では、**有限温度（$T > 0$）において自発的対称性の破れは起こらない**。  
> すなわち、熱ゆらぎ（長波長のスピン波励起）が長距離秩序を破壊するため、全磁化 $\langle \vec{M} \rangle = \frac{1}{N} \sum_i \langle \vec{S}_i \rangle = 0$ であり、有限温度での強磁性長距離秩序は存在しません。

### 2.2. BKT 転移（ベレジンスキー・コステリッツ・サウレス転移）
長距離秩序が存在しないにもかかわらず、2次元XY模型は **有限温度 $T_{\mathrm{BKT}} \approx 0.893 J$ において相転移を起こします**（2016年ノーベル物理学賞）。  
これは通常の「対称性の破れと秩序パラメータ」で記述される相転移ではなく、**トポロジカル欠陥である「渦（Vortex）」の振る舞いによって駆動されるトポロジカル相転移** です。

| 状態 | 低温相 ($T < T_{\mathrm{BKT}}$) | 高温相 ($T > T_{\mathrm{BKT}}$) |
| :--- | :--- | :--- |
| **主たる励起** | スピン波（調和振動）＋ **束縛された渦・反渦対** | **解離した自由な渦・反渦** |
| **スピン相関 $C(r)$** | **代数的減衰（べき乗則）** $C(r) \sim r^{-\eta(T)}$（準長距離秩序・臨界相） | **指数関数的減衰** $C(r) \sim \exp(-r/\xi)$（無秩序相） |
| **渦の振る舞い** | 渦（$+1$）と反渦（$-1$）が対（Pairs）を作り、遠距離からは相殺 | 対が解離（Unbinding）し、孤立した渦が自由に浮遊 |
| **秩序変数** | 自発磁化はゼロ $\langle \|\vec{m}\| \rangle \to 0$ | 自発磁化はゼロ $\langle \|\vec{m}\| \rangle \to 0$ |
| **スピン硬さ $\Upsilon$** | 有界（有限の剛性を持つ） | ゼロ（流体のように柔らかい） |

### 2.3. BKT 転移の重要な落とし穴：比熱ピークとのズレ
- 通常の 2 次相転移（イジング模型）では、臨界温度 $T_c$ で比熱 $C_V$ が発散ピークを示します。
- しかし、BKT 転移では、本質的な転移温度 $T_{\mathrm{BKT}} \approx 0.893$ において自由エネルギーの特異性は無限階微分可能（本質的特異点）であり、比熱の鋭い発散は生じません。
- 比熱のブロードな極大値（丸いピーク）は、渦対が大量に解離する熱的効果により **$T \approx 1.05$ 付近の高温側にズレて現れます**。
- したがって、**比熱ピークの位置から $T_{\mathrm{BKT}}$ を決定してはならない** という統計物理学上の重要な教訓が存在します。

---

## 3. トポロジカル欠陥：渦（Vortex）と渦度（Vorticity）の数理

正方格子の最小閉路（プラケット：$1 \times 1$ の正方形セル）を反時計回りに一周したときの、スピン角度の位相変化量を積算します。

サイト $1 \to 2 \to 3 \to 4 \to 1$ の反時計回りループにおいて、隣り合うボンドの角度差を $\Delta \theta_{j, i} = \theta_j - \theta_i$ とします。  
ここで、角度差は周期境界条件を考慮して主値区間 $[-\pi, \pi)$ に引き戻します：

$$
\delta \theta_{j, i} = \operatorname{wrap}(\theta_j - \theta_i) \equiv (\theta_j - \theta_i + \pi \pmod{2\pi}) - \pi
$$

このとき、プラケット周囲の循環和（Circulation）を $2\pi$ で割った値が **渦度（Topological Charge / Vorticity $q$）** です：

$$
q = \frac{1}{2\pi} \oint d\theta = \frac{1}{2\pi} \left( \delta \theta_{2, 1} + \delta \theta_{3, 2} + \delta \theta_{4, 3} + \delta \theta_{1, 4} \right)
$$

- **$q = +1$**: **渦（Vortex）**（スピンが反時計回りに $2\pi$ 渦巻いている）
- **$q = -1$**: **反渦（Antivortex）**（スピンが時計回りに $2\pi$ 渦巻いている）
- **$q = 0$**: 通常のプラケット（渦なし）

低温では渦と反渦が必ず隣接して中和（$+1$ と $-1$ のペア）していますが、高温では解離して孤立した渦がランダムに飛び交う様子を可視化できます。

---

## 4. モンテカルロアルゴリズムの設計

連続スピン系では、イジングのような離散スピン反転ではなく、角度の更新アルゴリズムが必要です。

### 4.1. 局所 Metropolis 法
サイト $i$ の現在のスピン角度 $\theta_i$ に対し、微小なランダム変位 $\Delta \theta \in [-\delta, +\delta]$ を加えた新候補 $\theta'_i = \theta_i + \Delta \theta$ を提案します。

局所有効磁場ベクトル：
$$
\vec{H}_{\mathrm{eff}} = J \sum_{j \in \text{nb}(i)} \vec{S}_j = \left( J \sum_{j} \cos \theta_j, \; J \sum_{j} \sin \theta_j \right)
$$

エネルギー変化量：
$$
\Delta E = -\vec{S}'_i \cdot \vec{H}_{\mathrm{eff}} - (-\vec{S}_i \cdot \vec{H}_{\mathrm{eff}}) = -(\vec{S}'_i - \vec{S}_i) \cdot \vec{H}_{\mathrm{eff}}
$$

Metropolis 受託確率 $P_{\mathrm{acc}} = \min(1, e^{-\beta \Delta E})$ で更新します。

### 4.2. 埋め込みスピン Wolff クラスター法 (Embedded Wolff Algorithm)
連続スピン系でも、ランダムな単位ベクトル $\vec{r} = (\cos \phi, \sin \phi)$ を選び、その射影軸に関するスピン反転（鏡映）を行うことで、Wolff クラスター法を適用できます：

1. 一様ランダムな方向ベクトル $\vec{r}$ を生成。
2. 各スピン $\vec{S}_i$ を $\vec{r}$ に直交する超平面に関して鏡映反転：
   $$
   \vec{S}'_i = \vec{S}_i - 2 (\vec{S}_i \cdot \vec{r}) \vec{r}
   $$
3. ボンド結着確率：
   $$
   P_{\mathrm{add}} = 1 - \exp\left( -2 \beta J (\vec{S}_i \cdot \vec{r}) (\vec{S}_j \cdot \vec{r}) \right) \quad (\text{if } (\vec{S}_i \cdot \vec{r})(\vec{S}_j \cdot \vec{r}) > 0)
   $$
これにより、BKT 転移付近の臨界減速（Critical Slowing Down）を劇的に抑制して超高速サンプリングが可能となります。

---

## 5. 観測すべき物理量と検証項目

1. **内部エネルギー $e = E / N$ および比熱 $C_V(T)$**:
   - $T \approx 1.05$ のブロードなピークと、$T_{\mathrm{BKT}} \approx 0.89$ との乖離の検証。
2. **渦度密度（Vortex Density $\rho_v$）**:
   - $\rho_v = \frac{1}{N} \sum_p |q_p|$。
   - 低温で急速に 0 に近づき、高温で急増する振る舞いの追跡。
3. **スピン相関関数 $C(r) = \langle \vec{S}_0 \cdot \vec{S}_r \rangle$**:
   - 低温相でのべき乗減衰 $r^{-\eta}$ vs 高温相での指数減衰 $e^{-r/\xi}$。
4. **渦・反渦の実空間ベクトル場 & トポロジカルチャージの可視化**:
   - ベクトル場（矢印）または角度カラーマップの上に、渦（赤丸 $\oplus$）と反渦（青四角 $\ominus$）をオーバーレイ描画。

---

## 6. ディレクトリ構造

```text
monte-carlo-simulation/
├── crates/mc_core/
│   ├── src/
│   │   ├── lib.rs
│   │   └── xy_2d.rs                 # [Rust] コア計算エンジン (Metropolis / Wolff / 渦度計算)
│   └── tests/
│       └── test_xy_2d.rs            # [Rust] コア単体テスト (幾何学的エネルギー・局所差分検証)
└── models/model_04_xy_2d/
    ├── README.md                    # 本ファイル (理論背景・数理定式化・検証計画)
    ├── src/
    │   ├── simulation.py            # [Python] 温度スイープ・サンプリング実行制御
    │   ├── observables.py           # [Python] 観測量集計 (エネルギー, 比熱, 渦密度)
    │   ├── plot_bkt_transition.py   # [Python] BKT転移解析プロット (比熱ピーク, 渦解離)
    │   └── animate_vortices.py      # [Python] スピンベクトル場 ＋ 渦対の実空間アニメーション
    └── tests/
        └── test_xy_model.py         # [Python] 統合テスト (PyO3 バインディング検証)
```

---

## 7. テスト・検証

```bash
# Rust 単体テスト
uv run cargo test --test test_xy_2d --no-default-features

# Python 統合テスト
uv run pytest models/model_04_xy_2d/tests/
```

