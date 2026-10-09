"""三角格子反強磁性イジング模型のスピン緩和、局所エネルギー、フラストレーション欠陥のアニメーションスクリプト."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# プロジェクトルートを sys.path に追加（直接実行対応）
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import mc_core


def create_triangular_mesh(L: int) -> tuple[np.ndarray, np.ndarray]:
    """三角格子の (L+1) x (L+1) 実空間頂点メッシュ座標 (X, Y) を生成する.

    並進ベクトル: a1 = (1, 0), a2 = (1/2, sqrt(3)/2)
    各サイト (x, y) が 60度/120度の菱形セルとして敷き詰められるよう、
    境界頂点座標を半格子分シフト (-0.5) して計算する。
    """
    i_coords = np.arange(L + 1) - 0.5
    j_coords = np.arange(L + 1) - 0.5
    i_grid, j_grid = np.meshgrid(i_coords, j_coords)

    X_mesh = i_grid + 0.5 * j_grid
    Y_mesh = (np.sqrt(3.0) / 2.0) * j_grid
    return X_mesh, Y_mesh


def compute_local_energy(spins: np.ndarray) -> np.ndarray:
    """各サイトの局所相互作用エネルギー e_i = 1/2 * s_i * sum_{j in nb} s_j を計算する.

    戻り値:
        e_i in {-3, -2, -1, 0, 1, 2, 3}
        全サイトの和 sum(e_i) は全体の全エネルギー E に厳密に一致する。
    """
    s = spins
    nb_sum = (
        np.roll(s, -1, axis=1)  # 右 (x+1, y)
        + np.roll(s, 1, axis=1)  # 左 (x-1, y)
        + np.roll(s, -1, axis=0)  # 上 (x, y+1)
        + np.roll(s, 1, axis=0)  # 下 (x, y-1)
        + np.roll(np.roll(s, -1, axis=1), 1, axis=0)  # 右下 (x+1, y-1)
        + np.roll(np.roll(s, 1, axis=1), -1, axis=0)  # 左上 (x-1, y+1)
    )
    return 0.5 * s * nb_sum


def compute_frustration_defects(spins: np.ndarray) -> np.ndarray:
    """各サイトが関与する『励起三角形 (+++ または ---)』の個数を集計する (0〜6).

    正三角形の 3 スピンの和 |s1 + s2 + s3|:
      - 1 のとき: 2アップ1ダウン / 1アップ2ダウン (基底状態: 不満ボンド1本) -> 欠陥なし
      - 3 のとき: 3アップ / 3ダウン (熱励起状態: 不満ボンド3本) -> 欠陥あり (高エネルギー)

    低温基底状態 (Wannier アンサンブル) では全三角形が 1 となり全系で欠陥が 0 (漆黒) となる。
    高温では確率 25% で励起三角形が生じ、激しい欠陥ノイズとなる。
    """
    s = spins
    s_right = np.roll(s, -1, axis=1)
    s_up = np.roll(s, -1, axis=0)
    s_right_down = np.roll(np.roll(s, -1, axis=1), 1, axis=0)

    # 1. 上向き三角形: (x, y), (x+1, y), (x, y+1)
    sum_up = s + s_right + s_up
    defect_up = (np.abs(sum_up) == 3).astype(np.float64)

    # 2. 下向き三角形: (x, y), (x+1, y), (x+1, y-1)
    sum_down = s + s_right + s_right_down
    defect_down = (np.abs(sum_down) == 3).astype(np.float64)

    # 各サイトが属する 6 つの三角形の欠陥数を集計
    site_defects = (
        defect_up
        + np.roll(defect_up, 1, axis=1)
        + np.roll(defect_up, 1, axis=0)
        + defect_down
        + np.roll(defect_down, 1, axis=1)
        + np.roll(np.roll(defect_down, 1, axis=1), -1, axis=0)
    )
    return site_defects


def main() -> None:
    parser = argparse.ArgumentParser(
        description="三角格子反強磁性イジング模型の実空間スピン緩和・局所エネルギー・欠陥アニメーション"
    )
    parser.add_argument("--L", type=int, default=40, help="格子の一辺のサイズ (default: 40)")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["all", "energy", "defect"],
        default="all",
        help="表示モード: 'all' (3段: スピン+エネルギー+欠陥), 'energy' (2段: スピン+エネルギー), 'defect' (2段: スピン+欠陥) (default: all)",
    )
    parser.add_argument(
        "--steps-per-frame",
        type=int,
        default=2,
        help="1フレームあたりに進める MCS 数 (default: 2)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=300,
        help="合計フレーム数 (計 frames * steps-per-frame MCS, default: 300)",
    )
    parser.add_argument("--fps", type=int, default=30, help="アニメーションの FPS (default: 30)")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="出力 GIF ファイルパス (省略時は outputs/triangular_af_relaxation[_mode].gif)",
    )
    args = parser.parse_args()

    L = args.L
    N = L * L
    mode = args.mode
    steps_per_frame = args.steps_per_frame
    num_frames = args.frames
    fps = args.fps

    output_dir = Path(__file__).resolve().parent.parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.output:
        gif_path = Path(args.output)
    else:
        suffix = "" if mode == "all" else f"_{mode}"
        gif_path = output_dir / f"triangular_af_relaxation{suffix}.gif"

    # 3つの代表温度
    temperatures = [0.2, 1.2, 3.5]
    temp_labels = [
        "Low Temp: $T = 0.2$ (Wannier Ensemble)",
        "Peak Temp: $T = 1.2$ ($C_V$ Peak / Schottky)",
        "High Temp: $T = 3.5$ (Paramagnetic)",
    ]

    print("=" * 70)
    print("Triangular AF Ising Model - Relaxation Dynamics Animation")
    print(f"Mode: {mode.upper()} ({'3-tier' if mode == 'all' else '2-tier'})")
    print(f"Lattice size: {L} x {L} (N = {N} sites)")
    print(f"Temperatures: {temperatures}")
    print(
        f"Frames: {num_frames} ({steps_per_frame} MCS/frame, total {num_frames * steps_per_frame} MCS)"
    )
    print(f"FPS: {fps} (Duration: {num_frames / fps:.1f} s)")
    print("=" * 70)

    # 実空間頂点メッシュの生成
    X_mesh, Y_mesh = create_triangular_mesh(L)

    # Rust コアエンジンの初期化 (全系ランダムスピンからスタート)
    engines = [
        mc_core.IsingTriangularRust(l=L, temperature=T, j_af=1.0, seed=100 + idx)
        for idx, T in enumerate(temperatures)
    ]
    for engine in engines:
        engine.initialize_spins("random")

    # 行数の決定
    n_rows = 3 if mode == "all" else 2
    row_height = 4.2
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, row_height * n_rows))
    if n_rows == 1:
        axes = np.array([axes])

    subtitle = {
        "all": "[Top: Spin Config | Middle: Local Energy Density | Bottom: Frustration Defect Map]",
        "energy": "[Top: Spin Config | Bottom: Local Energy Density]",
        "defect": "[Top: Spin Config | Bottom: Frustration Defect Map]",
    }[mode]

    fig.suptitle(
        f"Triangular Antiferromagnetic Ising Model Dynamics ($L = {L} \\times {L}$)\n{subtitle}",
        fontsize=14,
        fontweight="bold",
    )

    spin_meshes = []
    energy_meshes = []
    defect_meshes = []
    text_artists = []

    for idx, (T, label) in enumerate(zip(temperatures, temp_labels)):
        engine = engines[idx]
        spins = engine.get_spins()

        # 1. 上段 (Row 0): スピン配位 (+1: 赤, -1: 青)
        ax_spin = axes[0, idx]
        m_spin = ax_spin.pcolormesh(
            X_mesh, Y_mesh, spins, cmap="bwr", vmin=-1, vmax=1, shading="flat"
        )
        ax_spin.set_aspect("equal")
        ax_spin.set_title(f"{label}\nSpin Configuration $(\\pm 1)$", fontsize=11)
        ax_spin.set_xticks([])
        ax_spin.set_yticks([])
        spin_meshes.append(m_spin)

        # 物理量テキスト
        txt = ax_spin.text(
            0.03,
            0.05,
            "",
            transform=ax_spin.transAxes,
            color="white",
            fontsize=9.0,
            fontfamily="monospace",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.65),
        )
        text_artists.append(txt)

        # 2. 中段 / 下段の割り当て
        current_row = 1
        if mode in ["all", "energy"]:
            local_e = compute_local_energy(spins)
            ax_e = axes[current_row, idx]
            m_e = ax_e.pcolormesh(
                X_mesh,
                Y_mesh,
                local_e,
                cmap="coolwarm",
                vmin=-3.0,
                vmax=3.0,
                shading="flat",
            )
            ax_e.set_aspect("equal")
            ax_e.set_title(
                "Local Energy $e_i \\in [-3, +3]$\n[Blue: Stable | Red: Frustrated]",
                fontsize=10,
            )
            ax_e.set_xticks([])
            ax_e.set_yticks([])
            energy_meshes.append(m_e)
            current_row += 1

        if mode in ["all", "defect"]:
            defects = compute_frustration_defects(spins)
            ax_def = axes[current_row, idx]
            m_def = ax_def.pcolormesh(
                X_mesh,
                Y_mesh,
                defects,
                cmap="inferno",
                vmin=0,
                vmax=4,
                shading="flat",
            )
            ax_def.set_aspect("equal")
            ax_def.set_title(
                "Frustration Defects (Excited $\\Delta$ Count)\n[Black: Ground Rule | Bright: Defect]",
                fontsize=10,
            )
            ax_def.set_xticks([])
            ax_def.set_yticks([])
            defect_meshes.append(m_def)

    plt.tight_layout()

    def update(frame: int):
        current_mcs = (frame + 1) * steps_per_frame
        updated_artists = []

        for idx, engine in enumerate(engines):
            for _ in range(steps_per_frame):
                engine.step_metropolis()

            spins = engine.get_spins()
            spin_meshes[idx].set_array(spins.ravel())

            if mode in ["all", "energy"]:
                local_e = compute_local_energy(spins)
                energy_meshes[idx].set_array(local_e.ravel())
                updated_artists.append(energy_meshes[idx])

            if mode in ["all", "defect"]:
                defects = compute_frustration_defects(spins)
                defect_meshes[idx].set_array(defects.ravel())
                updated_artists.append(defect_meshes[idx])

            e = engine.energy_per_spin()
            m = engine.magnetization_per_spin()
            text_artists[idx].set_text(
                f"MCS: {current_mcs:03d}\ne: {e:+.2f} | m: {m:+.2f}"
            )

            updated_artists.extend([spin_meshes[idx], text_artists[idx]])

        return updated_artists

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=num_frames,
        interval=1000 // fps,
        blit=True,
    )

    print(f"\nRendering animation to: {gif_path}")
    writer = animation.PillowWriter(fps=fps)

    with tqdm(total=num_frames, desc="Rendering GIF") as pbar:
        ani.save(
            gif_path,
            writer=writer,
            progress_callback=lambda current_frame, total_frames: pbar.update(1),
        )

    plt.close()
    print(f"\n[Success] Animation successfully saved to: {gif_path}")


if __name__ == "__main__":
    main()
