"""2次元XY模型の 3層分離ダイナミクスアニメーションスクリプト.

第1層: スピン角度マップ (連続カラーマップ, マーカーなし)
第2層: スピンベクトル場 (白背景の矢印のみ)
第3層: トポロジカル欠陥マップ (白背景に渦・反渦のみ)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# プロジェクトルートのパス解決
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import Literal
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import mc_core

T_BKT = 0.8929


def main() -> None:
    parser = argparse.ArgumentParser(
        description="2D XY模型の 3層分離スピン & 渦ダイナミクスアニメーション"
    )
    parser.add_argument("--L", type=int, default=24, help="格子サイズ L (default: 24)")
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["wolff", "metropolis"],
        default="wolff",
        help="MCMC アルゴリズム (default: wolff)",
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
        default=240,
        help="合計フレーム数 (default: 240, 30fps で 8 秒)",
    )
    parser.add_argument("--fps", type=int, default=30, help="アニメーションの FPS (default: 30)")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="出力 GIF ファイルパス (省略時は outputs/xy_vortices_dynamics.gif)",
    )
    args = parser.parse_args()

    L = args.L
    N = L * L
    steps_per_frame = args.steps_per_frame
    num_frames = args.frames
    fps = args.fps
    algo = args.algorithm

    output_dir = Path(__file__).resolve().parent.parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    gif_path = (
        Path(args.output)
        if args.output
        else output_dir / "xy_vortices_dynamics.gif"
    )

    # 3つの代表温度
    temperatures = [0.5, 0.9, 1.4]
    temp_labels = [
        f"Low Temp: $T = 0.5 < T_{{\\mathrm{{BKT}}}}$",
        f"Critical: $T = 0.9 \\approx T_{{\\mathrm{{BKT}}}}$",
        f"High Temp: $T = 1.4 > T_{{\\mathrm{{BKT}}}}$",
    ]

    print("=" * 70)
    print("2D XY Model - 3-Tier Spin & Vortex Dynamics Animation")
    print(f"Lattice size: {L} x {L} (N = {N} sites)")
    print(f"Algorithm: {algo.upper()}")
    print(f"Temperatures: {temperatures}")
    print(
        f"Frames: {num_frames} ({steps_per_frame} MCS/frame, total {num_frames * steps_per_frame} MCS)"
    )
    print(f"FPS: {fps} (Duration: {num_frames / fps:.1f} s)")
    print("=" * 70)

    # 格子座標グリッド
    x_coords, y_coords = np.meshgrid(np.arange(L), np.arange(L))

    # Rust コアエンジンの初期化
    engines = [
        mc_core.XY2DRust(l=L, temperature=T, j=1.0, seed=100 + idx)
        for idx, T in enumerate(temperatures)
    ]
    # 初期平衡化
    for engine in engines:
        for _ in range(100):
            if algo == "wolff":
                engine.step_wolff_mcs()
            else:
                engine.step_metropolis()

    # 3行3列の描画ウィンドウ
    # 行 0: スピン角度 (カラーマップのみ)
    # 行 1: スピンベクトル場 (白背景の矢印のみ)
    # 行 2: トポロジカル欠陥マップ (白背景に渦・反渦のみ)
    fig, axes = plt.subplots(3, 3, figsize=(15, 13.5))
    fig.suptitle(
        f"2D XY Model Dynamics [{algo.upper()}] ($L = {L} \\times {L}$)\n"
        "[Top: Spin Wave (Angle) | Middle: Vector Field (Quiver) | Bottom: Topological Defects (Vortices)]",
        fontsize=14,
        fontweight="bold",
    )

    im_artists = []
    quiver_artists = []
    vortex_scatters = []
    antivortex_scatters = []
    text_artists = []

    # 矢印の間引き (見やすさ重視)
    step_skip = max(1, L // 20)
    x_sub = x_coords[::step_skip, ::step_skip]
    y_sub = y_coords[::step_skip, ::step_skip]

    for col, (T, label, engine) in enumerate(zip(temperatures, temp_labels, engines)):
        thetas = engine.get_thetas()
        vorts = engine.compute_vorticities()

        # ----------------------------------------------------
        # 第1層 (Row 0): スピン角度マップ (twilight: 方角を示す周期的色相環)
        # ----------------------------------------------------
        ax_angle = axes[0, col]
        im = ax_angle.imshow(
            thetas,
            cmap="twilight",
            vmin=-np.pi,
            vmax=np.pi,
            origin="lower",
            interpolation="nearest",
        )
        ax_angle.set_title(f"{label}\nSpin Angle $\\theta$ (Direction)", fontsize=11)
        ax_angle.set_xticks([])
        ax_angle.set_yticks([])
        im_artists.append(im)

        # 物理量テキスト表示
        txt = ax_angle.text(
            0.03,
            0.05,
            "",
            transform=ax_angle.transAxes,
            color="white",
            fontsize=9.0,
            fontfamily="monospace",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.7),
        )
        text_artists.append(txt)

        # ----------------------------------------------------
        # 第2層 (Row 1): スピンベクトル場 (白背景のクリアな矢印のみ)
        # ----------------------------------------------------
        ax_vec = axes[1, col]
        ax_vec.set_facecolor("white")
        u_sub = np.cos(thetas[::step_skip, ::step_skip])
        v_sub = np.sin(thetas[::step_skip, ::step_skip])
        q_artist = ax_vec.quiver(
            x_sub,
            y_sub,
            u_sub,
            v_sub,
            color="#1a237e",
            pivot="mid",
            scale=22,
            width=0.005,
        )
        ax_vec.set_title("Vector Field $(\\cos \\theta, \\sin \\theta)$", fontsize=10)
        ax_vec.set_xlim(-0.5, L - 0.5)
        ax_vec.set_ylim(-0.5, L - 0.5)
        ax_vec.set_xticks([])
        ax_vec.set_yticks([])
        quiver_artists.append(q_artist)

        # ----------------------------------------------------
        # 第3層 (Row 2): トポロジカル欠陥マップ (渦: 赤丸, 反渦: 青四角)
        # ----------------------------------------------------
        ax_def = axes[2, col]
        ax_def.set_facecolor("#fafafa")
        ax_def.grid(True, linestyle=":", color="#e0e0e0", alpha=0.7)

        v_y, v_x = np.where(vorts == 1)
        sc_v = ax_def.scatter(
            v_x + 0.5,
            v_y + 0.5,
            c="#ff1744",
            marker="o",
            s=80,
            edgecolors="black",
            linewidths=1.2,
            zorder=5,
        )
        vortex_scatters.append(sc_v)

        av_y, av_x = np.where(vorts == -1)
        sc_av = ax_def.scatter(
            av_x + 0.5,
            av_y + 0.5,
            c="#2979ff",
            marker="s",
            s=80,
            edgecolors="black",
            linewidths=1.2,
            zorder=5,
        )
        antivortex_scatters.append(sc_av)

        ax_def.set_title("Vortices [Red $\\oplus$: +1 | Blue $\\ominus$: -1]", fontsize=10)
        ax_def.set_xlim(0, L)
        ax_def.set_ylim(0, L)
        ax_def.set_xticks([])
        ax_def.set_yticks([])

    plt.tight_layout()

    def update(frame: int):
        current_mcs = (frame + 1) * steps_per_frame
        updated_artists = []

        for col, engine in enumerate(engines):
            for _ in range(steps_per_frame):
                if algo == "wolff":
                    engine.step_wolff_mcs()
                else:
                    engine.step_metropolis()

            thetas = engine.get_thetas()
            vorts = engine.compute_vorticities()

            # 1. 角度マップ更新
            im_artists[col].set_data(thetas)
            updated_artists.append(im_artists[col])

            # 2. 矢印更新
            u_sub = np.cos(thetas[::step_skip, ::step_skip])
            v_sub = np.sin(thetas[::step_skip, ::step_skip])
            quiver_artists[col].set_UVC(u_sub, v_sub)
            updated_artists.append(quiver_artists[col])

            # 3. 渦マーカー更新
            v_y, v_x = np.where(vorts == 1)
            v_offsets = (
                np.column_stack([v_x + 0.5, v_y + 0.5])
                if len(v_x) > 0
                else np.empty((0, 2))
            )
            vortex_scatters[col].set_offsets(v_offsets)
            updated_artists.append(vortex_scatters[col])

            av_y, av_x = np.where(vorts == -1)
            av_offsets = (
                np.column_stack([av_x + 0.5, av_y + 0.5])
                if len(av_x) > 0
                else np.empty((0, 2))
            )
            antivortex_scatters[col].set_offsets(av_offsets)
            updated_artists.append(antivortex_scatters[col])

            # 物理量テキスト更新
            e = engine.energy_per_spin()
            m = engine.magnetization()
            nv = len(v_x)
            nav = len(av_x)
            text_artists[col].set_text(
                f"MCS: {current_mcs:03d}\n"
                f"e: {e:+.2f} | |m|: {m:.2f}\n"
                f"Vortices: +{nv} / -{nav}"
            )
            updated_artists.append(text_artists[col])

        return updated_artists

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=num_frames,
        interval=1000 // fps,
        blit=True,
    )

    print(f"\nRendering 3-tier animation to: {gif_path}")
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
