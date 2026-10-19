"""2次元 q状態クロック模型の 3層分離ダイナミクスアニメーションスクリプト.

第1層 (Row 0): スピン角度マップ (周期的色相環: スピン波・ドメイン構造)
第2層 (Row 1): スピンベクトル場 (白背景のクリアな矢印ベクトル quiver)
第3層 (Row 2): トポロジカル欠陥マップ (渦: 赤丸 +1 / 反渦: 青四角 -1)

3つの代表的な温度領域を横並びで比較します：
1. 低温強磁性相 (T=0.4 < T_c1): 離散軸に揃った巨大ドメイン、欠陥なし
2. 中間 BKT 相 (T=0.8, T_c1 < T < T_c2): 擬似連続 U(1) スピン波、緊密な渦対
3. 高温無秩序相 (T=1.4 > T_c2): 激しい位相ゆらぎ、解離した自由渦プラズマ
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

from mc_core import Clock2DRust

T_C1_Q6 = 0.70
T_C2_Q6 = 0.90


def compute_vorticities(angles: np.ndarray) -> np.ndarray:
    """スピン角度配列 (L, L) から各プラケットの渦度 q_p ∈ {-1, 0, +1} を算出する."""
    th1 = angles
    th2 = np.roll(angles, shift=-1, axis=1)  # (x+1, y)
    th3 = np.roll(np.roll(angles, shift=-1, axis=1), shift=-1, axis=0)  # (x+1, y+1)
    th4 = np.roll(angles, shift=-1, axis=0)  # (x, y+1)

    # 角度差を [-π, π) に正規化
    def wrap_angle(d: np.ndarray) -> np.ndarray:
        return (d + np.pi) % (2.0 * np.pi) - np.pi

    d21 = wrap_angle(th2 - th1)
    d32 = wrap_angle(th3 - th2)
    d43 = wrap_angle(th4 - th3)
    d14 = wrap_angle(th1 - th4)

    circulation = d21 + d32 + d43 + d14
    return np.round(circulation / (2.0 * np.pi)).astype(np.int8)


def create_3tier_animation(
    L: int = 24,
    q: int = 6,
    frames: int = 240,
    steps_per_frame: int = 2,
    fps: int = 30,
    output_path: Path | None = None,
) -> None:
    """3層×3温度のダイナミクスアニメーションを生成する."""
    temperatures = [0.4, 0.8, 1.4]
    temp_labels = [
        f"Low-T (T=0.4 < $T_{{c1}}$)\n[Ordered Ferromagnetic]",
        f"Intermediate (T=0.8)\n[BKT Quasi-LRO / Spin-Wave]",
        f"High-T (T=1.4 > $T_{{c2}}$)\n[Disordered Paramagnetic]",
    ]

    print(f"=== Initializing 2D {q}-State Clock Model ({L}x{L}) for 3-Tier Animation ===")
    sims = [
        Clock2DRust(l=L, q=q, j=1.0, seed=100 + i * 50, initial="random")
        for i in range(3)
    ]
    betas = [1.0 / T for T in temperatures]

    # 初期熱平衡化 (各 1500 MCS)
    print("Thermalizing initial configurations...")
    for sim, beta in zip(sims, betas):
        for _ in range(1500):
            sim.step_metropolis(beta, "adjacent")

    # グリッド座標の準備
    x_coords, y_coords = np.meshgrid(np.arange(L), np.arange(L))
    step_skip = max(1, L // 20)
    x_sub = x_coords[::step_skip, ::step_skip]
    y_sub = y_coords[::step_skip, ::step_skip]

    # 3行3列の描画ウィンドウ
    fig, axes = plt.subplots(3, 3, figsize=(15, 13.5))
    fig.suptitle(
        f"2D {q}-State Clock Model Dynamics ($L = {L} \\times {L}$)\n"
        "[Top: Spin Wave (Angle) | Middle: Vector Field (Quiver) | Bottom: Topological Defects (Vortices)]",
        fontsize=14,
        fontweight="bold",
    )

    im_artists = []
    quiver_artists = []
    vortex_scatters = []
    antivortex_scatters = []
    text_angle_artists = []
    text_defect_artists = []

    for col, (T, label, sim) in enumerate(zip(temperatures, temp_labels, sims)):
        angles = sim.get_angles()
        vorts = compute_vorticities(angles)

        # ----------------------------------------------------
        # 第1層 (Row 0): スピン角度マップ (twilight: 方角を示す周期的色相環)
        # ----------------------------------------------------
        ax_angle = axes[0, col]
        im = ax_angle.imshow(
            angles,
            cmap="twilight",
            vmin=0.0,
            vmax=2.0 * np.pi,
            origin="lower",
            interpolation="nearest",
        )
        ax_angle.set_title(f"{label}\nSpin Angle $\\theta$", fontsize=11, fontweight="bold")
        ax_angle.set_xticks([])
        ax_angle.set_yticks([])
        im_artists.append(im)

        txt_angle = ax_angle.text(
            0.03,
            0.05,
            "",
            transform=ax_angle.transAxes,
            color="white",
            fontsize=8.5,
            fontfamily="monospace",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.7),
        )
        text_angle_artists.append(txt_angle)

        # ----------------------------------------------------
        # 第2層 (Row 1): スピンベクトル場 (白背景のクリアな矢印)
        # ----------------------------------------------------
        ax_vec = axes[1, col]
        ax_vec.set_facecolor("white")
        u_sub = np.cos(angles[::step_skip, ::step_skip])
        v_sub = np.sin(angles[::step_skip, ::step_skip])
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

        vy, vx = np.where(vorts > 0)
        sc_v = ax_def.scatter(
            vx,
            vy,
            c="#d32f2f",
            s=45,
            marker="o",
            edgecolors="black",
            linewidths=0.6,
            zorder=3,
            label="Vortex (+1)" if col == 0 else "",
        )
        vortex_scatters.append(sc_v)

        avy, avx = np.where(vorts < 0)
        sc_av = ax_def.scatter(
            avx,
            avy,
            c="#1976d2",
            s=45,
            marker="s",
            edgecolors="black",
            linewidths=0.6,
            zorder=3,
            label="Anti-Vortex (-1)" if col == 0 else "",
        )
        antivortex_scatters.append(sc_av)

        ax_def.set_xlim(-0.5, L - 0.5)
        ax_def.set_ylim(-0.5, L - 0.5)
        ax_def.set_xticks([])
        ax_def.set_yticks([])
        ax_def.set_title("Topological Defects", fontsize=10)

        if col == 0:
            ax_def.legend(loc="upper right", fontsize=8, framealpha=0.85)

        txt_def = ax_def.text(
            0.03,
            0.05,
            "",
            transform=ax_def.transAxes,
            color="#212121",
            fontsize=8.5,
            fontfamily="monospace",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#bdbdbd", alpha=0.85),
        )
        text_defect_artists.append(txt_def)

    plt.subplots_adjust(left=0.04, right=0.96, top=0.91, bottom=0.03, hspace=0.18, wspace=0.10)

    pbar = tqdm(total=frames, desc="Generating 3-Tier Animation Frames")

    def update(_frame_idx: int) -> list[plt.Artist]:
        updated_artists = []

        for col in range(3):
            sim = sims[col]
            beta = betas[col]

            # 1 フレームあたり指定 MCS だけ更新を進める
            for _ in range(steps_per_frame):
                sim.step_metropolis(beta, "adjacent")

            angles = sim.get_angles()
            vorts = compute_vorticities(angles)

            mx, my, norm_m = sim.magnetization()
            e_per_spin = sim.total_energy() / (L * L)
            psi = np.arctan2(my, mx)
            phi_q = np.cos(q * psi)

            # 1. 角度マップ更新
            im_artists[col].set_data(angles)
            updated_artists.append(im_artists[col])

            text_angle_artists[col].set_text(
                f"e={e_per_spin:+.2f} | |m|={norm_m:.2f}\nΦ_q={phi_q:+.2f}"
            )
            updated_artists.append(text_angle_artists[col])

            # 2. ベクトル場更新
            u = np.cos(angles[::step_skip, ::step_skip])
            v = np.sin(angles[::step_skip, ::step_skip])
            quiver_artists[col].set_UVC(u, v)
            updated_artists.append(quiver_artists[col])

            # 3. 欠陥マップ更新
            vy, vx = np.where(vorts > 0)
            if len(vx) > 0:
                vortex_scatters[col].set_offsets(np.column_stack([vx, vy]))
            else:
                vortex_scatters[col].set_offsets(np.empty((0, 2)))
            updated_artists.append(vortex_scatters[col])

            avy, avx = np.where(vorts < 0)
            if len(avx) > 0:
                antivortex_scatters[col].set_offsets(np.column_stack([avx, avy]))
            else:
                antivortex_scatters[col].set_offsets(np.empty((0, 2)))
            updated_artists.append(antivortex_scatters[col])

            n_v = len(vx)
            n_av = len(avx)
            text_defect_artists[col].set_text(f"Vortices: +{n_v} / -{n_av}")
            updated_artists.append(text_defect_artists[col])

        pbar.update(1)
        return updated_artists

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=1000 // fps,
        blit=True,
    )

    if output_path is None:
        output_path = (
            Path(__file__).resolve().parents[1] / "outputs" / "clock_dynamics.gif"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving 3-Tier animation to: {output_path} (FPS={fps}, Frames={frames})...")
    anim.save(str(output_path), writer="pillow", fps=fps)
    plt.close()
    pbar.close()
    print(f"[Done] 3-Tier animation successfully saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="2D クロック模型 3層分離ダイナミクスアニメーション生成")
    parser.add_argument("--L", type=int, default=24, help="格子サイズ L (default: 24)")
    parser.add_argument("--q", type=int, default=6, help="クロック状態数 q (default: 6)")
    parser.add_argument("--frames", type=int, default=200, help="フレーム数 (default: 200)")
    parser.add_argument("--steps-per-frame", type=int, default=2, help="1フレームあたりの MCS 数 (default: 2)")
    parser.add_argument("--fps", type=int, default=30, help="アニメーション FPS (default: 30)")
    parser.add_argument("--output", type=str, default=None, help="出力 GIF パス")

    args = parser.parse_args()

    output_path = Path(args.output) if args.output else None
    create_3tier_animation(
        L=args.L,
        q=args.q,
        frames=args.frames,
        steps_per_frame=args.steps_per_frame,
        fps=args.fps,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
