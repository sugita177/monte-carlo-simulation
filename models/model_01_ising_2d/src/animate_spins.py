"""Real-time animation of 2D Ising model spin relaxation at different temperatures."""

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
from models.model_01_ising_2d.src.ising import Ising2D

# 厳密な臨界温度
T_C = 2.0 / np.log(1.0 + np.sqrt(2.0))


def main() -> None:
    parser = argparse.ArgumentParser(description="2D Ising Model Spin Relaxation Animator")
    parser.add_argument(
        "--engine",
        type=str,
        choices=["rust", "python"],
        default="rust",
        help="Engine implementation to use (default: rust)",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["wolff", "metropolis"],
        default="wolff",
        help="Sampling algorithm to animate (default: wolff)",
    )
    args = parser.parse_args()

    # ガード条件: Python エンジンでの Wolff は未サポート
    if args.engine == "python" and args.algorithm == "wolff":
        raise NotImplementedError(
            "Wolff algorithm is currently only implemented in Rust engine. "
            "Please use --engine rust or specify --algorithm metropolis for python."
        )

    algo_name = args.algorithm
    algo_display = "Wolff Cluster Algorithm" if algo_name == "wolff" else "Metropolis Algorithm"
    engine_name = args.engine

    output_dir = Path(__file__).resolve().parent.parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    gif_path = output_dir / f"ising_relaxation_{engine_name}_{algo_name}.gif"

    # シミュレーション設定
    L = 50
    N = L * L
    steps_per_frame = 2  # 1フレームあたりに進める MCS 数
    num_frames = 500  # 合計フレーム数 (計 1000 MCS)
    fps = 30

    temperatures = [1.5, T_C, 3.5]
    titles = [
        f"Low Temp: $T = 1.5 < T_c$\n(Ferromagnetic Ordering)",
        f"Critical: $T = T_c \\approx {T_C:.3f}$\n(Critical Fluctuations)",
        f"High Temp: $T = 3.5 > T_c$\n(Paramagnetic / Disorder)",
    ]

    print("=" * 60)
    print("2D Ising Model - Spin Configuration Dynamics Animation")
    print(f"Engine: {engine_name.upper()}")
    print(f"Algorithm: {algo_display}")
    print(f"Lattice size: {L} x {L} (N = {N})")
    print(f"Temperatures: {temperatures}")
    print(f"Frames: {num_frames} ({steps_per_frame} MCS/frame, total {num_frames * steps_per_frame} MCS)")
    print("=" * 60)

    # 3つのシミュレーションエンジンの初期化
    if engine_name == "rust":
        engines = [
            mc_core.Ising2DRust(L, temperature=T, j=1.0, h=0.0, seed=42 + idx)
            for idx, T in enumerate(temperatures)
        ]
    else:
        engines = [
            Ising2D(L=L, temperature=T, J=1.0, h=0.0, seed=42 + idx)
            for idx, T in enumerate(temperatures)
        ]

    for engine in engines:
        engine.initialize_spins("random")

    # 描画ウィンドウの設定
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        f"2D Ising Model Relaxation Dynamics [{engine_name.upper()} | {algo_display}] ($L = {L} \\times {L}$)",
        fontsize=15,
        fontweight="bold",
    )

    im_artists = []
    text_artists = []

    for idx, (ax, engine, title) in enumerate(zip(axes, engines, titles)):
        # スピン (+1: 赤, -1: 青) の初期描画
        im = ax.imshow(
            engine.get_spins(),
            cmap="bwr",
            vmin=-1,
            vmax=1,
            interpolation="nearest",
            animated=True,
        )
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        im_artists.append(im)

        # 磁化・ステップ数のテキスト表示
        txt = ax.text(
            0.03,
            0.05,
            "",
            transform=ax.transAxes,
            color="white",
            fontsize=10,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6),
        )
        text_artists.append(txt)

    plt.tight_layout()

    def update(frame: int):
        current_mcs = (frame + 1) * steps_per_frame
        updated_artists = []

        for idx, (im, txt, engine) in enumerate(zip(im_artists, text_artists, engines)):
            # スピン配位を更新
            for _ in range(steps_per_frame):
                if algo_name == "metropolis":
                    engine.step_metropolis()
                elif algo_name == "wolff":
                    flipped = 0
                    while flipped < N:
                        flipped += engine.step_wolff()

            # 画像データとテキストを更新
            im.set_data(engine.get_spins())
            mag = engine.total_magnetization() / N
            txt.set_text(f"MCS: {current_mcs:03d} | m: {mag:+.2f}")

            updated_artists.extend([im, txt])

        return updated_artists

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=num_frames,
        interval=1000 // fps,
        blit=True,
    )

    print(f"\nGenerating and saving animation to: {gif_path}")
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
