"""Real-time animation of 2D Ising model spin relaxation at different temperatures."""

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

from models.model_01_ising_2d.src.ising import Ising2D

# 厳密な臨界温度
T_C = 2.0 / np.log(1.0 + np.sqrt(2.0))


def main() -> None:
    output_dir = Path(__file__).resolve().parent.parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    gif_path = output_dir / "ising_relaxation.gif"

    # シミュレーション設定
    L = 50
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
    print(f"Lattice size: {L} x {L} (N = {L*L})")
    print(f"Temperatures: {temperatures}")
    print(f"Frames: {num_frames} ({steps_per_frame} MCS/frame, total {num_frames * steps_per_frame} MCS)")
    print("=" * 60)

    # 3つのシミュレーションエンジンの初期化 (すべてランダム初期状態から)
    engines = [
        Ising2D(L=L, temperature=T, J=1.0, h=0.0, seed=42 + idx)
        for idx, T in enumerate(temperatures)
    ]
    for engine in engines:
        engine.initialize_spins(method="random")

    # 描画ウィンドウの設定
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        f"2D Ising Model Relaxation Dynamics ($L = {L} \\times {L}$)",
        fontsize=15,
        fontweight="bold",
    )

    im_artists = []
    text_artists = []

    for idx, (ax, engine, title) in enumerate(zip(axes, engines, titles)):
        # スピン (+1: 赤, -1: 青) の初期描画
        im = ax.imshow(
            engine.spins,
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
                engine.step_metropolis()

            # 画像データとテキストを更新
            im.set_data(engine.spins)
            mag = engine.total_magnetization() / engine.N
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
