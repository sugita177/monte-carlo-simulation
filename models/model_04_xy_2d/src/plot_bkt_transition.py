"""2次元XY模型の BKT 転移解析スクリプト (比熱ピークと渦密度の温度依存性)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# プロジェクトルートのパス解決
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import Literal
import matplotlib.pyplot as plt
import numpy as np

from models.model_04_xy_2d.src.simulation import run_temperature_sweep

# BKT 転移温度の理論値 (Gupta & Baillie 1992, Hasenbusch 2005)
T_BKT = 0.8929
# 比熱の丸い極大値が現れる温度 (渦対が大量解離する温度)
T_CV_PEAK = 1.05


def plot_bkt_physics(
    temperatures: np.ndarray,
    energies: np.ndarray,
    specific_heats: np.ndarray,
    magnetizations: np.ndarray,
    vortex_densities: np.ndarray,
    lattice_size: int,
    algorithm: str,
    output_path: Path,
) -> None:
    """4 パネル (2x2) で 2次元XY模型の BKT 転移物理量をプロットする."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(
        f"2D XY Model: BKT Topological Transition ($L = {lattice_size} \\times {lattice_size}$, {algorithm.upper()})",
        fontsize=15,
        fontweight="bold",
    )

    # (a) 比熱 C_V(T): T_BKT と比熱ピークの乖離
    ax_cv = axes[0, 0]
    ax_cv.plot(temperatures, specific_heats, "o-", color="#d62728", lw=2, ms=4)
    ax_cv.axvline(
        T_BKT,
        color="purple",
        linestyle="--",
        lw=2,
        label=f"Theoretical $T_{{\\mathrm{{BKT}}}} \\approx {T_BKT:.3f}$",
    )
    ax_cv.axvline(
        T_CV_PEAK,
        color="gray",
        linestyle=":",
        lw=1.5,
        label=f"Specific Heat Peak $T \\approx {T_CV_PEAK:.2f}$",
    )
    ax_cv.set_title("(a) Specific Heat $C_V(T)$", fontsize=13)
    ax_cv.set_xlabel("Temperature $T / J$", fontsize=11)
    ax_cv.set_ylabel("$C_V / (N k_B)$", fontsize=11)
    ax_cv.legend(fontsize=10, loc="upper right")
    ax_cv.grid(True, linestyle="--", alpha=0.6)

    # (b) 渦度密度 rho_v(T): T_BKT 以上での急激な渦解離
    ax_v = axes[0, 1]
    ax_v.plot(temperatures, vortex_densities, "s-", color="#1f77b4", lw=2, ms=4)
    ax_v.axvline(
        T_BKT,
        color="purple",
        linestyle="--",
        lw=2,
        label=f"$T_{{\\mathrm{{BKT}}}} \\approx {T_BKT:.3f}$",
    )
    ax_v.set_title("(b) Vortex Density $\\rho_v(T) = \\langle \\sum |q_p| \\rangle / N$", fontsize=13)
    ax_v.set_xlabel("Temperature $T / J$", fontsize=11)
    ax_v.set_ylabel("Vortex Density $\\rho_v$", fontsize=11)
    ax_v.legend(fontsize=10, loc="upper left")
    ax_v.grid(True, linestyle="--", alpha=0.6)

    # (c) 1サイトあたり内部エネルギー <e>(T): T -> 0 で -2.0 へ収束
    ax_e = axes[1, 0]
    ax_e.plot(temperatures, energies, "o-", color="#2ca02c", lw=2, ms=4)
    ax_e.axhline(
        -2.0,
        color="black",
        linestyle="--",
        lw=1.5,
        label="Ground State $e_0 = -2.0$",
    )
    ax_e.axvline(
        T_BKT,
        color="purple",
        linestyle="--",
        lw=2,
        label=f"$T_{{\\mathrm{{BKT}}}} \\approx {T_BKT:.3f}$",
    )
    ax_e.set_title("(c) Internal Energy $\\langle e \\rangle(T)$", fontsize=13)
    ax_e.set_xlabel("Temperature $T / J$", fontsize=11)
    ax_e.set_ylabel("Energy per spin $E / (N J)$", fontsize=11)
    ax_e.legend(fontsize=10, loc="lower right")
    ax_e.grid(True, linestyle="--", alpha=0.6)

    # (d) 磁化ノルム <|m|>(T): 有限サイズ効果と熱ゆらぎによる減衰
    ax_m = axes[1, 1]
    ax_m.plot(temperatures, magnetizations, "^-", color="#ff7f0e", lw=2, ms=4)
    ax_m.axvline(
        T_BKT,
        color="purple",
        linestyle="--",
        lw=2,
        label=f"$T_{{\\mathrm{{BKT}}}} \\approx {T_BKT:.3f}$",
    )
    ax_m.set_title("(d) Magnetization Norm $\\langle |\\vec{m}| \\rangle(T)$", fontsize=13)
    ax_m.set_xlabel("Temperature $T / J$", fontsize=11)
    ax_m.set_ylabel("Magnetization $\\langle |\\vec{m}| \\rangle$", fontsize=11)
    ax_m.set_ylim(-0.05, 1.05)
    ax_m.legend(fontsize=10, loc="upper right")
    ax_m.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"\n[Saved] Plot successfully saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="2D XY Model BKT Transition Plotter")
    parser.add_argument("--L", type=int, default=32, help="格子サイズ L (default: 32)")
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["wolff", "metropolis"],
        default="wolff",
        help="MCMC アルゴリズム (default: wolff)",
    )
    parser.add_argument(
        "--mcs-therm",
        type=int,
        default=2000,
        help="平衡化 MCS 数 (default: 2000)",
    )
    parser.add_argument(
        "--mcs-measure",
        type=int,
        default=5000,
        help="計測 MCS 数 (default: 5000)",
    )
    parser.add_argument(
        "--sample-interval",
        type=int,
        default=5,
        help="サンプリング間隔 (default: 5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="出力先画像パス (省略時は outputs/bkt_transition.png)",
    )
    args = parser.parse_args()

    # 温度グリッドの構築: T_BKT 付近と比熱ピーク付近 (0.4 から 1.8) を密にサンプリング
    # 低温側・ピーク付近をバランス良く配置
    temps = np.concatenate([
        np.linspace(0.4, 0.7, 5, endpoint=False),
        np.linspace(0.7, 1.3, 15, endpoint=False),  # 転移・ピーク付近を重点
        np.linspace(1.3, 1.8, 6),
    ])

    print("=" * 65)
    print("2D XY Model - BKT Topological Transition Simulation")
    print(f"Lattice size: {args.L} x {args.L} (N = {args.L * args.L})")
    print(f"Algorithm: {args.algorithm.upper()}")
    print(f"Temperature points: {len(temps)} (T from {temps.min():.2f} to {temps.max():.2f})")
    print(f"MCS: {args.mcs_therm} (thermalize) + {args.mcs_measure} (measure)")
    print("=" * 65)

    results = run_temperature_sweep(
        lattice_size=args.L,
        temperatures=temps,
        mcs_thermalize=args.mcs_therm,
        mcs_measure=args.mcs_measure,
        sample_interval=args.sample_interval,
        algorithm=args.algorithm,
    )

    t_arr = np.array([r.temperature for r in results])
    e_arr = np.array([r.mean_energy_per_spin for r in results])
    cv_arr = np.array([r.specific_heat for r in results])
    m_arr = np.array([r.mean_magnetization for r in results])
    v_arr = np.array([r.vortex_density for r in results])

    output_dir = Path(__file__).resolve().parent.parent / "outputs"
    output_path = Path(args.output) if args.output else output_dir / "bkt_transition.png"

    plot_bkt_physics(
        temperatures=t_arr,
        energies=e_arr,
        specific_heats=cv_arr,
        magnetizations=m_arr,
        vortex_densities=v_arr,
        lattice_size=args.L,
        algorithm=args.algorithm,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
