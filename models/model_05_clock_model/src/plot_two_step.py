"""2次元 q状態クロック模型の 2段階相転移解析・可視化スクリプト.

q >= 5 (特に q=6) における以下の 2 段階相転移を検証します：
1. T_c1 (≈ 0.70): 低温長距離秩序相 (Ferromagnetic) -> 中間 BKT 相 (Quasi-LRO / Spin-wave)
2. T_c2 (≈ 0.90): 中間 BKT 相 -> 高温無秩序相 (Paramagnetic)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# プロジェクトルートのパス解決
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from models.model_05_clock_model.src.observables import ClockSimulationResult
from models.model_05_clock_model.src.simulation import run_temperature_sweep

# q=6 の文献値転移温度
T_C1_Q6 = 0.70
T_C2_Q6 = 0.90


def plot_transitions(
    results: list[ClockSimulationResult],
    output_path: Path,
    q: int,
) -> None:
    """収集した観測量から 4 パネルの 2段階相転移プロットを生成・保存する."""
    temps = np.array([r.temperature for r in results])
    energies = np.array([r.mean_energy_per_spin for r in results])
    specific_heats = np.array([r.specific_heat for r in results])
    magnetizations = np.array([r.mean_magnetization for r in results])
    susceptibilities = np.array([r.susceptibility for r in results])
    clock_orders = np.array([r.clock_orientation for r in results])
    acceptances = np.array([r.acceptance_rate for r in results])

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=True)
    plt.subplots_adjust(hspace=0.32, wspace=0.30)

    # 3相の領域区分け用 (q=6 の場合)
    def add_phase_shading(ax: plt.Axes) -> None:
        if q == 6:
            ax.axvline(T_C1_Q6, color="crimson", linestyle="--", alpha=0.7, label=f"$T_{{c1}} \\approx {T_C1_Q6}$")
            ax.axvline(T_C2_Q6, color="navy", linestyle="--", alpha=0.7, label=f"$T_{{c2}} \\approx {T_C2_Q6}$")
            t_min, t_max = temps.min(), temps.max()
            ax.axvspan(t_min, T_C1_Q6, color="lightcoral", alpha=0.15)
            ax.axvspan(T_C1_Q6, T_C2_Q6, color="gold", alpha=0.15)
            ax.axvspan(T_C2_Q6, t_max, color="lightblue", alpha=0.15)

    # 1. 左上: 内部エネルギー <e> (左軸) & 比熱 C_v (右軸)
    ax1 = axes[0, 0]
    ax1.plot(temps, energies, "o-", color="tab:blue", label="Energy $\\langle e \\rangle$")
    ax1.set_ylabel("Energy per Spin $\\langle e \\rangle$", color="tab:blue", fontsize=11)
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.tick_params(axis="x", labelbottom=True)
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax1_twin = ax1.twinx()
    ax1_twin.plot(temps, specific_heats, "s-", color="tab:red", label="Specific Heat $C_v$")
    ax1_twin.set_ylabel("Specific Heat $C_v$", color="tab:red", fontsize=11)
    ax1_twin.tick_params(axis="y", labelcolor="tab:red")
    add_phase_shading(ax1)
    ax1.set_title("(a) Energy & Specific Heat", fontsize=12, fontweight="bold")

    # 2. 右上: 磁化ノルム <|m|> (左軸) & 帯磁率 χ (右軸)
    ax2 = axes[0, 1]
    ax2.plot(temps, magnetizations, "o-", color="tab:purple", label="Magnetization $\\langle |m| \\rangle$")
    ax2.set_ylabel("Magnetization $\\langle |m| \\rangle$", color="tab:purple", fontsize=11)
    ax2.tick_params(axis="y", labelcolor="tab:purple")
    ax2.tick_params(axis="x", labelbottom=True)
    ax2.grid(True, linestyle=":", alpha=0.6)

    ax2_twin = ax2.twinx()
    ax2_twin.plot(temps, susceptibilities, "^-", color="tab:orange", label="Susceptibility $\\chi$")
    ax2_twin.set_ylabel("Susceptibility $\\chi$", color="tab:orange", fontsize=11)
    ax2_twin.tick_params(axis="y", labelcolor="tab:orange")
    add_phase_shading(ax2)
    ax2.set_title("(b) Magnetization & Susceptibility", fontsize=12, fontweight="bold")

    # 3. 左下: クロック配向秩序変数 Φ_q = <cos(q * Ψ)>
    ax3 = axes[1, 0]
    ax3.plot(temps, clock_orders, "D-", color="darkgreen", label="Clock Order $\\Phi_q = \\langle \\cos(q\\Psi) \\rangle$")
    ax3.axhline(0.0, color="gray", linestyle=":", alpha=0.7)
    ax3.set_xlabel("Temperature $T$", fontsize=11)
    ax3.set_ylabel("Clock Order Parameter $\\Phi_q$", color="darkgreen", fontsize=11)
    ax3.tick_params(axis="y", labelcolor="darkgreen")
    ax3.grid(True, linestyle=":", alpha=0.6)
    add_phase_shading(ax3)
    ax3.set_title("(c) Clock Orientation (Discrete Anisotropy)", fontsize=12, fontweight="bold")

    # 4. 右下: メトロポリス受託率 (Acceptance Rate)
    ax4 = axes[1, 1]
    ax4.plot(temps, acceptances * 100.0, "v-", color="tab:brown", label="Acceptance Rate (%)")
    ax4.set_xlabel("Temperature $T$", fontsize=11)
    ax4.set_ylabel("Acceptance Rate (%)", color="tab:brown", fontsize=11)
    ax4.tick_params(axis="y", labelcolor="tab:brown")
    ax4.grid(True, linestyle=":", alpha=0.6)
    add_phase_shading(ax4)
    ax4.set_title("(d) Metropolis Acceptance Rate", fontsize=12, fontweight="bold")

    # 全体タイトルと注釈
    L = results[0].L
    phase_text = " [Phase I: Ferro | Phase II: BKT Quasi-LRO | Phase III: Para]" if q == 6 else ""
    fig.suptitle(
        f"2D {q}-State Clock Model Phase Transitions (L={L}, mode={results[0].num_samples} samples/T){phase_text}",
        fontsize=14,
        fontweight="bold",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n[Saved] Plot successfully saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="2D クロック模型 2段階相転移解析プロット")
    parser.add_argument("--L", type=int, default=24, help="格子サイズ L (default: 24)")
    parser.add_argument("--q", type=int, default=6, help="クロック状態数 q (default: 6)")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["random", "adjacent"],
        default="adjacent",
        help="提案方式 (default: adjacent)",
    )
    parser.add_argument("--mcs-thermalize", type=int, default=3000, help="平衡化 MCS 数")
    parser.add_argument("--mcs-measure", type=int, default=6000, help="計測 MCS 数")
    parser.add_argument("--sample-interval", type=int, default=2, help="計測サンプリング間隔")
    parser.add_argument("--t-min", type=float, default=0.2, help="最低温度 (default: 0.2)")
    parser.add_argument("--t-max", type=float, default=1.6, help="最高温度 (default: 1.6)")
    parser.add_argument("--n-temps", type=int, default=28, help="温度分割点数 (default: 28)")
    parser.add_argument("--output", type=str, default=None, help="出力画像パス")

    args = parser.parse_args()

    temperatures = np.linspace(args.t_min, args.t_max, args.n_temps)
    output_path = (
        Path(args.output)
        if args.output
        else Path(__file__).resolve().parents[1] / "outputs" / "clock_two_step_transitions.png"
    )

    print(f"=== 2D {args.q}-State Clock Model Simulation ===")
    print(f"Lattice Size: {args.L}x{args.L}, Mode: {args.mode}")
    print(f"Temperature Range: [{args.t_min:.2f}, {args.t_max:.2f}] ({args.n_temps} points)")
    print(f"MCS: {args.mcs_thermalize} thermalize, {args.mcs_measure} measure (interval={args.sample_interval})")

    results = run_temperature_sweep(
        lattice_size=args.L,
        q=args.q,
        temperatures=temperatures,
        mcs_thermalize=args.mcs_thermalize,
        mcs_measure=args.mcs_measure,
        sample_interval=args.sample_interval,
        mode=args.mode,
    )

    plot_transitions(results, output_path, args.q)


if __name__ == "__main__":
    main()
