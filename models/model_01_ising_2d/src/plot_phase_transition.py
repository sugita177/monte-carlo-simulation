"""Phase transition analysis and plotting against Onsager exact solution for 2D Ising model."""

import argparse
import sys
from pathlib import Path

# プロジェクトルートを sys.path に追加（直接実行対応）
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from models.model_01_ising_2d.src.simulation import run_simulation

# 理論上の臨界温度 (Onsager, 1944)
T_C_EXACT = 2.0 / np.log(1.0 + np.sqrt(2.0))


def onsager_exact_magnetization(temperatures: np.ndarray, J: float = 1.0) -> np.ndarray:
    """
    2次元正方格子イジング模型の自発磁化の厳密解 (Yang, 1952).

    M(T) = (1 - [sinh(2J/T)]^(-4))^(1/8) for T < Tc, 0 for T >= Tc
    """
    m_exact = np.zeros_like(temperatures, dtype=np.float64)
    below_tc = temperatures < T_C_EXACT

    T_sub = temperatures[below_tc]
    sinh_val = np.sinh(2.0 * J / T_sub)
    term = 1.0 - (1.0 / (sinh_val**4))
    term = np.maximum(term, 0.0)  # 数値誤差による負値を防止
    m_exact[below_tc] = term ** (1.0 / 8.0)
    return m_exact


def main() -> None:
    parser = argparse.ArgumentParser(description="2D Ising Model Phase Transition Plotter")
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
        help="Sampling algorithm to use (default: wolff)",
    )
    args = parser.parse_args()

    # ガード条件: Python エンジンでの Wolff は未サポート
    if args.engine == "python" and args.algorithm == "wolff":
        raise NotImplementedError(
            "Wolff algorithm is currently only implemented in Rust engine. "
            "Please use --engine rust or specify --algorithm metropolis for python."
        )

    engine_name = args.engine
    algo_name = args.algorithm
    algo_display = "Wolff Cluster Algorithm" if algo_name == "wolff" else "Metropolis Algorithm"

    # 出力先ディレクトリの確保
    output_dir = Path(__file__).resolve().parent.parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 実験パラメータ
    lattice_sizes = [8, 16, 24, 32, 40, 48]
    temperatures = np.linspace(1.5, 3.5, 26)
    mcs_thermalize = 1000
    mcs_measure = 4000
    sample_interval = 4

    # 結果保持用辞書 {L: {obs_name: [values]}}
    results: dict[int, dict[str, list[float]]] = {}

    print("=" * 60)
    print("2D Square Lattice Ising Model - Temperature Sweep Simulation")
    print(f"Engine: {engine_name.upper()}")
    print(f"Algorithm: {algo_display}")
    print(f"Lattice sizes: {lattice_sizes}")
    print(f"Temperature range: [{temperatures[0]:.2f}, {temperatures[-1]:.2f}] (26 points)")
    print(f"Exact critical temperature Tc = {T_C_EXACT:.6f}")
    print("=" * 60)

    for L in lattice_sizes:
        print(f"\n[Simulation] Running L = {L} ...")
        res_L: dict[str, list[float]] = {
            "mean_energy": [],
            "mean_mag": [],
            "specific_heat": [],
            "susceptibility": [],
            "binder": [],
        }

        for T in tqdm(temperatures, desc=f"L={L}"):
            sim_res = run_simulation(
                L=L,
                temperature=T,
                mcs_thermalize=mcs_thermalize,
                mcs_measure=mcs_measure,
                sample_interval=sample_interval,
                init_method="random",
                engine_type=engine_name,
                algorithm=algo_name,
                seed=42,
            )
            res_L["mean_energy"].append(sim_res.mean_energy_per_spin)
            res_L["mean_mag"].append(sim_res.mean_magnetization_per_spin)
            res_L["specific_heat"].append(sim_res.specific_heat)
            res_L["susceptibility"].append(sim_res.susceptibility)
            res_L["binder"].append(sim_res.binder_cumulant)

        results[L] = res_L

    # ----------------------------------------------------
    # プロット描画 (論文クオリティの 2x3 グリッドスタイル)
    # ----------------------------------------------------
    print("\nGenerating phase transition plots...")
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        f"2D Ising Model Phase Transition [{engine_name.upper()} | {algo_display}] (Onsager Exact $T_c \\approx {T_C_EXACT:.4f}$)",
        fontsize=16,
        fontweight="bold",
    )

    colors = {8: "#1f77b4", 16: "#ff7f0e", 24: "#2ca02c", 32: "#d62728", 40: "#9467bd", 48: "#8c564b"}
    markers = {8: "o", 16: "s", 24: "^", 32: "d", 40: "p", 48: "*"}

    # 1. 自発磁化 <|m|> vs T
    ax = axes[0, 0]
    T_dense = np.linspace(1.5, 3.5, 300)
    ax.plot(
        T_dense,
        onsager_exact_magnetization(T_dense),
        "k-",
        linewidth=2,
        label="Onsager Exact (1944)",
        zorder=2,
    )
    for L in lattice_sizes:
        ax.plot(
            temperatures,
            results[L]["mean_mag"],
            marker=markers[L],
            color=colors[L],
            markersize=5,
            linestyle="--",
            alpha=0.85,
            label=f"$L = {L}$",
            zorder=3,
        )
    ax.axvline(T_C_EXACT, color="red", linestyle=":", label=f"$T_c = {T_C_EXACT:.3f}$")
    ax.set_xlabel("Temperature $T$")
    ax.set_ylabel("Magnetization $\\langle |m| \\rangle$")
    ax.set_title("Spontaneous Magnetization")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # 2. 内部エネルギー <e> vs T
    ax = axes[0, 1]
    for L in lattice_sizes:
        ax.plot(
            temperatures,
            results[L]["mean_energy"],
            marker=markers[L],
            color=colors[L],
            markersize=5,
            linestyle="--",
            alpha=0.85,
            label=f"$L = {L}$",
        )
    ax.axvline(T_C_EXACT, color="red", linestyle=":", label=f"$T_c$")
    ax.set_xlabel("Temperature $T$")
    ax.set_ylabel("Energy per spin $\\langle e \\rangle$")
    ax.set_title("Internal Energy")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # 3. 比熱 C_V vs T
    ax = axes[0, 2]
    for L in lattice_sizes:
        ax.plot(
            temperatures,
            results[L]["specific_heat"],
            marker=markers[L],
            color=colors[L],
            markersize=5,
            linestyle="-",
            alpha=0.85,
            label=f"$L = {L}$",
        )
    ax.axvline(T_C_EXACT, color="red", linestyle=":", label=f"$T_c$")
    ax.set_xlabel("Temperature $T$")
    ax.set_ylabel("Specific Heat $C_V$")
    ax.set_title("Specific Heat (Fluctuation)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # 4. 帯磁率 chi vs T
    ax = axes[1, 0]
    for L in lattice_sizes:
        ax.plot(
            temperatures,
            results[L]["susceptibility"],
            marker=markers[L],
            color=colors[L],
            markersize=5,
            linestyle="-",
            alpha=0.85,
            label=f"$L = {L}$",
        )
    ax.axvline(T_C_EXACT, color="red", linestyle=":", label=f"$T_c$")
    ax.set_xlabel("Temperature $T$")
    ax.set_ylabel("Magnetic Susceptibility $\\chi$")
    ax.set_title("Magnetic Susceptibility")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # 5. ビンダー比 U_4 vs T
    ax = axes[1, 1]
    for L in lattice_sizes:
        ax.plot(
            temperatures,
            results[L]["binder"],
            marker=markers[L],
            color=colors[L],
            markersize=5,
            linestyle="-",
            alpha=0.85,
            label=f"$L = {L}$",
        )
    ax.axvline(T_C_EXACT, color="red", linestyle=":", label=f"$T_c = {T_C_EXACT:.3f}$")
    ax.axhline(2.0 / 3.0, color="gray", linestyle="--", alpha=0.5, label="$U_4 \\to 2/3$ (Low T)")
    ax.axhline(0.0, color="gray", linestyle=":", alpha=0.5, label="$U_4 \\to 0$ (High T)")
    ax.set_xlabel("Temperature $T$")
    ax.set_ylabel("Binder Cumulant $U_4$")
    ax.set_title("Binder Cumulant (Curves Cross at $T_c$)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # 6. 要約メモ
    ax = axes[1, 2]
    ax.axis("off")
    summary_text = (
        "Key Observations in 2D Ising:\n\n"
        f"1. Critical Temp: Tc = 2 / ln(1+sqrt(2)) ~ {T_C_EXACT:.4f}\n\n"
        "2. Magnetization <|m|>:\n"
        "   Approaches Onsager exact curve as L increases.\n"
        "   Sharp drop near Tc (beta = 1/8).\n\n"
        "3. Specific Heat C_V & Susceptibility chi:\n"
        "   Peak heights grow and sharpen with L,\n"
        "   demonstrating divergence at thermodynamic limit.\n\n"
        "4. Binder Cumulant U_4:\n"
        "   Curves for different L intersect uniquely at Tc!\n"
        "   U_4 -> 2/3 as T -> 0, U_4 -> 0 as T -> inf."
    )
    ax.text(
        0.05,
        0.5,
        summary_text,
        fontsize=11,
        verticalalignment="center",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.8", facecolor="#f5f5f5", edgecolor="#cccccc"),
    )

    plt.tight_layout()
    plot_path = output_dir / f"phase_transition_{engine_name}_{algo_name}.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"\nPlot successfully saved to: {plot_path}")


if __name__ == "__main__":
    main()
