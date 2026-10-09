"""三角格子反強磁性イジング模型の物理量および零点残余エントロピーの可視化スクリプト."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# プロジェクトルートを sys.path に追加（スクリプト直接実行対応）
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simpson

from models.model_02_ising_triangular_af.src.simulation import run_temperature_sweep

# Gregory Wannier の零点残余エントロピー厳密解 (1950年)
# S_0 / (N * k_B) = (3 / pi) * integral_0^{pi/6} ln(2 cos theta) d theta ≈ 0.3230659
WANNIER_ENTROPY = 0.3230659455394249
GROUND_STATE_ENERGY = -1.0


def compute_entropy(
    temperatures: np.ndarray, specific_heat: np.ndarray
) -> np.ndarray:
    """比熱曲線 C_V(T) から Simpson 則による熱力学積分で各温度のエントロピー s(T) を算出する.

    熱力学関係式:
        s(T) = ln(2) - ∫_{T}^{∞} (C_V(T') / T') dT'
             = ln(2) - Δs_tail - ∫_{T}^{T_max} (C_V(T') / T') dT'

    【実測値ベースの高温テイル補正 (Continuous High-T Tail Correction)】:
    三角格子反強磁性では三角形内のフラストレーション短距離相関が高温域でも残存するため、
    完全無相関極限 (3/T^2) ではなく、測定上限 T_max での実測比熱 C_V(T_max) を基準として
    漸近減衰 C_V(T) ≈ C_V(T_max) * (T_max / T)^2 を仮定する。
    これを T_max から ∞ まで積分すると:
        Δs_tail = ∫_{T_max}^{∞} (C_V(T_max) * T_max^2 / T'^3) dT' = C_V(T_max) / 2
    となり、測定データと滑らかに連続する極めて整合的なテイル補正が得られる。

    Args:
        temperatures: 昇順 (低温 -> 高温) に整列された温度配列 (長さ M)
        specific_heat: 各温度における 1 サイトあたり比熱 C_V の配列 (長さ M)

    Returns:
        各温度における 1 サイトあたりエントロピー s(T) の配列 (長さ M)
    """
    y = specific_heat / temperatures
    n_points = len(temperatures)

    # 実測比熱に基づく連続高温テイル補正: Δs_tail = C_V(T_max) / 2
    s_tail_correction = float(specific_heat[-1]) / 2.0
    s_at_t_max = np.log(2.0) - s_tail_correction

    # 各温度点 T_k から T_max までの Simpson 積分
    integrals = np.zeros(n_points, dtype=np.float64)
    for k in range(n_points):
        if k == n_points - 1:
            integrals[k] = 0.0
        elif k == n_points - 2:
            # 2点のみの末尾区間は台形則
            dt = temperatures[-1] - temperatures[-2]
            integrals[k] = 0.5 * (y[-2] + y[-1]) * dt
        else:
            # 3点以上は Simpson 則 (不等間隔グリッド対応)
            integrals[k] = float(simpson(y=y[k:], x=temperatures[k:]))

    return s_at_t_max - integrals


def plot_triangular_af_physics(
    temperatures: np.ndarray,
    energies: np.ndarray,
    specific_heats: np.ndarray,
    magnetizations: np.ndarray,
    entropies: np.ndarray,
    lattice_size: int,
    output_path: Path,
) -> None:
    """4 パネル (2x2) で三角格子反強磁性模型の熱力学的挙動をプロットする."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # (a) 比熱 C_V(T): ショットキー型異常 (丸い山)
    ax_cv = axes[0, 0]
    ax_cv.plot(temperatures, specific_heats, "o-", color="#d62728", lw=2, ms=4)
    ax_cv.set_title("(a) Specific Heat $C_V(T)$ (Schottky-like Anomaly)", fontsize=13)
    ax_cv.set_xlabel("Temperature $T / J_{\\mathrm{AF}}$", fontsize=11)
    ax_cv.set_ylabel("$C_V / (N k_B)$", fontsize=11)
    ax_cv.grid(True, linestyle="--", alpha=0.6)

    # (b) 残余エントロピー s(T): Wannier 厳密解との比較
    ax_s = axes[0, 1]
    ax_s.plot(temperatures, entropies, "o-", color="#1f77b4", lw=2, ms=4, label="Simulation $s(T)$")
    ax_s.axhline(
        WANNIER_ENTROPY,
        color="purple",
        linestyle="--",
        lw=2,
        label=f"Wannier $S_0/N \\approx {WANNIER_ENTROPY:.4f}$",
    )
    ax_s.axhline(
        np.log(2.0),
        color="gray",
        linestyle=":",
        lw=1.5,
        label="High-T limit $\\ln 2 \\approx 0.693$",
    )
    ax_s.set_title("(b) Residual Entropy $s(T)$ via Thermodynamic Integration", fontsize=13)
    ax_s.set_xlabel("Temperature $T / J_{\\mathrm{AF}}$", fontsize=11)
    ax_s.set_ylabel("Entropy per spin $s / k_B$", fontsize=11)
    ax_s.legend(fontsize=10, loc="lower right")
    ax_s.grid(True, linestyle="--", alpha=0.6)

    # (c) 内部エネルギー <e>(T): T->0 で -1.0 へ収束
    ax_e = axes[1, 0]
    ax_e.plot(temperatures, energies, "s-", color="#2ca02c", lw=2, ms=4)
    ax_e.axhline(
        GROUND_STATE_ENERGY,
        color="black",
        linestyle="--",
        lw=1.5,
        label="Ground State $e_0 = -1.0$ (Frustrated)",
    )
    ax_e.set_title("(c) Internal Energy $\\langle e \\rangle(T)$", fontsize=13)
    ax_e.set_xlabel("Temperature $T / J_{\\mathrm{AF}}$", fontsize=11)
    ax_e.set_ylabel("Energy per spin $E / (N J_{\\mathrm{AF}})$", fontsize=11)
    ax_e.legend(fontsize=10, loc="lower right")
    ax_e.grid(True, linestyle="--", alpha=0.6)

    # (d) 自発磁化 <|m|>(T): 全温度で 0
    ax_m = axes[1, 1]
    ax_m.plot(temperatures, magnetizations, "^-", color="#ff7f0e", lw=2, ms=4)
    ax_m.axhline(0.0, color="black", linestyle="--", lw=1.5, label="No spontaneous magnetization")
    ax_m.set_title("(d) Spontaneous Magnetization $\\langle |m| \\rangle(T) \\approx 0$", fontsize=13)
    ax_m.set_xlabel("Temperature $T / J_{\\mathrm{AF}}$", fontsize=11)
    ax_m.set_ylabel("Magnetization per spin $\\langle |m| \\rangle$", fontsize=11)
    ax_m.set_ylim(-0.05, 1.05)
    ax_m.legend(fontsize=10, loc="upper right")
    ax_m.grid(True, linestyle="--", alpha=0.6)

    fig.suptitle(
        f"Triangular Antiferromagnetic Ising Model (L={lattice_size})\n"
        f"Geometric Frustration & Wannier Residual Entropy Verification",
        fontsize=15,
        fontweight="bold",
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Plot saved successfully to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate Triangular AF Ising model and verify Wannier residual entropy."
    )
    parser.add_argument("--L", type=int, default=24, help="Lattice size L (default: 24)")
    parser.add_argument(
        "--t-min", type=float, default=0.1, help="Minimum temperature (default: 0.1)"
    )
    parser.add_argument(
        "--t-max", type=float, default=5.0, help="Maximum temperature (default: 5.0)"
    )
    parser.add_argument(
        "--num-t", type=int, default=45, help="Number of temperature points (default: 45)"
    )
    parser.add_argument(
        "--mcs-thermalize", type=int, default=3000, help="Thermalization steps (default: 3000)"
    )
    parser.add_argument(
        "--mcs-measure", type=int, default=6000, help="Measurement steps (default: 6000)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/model_02_ising_triangular_af/outputs/residual_entropy.png",
        help="Output image path",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()

    # 温度グリッド (低温側を少し細かくするため linspace または幾何的配置)
    temperatures = np.linspace(args.t_min, args.t_max, args.num_t)

    print(f"Running temperature sweep for L={args.L} ({len(temperatures)} points)...")
    results = run_temperature_sweep(
        lattice_size=args.L,
        temperatures=temperatures,
        mcs_thermalize=args.mcs_thermalize,
        mcs_measure=args.mcs_measure,
        seed=args.seed,
    )

    t_arr = np.array([r.temperature for r in results])
    e_arr = np.array([r.mean_energy_per_spin for r in results])
    cv_arr = np.array([r.specific_heat for r in results])
    m_arr = np.array([r.mean_magnetization_per_spin for r in results])

    # エントロピーの熱力学積分 (実測比熱ベースの連続高温テイル補正付き)
    s_arr = compute_entropy(t_arr, cv_arr)

    output_path = Path(args.output)
    plot_triangular_af_physics(
        temperatures=t_arr,
        energies=e_arr,
        specific_heats=cv_arr,
        magnetizations=m_arr,
        entropies=s_arr,
        lattice_size=args.L,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
