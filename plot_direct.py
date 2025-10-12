import argparse
import os
import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
from qiskit.quantum_info import Pauli
from uncertainties import ufloat

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str)
parser.add_argument("--save", action="store_true")
args = parser.parse_args()

with open(args.filename, "rb") as f:
    data = pkl.load(f)

basename = os.path.splitext(os.path.basename(args.filename))[0]
n = data["n"]
observables = data["observables"]
ghz_qubits = data["ghz_qubits"]
shots = data["shots"]
ref_expvals = data["ref_expvals"]
main_expvals = data["main_expvals"]
ref_psrs = data["ref_psrs"]
main_psrs = data["main_psrs"]
backend = data["backend"]


def reduce_obs(obs):
    """reduce a Pauli observable to only the support on ghz qubits."""
    return Pauli("".join(obs.to_label()[::-1][j] for j in ghz_qubits))


def truncate_string(s):
    """truncate a large stabilizer string"""
    if len(s) > 10:
        return s[:3] + "..." + s[-2:]
    return s


twirl_lim = 128

# go through different post-processing scenarios
scenario_expvals, scenario_error_bars, scenario_psrs = {}, {}, {}
scenarios = [
    (False, False, False),
    (True, False, False),
    # (False, True, False),
    # (True, True, False),
    #(False, False, True),
    (True, False, True),
    # (False, True, True),
    # (True, True, True)
]
for paulicheck, markoviancheck, trex in scenarios:
    expvals, error_bars, psrs = [], [], []
    for obs_index, obs in enumerate(observables):
        obs_expvals = main_expvals[(paulicheck, markoviancheck)][obs_index][:twirl_lim]
        obs_psrs = main_psrs[(paulicheck, markoviancheck)][obs_index][:twirl_lim]
        obs_psr = np.mean(
            obs_psrs
        )  # roughly the same number of shots survive, regardless of twirl
        # standard error = 1/num_twirls [ (num_shots-1)/num_shots * variance_over_twirl + variance_over_shots]
        num_twirls = len(obs_expvals)
        survived_shots = shots * obs_psr
        mean_over_twirls = np.mean(obs_expvals)
        var_over_twirls = np.var(obs_expvals)
        var_over_shots = (1 - mean_over_twirls**2) / survived_shots
        obs_var = (
            1
            / num_twirls
            * ((survived_shots - 1) / survived_shots * var_over_twirls + var_over_shots)
        )
        obs_error_bar = np.sqrt(obs_var)
        # scale expval and error bar if trex is being used
        if trex:
            obs_expval = np.mean(obs_expvals) / np.mean(
                ref_expvals[markoviancheck][obs_index][:twirl_lim]
            )
            obs_error_bar /= np.mean(ref_expvals[markoviancheck][obs_index][:twirl_lim])
        else:
            obs_expval = np.mean(obs_expvals)
        # print(f'num_twirls = {num_twirls}, survived_shots = {survived_shots:.2f}')
        # print(f'expval = {obs_expval:.4f} +/- {obs_error_bar:.4f}')
        # print(f'var_over_twirls = {var_over_twirls}, var_over_shots = {var_over_shots}')

        expvals.append(np.abs(obs_expval))
        error_bars.append(obs_error_bar)
        psrs.append(obs_psr)

    scenario_expvals[(paulicheck, markoviancheck, trex)] = expvals
    scenario_error_bars[(paulicheck, markoviancheck, trex)] = error_bars
    scenario_psrs[(paulicheck, markoviancheck, trex)] = psrs

labels = list(scenario_expvals.keys())


fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.2
x_pos = np.arange(len(observables))
colors = plt.cm.viridis(np.linspace(0, 1, len(scenarios) + 1))
for i, key in enumerate(scenarios):
    exp_list = scenario_expvals[key]
    var_list = np.array(scenario_error_bars[key]) ** 2
    m = len(scenario_expvals[key])
    var_F_1 = np.var(exp_list) / m
    var_F_2 = 1 / m**2 * np.sum(var_list)
    var_F_total = var_F_1 + var_F_2
    F_err = np.sqrt(var_F_total)
    ax.bar(
        x_pos,
        exp_list,
        0.5,
        yerr=scenario_error_bars[key],
        zorder=5 - i,
        # label=f'{str(key)} (psr: {np.mean(scenario_psrs[key]):.2f})',
        # label=f"{str((key[0], key[2]))} (psr: {np.mean(scenario_psrs[key]):.2f})",
        label=f"{str((key[0], key[2]))}",
        color=colors[i],
        alpha=1.
    )
    # ax.bar(x_pos, exp_list, 0, yerr=scenario_error_bars[key],
    #     #label=f'{str(key)} (psr: {np.mean(scenario_psrs[key]):.2f})',
    #     color=colors[i],zorder=0)
    ax.axhline(np.mean(exp_list), color=colors[i])
    ax.axhspan(
        np.mean(exp_list) - F_err,
        np.mean(exp_list) + F_err,
        facecolor=colors[i],
        alpha=0.5,
    )
    x_err = ufloat(np.mean(exp_list), F_err)
    err_str = f"{x_err:.1uS}"
    ax.text(len(observables), np.mean(exp_list), f"{err_str}", fontsize=12, ha="left", c=colors[i],
    )
    print(f'{np.mean(exp_list):.3f} +/- {F_err:.3f}')
ax.set_xticks(x_pos)
ax.set_xticklabels(
    [truncate_string(reduce_obs(obs).to_label()) for obs in observables], rotation=45
)
ax.set_xlabel("stabilizer", fontsize=14)
ax.set_ylabel("expectation value", fontsize=14)
#ax.set_title(
#    f"Direct fidelity estimation of {n}-qubit GHZ on {backend} ({len(ref_expvals[True][0])} twirls, {shots} shots)"
#)
ax.set_ylim(0, 1.0)
ax.axhline(0.5, linestyle="dashed", lw=1.5, color="k")
ax.text(len(observables), 0.5, "Threshold", fontsize=12, ha="left")
# ax.legend(title='(paritycheck, markoviancheck, trex)')
ax.legend(title="(paritycheck, readout)")


if args.save:
    plt.savefig(f"./plots_direct/{basename}.pdf", bbox_inches="tight")
    print(f"saved plot to ./plots_direct/{basename}.pdf")
else:
    plt.tight_layout()
    plt.show()
