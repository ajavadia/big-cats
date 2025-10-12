import argparse
import os
import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from uncertainties import ufloat

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str)
parser.add_argument("--limit", type=int, default=None)
parser.add_argument("--save", action="store_true")
args = parser.parse_args()

with open(args.filename, "rb") as f:
    data = pkl.load(f)

basename = os.path.splitext(os.path.basename(args.filename))[0]
backend = data["backend"]
n = data["n"]
angles = data["angles"]
shots = data["shots"]
ref_expvals = data["ref_expvals"]
main_expvals = data["main_expvals"]
ref_psrs = data["ref_psrs"]
main_psrs = data["main_psrs"]

if args.limit:
    angles = angles[: args.limit]

# go through different post-processing scenarios
scenario_expvals, scenario_error_bars, scenario_psrs = {}, {}, {}
scenarios = [
    (False, False, False),
    (True, False, False),
    # (False, True, False),
    # (True, True, False),
    # (False, False, True),
    (True, False, True),
    # (False, True, True),
    # (True, True, True)
]
for paulicheck, markoviancheck, trex in scenarios:
    expvals, error_bars, psrs = [], [], []
    for angle_index, angle in enumerate(angles):
        angle_expvals = main_expvals[(paulicheck, markoviancheck)][angle_index]
        angle_psrs = main_psrs[(paulicheck, markoviancheck)][angle_index]
        # roughly the same number of shots survive, regardless of twirl
        angle_psr = np.mean(angle_psrs)
        # standard error = 1/num_twirls [ (num_shots-1)/num_shots * variance_over_twirl + variance_over_shots]
        num_twirls = len(angle_expvals)
        survived_shots = shots * angle_psr
        mean_over_twirls = np.mean(angle_expvals)
        var_over_twirls = np.var(angle_expvals)
        var_over_shots = (1 - mean_over_twirls**2) / survived_shots
        angle_var = (
            1
            / num_twirls
            * ((survived_shots - 1) / survived_shots * var_over_twirls + var_over_shots)
        )
        angle_error_bar = np.sqrt(angle_var)
        # scale expval and error bar if trex is being used
        if trex:
            angle_expval = np.mean(angle_expvals) / np.mean(ref_expvals[markoviancheck])
            angle_error_bar /= np.mean(ref_expvals[markoviancheck])
        else:
            angle_expval = np.mean(angle_expvals)
        # print(f'num_twirls = {num_twirls}, survived_shots = {survived_shots:.2f}')
        # print(f'var_over_twirls = {var_over_twirls}, var_over_shots = {var_over_shots}')
        # print(f'expval = {angle_expval:.4f} +/- {angle_error_bar:.4f}')

        expvals.append(angle_expval)
        error_bars.append(angle_error_bar)
        psrs.append(angle_psr)

    scenario_expvals[(paulicheck, markoviancheck, trex)] = expvals
    scenario_error_bars[(paulicheck, markoviancheck, trex)] = error_bars
    scenario_psrs[(paulicheck, markoviancheck, trex)] = psrs

labels = list(scenario_expvals.keys())
values = list(scenario_expvals.values())
colors = plt.cm.viridis(np.linspace(0, 1, len(scenario_expvals) + 1))

fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))
#fig.suptitle(
#    f"Parity of {n}-qubit GHZ on {backend} ({len(ref_expvals[True])} twirls, {shots} shots)"
#)


# Define the cosine function to fit
def model(phi, C, theta):
    return C * np.cos(n * phi - theta)


last_data = scenario_expvals[scenarios[-1]]
last_error_bar = scenario_error_bars[scenarios[-1]]
params, params_covariance = curve_fit(model, angles, last_data, p0=[1, 0])
C_fit, theta_fit = params
C_err, theta_err = np.sqrt(np.diag(params_covariance))

# plot oscillations
for i, (key, value) in enumerate(scenario_expvals.items()):
    # ax0.plot(angles, value, label=f'{str(key)} (psr = {np.mean(scenario_psrs[key]):.2f})', color=colors[i], markersize=4, linewidth=.2)
    ax0.plot(
        angles,
        value,
        label=f"{str((key[0], key[2]))}",
        color=colors[i],
        markersize=4,
        linewidth=0.5,
    )
    ax0.errorbar(angles, value, yerr=scenario_error_bars[key], fmt=".", color=colors[i])

c_fit_var = ufloat(C_fit, C_err)
c_fit_str = f"{c_fit_var:.1uS}"

ax0.scatter(
    angles,
    model(angles, *params),
    label=f"fit: {c_fit_str} cos({n}ϕ - {theta_fit:.3f})",
    # s=10,
    color="grey",
    alpha=0.5,
)
# ax0.scatter(angles, model(angles, 1, 0), label=f'ideal: cos({n}ϕ)', s=10, color='green')

ax0.set_xlabel("ϕ", fontsize=12)
ax0.set_ylabel("$S_ϕ$", fontsize=12)
ax0.set_ylim(-1.0, 1.0)


# plot frequencies
for i, (key, value) in enumerate(scenario_expvals.items()):
    fourier = np.roll(np.abs(np.fft.fft(value) / len(angles)), int(len(angles) / 2))
    # ax1.plot(range(-int(len(angles)/2), int(len(angles)/2)), fourier, label=str(key), color=colors[i])
    ax1.plot(
        range(-int(len(angles) / 2), int(len(angles) / 2)),
        fourier,
        label=str((key[0], key[2])),
        color=colors[i],
    )

print(C_fit, C_err)
ax1.set_xlabel("$q$", fontsize=14)
ax1.set_ylabel("$I_q$", fontsize=14)
handles, labels = ax0.get_legend_handles_labels()
print(handles)
# leg = ax1.legend(handles, labels, title="(paritycheck, markoviancheck, trex)")
leg = ax1.legend(handles, labels, title="(paritycheck, readout)")
for legobj in leg.legend_handles:
    legobj.set_linewidth(3.0)

if args.save:
    plt.savefig(f"./plots_parity/{basename}.pdf", bbox_inches="tight")
    print(f"saved plot to ./plots_parity/{basename}.pdf")
else:
    plt.tight_layout()
    plt.show()
