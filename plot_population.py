import argparse
import os
import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
from qiskit.visualization import plot_histogram
from uncertainties import ufloat

parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str)
parser.add_argument("--save", action="store_true")
args = parser.parse_args()

with open(args.filename, "rb") as f:
    data = pkl.load(f)

basename = os.path.splitext(os.path.basename(args.filename))[0]
backend = data["backend"]
n = data["n"]
shots = data["shots"]
probs = data["probs"]
psrs = data["psrs"]
hfs = data["hfs"]
ebs = data["ebs"]
pops = data["pops"]

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


fig, ax = plt.subplots(figsize=(12, 6))
bar_width = 0.5
x_pos = np.arange(3)  # * len(scenarios) * 0.3
colors = plt.cm.viridis(np.linspace(0, 1, len(scenarios) + 1))
target_bitstrings = ["0" * n, "1" * n]
print(ebs)
for i, key in enumerate(scenarios):
    prob = probs[key]
    zeros = prob.get(target_bitstrings[0], 0)
    ones = prob.get(target_bitstrings[1], 0)
    eb_zeros = ebs[key] * np.sqrt(zeros * (1-zeros))
    eb_ones = ebs[key] * np.sqrt(ones * (1-ones))

    y_err = ufloat(pops[key], ebs[key])
    y_err_str = f"{y_err:.1uS}"

    rest = sum(prob[bs] for bs in prob.keys() if bs not in target_bitstrings)
    eb_rest = ebs[key] * np.sqrt(rest * (1-rest))

    ax.bar(
        x_pos[[0, 2]],
        [zeros, ones],
        bar_width,
        yerr=[eb_zeros, eb_ones],
        zorder=3 - i,
        # label=f'{str(key)} -- pop = {pops[key]:.2f} -- hf = {hfs[key]:.2f} (psr: {psrs[key]:.2f})', color=colors[i])
        # label=f"{str((key[0], key[2]))} -- population = {y_err_str} (psr: {psrs[key]:.2f})",
        label=f"{str((key[0], key[2]))} P={y_err_str}",
        color=colors[i],
    )
    ax.bar(
        x_pos[[1]],
        [rest],
        bar_width,
        yerr=[eb_rest],
        zorder=i,
        # label=f'{str(key)} -- pop = {pops[key]:.2f} -- hf = {hfs[key]:.2f} (psr: {psrs[key]:.2f})', color=colors[i])
        # label=f"{str((key[0], key[2]))} -- population = {y_err_str} (psr: {psrs[key]:.2f})",
        color=colors[i],
    )
    print(f"{zeros} +/- {eb_zeros}")
    print(f"{ones} +/- {eb_ones}")
    print(f"{pops[key]} +/- {ebs[key]}")
    print()

ax.set_xticks(x_pos)
ax.set_xticklabels(["00..0", "rest", "11..1"], fontsize=12)
ax.set_ylabel("population fraction", fontsize=12)
#ax.set_title(f"Population of {n}-qubit GHZ on {backend} ({shots} shots)")
# ax.legend(title='(paritycheck, markoviancheck, m3)')
ax.legend(title="(paritycheck, readout)")

if args.save:
    plot_path = f"./plots_population/{basename}.pdf"
    plt.savefig(f"{plot_path}", bbox_inches="tight")
    print(f"saved plot to {plot_path}")
else:
    plt.tight_layout()
    plt.show()
