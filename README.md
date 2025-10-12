# big cats

This repository contains code to compile circuits for GHZ state preparation on a given backend.

`big_ghz.py` takes a backend (and optionally a qiskit-ibm-runtime instance `name` to retrieve that backend from).
It also takes thresholds to filter out bad qubits and gates, and some hyperparameters for the search algorithm.

Three different experiments are included for certifying the state fidelity, each doable in 3 steps of `submit_*`, `fetch_*`, `plot_*`.
- `parity` for measuring parity oscillations.
- `population` for measuring population of all-zero and all-one bitstrings.
- `direct` for measuring direct fidelity estimates (DFE).

Various other plotting scripts are included to visualize the circuits and the hardware, calibration history, etc.

The code is tested with `qiskit 2.1.0`.

Please refer to the corresponding paper for more details.
