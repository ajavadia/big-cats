import argparse
import mthree
import datetime
import random
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
from tqdm import tqdm
from qiskit.visualization import timeline_drawer
from qiskit_ibm_runtime import QiskitRuntimeService, Batch
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit.circuit import ClassicalRegister, Gate
from qiskit.circuit.library import XGate, RZGate
from qiskit.quantum_info import Pauli
from qiskit.transpiler import PassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler.passes import ALAPScheduleAnalysis, PadDynamicalDecoupling, PadDelay, RemoveBarriers
from qiskit_device_benchmarking.utilities.gate_map import plot_gate_map

from bad_qubits import BAD_QUBITS


"""
We load a previously compiled circuit, rotate it into different stabilizer states, twirl the readout on each of the rotated circuits,
add measurements, translate circuits to the backend basis, add dynamical decoupling, and run.

Before submitting the experiment, we add measure-xslow-measure to all qubits. There are 4 classical register which serve the following purpose:
    - c1: measurement of the GHZ payload
    - c2: measurement of pauli checks. Expect all-zero, otherwise will discard shot as it indicates an error in the payload.
    - c3: measurement of spectators. Will use these for Markovianity post-selection as well.
    - c4: repetition of c1 but after a xslow. Expect disagreement with c1, otherwise discard as it indicates a non-Markovian errors.
    - c5: repetition of c2 (similar to above line).
    - c6: repetition of c3 (similar to above line).
"""

BACKEND = 'ibm_kingston'
SHOTS = 10000           # choose many shots if going to post-select a lot
TWIRLS = 15             # number of twirls in TREX protocol (actual number is twice this)
INSTANCE = None         # specify an instance "name" for the QiskitRuntimeService (otherwise the default will be used)
XSLOW_TRAIN = 25        # how many Rx rotations to chain together to implement a "slow" pi rotation


def add_measurements(qc, ghz_qubits, checks=None, spectators=None, do_xslow=False):
    # measure each set of qubits into a different classical register to facilitate post-processing
    c1 = ClassicalRegister(len(ghz_qubits), 'c1')
    qc.add_register(c1)
    for q, c in zip(ghz_qubits, c1):
        qc.measure(q, c)
    if checks:
        c2 = ClassicalRegister(len(checks), 'c2')
        qc.add_register(c2)
        for q, c in zip(checks, c2):
            qc.measure(q, c)
    # for the experiment we are going to post-select non-Markovian errors away (measure again after xslow, into a separate register)
    if do_xslow:
        c4 = ClassicalRegister(len(ghz_qubits), name='c4')
        qc.add_register(c4)
        for q, c in zip(ghz_qubits, c4):
            for _ in range(XSLOW_TRAIN):
                qc.rx(np.pi/XSLOW_TRAIN, [q])
                qc.barrier([q]
            qc.measure(q, c)
        if checks:
            c5 = ClassicalRegister(len(checks), name='c5')
            qc.add_register(c5)
            for q, c in zip(checks, c5):
                for _ in range(XSLOW_TRAIN):
                    qc.rx(np.pi/XSLOW_TRAIN, [q])
                    qc.barrier([q]
                qc.measure(q, c)
        if spectators:
            c3 = ClassicalRegister(len(spectators), 'c3')
            qc.add_register(c3)
            for q, c in zip(spectators, c3):
                qc.measure(q, c)
            c6 = ClassicalRegister(len(spectators), name='c6')
            qc.add_register(c6)
            for q, c in zip(spectators, c6):
                for _ in range(XSLOW_TRAIN):
                    qc.rx(np.pi/XSLOW_TRAIN, [q])
                    qc.barrier([q]
                qc.measure(q, c)
    return qc


parser = argparse.ArgumentParser()
parser.add_argument('filename', type=str)
parser.add_argument('--do-xslow', action="store_true", default=False) # apply trex readout mitigation
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()
print(f'do-xslow={args.do_xslow}')

if not BACKEND.removeprefix('alt_').removeprefix('ibm_') in args.filename:
    raise Exception("WARNING: trying to run a circuit compiled for a different backend.")

circuits_path = './circuits'
with open(args.filename, 'rb') as f:
    circuits_info = pkl.load(f)
base_circuit = circuits_info['circuit']
ghz_qubits = circuits_info['ghz_qubits']
checks = circuits_info['checks']
depth = circuits_info['depth']
coverage = circuits_info['coverage']

answer = input(
        f"---------------------------\n"
        f"Backend: {BACKEND}\n"
        f"XSLOW: {args.do_xslow}\n"
        f"Total experiment time: {(SHOTS * 2 * TWIRLS * (2 * len(ghz_qubits) + 2) * 250 * 1e-6 / 60):.2f} mins.\n"
        f"Continue [y/n]\n")
if answer.lower() in ["n", "no"]:
    sys.exit()


service = QiskitRuntimeService(name=INSTANCE)
backend = service.backend(f"{BACKEND}")

spectators = list(set(neighbor for q in ghz_qubits+checks for neighbor in backend.coupling_map.neighbors(q) if neighbor not in ghz_qubits+checks))
bad_qubits = BAD_QUBITS[BACKEND.removeprefix('alt_').removeprefix('ibm_')]
spectators = [s for s in spectators if s not in bad_qubits]

# plot the layout of ghz, check and spectator qubits
plot_gate_map(backend, label_qubits=True, line_width=20,
              line_color = ['black' if edge[0] in ghz_qubits+checks and edge[1] in ghz_qubits+checks else
                            'lightgrey' for edge in backend.coupling_map.graph.edge_list()],
              qubit_color=['blue' if i in ghz_qubits else 
                           'salmon' if i in checks else 
                           'lightblue' if i in spectators and args.do_xslow else
                           'lightgrey' for i in range(0, backend.num_qubits)]
              )
plt.savefig('./plots/expt_layout.pdf', bbox_inches='tight')


base_circuit.draw('mpl', fold=-1, idle_wires=False, filename='./plots/ghz_base.pdf')

# rotation by an angle and measuring in X basis
angles = np.linspace(0, 2*np.pi, 2*(len(ghz_qubits)+1), endpoint=False)
rotated_circuits = []
for phi in angles:
    rotated_circuit = base_circuit.copy()
    rotated_circuit.rz(phi, ghz_qubits)
    rotated_circuit.sx(ghz_qubits)
    rotated_circuits.append(rotated_circuit)
rotated_circuits[-1].draw('mpl', fold=-1, idle_wires=False, filename='./plots/ghz_rotated.pdf')

# twirling each rotated circuit
all_circuits = []
print("selecting random twirls")
twirl_xs, twirl_zs = [], []
for _ in range(TWIRLS):
    x_vector = [random.choice([True, False]) for _ in range(len(ghz_qubits))]
    z_vector = [random.choice([True, False]) for _ in range(len(ghz_qubits))]
    twirl_xs.append(x_vector)
    twirl_zs.append(z_vector)
    twirl_xs.append(np.bitwise_not(x_vector))  # ensure each qubit is flipped/not flipped equally often
    twirl_zs.append(np.bitwise_not(z_vector))

print("building trex reference circuits")
trex_cals = []
for x_vector, z_vector in zip(twirl_xs, twirl_zs):
    cal_circ = base_circuit.copy_empty_like()
    for do_x, do_z, q in zip(x_vector, z_vector, ghz_qubits):
        if do_z:
            cal_circ.rz(np.pi, q)
        if do_x:
            cal_circ.x(q)
    trex_cals.append(cal_circ)
trex_cals[-1].draw('mpl', fold=-1, idle_wires=False, filename='./plots/trex_cals.pdf')

print("twirling readout for every angle")
for rotated_circuit in tqdm(rotated_circuits):
    rotated_twirled_circuits = []
    for x_vector, z_vector in zip(twirl_xs, twirl_zs):
        rotated_twirled_circuit = rotated_circuit.copy()
        for do_x, do_z, q in zip(x_vector, z_vector, ghz_qubits):
            if do_z:
                rotated_twirled_circuit.z(q)
            if do_x:
                rotated_twirled_circuit.x(q)
        rotated_twirled_circuits.append(rotated_twirled_circuit)
    all_circuits.append(rotated_twirled_circuits)

#all_circuits[-1][-1].draw('mpl', fold=-1, idle_wires=False, filename='./plots/ghz_rotated_twirled.pdf')


print("adding measurements")
for i in range(len(all_circuits)):
    all_circuits[i] = [add_measurements(qc, ghz_qubits, checks, spectators, args.do_xslow) for qc in all_circuits[i]]

trex_cals[:] = [add_measurements(qc, ghz_qubits, checks, spectators, args.do_xslow) for qc in trex_cals]

#all_circuits[-1][-1].draw('mpl', fold=-1, idle_wires=False, filename='./plots/ghz_measured.pdf')
#trex_cals[-1].draw('mpl', fold=-1, idle_wires=False, filename='./plots/trex_cal_measured.pdf')

# transpile the circuits for the backend
print("scheduling circuits")
pm = generate_preset_pass_manager(
        target=backend.target,
        layout_method='trivial',
        optimization_level=2,
        routing_method='none'
    )
pm.scheduling = PassManager([
        RemoveBarriers(),
        ALAPScheduleAnalysis(target=backend.target),
        PadDynamicalDecoupling(dd_sequence=[XGate(), RZGate(-np.pi), XGate(), RZGate(np.pi)],
                                spacing=[1/4, 1/2, 0, 0, 1/4],
                                target=backend.target)
        #PadDelay(target=backend.target)
        ])

all_circuits_isa = [pm.run(circuits) for circuits in tqdm(all_circuits)]

#timeline_drawer(all_circuits_isa[0][0], show_idle=False, filename='./plots/ghz_timeline.pdf')

batch = Batch(backend=backend)
sampler = Sampler(mode=batch)
sampler.options.default_shots = SHOTS
sampler.options.dynamical_decoupling.enable = False
sampler.options.execution.rep_delay = 0.00025

trex_cal_jobs = []
print("Submitting trex cals")
trex_cal_job = sampler.run(trex_cals)
trex_cal_job.update_tags(['TREX cal'])
trex_cal_jobs.append(trex_cal_job)

print("Submitting sampler jobs")
ghz_jobs = [sampler.run(circs) for circs in all_circuits_isa]
for i, j in enumerate(ghz_jobs):
    j.update_tags([f'ghz {len(ghz_qubits)} (angle {i})'])


now = datetime.datetime.now()
today = now.strftime("%y%m%d")
job_metadata = {
        'backend': BACKEND,
        'instance_name': INSTANCE,
        'batch_id': batch.session_id,
        'ghz_qubits': ghz_qubits,
        'checks': checks,
        'spectators': spectators,
        'depth': depth,
        'coverage': coverage,
        'angles': angles,
        'job_ids': [job.job_id() for job in ghz_jobs],
        'trex_cal_job_ids': [j.job_id() for j in trex_cal_jobs],
        'shots': SHOTS,
        'twirl_xs': twirl_xs
}

job_name = f'{today}_{BACKEND}_ghz{len(ghz_qubits)}_depth{depth}_coverage{100*coverage:.0f}_twirls{2*TWIRLS}'
jobs_path = './jobs_parity'
with open(f'{jobs_path}/{job_name}.pkl', 'wb') as f:
    pkl.dump(job_metadata, f)
batch.close()
print(f'jobs metadata written to {jobs_path}/{job_name}.pkl')
print('--- TERMINATE SCRIPT! ---')
