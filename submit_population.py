import argparse
import mthree
import datetime
import pickle as pkl
import numpy as np
from qiskit.visualization import timeline_drawer
from qiskit_ibm_runtime import QiskitRuntimeService, Batch
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit.circuit import ClassicalRegister, Gate
from qiskit.circuit.library import XGate, RZGate
from qiskit.transpiler import PassManager

from schedule import build_passmanager
from bad_qubits import BAD_QUBITS


"""
We load a previously compiled circuit, translate it to the backend basis and add dynamical decoupling.

Before submitting the experiment, we add measure-xslow-measure to all qubits. There are 4 classical register which serve the following purpose:
    - c1: measurement of the GHZ payload
    - c2: measurement of parity checks. Expect all-zero, otherwise will discard shot as it indicates an error in the payload.
    - c3: repetition of c1 but after a xslow. Expect disagreement with c1, otherwise discard as it indicates a non-Markovian error.
    - c4: repetition of c2 (similar to above line).
"""

BACKEND = 'ibm_kingston'
SHOTS = 50000           # choose many shots if going to post-select a lot
SHOTS_CALS = 30000      # shots for calibrating M3 readout model
INSTANCE = None         # specify an instance "name" for the QiskitRuntimeService (otherwise the default will be used)
XSLOW_TRAIN = 25        # how many Rx rotations to chain together to implement a "slow" pi rotation


parser = argparse.ArgumentParser()
parser.add_argument('filename', type=str)
parser.add_argument('--do-m3-cals', action="store_true", default=False) # apply m3 readout mitigation
parser.add_argument('--do-xslow', action="store_true", default=False) # apply m3 readout mitigation
args = parser.parse_args()
print(f'do-m3-cals={args.do_m3_cals}, do-xslow={args.do_xslow}')

if not BACKEND.removeprefix('alt_').removeprefix('ibm_') in args.filename:
    raise Exception("WARNING: trying to run a circuit compiled for a different backend.")

circuits_path = './circuits'
with open(args.filename, 'rb') as f:
    circuits_info = pkl.load(f)
qc = circuits_info['circuit']
ghz_qubits = circuits_info['ghz_qubits']
checks = circuits_info['checks']
depth = circuits_info['depth']
coverage = circuits_info['coverage']
undo = circuits_info.get('undo', 0)

service = QiskitRuntimeService(name=INSTANCE)
backend = service.backend(f"{BACKEND}")

spectators = list(set(neighbor for q in ghz_qubits+checks for neighbor in backend.coupling_map.neighbors(q) if neighbor not in ghz_qubits+checks))
bad_qubits = BAD_QUBITS[BACKEND.removeprefix('alt_').removeprefix('ibm_')]
spectators = [s for s in spectators if s not in bad_qubits]

# add measurements of ghz qubits and check qubits into their respective registers
c1 = ClassicalRegister(len(ghz_qubits), 'c1')
qc.add_register(c1)
for q, c in zip(ghz_qubits, c1):
    qc.measure(q, c)
if len(checks) > 0:
    c2 = ClassicalRegister(len(checks), 'c2')
    qc.add_register(c2)
    for q, c in zip(checks, c2):
        qc.measure(q, c)

# for the experiment we are going to post-select non-Markovian errors away
if args.do_xslow:
    c3 = ClassicalRegister(len(ghz_qubits), name='c3')
    qc.add_register(c3)
    for q, c in zip(ghz_qubits, c3):
        for _ in range(XSLOW_TRAIN):
            qc.rx(np.pi/XSLOW_TRAIN, [q])
            qc.barrier([q]
        qc.measure(q, c)
    if len(checks) > 0:
        c4 = ClassicalRegister(len(checks), name='c4')
        qc.add_register(c4)
        for q, c in zip(checks, c4):
            for _ in range(XSLOW_TRAIN):
                qc.rx(np.pi/XSLOW_TRAIN, [q])
                qc.barrier([q]
            qc.measure(q, c)
    if len(spectators) > 0:
        s1 = ClassicalRegister(len(spectators), 's1')
        qc.add_register(s1)
        for q, c in zip(spectators, s1):
            qc.measure(q, c) 

        s2 = ClassicalRegister(len(spectators), name='s2')
        qc.add_register(s2)
        for q, c in zip(spectators, s2):
            for _ in range(XSLOW_TRAIN):
                qc.rx(np.pi/XSLOW_TRAIN, [q])
                qc.barrier([q]
            qc.measure(q, c)

# transpile the circuit for the backend
pm = build_passmanager(backend, ghz_qubits[undo:])

circuit = pm.run(qc)

timeline_drawer(circuit, show_idle=False, time_range=(0, 1000), target=backend.target, filename='./plots/ghz_timeline.pdf')

m3_cal_jobs = []
if args.do_m3_cals:
    print("Submitting m3 cals")
    mit = mthree.M3Mitigation(system=backend)
    m3_cal_jobs = mit.cals_from_system(qubits=ghz_qubits, shots=SHOTS_CALS, method='balanced',
                                       initial_reset=False, rep_delay=0.0005, async_cal=True,
                                       runtime_mode='batch')
    batch = mit.system.get_mode()
    print('m3_cal_jobs: ', [j.job_id() for j in m3_cal_jobs])
else:
    batch = Batch(backend=backend)

print("Submitting sampler jobs")
sampler = Sampler(mode=batch)
sampler.options.default_shots = SHOTS
sampler.options.dynamical_decoupling.enable = False
sampler.options.execution.rep_delay = 0.0005

ghz_job = sampler.run([circuit])
ghz_job.update_tags([f'ghz {len(ghz_qubits)}'])

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
        'job_ids': [ghz_job.job_id()],
        'm3_cal_job_ids': [j.job_id() for j in m3_cal_jobs],
        'shots': SHOTS,
        'shots_cals': SHOTS_CALS
}

job_name = f'{today}_{BACKEND}_ghz{len(ghz_qubits)}_root{root}_undo{undo}_depth{depth}_coverage{100*coverage:.0f}_twirls{2*TWIRLS}'
jobs_path = './jobs_population'
with open(f'{jobs_path}/{job_name}.pkl', 'wb') as f:
    pkl.dump(job_metadata, f)
batch.close()
print(f'jobs metadata written to {jobs_path}/{job_name}.pkl')
print('--- TERMINATE SCRIPT! ---')
