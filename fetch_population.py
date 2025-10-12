import os
import re
import argparse
import itertools
import pickle as pkl
import numpy as np
import itertools as it
from qiskit_ibm_runtime import QiskitRuntimeService
import mthree
from mthree.mitigation import _job_thread
from mthree.generators import HadamardGenerator
from runningman.job import RunningManJob
from qiskit.quantum_info import hellinger_fidelity


def no_flag(check, do_check):
    if not do_check:
        return True
    return np.all([b=='0' for b in check])


def no_nonmarkovian(backend, ghz_qubits, check_qubits, spectator_qubits, meas, check, spec, meas2, check2, spec2, do_check):
    if not do_check:
        return True
    bad_qubits = []
    for i, (b1, b2) in enumerate(zip(meas, meas2)):
        if b1 == b2:
            bad_qubits.append(ghz_qubits[-i-1])
    for i, (b1, b2) in enumerate(zip(check, check2)):
        if b1 == b2:
            bad_qubits.append(check_qubits[-i-1])
    for i, (b1, b2) in enumerate(zip(spec, spec2)):
        if b1 == b2:
            bad_qubits.append(spectator_qubits[-i-1])
    return not any(e[0] in bad_qubits and e[1] in bad_qubits for e in backend.coupling_map)


parser = argparse.ArgumentParser()
parser.add_argument('filename', type=str)
args = parser.parse_args()

with open(args.filename, 'rb') as f:
    job_metadata = pkl.load(f)
basename = os.path.splitext(os.path.basename(args.filename))[0]
print(basename)

backend_name = job_metadata['backend']
service = QiskitRuntimeService(name=job_metadata['instance_name']) 
full_jobs = [service.job(job_id) for job_id in job_metadata['job_ids']]
backend = full_jobs[0].backend()
num_shots = job_metadata['shots']
ghz_qubits = job_metadata['ghz_qubits']
check_qubits = job_metadata['checks']
spectator_qubits = job_metadata['spectators']
print(f'backend: {backend.name}')

# ideal probability distribution
ans = {'0'*len(ghz_qubits): 0.5, '1'*len(ghz_qubits): 0.5}

# build an M3 mitigator based on the readout errors of the experiment
m3mit = None
m3_cal_job_ids = job_metadata['m3_cal_job_ids']
print('building m3 mitigator')
generator = HadamardGenerator(len(ghz_qubits))    
m3_cal_jobs = [RunningManJob(service.job(job_id)) for job_id in m3_cal_job_ids]
m3mit = mthree.M3Mitigation(system=backend)
m3mit.cal_shots = job_metadata['shots_cals']
shots = 2 * m3mit.cal_shots // generator.length
if 2 * m3mit.cal_shots / generator.length != shots:
    shots += 1
m3mit._balanced_shots = shots * generator.length
m3mit.single_qubit_cals = [None for _ in range(backend.num_qubits)]
_job_thread(m3_cal_jobs, m3mit, ghz_qubits, len(ghz_qubits), generator)
m3mit.num_qubits = len(ghz_qubits)
m3mit.cals_from_matrices([x for x in m3mit.single_qubit_cals if x is not None])


# get raw bitstrings from the experiment
res = full_jobs[0].result()[0].data
meas_bitstrings = res.c1.get_bitstrings()
check_bitstrings = res.c2.get_bitstrings() if getattr(res, 'c2', None) else ['0'] * num_shots
spec_bitstrings = res.s1.get_bitstrings() if getattr(res, 's1', None) else ['0'] * num_shots
meas_bitstrings_2 = res.c3.get_bitstrings() if getattr(res, 'c3', None) else [''.join('0' if s=='1' else '1' for s in bs) for bs in meas_bitstrings]
check_bitstrings_2 = res.c4.get_bitstrings() if getattr(res, 'c4', None) else ['1'] * num_shots
spec_bitstrings_2 = res.s2.get_bitstrings() if getattr(res, 's2', None) else ['1'] * num_shots

# get probabilities and post-selection rate after different kinds of post-selection (paulicheck, markoviancheck)
scenarios = [
        (False, False, False),
        (True, False, False),
        #(False, True, False),
        #(True, True, False),
        (False, False, True),
        (True, False, True),
        #(False, True, True),
        #(True, True, True)
        ]
hfs, psrs, pops, ebs, probs = [{k: [] for k in scenarios} for _ in range(5)]
for paulicheck, markoviancheck, m3 in probs.keys():
    print(paulicheck, markoviancheck, m3)
    accepted_meas = {}
    accepted_count = 0
    for meas, check, spec, meas2, check2, spec2 in zip(meas_bitstrings, check_bitstrings, spec_bitstrings, meas_bitstrings_2, check_bitstrings_2, spec_bitstrings_2):
        if no_flag(check, paulicheck) and no_nonmarkovian(backend, ghz_qubits, check_qubits, spectator_qubits, meas, check, spec, meas2, check2, spec2, do_check=markoviancheck):
            accepted_meas[meas] = accepted_meas.get(meas, 0) + 1
            accepted_count += 1
    psr = sum(accepted_meas.values()) / num_shots
    error_bar = 1/np.sqrt(accepted_count)

    if m3:
        roe = [1 - f for f in m3mit.readout_fidelity(range(len(ghz_qubits)))]
        m3_quasi = m3mit.apply_correction(accepted_meas, range(len(ghz_qubits)), distance=3, max_iter=100, return_mitigation_overhead=True)
        error_bar *= np.sqrt(m3_quasi.mitigation_overhead)
        probabilities = m3_quasi.nearest_probability_distribution()
    else:
        probabilities = {meas: count/accepted_count for meas, count in accepted_meas.items()}

    hfs[(paulicheck, markoviancheck, m3)] = hellinger_fidelity(ans, probabilities)
    psrs[(paulicheck, markoviancheck, m3)] = accepted_count / num_shots
    pops[(paulicheck, markoviancheck, m3)] = probabilities.get('0'*len(ghz_qubits), 0) + probabilities.get('1'*len(ghz_qubits), 0)
    ebs[(paulicheck, markoviancheck, m3)] = error_bar
    probs[(paulicheck, markoviancheck, m3)] = probabilities

print(hfs)
data = {
    'backend': backend.name,
    'n': len(ghz_qubits),
    'ghz_qubits': ghz_qubits,
    'check_qubits': check_qubits,
    'spectator_qubits': spectator_qubits,
    'depth': job_metadata["depth"],
    'coverage': job_metadata["coverage"] if job_metadata.get('coverage') else 0,
    'shots': job_metadata["shots"],
    'hfs': hfs,
    'psrs': psrs,
    'pops': pops,
    'ebs': ebs,
    'probs': probs
}

data_path = './data_population'
with open(f'{data_path}/{basename}.pkl', 'wb') as f:
    pkl.dump(data, f)
print(f'saved data to {data_path}/{basename}.pkl')
