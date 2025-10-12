import os
import argparse
import random
import pickle as pkl
import numpy as np
import itertools as it
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.primitives import BitArray
from qiskit.quantum_info import Pauli


def no_flag(check, do_check):
    if not do_check:
        return True
    return np.all([b=='0' for b in check])


def no_nonmarkovian(backend, ghz_qubits, check_qubits, spectator_qubits, meas, check, spec, meas2, check2, spec2, do_check):
    if not do_check:
        return True
    wrong_qubits = []
    for i, (b1, b2) in enumerate(zip(meas, meas2)):
        if b1 == b2:
            wrong_qubits.append(ghz_qubits[-i-1])
    for i, (b1, b2) in enumerate(zip(check, check2)):
        if b1 == b2:
            wrong_qubits.append(check_qubits[-i-1])
    for i, (b1, b2) in enumerate(zip(spec, spec2)):
        if b1 == b2:
            wrong_qubits.append(spectator_qubits[-i-1])
    return not any(e[0] in wrong_qubits and e[1] in wrong_qubits for e in backend.coupling_map)

np.set_printoptions(linewidth=200)

parser = argparse.ArgumentParser()
parser.add_argument('filename', type=str)
parser.add_argument('--limit', type=int, default=None)
parser.add_argument('--pad-diagonals', action='store_true')  # if True, compute some {I, Z}^n obseravbles from the all-Z experiment
args = parser.parse_args()

with open(args.filename, 'rb') as f:
    job_metadata = pkl.load(f)

basename = os.path.splitext(os.path.basename(args.filename))[0]
print(basename)

backend_name = job_metadata['backend']
service = QiskitRuntimeService(name=job_metadata['instance_name'])
full_jobs = [service.job(job_id) for job_id in job_metadata['job_ids']]
backend = full_jobs[0].backend()
trex_cal_jobs = [service.job(job_id) for job_id in job_metadata['trex_cal_job_ids']]
num_shots = job_metadata['shots']
ghz_qubits = job_metadata['ghz_qubits']
root = job_metadata.get('root', -1)
checks = job_metadata['checks']
spectators = job_metadata['spectators']
observables = job_metadata['observables']
twirl_xs = job_metadata['twirl_xs']
print(f'backend: {backend.name}, shots: {num_shots}')


# Add some diagonal stabilizers (those with even Z weight) which we can obtain from the full-weight diagonal measurement
if args.pad_diagonals:
    diagonal_paulis = []
    for _ in range(len(observables) - 2):
        weight = random.randrange(0, len(ghz_qubits) + 1, 2)
        positions = random.sample(ghz_qubits, weight)
        string = ''.join('Z' if i in positions else 'I' for i in range(len(observables[0])))
        observables.append(Pauli(string[::-1]))
        full_jobs.append(full_jobs[0])

if args.limit:
    full_jobs = full_jobs[:args.limit]
    observables = observables[:args.limit]

# filter out failed jobs
obs_results, succeeded_observables = [], []
for obs, job in zip(observables, full_jobs):
    try:
        obs_results.append(job.result(timeout=5))
        succeeded_observables.append(obs)
    except:
        pass
observables = succeeded_observables

# record the observable phases then set them to zero
obs_phases = [obs.phase for obs in observables]
for obs in observables:
    obs.phase = 0

# make a bitarray for the trex ref, which we will use to calculate different diagonal expectation values from
print("building trex ref shots")
trex_res = trex_cal_jobs[0].result() if trex_cal_jobs else None
ref_bit_arrays, ref_psrs = [{k: [] for k in [False, True]} for _ in range(2)]
for twirl_index, twirl in enumerate(twirl_xs):
    ref_res = trex_res[twirl_index].data
    meas_bitstrings_trex = ref_res.c1.get_bitstrings()
    check_bitstrings_trex = ref_res.c2.get_bitstrings() if getattr(ref_res, 'c2', None) else ['0'] * num_shots
    spec_bitstrings_trex = ref_res.c3.get_bitstrings() if getattr(ref_res, 'c3', None) else ['0'] * num_shots
    meas_bitstrings_2_trex = ref_res.c4.get_bitstrings() if getattr(ref_res, 'c4', None) else [''.join('0' if s=='1' else '1' for s in bs) for bs in meas_bitstrings_trex]
    check_bitstrings_2_trex = ref_res.c5.get_bitstrings() if getattr(ref_res, 'c5', None) else ['1'] * num_shots
    spec_bitstrings_2_trex = ref_res.c6.get_bitstrings() if getattr(ref_res, 'c6', None) else ['1'] * num_shots
    # without post-selection
    ref_bit_arrays[False].append(BitArray.from_samples(meas_bitstrings_trex, num_bits=len(ghz_qubits)))
    ref_psrs[False].append(1.)
    # with post-selection
    accepted_meas_trex = []
    for meas, check, spec, meas2, check2, spec2 in zip(meas_bitstrings_trex, check_bitstrings_trex, spec_bitstrings_trex,
                                                       meas_bitstrings_2_trex, check_bitstrings_2_trex, spec_bitstrings_2_trex):
        if no_nonmarkovian(backend, ghz_qubits, checks, spectators, meas, check, spec, meas2, check2, spec2, do_check=True):
            accepted_meas_trex.append(meas)
    ref_bit_arrays[True].append(BitArray.from_samples(accepted_meas_trex, num_bits=len(ghz_qubits)))
    ref_psrs[True].append(len(accepted_meas_trex) / num_shots)

# calculate expectation value and reference for each observable measured
print("getting expval for each obs and ref_obs")
ref_expvals = {k: [] for k in [False, True]}
main_expvals, main_psrs = [{k: [] for k in it.product([False, True], repeat=2)} for _ in range(2)]
for obs_index, obs in enumerate(observables):
    sign = 0 if obs_phases[obs_index] == 0 else 1
    obs = Pauli(''.join(obs.to_label()[::-1][j] for j in ghz_qubits))                   # reduce from whole device to only the ghz qubits
    print('    obs:', obs.to_label())
    ref_obs = Pauli(''.join('Z' if j in ['X', 'Y'] else j for j in obs.to_label()))     # convert to diagonal
    res = obs_results[obs_index]
    ref_obs_expvals = {k: [] for k in [False, True]}
    obs_expvals, obs_psrs = [{k: [] for k in it.product([False, True], repeat=2)} for _ in range(2)]
    for twirl_index, twirl in enumerate(twirl_xs):
        flips = [t and p=='Z' for t, p in zip(twirl[::-1], ref_obs.to_label())]         # if twirl flipped a qubit and we need its Z expval, record a sign
        # ref expval under different post-selections
        flip = np.count_nonzero(flips) % 2                                              # flip the ref expval based on twirl considered
        for markoviancheck in ref_obs_expvals.keys():
            ref_expval = (-1)**flip * ref_bit_arrays[markoviancheck][twirl_index].expectation_values(ref_obs)
            ref_obs_expvals[markoviancheck].append(ref_expval)
        # obs expvals under different post-selections
        flip = sign ^ (np.count_nonzero(flips) % 2)                                     # flip the expval based on sign of observable and twirl considered
        main_res = res[twirl_index].data
        meas_bitstrings = main_res.c1.get_bitstrings()
        check_bitstrings = main_res.c2.get_bitstrings() if getattr(main_res, 'c2', None) else ['0'] * num_shots
        spec_bitstrings = main_res.c3.get_bitstrings() if getattr(main_res, 'c3', None) else ['0'] * num_shots
        meas_bitstrings_2 = main_res.c4.get_bitstrings() if getattr(main_res, 'c4', None) else [''.join('0' if s=='1' else '1' for s in bs) for bs in meas_bitstrings]
        check_bitstrings_2 = main_res.c5.get_bitstrings() if getattr(main_res, 'c5', None) else ['1'] * num_shots
        spec_bitstrings_2 = main_res.c6.get_bitstrings() if getattr(main_res, 'c6', None) else ['1'] * num_shots
        accepted_meas = {k: [] for k in it.product([False, True], repeat=2)}
        for paulicheck, markoviancheck in accepted_meas.keys():
            for meas, check, spec, meas2, check2, spec2 in zip(meas_bitstrings, check_bitstrings, spec_bitstrings,
                                                               meas_bitstrings_2, check_bitstrings_2, spec_bitstrings_2):
                if no_flag(check, paulicheck) and no_nonmarkovian(backend, ghz_qubits, checks, spectators, meas, check, spec, meas2, check2, spec2, markoviancheck):
                    accepted_meas[(paulicheck, markoviancheck)].append(meas)

            expval = (-1)**flip * BitArray.from_samples(accepted_meas[(paulicheck, markoviancheck)], num_bits=len(ghz_qubits)).expectation_values(ref_obs)
            psr = len(accepted_meas[(paulicheck, markoviancheck)]) / num_shots
            obs_expvals[(paulicheck, markoviancheck)].append(expval)
            obs_psrs[(paulicheck, markoviancheck)].append(psr)

    for markoviancheck in ref_expvals.keys():
        ref_expvals[markoviancheck].append(ref_obs_expvals[markoviancheck])
    for paulicheck, markoviancheck in main_expvals.keys():
        main_expvals[(paulicheck, markoviancheck)].append(obs_expvals[(paulicheck, markoviancheck)])
        main_psrs[(paulicheck, markoviancheck)].append(obs_psrs[(paulicheck, markoviancheck)])

    test_paulicheck = True
    test_markoviancheck = False
    nom = main_expvals[(test_paulicheck, test_markoviancheck)][obs_index]
    denom = ref_expvals[test_markoviancheck][obs_index]
    #print(f'obs fids: {np.round(nom, 3)}')
    #print(f'ref fids: {np.round(denom, 3)}')
    #print(f'    fids: {np.round(np.array(nom) / np.array(denom), 3)}')    
    print(f'mean obs: {np.round(np.mean(nom) / np.mean(denom), 3)}')


data = {
    'backend': backend.name,
    'n': len(ghz_qubits),
    'ghz_qubits': ghz_qubits,
    'root': root,
    'checks': checks,
    'spectators': spectators,
    'observables': observables,
    'depth': job_metadata["depth"],
    'coverage': job_metadata.get("coverage", 0),
    'shots': num_shots,
    'ref_expvals': ref_expvals,
    'main_expvals': main_expvals,
    'ref_psrs': ref_psrs,
    'main_psrs': main_psrs
}


data_path = './data_direct'
with open(f'{data_path}/{basename}.pkl', 'wb') as f:
    pkl.dump(data, f)
print(f'saved data to {data_path}/{basename}.pkl')
