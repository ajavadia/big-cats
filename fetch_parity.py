import os
import argparse
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

np.set_printoptions(linewidth=300)

parser = argparse.ArgumentParser()
parser.add_argument('filename', type=str)
parser.add_argument('--limit', type=int, default=None)
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
checks = job_metadata['checks']
spectators = job_metadata['spectators']
angles = job_metadata['angles']
twirl_xs = job_metadata['twirl_xs']
num_twirls = len(twirl_xs)
print(f'backend: {backend.name}, shots: {num_shots}')

if args.limit:
    full_jobs = full_jobs[:args.limit]
    angles = angles[:args.limit]

# observable for all experiments
all_z = Pauli('Z' * len(ghz_qubits))
angle_results = [job.result() for job in full_jobs]

# get trex reference expvals, in both raw and markoviancheck post-selected
print("building trex refs")
trex_res = trex_cal_jobs[0].result()
ref_expvals, ref_psrs = [{k: [] for k in [False, True]} for _ in range(2)]
for twirl_index, twirl in enumerate(twirl_xs):
    # need to flip the expval of all_z if there are an odd number of Xs in the twirls
    flip = np.count_nonzero(twirl) % 2
    ref_res = trex_res[twirl_index].data
    meas_bitstrings_trex = ref_res.c1.get_bitstrings()
    check_bitstrings_trex = ref_res.c2.get_bitstrings() if getattr(ref_res, 'c2', None) else ['0'] * num_shots
    spec_bitstrings_trex = ref_res.c3.get_bitstrings() if getattr(ref_res, 'c3', None) else ['0'] * num_shots
    meas_bitstrings_2_trex = ref_res.c4.get_bitstrings() if getattr(ref_res, 'c4', None) else [''.join('0' if s=='1' else '1' for s in bs) for bs in meas_bitstrings_trex]
    check_bitstrings_2_trex = ref_res.c5.get_bitstrings() if getattr(ref_res, 'c5', None) else ['1'] * num_shots
    spec_bitstrings_2_trex = ref_res.c6.get_bitstrings() if getattr(ref_res, 'c6', None) else ['1'] * num_shots
    # expvals under different post-selections
    accepted_meas_trex = {k: [] for k in [False, True]}
    for markoviancheck in accepted_meas_trex.keys():
        for meas, check, spec, meas2, check2, spec2 in zip(meas_bitstrings_trex, check_bitstrings_trex, spec_bitstrings_trex,
                                                           meas_bitstrings_2_trex, check_bitstrings_2_trex, spec_bitstrings_2_trex):
            if no_nonmarkovian(backend, ghz_qubits, checks, spectators, meas, check, spec, meas2, check2, spec2, do_check=markoviancheck):
                accepted_meas_trex[markoviancheck].append(meas)
        ref_expval = (-1)**flip * BitArray.from_samples(accepted_meas_trex[markoviancheck], num_bits=len(ghz_qubits)).expectation_values(all_z)
        ref_psr = len(accepted_meas_trex[markoviancheck]) / num_shots
        ref_expvals[markoviancheck].append(ref_expval)
        ref_psrs[markoviancheck].append(ref_psr)

# get main expvals, in both raw and post-selected (with paulicheck and/or markoviancheck)
print("calculating main expvals")
main_expvals, main_psrs = [{k: [] for k in it.product([False, True], repeat=2)} for _ in range(2)]
for angle_index, angle in enumerate(angles):
    res = angle_results[angle_index]
    print('    angle:', np.round(angle, 3))
    angle_expvals, angle_psrs = [{k: [] for k in it.product([False, True], repeat=2)} for _ in range(2)]
    for twirl_index, twirl in enumerate(twirl_xs):
        # need to flip the expval of all_z if there are an odd number of Xs in the twirls
        flip = np.count_nonzero(twirl) % 2
        main_res = res[twirl_index].data
        meas_bitstrings = main_res.c1.get_bitstrings()
        check_bitstrings = main_res.c2.get_bitstrings() if getattr(main_res, 'c2', None) else ['0'] * num_shots
        spec_bitstrings = main_res.c3.get_bitstrings() if getattr(main_res, 'c3', None) else ['0'] * num_shots
        meas_bitstrings_2 = main_res.c4.get_bitstrings() if getattr(main_res, 'c4', None) else [''.join('0' if s=='1' else '1' for s in bs) for bs in meas_bitstrings]
        check_bitstrings_2 = main_res.c5.get_bitstrings() if getattr(main_res, 'c5', None) else ['1'] * num_shots
        spec_bitstrings_2 = main_res.c6.get_bitstrings() if getattr(main_res, 'c6', None) else ['1'] * num_shots
        # expvals under different post-selections
        accepted_meas = {k: [] for k in it.product([False, True], repeat=2)}
        for paulicheck, markoviancheck in accepted_meas.keys():
            for meas, check, spec, meas2, check2, spec2 in zip(meas_bitstrings, check_bitstrings, spec_bitstrings,
                                                               meas_bitstrings_2, check_bitstrings_2, spec_bitstrings_2):
                if no_flag(check, paulicheck) and no_nonmarkovian(backend, ghz_qubits, checks, spectators, meas, check, spec, meas2, check2, spec2, markoviancheck):
                    accepted_meas[(paulicheck, markoviancheck)].append(meas)

            expval = (-1)**flip * BitArray.from_samples(accepted_meas[(paulicheck, markoviancheck)], num_bits=len(ghz_qubits)).expectation_values(all_z)
            psr = len(accepted_meas[(paulicheck, markoviancheck)]) / num_shots
            angle_expvals[(paulicheck, markoviancheck)].append(expval)
            angle_psrs[(paulicheck, markoviancheck)].append(psr)

    for paulicheck, markoviancheck in main_expvals.keys():
        main_expvals[(paulicheck, markoviancheck)].append(angle_expvals[(paulicheck, markoviancheck)])
        main_psrs[(paulicheck, markoviancheck)].append(angle_psrs[(paulicheck, markoviancheck)])


data = {
    'backend': backend.name,
    'n': len(ghz_qubits),
    'ghz_qubits': len(ghz_qubits),
    'checks': len(checks),
    'spectators': len(spectators),
    'depth': job_metadata["depth"],
    'coverage': job_metadata["coverage"],
    'angles': job_metadata["angles"],
    'shots': num_shots,
    'ref_expvals': ref_expvals,
    'main_expvals': main_expvals,
    'ref_psrs': ref_psrs,
    'main_psrs': main_psrs,
}


data_path = './data_parity'
with open(f'{data_path}/{basename}.pkl', 'wb') as f:
    pkl.dump(data, f)
print(f'saved data to {data_path}/{basename}.pkl')
