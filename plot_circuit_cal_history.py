import os
import argparse
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_device_benchmarking.utilities.gate_map import plot_gate_map


INSTANCE = None

def edge_errors(props, edges):
    cz_errors = []
    for edge in edges:
        cz_errors.append(props.gate_error('cz', edge))
    return cz_errors


def node_errors(props, nodes):
    meas_errors = []
    for node in nodes:
        meas_errors.append(props.readout_error(node))
    return meas_errors


parser = argparse.ArgumentParser()
parser.add_argument('filename', type=str)
parser.add_argument('--days', type=int, default=1)
args = parser.parse_args()


service = QiskitRuntimeService(name=INSTANCE)
backend = 'ibm_' + os.path.dirname(args.filename).removeprefix('circuits_')
backend = service.backend(backend)
coupling_map = backend.target.build_coupling_map()

circuits_path = './circuits'
with open(args.filename, 'rb') as f:
    circuits_info = pkl.load(f)
qc = circuits_info['circuit']
ghz_qubits = circuits_info['ghz_qubits']
checks = circuits_info['checks']
depth = circuits_info['depth']
coverage = circuits_info['coverage']
print(f'{depth} depth, {len(checks)} checks covering {100*coverage:.0f}%')

circuit_edges = {}
circuit_nodes = {}
for instruction in qc.data:
    if instruction.operation.name in ['cz', 'cx']:
        edge = tuple(sorted([qc.qubits.index(q) for q in instruction.qubits]))
        if edge not in circuit_edges:
            circuit_edges[edge] = []
    elif instruction.operation.num_qubits == 1 and instruction.name != 'delay':
        node = qc.qubits.index(instruction.qubits[0])
        if node not in circuit_nodes:
            circuit_nodes[node] = []

def get_color(i):
    if i in ghz_qubits:
        return 'blue'
    elif i in checks:
        return 'salmon'
    else:
        return 'white'
qubit_color = [get_color(i) for i in range(0, backend.num_qubits)]
line_color = ['black' if ((edge[0], edge[1]) in circuit_edges or (edge[1], edge[0]) in circuit_edges) else 'white' for edge in coupling_map.graph.edge_list()]
plot_gate_map(backend, label_qubits=True, qubit_color=qubit_color, line_color=line_color, line_width=20)
plt.savefig('./plots/ghz_layout.pdf', bbox_inches='tight')


now = datetime.datetime.now()
days_ago = now - datetime.timedelta(days=args.days)
dates = [days_ago + datetime.timedelta(hours=12*(i+1)) for i in range(2*(now - days_ago).days)]  # 12-hour increments (=1 cal cycle)
print(
        'date   ', '\t', 'time', '\t\t'
        'cz (min, avg, max)', '\t',
        'ro (min, avg, max)', '\t',
        'cz checks (min, avg, max)'
)
for date in dates:
    props = backend.properties(datetime=date)
    cz_errors = edge_errors(props, circuit_edges)
    meas_errors_ghz = node_errors(props, ghz_qubits)
    meas_errors_checks = node_errors(props, checks)
    print(
            date.strftime("%Y-%m-%d %H:%M"), '\t',
            np.round(np.min(cz_errors), 4), np.round(np.mean(cz_errors), 4), np.round(np.max(cz_errors), 4), '\t',
            np.round(np.min(meas_errors_ghz), 4), np.round(np.mean(meas_errors_ghz), 4), np.round(np.max(meas_errors_ghz), 4), '\t',
            np.round(np.min(meas_errors_checks), 4), np.round(np.mean(meas_errors_checks), 4), np.round(np.max(meas_errors_checks), 4)
    )
    for edge in circuit_edges:
        fid = props.gate_error('cz', edge)
        circuit_edges[edge].append(fid)
    for node in circuit_nodes:
        fid = props.readout_error(node)
        circuit_nodes[node].append(fid)

all_circuit_edges = sorted(list(circuit_edges.keys()))

# Plot the data
num_gates = len(all_circuit_edges)
indices = range(0, num_gates)
fig, axs = plt.subplots(num_gates, 1, figsize=(10, num_gates*2), sharex=True)
for idx, ax in enumerate(axs):
    edge = all_circuit_edges[idx]
    ax.plot(dates, circuit_edges[edge], marker='o', linestyle='-', color='b')
    ax.set_title(f'{backend.name} CZ ({edge[0]}, {edge[1]})')
    ax.set_ylabel('Error')
    ax.set_ylim(0, .025)
    for label in ax.get_xticklabels():
        label.set_rotation(45)
axs[-1].set_xlabel("Date")

# Display the plot
plt.tight_layout()
plt.savefig('cal_history.pdf')
plt.close()
os.system('open cal_history.pdf')

