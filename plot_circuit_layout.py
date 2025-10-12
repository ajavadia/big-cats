# > python plot_circuit_layout.py circuits_kingston/ghz100_undo0_depth20_coverage31.pk
import os
import argparse
import matplotlib.pyplot as plt
import pickle as pkl
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_device_benchmarking.utilities.gate_map import plot_gate_map

INSTANCE=None

parser = argparse.ArgumentParser()
parser.add_argument('filename', type=str)
args = parser.parse_args()

backend = 'ibm_' + os.path.dirname(args.filename).removeprefix('circuits_')

service = QiskitRuntimeService(name=INSTANCE) 
backend = service.backend(f"{backend}")

with open(args.filename, 'rb') as f:
    circuits_info = pkl.load(f)
base_circuit = circuits_info['circuit']


import datetime
date=datetime.datetime(2025, 6, 4, 21, 31)
props = backend.properties(datetime=date)
def bad_cz(target, threshold=.01):
    undirected_edges = []
    for edge in target.build_coupling_map().get_edges():
        if (edge[1], edge[0]) not in undirected_edges:
            undirected_edges.append(edge)
    edges = undirected_edges
    cz_errors = {}
    for edge in edges:
        cz_errors[edge] = props.gate_error('cz', edge)
    worst_edges = sorted(cz_errors.items(), key=lambda x: x[1], reverse=True)
    return [list(edge) for edge, error in worst_edges if error > threshold]
def bad_readout(target, threshold=.01):
    meas_errors = {}
    for node in range(backend.num_qubits):
        meas_errors[node] = props.readout_error(node)
    worst_nodes = sorted(meas_errors.items(), key=lambda x: x[1], reverse=True)
    return [node for node, error in worst_nodes if error > threshold]
bad_edges = bad_cz(backend.target, threshold=0.01)
bad_nodes = bad_readout(backend.target, threshold=0.08)
print(bad_nodes)
print(bad_edges)
root = circuits_info.get('root', None)
ghz_qubits = circuits_info['ghz_qubits']
checks = circuits_info['checks']

# plot the layout of ghz, check and spectator qubits
plot_gate_map(backend, label_qubits=True, line_width=20,
              line_color = ['whitesmoke' if ([edge[0], edge[1]] in bad_edges or [edge[1], edge[0]] in bad_edges) else
                            'black' for edge in backend.coupling_map.graph.edge_list()],
              qubit_color=['green' if i == root else
                           'blue' if i in ghz_qubits else 
                           'salmon' if i in checks else 
                           'whitesmoke' if i in bad_nodes else
                           'lightgrey' for i in range(0, backend.num_qubits)],
              #qubit_labels=[str(i) if i in [3, 4, 5, 6, 7, 16, 17, 21, 22, 23, 27, 28, 29, 36, 38, 41, 42, 43, 47, 48, 49, 56, 57, 63, 64, 65, 66, 67, 78] else '' for i in range(backend.num_qubits)]
              qubit_labels=[str(i) for i in range(backend.num_qubits)]
              )
plt.savefig('./plots/layout.pdf', bbox_inches='tight')
print(f'saved ./plots/layout.pdf')
