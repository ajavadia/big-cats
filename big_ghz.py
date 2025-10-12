import copy
import sys
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
from collections import deque, defaultdict
from qiskit.circuit import QuantumCircuit, ClassicalRegister
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import RemoveBarriers, RemoveFinalMeasurements
from qiskit_device_benchmarking.utilities.gate_map import plot_gate_map
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService


"""Create a big GHZ state on a given backend.

This code starts by eliminating some edges with bad CZ and some nodes with bad readouts (above given thresholds).
Then it iterates on all the nodes, and attempts to make a GHZ of a given size starting at the node as root and BFS-ing outwards.
The node with the shortest depth GHZ is selected as root.
From this root, we randomly block some subset of the qubits belonging to the GHZ and rerun the BFS search. 
Any node that does not belong to the GHZ but is adjacent to 2 of the GHZ qubits may act as a check for a ZZ parity measurement of the GHZ.
We aim to maximize the ``coverage'' of checks that we can find through this randomization, while not increasing the depth beyond a given threshold
above the best depth that we already found. 

The coverage of a check are the spacetime locations that lie between the two checked leaves and their lowest common ancestor. 
"""

BACKEND = 'ibm_kingston' # choose a backend to build GHZ on.
INSTANCE = None          # specify an instance "name" for the QiskitRuntimeService (otherwise the default will be used)

ROOT = None              # root for BFS search
GHZ_SIZE = 100           # number of (data) qubits in the GHZ state
SKIP = []                # nodes to intentionally skip so that we have a better chance for finding checks
THRESH_CZ = .025         # exclude from BFS those edges whose CZ error is worse than this threshold
THRESH_MEAS = .15        # exclude from BFS those nodes whose measurement error is worse than this threshold
THRESH_T2 = 10           # exclude from BFS those nodes whose T2 error is lower than this threshold
UNDO_NODES = 0           # how many of the earliest visited nodes to undo, before entangling them again towards the end of the circuit

SEED = 1                 # seed of randomization
MAX_SKIPS = 10           # at most how many qubits to skip (i.e. block) while making GHZ (in addition to the bad ones and the ones forced to skip above)
SHUFFLES = 200           # how many times to try removing nodes for checks
MAX_DEPTH_INCREASE = 10  # how far from the base ghz depth do we go in order to include checks (increase if you want more checks at expense of depth)

W_IDLE = 0.2             # weight of errors to consider during idle timesteps
W_GATE = 0.8             # weight of errors to consider during gates


def active_qubits(circ):
    active_qubits = set()
    for inst in circ.data:
        if inst.operation.name != "delay" and inst.operation.name != "barrier":
            for qubit in inst.qubits:
                q = circ.find_bit(qubit).index
                active_qubits.add(q)
    return list(active_qubits)


def active_gates(circ):
    used_2q_gates = set()
    for inst in circ:
        if inst.operation.num_qubits == 2:
            qs = inst.qubits
            qs = sorted([circ.find_bit(q).index for q in qs])
            used_2q_gates.add(tuple(sorted(qs)))
    return used_2q_gates


def active_wires(layers):
    """
    Returns per-layer dict with two sets:
    - 'idle': activated wires that are idle in this layer
    - 'gate': activated wires that are control/target of a CNOT at this layer
    """
    first_activation = {}
    for l, layer in enumerate(layers):
        for c, t in layer:
            first_activation.setdefault(c, l)
            first_activation.setdefault(t, l)
    result = {}
    for l in range(len(layers)):
        active = {q for q, l0 in first_activation.items() if l >= l0}
        gate = {q for c, t in layers[l] for q in (c, t)}
        idle = active - gate
        result[l] = {"idle": idle, "gate": gate}
    return result


def z_trace_backward(layers, initial_Zs):
    """
    Backward propagate Zs with parity cancellation.
    Returns {layer: set of qubits with odd parity Z at that layer}.
    """
    wires = active_wires(layers)
    support = set(initial_Zs)
    trace = {}
    for l in range(len(layers)-1, -1, -1):
        active = wires[l]["idle"] | wires[l]["gate"]
        trace[l] = support & active
        # propagate backwards
        new_support = set()
        for q in support:
            hit = False
            for c, t in layers[l]:
                if q == t:         # Z on target: copy to control
                    new_support ^= {t, c}   # toggle both
                    hit = True
                    break
                elif q == c:       # Z on control: passes through
                    new_support ^= {c}
                    hit = True
                    break
            if not hit:            # unaffected
                new_support ^= {q}
        support = new_support
    return trace


def weighted_coverage(layers, parities, w_idle=W_IDLE, w_gate=W_GATE):
    """
    Compute weighted fraction (idle + gate) of wires that are
    covered by at least one parity to all active wires.
    """
    wires = active_wires(layers)
    covered_by_any = {l: set() for l in range(len(layers))}
    for parity in parities:
        trace = z_trace_backward(layers, parity)
        for l, qs in trace.items():
            covered_by_any[l] |= qs
    covered_weight = 0
    total_weight = 0
    for l in range(len(layers)):
        idle = wires[l]["idle"]
        gate = wires[l]["gate"]
        total_weight += w_idle * len(idle) + w_gate * len(gate)
        covered_idle = covered_by_any[l] & idle
        covered_gate = covered_by_any[l] & gate
        covered_weight += w_idle * len(covered_idle) + w_gate * len(covered_gate)
    return covered_weight / total_weight if total_weight > 0 else 0


def bad_cz(target, threshold=.01):
    undirected_edges = []
    for edge in backend.target.build_coupling_map().get_edges():
        if (edge[1], edge[0]) not in undirected_edges:
            undirected_edges.append(edge)
    edges = undirected_edges
    cz_errors = {}
    for edge in edges:
        cz_errors[edge] = target['cz'][edge].error
    worst_edges = sorted(cz_errors.items(), key=lambda x: x[1], reverse=True)
    return [list(edge) for edge, error in worst_edges if error > threshold]


def bad_readout(target, threshold=.01):
    meas_errors = {}
    for node in range(backend.num_qubits):
        meas_errors[node] = target['measure'][(node, )].error
    worst_nodes = sorted(meas_errors.items(), key=lambda x: x[1], reverse=True)
    return [node for node, error in worst_nodes if error > threshold]


def bad_coherence(target, threshold=60):
    t2s = {}
    for node in range(backend.num_qubits):
        t2 = target.qubit_properties[node].t2
        t2s[node] = t2 * 1e6 if t2 else 0
    worst_nodes = sorted(t2s.items(), key=lambda x: x[1])
    return [node for node, val in worst_nodes if val < threshold]


def circuit_errors(target, circ, error_type="cz"):
    active_edges = active_gates(circ)
    edges = [edge for edge in target.build_coupling_map().get_edges() if tuple(sorted(edge)) in active_edges]
    undirected_edges = []
    for edge in edges:
        if (edge[1], edge[0]) not in undirected_edges:
            undirected_edges.append(edge)
    edges = undirected_edges
    cz_errors, meas_errors, t1_errors, t2_errors = [], [], [], []
    for edge in active_gates(circ):
        cz_errors.append(target['cz'][edge].error)
    for qubit in active_qubits(circ):
        meas_errors.append(target['measure'][(qubit, )].error)
        t1_errors.append(target.qubit_properties[qubit].t1 * 1e6)
        t2_errors.append(target.qubit_properties[qubit].t2 * 1e6)
    if error_type == "cz":
        return cz_errors
    elif error_type == "meas":
        return meas_errors
    elif error_type == "t1":
        return t1_errors
    else:
        return t2_errors


def sched_asap(ops):
    """
    Schedule a list of two-qubit gates (i.e. an (int, int) pair) as soon as possible,
    respecting dependencies and minimizing layers.
    """
    n = len(ops)
    deps = defaultdict(set)
    last_use = {}

    # build implicit dependency graph
    for i, (a, b) in enumerate(ops):
        for node in (a, b):
            if node in last_use:
                deps[i].add(last_use[node])
            last_use[node] = i

    # topological sort
    in_deg = [0] * n
    for i in range(n):
        for d in deps[i]:
            in_deg[i] += 1
    queue = deque([i for i in range(n) if in_deg[i] == 0])
    order = []

    while queue:
        u = queue.popleft()
        order.append(u)
        for v in range(n):
            if u in deps[v]:
                in_deg[v] -= 1
                if in_deg[v] == 0:
                    queue.append(v)

    # Schedule into layers
    layers = []
    op_layer = {}

    for i in order:
        a, b = ops[i]
        min_layer = max((op_layer[d] + 1 for d in deps[i]), default=0)
        while True:
            if len(layers) <= min_layer:
                layers.append([])
            used = {n for op in layers[min_layer] for n in op}
            if a not in used and b not in used:
                layers[min_layer].append((a, b))
                op_layer[i] = min_layer
                break
            min_layer += 1

    return layers


def parallel_ghz(root, num_qubits, backend, bad_edges, skip, undo_nodes=UNDO_NODES):
    """
    Build a GHZ state of size num_qubits on a backend, starting at root and exploring outwards in BFS order.
    BFS here is layer-wise such that at each layer only one neighbor can be explored (to not cause qubit conflict).

    Avoid bad_edges and skip nodes during the BFS search (the skip might later be used for checks).

    Undo the entanglement of some earliest-visited nodes as soon as possible (once they are not used anymore), using one of their neighbors.
    This lets those qubits remain in the ground state while the rest of the circuit is being built.
    Redo their entanglement towards the end of the circuit. We must start redoing early enough to finish restoring by the last layer.
    """
    cmap = backend.configuration().coupling_map
    edges = [
        e for e in cmap
        if e not in bad_edges and [e[1], e[0]] not in bad_edges
        and e[0] not in skip and e[1] not in skip
    ]

    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    qc = QuantumCircuit(backend.configuration().num_qubits)
    visited = [root]
    queue = deque([root])
    undone = set([])
    explored = defaultdict(set)
    layers = []
    undo_targets = [root] if undo_nodes > 0 else []
    qc.h(root)
    found_undo_nodes = False
    undo_layers = []
    undo_gates = []
    while queue and len(visited) < num_qubits:
        # Explore: each node explores one unvisited neighbor
        layer = []
        current = list(queue)
        busy = set()  # Track qubits used in this layer to avoid conflicts
        for node in current:
            queue.popleft()
            unvisited_neighbors = [n for n in adj[node] if n not in visited and n not in explored[node]]
            if unvisited_neighbors:
                neighbor = unvisited_neighbors[0]
                visited.append(neighbor)

                if not found_undo_nodes:
                    if len(undo_targets) >= undo_nodes:
                        found_undo_nodes = True
                    else:
                        undo_targets.append(neighbor)

                queue.append(node)     # Keep node to explore more neighbors
                queue.append(neighbor) # Add new neighbor to queue
                explored[node].add(neighbor)
                layer.append((node, neighbor))
                busy.update([node, neighbor])
                if len(visited) == num_qubits:
                    break
            # else: do not reappend node if fully explored

        # Undo: try to undo earliest undo_nodes visited qubits
        undo_layer = []
        for q in list(undo_targets):  # Copy to allow safe removal
            for neighbor in adj[q]:
                if neighbor in visited and q not in busy and neighbor not in busy and neighbor not in undone:
                    undo_layer.append((neighbor, q))
                    undo_gates.append((neighbor, q))
                    undone.add(q)
                    busy.update([neighbor, q])
                    undo_targets.remove(q)  # Remove once undone
                    break
        if undo_layer:
            undo_layers.append(undo_layer)

        full_layer = layer + undo_layer
        if full_layer:
            layers.append(full_layer)
        else:
            break

    # now build the circuit based on the layers, and do as-late-as-possible redo of every undo_layer
    undo_layers = sched_asap(undo_gates)
    max_depth = len(layers)
    redo_depth = len(undo_layers)
    for i, full_layer in enumerate(layers):
        for q1, q2 in full_layer:
            qc.cx(q1, q2)
        if i >= max_depth - redo_depth:
            for q1, q2 in undo_layers[max_depth - i - 1]:
                qc.cx(q1, q2)
        qc.barrier()

    return qc, visited, layers


if SEED:
    np.random.seed(SEED)


service = QiskitRuntimeService(name=INSTANCE)
backend = service.backend(f"{BACKEND}")
coupling_map = backend.target.build_coupling_map()
bad_edges = bad_cz(backend.target, threshold=THRESH_CZ)
bad_nodes_readout = bad_readout(backend.target, threshold=THRESH_MEAS)
dead_qubits = bad_readout(backend.target, threshold=.4)
bad_nodes_coherence = bad_coherence(backend.target, threshold=THRESH_T2)
bad_nodes = list(set(bad_nodes_readout) | set(bad_nodes_coherence))
print(f'{len(bad_edges)} bad edges: \n{bad_edges}')
print(f'{len(bad_nodes)} bad nodes: \n{bad_nodes}')


# search for the best root (yieling the shallowest GHZ)
if ROOT is None:
    best_root = -1
    base_depth = 100
    for root in range(backend.num_qubits):
        qc, ghz_qubits, _ = parallel_ghz(root, GHZ_SIZE, backend, bad_edges, SKIP, UNDO_NODES)
        if len(ghz_qubits) != GHZ_SIZE:
            continue
        depth = qc.depth(lambda x: x.operation.num_qubits==2)
        if depth < base_depth:
            best_root = root
            base_depth = depth
    ROOT = best_root

# build a ghz starting at the best root
qc, ghz_qubits, _ = parallel_ghz(ROOT, GHZ_SIZE, backend, bad_edges, SKIP+bad_nodes, UNDO_NODES)
base_depth = qc.depth(lambda x: x.operation.num_qubits==2)
base_count = qc.size(lambda x: x.operation.num_qubits==2)
print(f'base depth: {base_depth}, base count: {base_count}')
print(f'ROOT: {ROOT}')
if len(ghz_qubits) != GHZ_SIZE:
    raise Exception("No GHZ found. Relax error thresholds.")

# remove random nodes from the ghz and build from the root again in order to increase checks
degree_two_nodes = [i for i in ghz_qubits
                    if all(n in ghz_qubits for n in coupling_map.neighbors(i)) 
                    and len(coupling_map.neighbors(i))>=2
                    ]
num_checks = 0
best_covered_fraction = -1
best_qc = qc
best_checks = []
best_parities = []
best_layers = []
for num_skips in range(MAX_SKIPS):
    for _ in range(SHUFFLES):
        skip = SKIP + list(np.random.choice(degree_two_nodes, num_skips))
        qc, ghz_qubits, layers = parallel_ghz(ROOT, GHZ_SIZE, backend, bad_edges, skip+bad_nodes, UNDO_NODES)
        depth = qc.depth(lambda x: x.operation.num_qubits==2)
        if len(ghz_qubits) != GHZ_SIZE:
            continue
        checks = []
        parities = []
        for i in range(backend.num_qubits):
            neighbors = [n for n in coupling_map.neighbors(i) if n in ghz_qubits]
            if (i not in ghz_qubits and 
                i not in dead_qubits and
                len(neighbors)>=2 and
                not any([[neighbor, i] in bad_edges or [i, neighbor] in bad_edges for neighbor in neighbors])
                ):
                checks.append(i)
                parities.append((neighbors[0], neighbors[1]))
                qc.cx(neighbors[0], i)
                qc.cx(neighbors[1], i)
        covered_fraction = weighted_coverage(layers=layers, parities=parities)
        if covered_fraction > best_covered_fraction and depth <= base_depth + MAX_DEPTH_INCREASE:
            best_covered_fraction = covered_fraction
            best_qc = qc
            best_ghz_qubits = ghz_qubits
            best_checks = checks
            best_parities = parities
            best_layers = layers

qc = best_qc
checks = best_checks
parities = best_parities
layers = best_layers
ghz_qubits = best_ghz_qubits
if len(ghz_qubits) != GHZ_SIZE:
    raise Exception("No GHZ found. Relax error thresholds.")

print(f'ghz qubits: {ghz_qubits} {len(ghz_qubits)}')
print(f'check qubits: {checks} {len(checks)}')

covered_fraction = weighted_coverage(layers=layers, parities=parities)
print(f'covered fraction (no idle): ', weighted_coverage(layers=layers, parities=parities, w_idle=0., w_gate=1.))

cz_errors = circuit_errors(backend.target, qc, error_type="cz")
meas_errors = circuit_errors(backend.target, qc, error_type="meas")
t1_errors = circuit_errors(backend.target, qc, error_type="t1")
t2_errors = circuit_errors(backend.target, qc, error_type="t2")
np.set_printoptions(linewidth=np.inf)
print(f'cz errors:\n{np.round(cz_errors, 3)} \n mean: {np.round(np.mean(cz_errors), 3)}, max: {np.round(np.max(cz_errors), 3)}') 
print(f'meas errors:\n{np.round(meas_errors, 3)} \n mean: {np.round(np.mean(meas_errors), 3)}, max: {np.round(np.max(meas_errors), 3)}')
print(f't1 errors:\n{np.round(t1_errors, 1)} \n mean: {np.round(np.mean(t1_errors), 1)}, min: {np.round(np.min(t1_errors), 1)}')
print(f't2 errors:\n{np.round(t2_errors, 1)} \n mean: {np.round(np.mean(t2_errors), 1)}, min: {np.round(np.min(t2_errors), 1)}')


# simulate to ensure correctness
qc_meas = qc.copy()
c1 = ClassicalRegister(len(ghz_qubits), 'c1')
qc_meas.add_register(c1)
for q, c in zip(ghz_qubits, c1):
    qc_meas.measure(q, c)
if len(checks) > 0:
    c2 = ClassicalRegister(len(checks), 'c2')
    qc_meas.add_register(c2)
    for q, c in zip(checks, c2):
        qc_meas.measure(q, c)
sim_stab = AerSimulator(method='stabilizer')
res = sim_stab.run(qc_meas, shots=1000).result()
counts = res.get_counts()
print('stabilizer simulation result:')
print(counts)


def get_color(i):
    if i == ROOT:
        return 'green'
    elif i in ghz_qubits:
        return 'blue'
    elif i in checks:
        return 'salmon'
    elif i in bad_nodes:
        return 'grey'
    else:
        return 'black'
qubit_color = [get_color(i) for i in range(0, backend.num_qubits)]
line_color = ['white' if ([edge[0], edge[1]] in bad_edges or [edge[1], edge[0]] in bad_edges) else 'black' for edge in coupling_map.graph.edge_list()]
plot_gate_map(backend, label_qubits=True, qubit_color=qubit_color, line_color=line_color, line_width=20)
depth = qc.depth(lambda x: x.operation.num_qubits==2)
count = qc.size(lambda x: x.operation.num_qubits==2)
coverage = covered_fraction
plt.savefig('./plots/ghz_layout.pdf', bbox_inches='tight')
qc.draw('mpl', fold=-1, idle_wires=False, filename='./plots/ghz.pdf')

answer = input(
        f"---------------------------\n"
        f"Found {len(ghz_qubits)}-qubit GHZ.\n"
        f"depth = {depth}\n"
        f"count = {count}\n"
        f"coverage = {coverage}\n"
        f"checks = {len(checks)}\n"
        f"CZ errors: mean = {np.round(np.mean(cz_errors), 3)}, max = {np.round(np.max(cz_errors), 3)}\n"
        f"RO errors: mean = {np.round(np.mean(meas_errors), 3)}, max = {np.round(np.max(meas_errors), 3)}\n"
        f"Save circuit? [y/n]\n")
if answer.lower() in ["n", "no"]:
    sys.exit()


circuits_info = {
        'circuit': qc,
        'ghz_qubits': ghz_qubits,
        'checks': checks,
        'depth': depth,
        'coverage': coverage,
        'undo': UNDO_NODES,
        'root': ROOT,
        'skip': SKIP,
        'bad_nodes': bad_nodes,
        'bad_edges': bad_edges,
        'thresh_cz': THRESH_CZ,
        'thresh_meas': THRESH_MEAS
}

circuits_name = f'ghz{len(ghz_qubits)}_undo{UNDO_NODES}_depth{depth}_coverage{100*coverage:.0f}'
circuits_path = f'./circuits_{BACKEND.removeprefix("alt_").removeprefix("ibm_")}'
with open(f'{circuits_path}/{circuits_name}.pkl', 'wb') as f:
    pkl.dump(circuits_info, f)
print(f'saved circuits to {circuits_path}/{circuits_name}.pkl')
