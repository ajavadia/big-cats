import numpy as np
from functools import partial
from qiskit.circuit import Delay
from qiskit.circuit.library import XGate, RZGate
from qiskit.transpiler import PassManager, AnalysisPass
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler.passes import ALAPScheduleAnalysis, PadDynamicalDecoupling, PadDelay, RemoveBarriers, CollectAndCollapse
from qiskit.transpiler.passes.optimization.collect_and_collapse import collect_using_filter_function, collapse_to_operation

# the scheduling consists of first inserting delays while barriers are still there
# then removing the barriers and consolidating the delays, so that the operations do not move in time
# lastly we replace delays with dynamical decoupling
collect_function = partial(
    collect_using_filter_function,
    filter_function=(lambda node: node.op.name=='delay'),
    split_blocks=True,
    min_block_size=2,
    split_layers=False,
    collect_from_back=False,
    max_block_width=None,
)

collapse_function = partial(
    collapse_to_operation,
    collapse_function=(lambda circ: Delay(sum(inst.operation.duration for inst in circ)))
)

class Unschedule(AnalysisPass):
    """Removes a property from the passmanager property set so that the circuit looks unscheduled, so we can schedule it again."""
    def run(self, dag):
        del self.property_set["node_start_time"]

def build_passmanager(backend, dd_qubits=None):
    pm = generate_preset_pass_manager(
            target=backend.target,
            layout_method='trivial',
            optimization_level=2,
            routing_method='none'
        )

    pm.scheduling = PassManager([
            ALAPScheduleAnalysis(target=backend.target),
            PadDelay(target=backend.target),
            RemoveBarriers(),
            Unschedule(),
            CollectAndCollapse(collect_function=collect_function, collapse_function=collapse_function),
            ALAPScheduleAnalysis(target=backend.target),
            PadDynamicalDecoupling(dd_sequence=[XGate(), RZGate(-np.pi), XGate(), RZGate(np.pi)],
                                   spacing=[1/4, 1/2, 0, 0, 1/4],
                                   target=backend.target,
                                   qubits=dd_qubits),
            ])

    return pm
