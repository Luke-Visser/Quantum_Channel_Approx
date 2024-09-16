import itertools
from operator import add
from typing import Callable, NamedTuple

import numpy as np

from q_channel_approx.qubit_layouts import QubitLayout
from q_channel_approx.gate_operations import (
    H_fac,
    rx,
    ryd_vdw_fac,
    ryd_dipole_fac,
    xy_fac,
    rz,
    matmul_l,
    CNOT_fac,
    matmul_acc_ul
)


class GateCircuit(NamedTuple):
    U: Callable[[np.ndarray], np.ndarray]
    U_full: Callable[[np.ndarray], np.ndarray]
    qubit_layout: QubitLayout
    P: int
    operations: list[tuple[str, str | np.ndarray]]

    def __repr__(self) -> str:
        return f"Circuit; qubits layout: \n {self.qubit_layout} \n Parameters: {self.P} \n Operations {self.operations}"

    def __call__(self, theta: np.ndarray) -> np.ndarray:
        return self.U(theta)
    


def count_qubits(dims: int) -> int:
    return dims.bit_length() - 1


def unitary_circuit_fac(
    qubit_layout: QubitLayout, operations, repeats: int = 1
) -> GateCircuit:

    dims_A = qubit_layout.dims_A
    dims_AB = qubit_layout.dims_AB
    connections = qubit_layout.gate_connections

    DIMS_MAP = {
        "A": count_qubits(dims_A),
        "B": count_qubits(dims_AB // dims_A),
        "AB": count_qubits(dims_AB),
    }

    def init_gate(operation) -> tuple[Callable[[np.ndarray], np.ndarray], int]:
        match operation:
            case "rz", dims:
                return rz, DIMS_MAP[dims]
            case "rx", dims:
                return rx, DIMS_MAP[dims]
            case "ham", H:
                return H_fac(H, dims_AB), 0
            case "ryd-vdw", _:
                return ryd_vdw_fac(connections, dims_AB), 1
            case "ryd-dipole", _:
                return ryd_dipole_fac(connections, dims_AB), 1
            case "xy", _:
                return xy_fac(connections, dims_AB), len(connections)
            case "cnot", _:
                return CNOT_fac(connections, dims_AB), 0
            case _:
                raise ValueError(f"unknown gate: {operation}")

    _operations = [init_gate(operation) for operation in operations]

    D = len(_operations)

    params = [params for gate, params in _operations]
    params_acc = [0] + list(itertools.accumulate(params, add))
    P = sum(params)

    def unitary(theta):

        Us = np.zeros((D, dims_AB, dims_AB), dtype=np.complex128)

        for d, operation in enumerate(_operations):
            gate, params = operation
            Us[d, :, :] = gate(theta[params_acc[d] : params_acc[d + 1]])

        U = matmul_l(Us)
# =============================================================================
#         U_forward = np.array([matmul_l(Us[:d]) for d in range(D)])
#         U_backward = np.array([matmul_l(Us[d:]) for d in range(D)])
# =============================================================================

        return np.linalg.matrix_power(U, repeats)
    
    def unitary_full(theta):

        Us = np.zeros((D, dims_AB, dims_AB), dtype=np.complex128)

        for d, operation in enumerate(_operations):
            gate, params = operation
            Us[d, :, :] = gate(theta[params_acc[d] : params_acc[d + 1]])
            
        U_forward, _, U_backward = matmul_acc_ul(Us)
        U = U_forward[-1]

        if repeats !=1:
            print("note: repeats not yet implemented for U_forward and U_backward ")

        return U_forward, Us, U_backward

    return GateCircuit(unitary, unitary_full, qubit_layout, P, operations)


def HEA_fac(qubit_layout: QubitLayout, circuit_depth: int, ent_type: str = "cnot") -> GateCircuit:
    operations = [
        ("rz", "AB"),
        ("rx", "AB"),
        ("rz", "AB"),
        (ent_type, "AB"),
    ] * circuit_depth

    return unitary_circuit_fac(qubit_layout, operations)


def SHEA_trot_fac(
    qubit_layout: QubitLayout,
    H: np.ndarray,
    t: float,
    circuit_depth: int,
    q: int,
    ent_type: str = "cnot",
) -> GateCircuit:
    """Trotterized H, does a small H block for time `t` followed by one HEA cycle (ZXZ, ent)
    This sequence is repeated `depth` times.

    Args:
        qubit_layout (QubitLayout): _description_
        H (np.ndarray): _description_
        t (float): _description_
        depth (int): _description_

    Returns:
        GateCircuit: _description_
    """

    operations = [("ham", (H, t / circuit_depth))] + [
        ("rz", "AB"),
        ("rx", "AB"),
        ("rz", "AB"),
        (ent_type, "AB"),
    ] * q

    return unitary_circuit_fac(qubit_layout, operations, repeats=circuit_depth)


def SHEA_fac(
    qubit_layout: QubitLayout,
    H: np.ndarray,
    t: float,
    circuit_depth: int,
    ent_type: str = "cnot",
) -> GateCircuit:
    """Starts with H block for `t`, them does HEA with `depth`.

    Args:
        qubit_layout (QubitLayout): _description_
        H (np.ndarray): _description_
        t (float): _description_
        depth (int): _description_

    Returns:
        GateCircuit: _description_
    """

    operations = [("ham", (H, t))] + [
        ("rz", "AB"),
        ("rx", "AB"),
        ("rz", "AB"),
        (ent_type, "AB"),
    ] * circuit_depth

    return unitary_circuit_fac(qubit_layout, operations)

def gate_circuit_fac(qubits, **kwargs):
    
    
    try:
        match kwargs['method']:
            case 'HEA':
                par_names = ['circuit_depth', 'ent_type']
                pars = {key: kwargs[key] for key in par_names}
                circuit = HEA_fac(qubits, **pars)
                
            case 'SHEA':
                par_names = ['circuit_depth', 'ent_type', 'H', 't']
                pars = {key: kwargs[key] for key in par_names}
                circuit = SHEA_fac(qubits, **pars)
                
            case 'SHEA':
                par_names = ['circuit_depth', 'ent_type', 'H', 't', 'q']
                pars = {key: kwargs[key] for key in par_names}
                circuit = SHEA_trot_fac(qubits, **pars)    
        
                
    except KeyError:
        print(f"Some of the required variables {par_names} are not given in circuit parameters.")
        raise
        
    return circuit


# =============================================================================
# class Circuit():
#     theta: np.ndarray
#     new_theta: bool
#     U_func: Callable[[np.ndarray], np.ndarray]
#     qubit_layout: QubitLayout
#     P: int
#     operations: list[tuple[str, str | np.ndarray]]
#     
#     def __init__(self, theta, U, qubit_layout, P, operations):
#         self._theta = 0
#         self._U = U
#         self.qubit_layout = qubit_layout
#         self.P = P
#         self.operations = operations
#         self.new_theta = True
#         
#     def __call__(self, theta: np.ndarray = None) -> np.ndarray:
#         if theta != None:
#             self.theta = theta
#         return self.U
#             
#         
#     @property
#     def theta(self):
#         return self._theta
#     
#     @theta.setter
#     def theta(self, theta):
#         self.new_theta = True
#         self._theta = theta
#     
#     @property
#     def U(self):
#         if self.new_theta:
#             self._U = self.U_func(self.theta)
#             self.new_theta = False
#             return self._U
#         else:
#             return self._U
#     
#     @U.setter
#     def U(self, theta):
#         self.theta = theta
#         self._U = self.U_func(self.theta)
# =============================================================================
