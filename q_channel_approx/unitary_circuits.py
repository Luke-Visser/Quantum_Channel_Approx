import itertools
from operator import add
from typing import Callable, NamedTuple

import numpy as np
import scipy as sc
import qutip as qt

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
    driving_H_fac,
    control_H_fac
)


class Circuit(NamedTuple):
    U: Callable[[np.ndarray], np.ndarray]
    Ugrad: Callable[[np.ndarray], np.ndarray]
    qubit_layout: QubitLayout
    P: int
    operations: list[tuple[str, str | np.ndarray]]

    def __repr__(self) -> str:
        return f"Circuit; qubits layout: \n {self.qubit_layout} \n Parameters: {self.P} \n Operations {self.operations}"

    def __call__(self, theta: np.ndarray) -> np.ndarray:
        return self.U(theta)
    


def count_qubits(dims: int) -> int:
    return dims.bit_length() - 1

class Function:
    def __init__(self, qubit, sign, endtime, values):
        """
        Initialize the function class

        Parameters
        ----------
        qubit : int
            gives the control number.
        sign : -1,1
            sign of the function value.
        endtime : float
            total evolution time.
        values : np..ndarray, Zdt
            values for the new pulse

        Returns
        -------
        None.

        """
        self.qubit = qubit
        self.sign = sign
        self.values = values
        self.T = endtime

    def update(self, values):
        """
        Update the function parameters

        Parameters
        ----------
        values : np..ndarray, Zdt
            values for the new pulse

        Returns
        -------
        None.

        """
        self.values = values

    def f_t(self, t,args):
        """
        Returns the complex function value at the specified time

        Parameters
        ----------
        t : float
            specified time
        args : np.ndarray, Zdt x2
            functions values

        Returns
        -------
        float complex
            complex function value

        """
        index = int((t % self.T) // (self.T / len(self.values)))
        if index>(len(self.values)-1):
            index=index-1
        return self.values[index,0]+self.sign*1j*self.values[index,1]

def unitary_pulsed_fac(
        qubit_layout: QubitLayout, 
        control_H: str, 
        driving_H: str, 
        Zdt: int, 
        t_max: float
        ) -> Circuit:
    
    m = qubit_layout.m + qubit_layout.n_ancilla
    control_H = control_H_fac(m, control_H)
    driving_H = driving_H_fac(m, driving_H, qubit_layout)
    
    m = len(control_H[0,0].dims[0])
    
    def unitary(theta: np.array):
        """
        Determines the propagator as in the Fréchet derivatives

        Parameters
        ----------
        argsc : np.ndarray complex, num_control x Zdt x 2
            Describes the (complex) pulse parameters of the system throughout the process
        T : float
            Total evolution time.
        control_H : np.ndarray Qobj 2**m x 2**m, num_control x 2 
            array of Qobj describing the control operators
        driving_H : Qobj 2**m x 2**m 
            Hamiltonian describing the drift of the system

        Returns
        -------
        U : np.ndarray Qobjs, Zdt
             Array describing unitary evolution at each timestep
        """
        argsc = np.reshape(theta,[m,Zdt,2])
        options = qt.Options()
        functions=np.ndarray([m,2,],dtype=object)
        for k in range(m):
            functions[k,0]=Function(k+1,1,t_max,argsc[k,:,:])
            functions[k,1]=Function(k+1,-1,t_max,argsc[k,:,:])
        H=[driving_H]
        for k in range(m):
            H.append([control_H[k,0],functions[k,0].f_t])
            H.append([control_H[k,1],functions[k,1].f_t])
        
        U = qt.propagator(H, t = np.linspace(0, t_max, len(argsc[0,:,0])+1), 
                          options=options,args = {"_step_func_coeff": True})

        return U[-1].full()
    
    P = m*Zdt*2
    operations = 0
    return Circuit(unitary, qubit_layout, P, operations)


def unitary_circuit_fac(
    qubit_layout: QubitLayout, operations, repeats: int = 1
) -> Circuit:

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

        return np.linalg.matrix_power(U, repeats)

    return Circuit(unitary, qubit_layout, P, operations)


def HEA_fac(qubit_layout: QubitLayout, circuit_depth: int, ent_type: str = "cnot") -> Circuit:
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
) -> Circuit:
    """Trotterized H, does a small H block for time `t` followed by one HEA cycle (ZXZ, ent)
    This sequence is repeated `depth` times.

    Args:
        qubit_layout (QubitLayout): _description_
        H (np.ndarray): _description_
        t (float): _description_
        depth (int): _description_

    Returns:
        Circuit: _description_
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
) -> Circuit:
    """Starts with H block for `t`, them does HEA with `depth`.

    Args:
        qubit_layout (QubitLayout): _description_
        H (np.ndarray): _description_
        t (float): _description_
        depth (int): _description_

    Returns:
        Circuit: _description_
    """

    operations = [("ham", (H, t))] + [
        ("rz", "AB"),
        ("rx", "AB"),
        ("rz", "AB"),
        (ent_type, "AB"),
    ] * circuit_depth

    return unitary_circuit_fac(qubit_layout, operations)

def circuit_fac(qubits, **kwargs):
    
    
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
            
            case 'pulse':
                par_names = ['control_H', 'driving_H', 'Zdt', 't_max']
                pars = {key: kwargs[key] for key in par_names}
                circuit = unitary_pulsed_fac(qubits, **pars)
                
    except KeyError:
        print(f"keys {par_names} not given in circuit parameters.")
        raise
        
    return circuit
