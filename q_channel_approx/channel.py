import numpy as np
from q_channel_approx.unitary_circuits import GateCircuit


def channel_fac(circuit: GateCircuit):
    """
    Generates a function phi that inputs 
        [theta] 
    and outputs
        a function approx_phi that inputs 
            [rho]
        and outputs
            Tr_b[U[theta] rho \tensor |0><0|_b U[theta]^\dagger]

    Parameters
    ----------
    circuit : GateCircuit
        The corresponding circuit, determines what theta can be input and the 
        relation between theta and the resulting unitary U[theta]

    Returns
    -------
    function

    """
    unitary, qubits = circuit.U, circuit.qubit_layout
    dims_A = qubits.dims_A
    dims_B = qubits.dims_B

    ancilla = np.zeros((dims_B, dims_B))
    ancilla[0, 0] = 1

    def phi(theta):

        U = unitary(theta)
        U_dag = np.transpose(U.conj())

        def approx_phi(rho):
            rho_AB = np.kron(rho, ancilla)
            rho_tensor = (U @ rho_AB @ U_dag).reshape(dims_A, dims_B, dims_A, dims_B)
            return np.trace(rho_tensor, axis1=1, axis2=3)

        return approx_phi

    return phi


def evolver_fac(circuit: GateCircuit, N: int):
    """
    Same as channel_fac, but the inner function outputs N consecutive applications
    of the quantum channel

    Parameters
    ----------
    circuit : GateCircuit
        The corresponding circuit, determines what theta can be input and the 
        relation between theta and the resulting unitary U[theta].
    N : int
        The required number of reapplications

    Returns
    -------
    function

    """

    dims_A = circuit.qubit_layout.dims_A

    phi_fac = channel_fac(circuit)

    def evolve_N_times_fac(theta: np.ndarray):

        phi_prime = phi_fac(theta=theta)

        def evolve_n_times(rho: np.ndarray):
            rho_acc = rho
            rhos = np.zeros((N + 1, dims_A, dims_A), dtype=np.complex128)
            rhos[0, :, :] = rho_acc
            for i in range(1, N + 1):
                rho_acc = phi_prime(rho_acc)
                rhos[i, :, :] = rho_acc

            return np.array(rhos)

        return evolve_n_times

    return evolve_N_times_fac
