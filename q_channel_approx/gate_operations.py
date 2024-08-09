import numpy as np
import qutip as qt
import scipy as sc

from q_channel_approx.qubit_layouts import QubitLayout


def kron_gates_l(single_gates):
    result = single_gates[0]
    for gate in single_gates[1:]:
        result = np.kron(result, gate)

    return result


def kron_neighbours_even(single_gates):

    l, dims, _ = single_gates.shape
    double_gates = np.zeros((l // 2, dims**2, dims**2), dtype=np.complex128)

    for i in range(0, l // 2):
        double_gates[i, :, :] = np.kron(single_gates[i * 2], single_gates[i * 2 + 1])

    return double_gates


def kron_gates_r(single_gates):
    """Recursively multiply the neighbouring gates.
    When the block size gets below the turnover point the linear
    kron_gates_l is used as it is more efficient in this usecase."""
    TURNOVER = 3

    l = len(single_gates)

    if l > TURNOVER:
        if l % 2 == 0:
            return kron_gates_r(kron_neighbours_even(single_gates))
        return np.kron(
            kron_gates_r(kron_neighbours_even(single_gates[:-1])),
            single_gates[-1],
        )

    return kron_gates_l(np.array(single_gates))


def rz(theta):
    zero = np.zeros(theta.shape)
    exp_m_theta = np.exp(-1j * theta / 2)
    exp_theta = np.exp(1j * theta / 2)

    single_gates = np.einsum(
        "ijk->kij", np.array([[exp_m_theta, zero], [zero, exp_theta]])
    )

    u_gates = kron_gates_l(single_gates)

    return u_gates


def rx(theta):
    costheta = np.cos(theta / 2)
    sintheta = np.sin(theta / 2)

    single_gates = np.einsum(
        "ijk->kij", np.array([[costheta, -sintheta], [sintheta, costheta]])
    )

    u_gates = kron_gates_l(single_gates)

    return u_gates


def H_fac(H, dims_AB):

    H, t = H

    if isinstance(H, qt.Qobj):
        H = H.full()

    dims, _ = H.shape
    dims_expand = dims_AB // dims

    def U(
        foo,
    ):  # needs a throwaway argument because we are going to pass an empty array in the unitary_fac
        e_H = sc.linalg.expm((-1j) * t * H)
        e_H_exp = np.kron(e_H, np.identity(dims_expand))

        return e_H_exp

    return U

def driving_H_fac(m, interaction, qubits: QubitLayout):
    """
    Creates the driving Hamiltonian describing the drift of the system

    Parameters
    ----------
    m : int
        number of qubits.
    type : string
        selection of drive Hamiltonian.

    Returns
    -------
    Hamiltonian : QObj, 2**m x 2**m
          Hamiltonian describing the natural drift of the system

    """
    
    Hamiltonian = qt.Qobj(dims=[[2] * m, [2] * m])
    project1111 = qt.Qobj(np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]]), dims = [[2]*2,[2]*2])
    project0110_1001 = qt.Qobj(np.array([[0,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,0]]), dims = [[2,2],[2,2]])
    
    pairs_distance = qubits.find_qubit_distances()
    
    print(m)
    for (i, j, d) in pairs_distance:
        print(f"index i {i}, j {j}, with distance {d}")
        
    if interaction=='basic11':
        for (i, j, d) in pairs_distance:
            if d <=1:
                Hamiltonian=Hamiltonian +qt.qip.operations.gates.expand_operator(project1111,m,[i,j])
        return Hamiltonian
    
    elif interaction=='rydberg11':
        for (i, j, d) in pairs_distance:
            Hamiltonian += d**(-3) *qt.qip.operations.gates.expand_operator(project1111,m,[i,j])         
        return Hamiltonian
    
    elif interaction == 'dipole0110':
        for (i, j, d) in pairs_distance:
            Hamiltonian += d**(-3/2) *qt.qip.operations.gates.expand_operator(project0110_1001,m,[i,j])
        return Hamiltonian
    
    else:
        raise ValueError(interaction +' is not a specified driving Hamiltonian interaction')

    

def control_H_fac(m,type_h):
    """
    Creates the control Hamiltonian operators

    Parameters
    ----------
    m : int
        number of qubits.
    type_h : string
        describes the type of control Hamiltonian.

    Returns
    -------
    Hamiltonians : np.ndarray Qobj's, num_controls x 2
        array of control Hamiltonians Ql to be influenced

    """
    if type_h=='rotations':
        Hamiltonians=np.ndarray([m,2,],dtype=object)
        project10op = qt.Qobj(np.array([[0,0],[1,0]]),dims = [[2],[2]])
        project01op = qt.Qobj(np.array([[0,1],[0,0]]),dims = [[2],[2]])
        for k in range(m):
            Hamiltonians[k,0]=qt.qip.operations.gates.expand_operator(project10op, m, k)
            Hamiltonians[k,1]=qt.qip.operations.gates.expand_operator(project01op, m, k)
        return Hamiltonians
    elif type_h=='realrotations':
        Hamiltonians=np.ndarray([m,2,],dtype=object)
        project10op = qt.Qobj(np.array([[0,0],[1,0]]),dims = [[2],[2]])
        project01op = qt.Qobj(np.array([[0,1],[0,0]]),dims = [[2],[2]])
        for k in range(m):
            gate = qt.qip.operations.gates.expand_operator(project01op, m, k)
            gate += qt.qip.operations.gates.expand_operator(project10op, m, k)
            Hamiltonians[k,0]=gate
            Hamiltonians[k,1]=gate
        return Hamiltonians
    
    elif type_h=='rotations+11':
        Hamiltonians=np.ndarray([2*m,2,],dtype=object)
        project10op = qt.Qobj(np.array([[0,0],[1,0]]),dims = [[2],[2]])
        project01op = qt.Qobj(np.array([[0,1],[0,0]]),dims = [[2],[2]])
        project11op = qt.Qobj(np.array([[0,0],[0,1]]), dims = [[2],[2]])
        for k in range(m):
            Hamiltonians[k,0]=qt.qip.operations.gates.expand_operator(project10op, m, k)
            Hamiltonians[k,1]=qt.qip.operations.gates.expand_operator(project01op, m, k)
            Hamiltonians[m+k,0]=qt.qip.operations.gates.expand_operator(project11op, m, k)
            Hamiltonians[m+k,1]=qt.qip.operations.gates.expand_operator(project11op, m, k)
        return Hamiltonians
    
    elif type_h=='realrotations+11':
        Hamiltonians=np.ndarray([2*m,2,],dtype=object)
        project10op = qt.Qobj(np.array([[0,0],[1,0]]),dims = [[2],[2]])
        project01op = qt.Qobj(np.array([[0,1],[0,0]]),dims = [[2],[2]])
        project11op = qt.Qobj(np.array([[0,0],[0,1]]), dims = [[2],[2]])
        for k in range(m):
            gate = qt.qip.operations.gates.expand_operator(project01op, m, k)
            gate += qt.qip.operations.gates.expand_operator(project10op, m, k)
            Hamiltonians[k,0]=gate
            Hamiltonians[k,1]=gate
            Hamiltonians[m+k,0]=qt.qip.operations.gates.expand_operator(project11op, m, k)
            Hamiltonians[m+k,1]=qt.qip.operations.gates.expand_operator(project11op, m, k)
        return Hamiltonians
    
    else:
        raise ValueError(type_h+' is not a specified way of creating control Hamiltonians.')


def ryd_dipole_fac(connections, dims_AB):

    rydberg = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    n_qubits = dims_AB.bit_length() - 1

    rydberg_2gate = qt.Qobj(rydberg, dims=[[2] * 2, [2] * 2])
    rydberg_gate = np.zeros([dims_AB, dims_AB], dtype=np.complex128)
    for connection in connections:

        id1, id2, d = connection
        ham = qt.expand_operator(
            oper=rydberg_2gate, N= n_qubits, dims=[2] * n_qubits, targets=[id1, id2]
        ).full()
        rydberg_gate += ham / d**3  # distance to the power -6

    def ryd_ent(theta):
        return sc.linalg.expm(-1j * theta * rydberg_gate)

    return ryd_ent


def ryd_vdw_fac(connections, dims_AB):

    rydberg = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
        ]
    )
    n_qubits = dims_AB.bit_length() - 1

    rydberg_2gate = qt.Qobj(rydberg, dims=[[2] * 2, [2] * 2])
    rydberg_gate = np.zeros([dims_AB, dims_AB], dtype=np.complex128)
    for connection in connections:

        id1, id2, d = connection
        ham = qt.expand_operator(
            oper=rydberg_2gate, N= n_qubits, dims=[2] * n_qubits, targets=[id1, id2]
        ).full()
        rydberg_gate += ham / d**3  # distance to the power -6

    def ryd_ent(theta):
        return sc.linalg.expm(-1j * theta * rydberg_gate)

    return ryd_ent


def CNOT_fac(connections, dims_AB):
    H_CNOT = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ]
    )
    n_qubits = dims_AB.bit_length() - 1

    CNOT_gate = qt.Qobj(H_CNOT, dims=[[2] * 2, [2] * 2])
    gates = np.identity(dims_AB, dtype=np.complex128)
    for connection in connections:
        id1, id2, d = connection
        gate = qt.expand_operator(
            oper=CNOT_gate, N= n_qubits, dims=[2] * n_qubits, targets=[id1, id2]
        ).full()
        gates = gate @ gates

    def CNOT(foo):
        return gates

    return CNOT


def xy_fac(connections, dims_AB):
    rydberg = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    n_qubits = dims_AB.bit_length() - 1
    xy_gate = qt.Qobj(rydberg, dims=[[2] * 2, [2] * 2])


    def ryd_ent(theta):
        gates = np.identity(dims_AB, dtype=np.complex128)
        for i, connection in enumerate(connections):

            id1, id2, d = connection
            ham = qt.expand_operator(
                oper=xy_gate, N= n_qubits, dims=[2] * n_qubits, targets=[id1, id2]
            ).full()
            gates += ham / d**3  # distance to the power -6

            gates = sc.linalg.expm(-1j * theta[i] * gates) @ gates
    
        return gates

    return ryd_ent


def matmul_acc_ul(Us: np.ndarray) -> np.ndarray:

    w, dims, _ = Us.shape

    U_lower = np.zeros((w, dims, dims), dtype=np.complex128)
    U_upper = np.zeros((w, dims, dims), dtype=np.complex128)

    U_l_acc = np.identity(dims)
    U_u_acc = np.identity(dims)

    for i, U in enumerate(Us):
        U_l_acc = U_l_acc @ U
        U_lower[i, :, :] = U_l_acc

    for i, U in enumerate(Us[::-1]):
        U_u_acc = U @ U_u_acc
        U_upper[-i - 1, :, :] = U_u_acc

    return U_lower, Us, U_upper


def matmul_acc(Us: np.ndarray) -> np.ndarray:
    Ul, Us, Uu = matmul_acc(Us)
    U = Ul[-1]
    return U


def matmul_l(Us: np.ndarray) -> np.ndarray:
    U_acc = Us[0]

    for U in Us[1:]:
        U_acc = U @ U_acc

    return U_acc
