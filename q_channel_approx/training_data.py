from dataclasses import dataclass

import numpy as np
import qutip as qt
import scipy as sc

from q_channel_approx.physics_defns.target_systems import TargetSystem
from q_channel_approx.physics_defns.hamiltonians import create_hamiltonian, create_jump_opers
from q_channel_approx.physics_defns.initial_states import rho_rand_haar
from q_channel_approx.training_observables import Observables


@dataclass
class TrainingData:
    """Training data is defined as a list of observales Os together with
    a list of initial states rho0s and a grid of expectation values Ess.
    The class automatically extracts some handy variables such as the dimensions of the
    underlying Hilbert space `dims`, and the indexing variables `K, L, N`.

    Args:
    -----
    Os (np.ndarray): "list of observables", but an observables
    is a matrix so this should be a 3D array indexed by `(k, a, b)`
    where `k` indexes the observable, and `a` and `b` are the row and column
    index of the matrix respectively.
    `[O_0, O_1, O_2, ..., O_K]`

    rho0s (np.ndarray): "matrix of states" each row gives the
    evolution of a particular initial state, but since a state is a density matrix
    this is a 4D array indexed by `(l, n, a, b)` where `l` indexes the initial state
    `n` indexes the time step and `a` and `b` respectively index the row and column
    of the density matrix.
    `       N ->`\n
    `L   [[rho00, rho01, ..., rho0N],`\n
    `|    [rho10, rho11, ..., rho1N],`\n
    `v     ...`\n
    `[rhoL0, rhoL1, ..., rhoLN]]`

    Ess (np.ndarray): "list of expectation values of each states with each observable"
    but since there are `L` initial states and `K` observables it is a list of matrices
    or a 3D array. The indexing convention is (l, k, n).
    """

    Os: Observables
    #Os_names: np.ndarray
    rho0s: np.ndarray
    Esss: np.ndarray

    def __post_init__(self):
        """Determine the indexing variables `N, K, L`,
        the dimension of the underlying Hilbert space.
        """
        K_Os = len(self.Os())
        self.dims_A, _ = self.Os()[0].shape
        self.N_, K_Esss, self.L = self.Esss.shape
        self.N = self.N_ - 1

        assert (
            K_Os == K_Esss
        ), f"Number of observables {K_Os} does not match number of expecation values {K_Esss}"

        self.K = K_Os
        self.m = self.dims_A.bit_length() - 1


def random_rho0s(m: int, L: int, seed: int = None) -> list[qt.Qobj]:
    """Generate a list of `L` initial states on `m` qubits.

    Args:
        m (int): number of qubits.
        L (int): number of initial states.
        seed (int, optional): used for the generation of L seed values
        which are passed to `rho_rand_haar`. Defaults to None.

    Returns:
        list[qt.Qobj]: list of `L` randomly generated initial states.
    """
    
    rho_rand_haar(m=m, seed=seed)
    
    rho0s = [rho_rand_haar(m=m) for _ in range(L)]

    return rho0s

def deterministic_rho0s(m: int, L: int, seed: int = None) -> list[qt.Qobj]:
    ket0 = qt.states.basis(2,0)
    ket1 = qt.states.basis(2,1)
    if m == 2:
        basis = [qt.tensor(ket0, ket0), qt.tensor(ket0, ket1), qt.tensor(ket1, ket0), qt.tensor(ket1, ket1)]
        superpositions = [(a+b)*(2)**(-1/2) for idx, a in enumerate(basis) for b in basis[idx + 1:]]
        rho0s = basis+superpositions
    else:
        print("No deterministic rho0s given for these settings, using random rhos")
        rho0s = random_rho0s(m,L,seed)
        
    if L > len(rho0s):
        print("Not enough deterministic rho0s set")
        raise NotImplementedError
        
    rho0s = [rho * rho.dag() for rho in rho0s]
    return rho0s

def solve_lindblad_rho0(
    rho0: qt.Qobj,
    delta_t: float,
    N: int,
    s: TargetSystem,
) -> tuple[np.ndarray, np.ndarray]:
    """Evolve a single initial state `rho0` for `N` timesteps of `delta_t` according the
    Lindblad equation with Hamiltonian defined by `s` and using
    jump operators `jump_opers`

    Args:
        rho0 (qt.Qobj): initial state
        delta_t (float): time step
        N (int): number of time steps
        s (TargetSystem): settings object used to create Hamiltonian
        jump_opers (list[qt.Qobj]): list of jump operators

    Returns:
        tuple[list[qt.Qobj], np.ndarray]: evolution of the initial state,
        list of timesteps at which the states are given
    """

    H = create_hamiltonian(s)
    jump_opers = create_jump_opers(s)

    ts = np.arange(N + 1) * delta_t

    rhoss = qt.mesolve(H=H, rho0=rho0, tlist=ts, c_ops=jump_opers).states

    return rhoss, ts


def solve_lindblad_rho0s(
    rho0s: list[qt.Qobj],
    delta_t: float,
    N: int,
    s: TargetSystem,
) -> tuple[np.ndarray, np.ndarray]:
    """Evolve all `rho0s` for `N` timesteps of `delta_t` according the
    Lindblad equation with Hamiltonian defined by `s` and using
    jump operators `jump_opers`

    Args:
        rho0s (list[qt.Qobj]): list of initial states
        delta_t (float): time step between states
        N (int): Number of evolutions of delta_t to make
        s (TargetSystem): settings object used to create the Hamiltonian
        jump_opers (list[qt.Qobj]): jump operators for the Lindbladian

    Returns:
        tuple[np.ndarray, np.ndarray]: L x N matrix
        where each entry is a density matrix itself, ts
        which is a list of time steps at which the states are given.

    """

    H = create_hamiltonian(s)
    jump_opers = create_jump_opers(s)

    L = len(rho0s)
    dims, _ = rho0s[0].shape

    ts = np.arange(N + 1) * delta_t

    rhoss = np.zeros((N + 1, L, dims, dims), dtype=np.complex128)

    for l in range(L):
        rhoss[:, l, :, :] = np.array(
            [
                state.full()
                for state in qt.mesolve(
                    H=H, rho0=rho0s[l], tlist=ts, c_ops=jump_opers
                ).states
            ]
        )

    return rhoss, ts


def measure_rhos(rhos: np.ndarray, Os: np.ndarray) -> np.ndarray:
    """Create a matrix of expectation values by measuring (i.e. trace of O rho)
    a list of density matrices with a list of observables.
    If there are `K` observables in `Os` and `N` states in `rhos`
    then the resulting matrix is of dimension `K` by `N`.

    Args:
        rhos (np.ndarray): think of it as a list of density matrices (length `N`).
        Os (list[np.ndarray]): think of it as a list of observables (length `K`).

    Returns:
        np.ndarray: matrix of expectation values of dimension `N` by `K`.
    """
    result = np.einsum("kab,nab -> nk", Os, rhos, dtype=np.float64, optimize="greedy")
    
    max_imag = np.max(np.imag(result))
    if max_imag >= 10**-4:
        print(f"Significant imaginary measurement component of value {max_imag} discarded")
    
    return np.real(result)


def measure_rhoss(rhoss: np.ndarray, Os: np.ndarray) -> np.ndarray:
    """Create a holor of expectation values by measuring (i.e. trace of O rho)
    a matrix of density matrices with a list of observables.
    If there are `K` observables in `Os` and `rhoss` is of dimension (`L`, `N`)
    then the resulting holor has dimension `L` by `K` by `N`.

    Args:
        rhoss (np.ndarray): think of it as a list of density matrices (dims `L` by `N`).
        Os (list[np.ndarray]): think of it as a list of observables (length `K`).

    Returns:
        np.ndarray: holor of expectation values (dimension (`N`, `K`, `L`)).
    """
    result = np.einsum("kab, nlba -> nkl", Os, rhoss, dtype=np.float64, optimize="greedy")
    
    max_imag = np.max(np.imag(result))
    if max_imag >= 10**-4:
        print(f"Significant imaginary measurement component of value {max_imag} discarded")
    
    return np.real(result)


def mk_training_data(rhoss: np.ndarray, Os: Observables) -> TrainingData:
    """Create training data object from a matrix of states where each row
    gives the evolution of its zeroth state and a list of observables.

    Args:
        rhoss (np.ndarray): matrix of states
        Os (list[qt.Qobj]): list of observables

    Returns:
        TrainingData: the corresponding TrainingData object
        which can be used to optimize a gate sequence.
    """

    rho0s = rhoss[0, :, :, :]
    Esss = measure_rhoss(rhoss, Os())

    return TrainingData(Os, rho0s, Esss)

def wasserstein1(rho1, rho2, pauli_tuple):
    """
    Maximizes SUM_j c_j w_j over w_j with 
    SUM_{j s.t. P_j acts as X,Y,Z on qubit i} |w_j|<=1   : for i =1,...,n
    and c_j = Tr[(rho1-rho2) P_j]
    with P_j pauli spin matrices given in pauli_tuple

    Parameters
    ----------
    rho1 : Qobj, matrix
        Density matrix 1.
    rho2 : Qobj, matrix
        Density matrix 2.
    pauli_tuple : (np.array, np.array)
        tuple with pauli spin matrices as np.arrays, and list of qubits it acts on.

    Returns
    -------
    max_expectation : float
        maximal expectation that was found.
    weights : np.array
        weights that maximize the expectation

    """
    paulis, id_qubit_list = pauli_tuple
    max_expectation = 0
    
    num_pauli = len(paulis)
    num_bit = id_qubit_list.shape[0]
    traces = np.zeros([num_pauli])
    for i in range(num_pauli):
        traces[i] = np.real((paulis[i]*(rho1-rho2)).trace())
    
    # Include the absolute value by setting -w'_j <= w_j <= w'_j
    # This gives |w_j| = w'_j. w'_j in [0,1], w_j in [-1,1]
    # parameter list is first all w'_j, then all w_j 
    # So #pars = 2 * #pauli-matrices 
    # And #constraints = m + #pars
    
    # Minimize MINUS w_j * P_j (rho1 - rho2) (is max over plus)
    obj = np.append(np.zeros(num_pauli),-traces)
    lhs_ineq = np.zeros([num_bit+2*num_pauli, 2*num_pauli])
    lhs_ineq[0:num_bit,0:num_pauli] = id_qubit_list
    for i in range(num_pauli):
        
        # -w'_i -w_i <= 0
        lhs_ineq[num_bit + 2*i,     i] = -1 
        lhs_ineq[num_bit + 2*i,     num_pauli +i] = -1
        
        # -w'_i +w_i <=0
        lhs_ineq[num_bit + 2*i +1,  i] = -1
        lhs_ineq[num_bit + 2*i +1,  num_pauli +i] = 1
        
    rhs_ineq = np.append(np.ones(num_bit)/2, np.zeros(2*num_pauli))
    bnd = [(0,0.5) for _ in range(num_pauli)]
    bnd = bnd + [(-0.5,0.5) for _ in range(num_pauli)]
    
    opt = sc.optimize.linprog(c=obj, A_ub=lhs_ineq, b_ub=rhs_ineq, bounds=bnd)
    
    if opt.success:
        max_expectation = -opt.fun
        x = opt.x
        weights = x[num_pauli:] #get non-absolute values.
        #p_print(opt.x)
        #p_print(max_expectation)
    else:
        print("Wasserstein optimum finding did not converge")
        max_expectation = 0
        weights = np.zeros(num_pauli)
        
    return max_expectation, weights

def wasserstein_all(meas1, meas2, id_qubit_list):
    """
    Maximizes SUM_j c_j w_j over w_j with 
    SUM_{j s.t. P_j acts as X,Y,Z on qubit i} |w_j|<=1   : for i =1,...,n
    and c_j = Tr[(rho1-rho2) P_j]
    with P_j pauli spin matrices given in pauli_tuple

    Parameters
    ----------
    meas1 : np.array
        matrix of measurements (# observables x # rho's')
    rho2 : np.array
        matrix of measurements (# observables x # rho's')
    pauli_tuple : (np.array, np.array)
        tuple list of qubits each pauli matrix acts on. 
        (in same order as # observables in the measurements)

    Returns
    -------
    max_expectation : float
        maximal expectation that was found.
    weights : np.array
        weights that maximize the expectation

    """
    id_qubit_list = np.transpose(id_qubit_list)
    
    max_expectation = 0
    
    num_bit, num_pauli = id_qubit_list.shape
    traces = meas1 - meas2
    
    # Include the absolute value by setting -w'_j <= w_j <= w'_j
    # This gives |w_j| = w'_j. w'_j in [0,1], w_j in [-1,1]
    # parameter list is first all w'_j, then all w_j 
    # So #pars = 2 * #pauli-matrices 
    # And #constraints = m + #pars
    
    # Minimize MINUS w_j * P_j (rho1 - rho2) (is max over plus)
    obj = np.append(np.zeros(num_pauli),-traces)
    lhs_ineq = np.zeros([num_bit+2*num_pauli, 2*num_pauli])
    lhs_ineq[0:num_bit,0:num_pauli] = id_qubit_list
    for i in range(num_pauli):
        
        # -w'_i -w_i <= 0
        lhs_ineq[num_bit + 2*i,     i] = -1 
        lhs_ineq[num_bit + 2*i,     num_pauli +i] = -1
        
        # -w'_i +w_i <=0
        lhs_ineq[num_bit + 2*i +1,  i] = -1
        lhs_ineq[num_bit + 2*i +1,  num_pauli +i] = 1
        
    rhs_ineq = np.append(np.ones(num_bit)/2, np.zeros(2*num_pauli))
    bnd = [(0,0.5) for _ in range(num_pauli)]
    bnd = bnd + [(-0.5,0.5) for _ in range(num_pauli)]
    
    opt = sc.optimize.linprog(c=obj, A_ub=lhs_ineq, b_ub=rhs_ineq, bounds=bnd)
    
    if opt.success:
        max_expectation = -opt.fun
        x = opt.x
        weights = x[num_pauli:] #get non-absolute values.
        #p_print(opt.x)
        #p_print(max_expectation)
    else:
        print("Wasserstein optimum finding did not converge")
        max_expectation = 0
        weights = np.zeros(num_pauli)
        
    return max_expectation, weights
