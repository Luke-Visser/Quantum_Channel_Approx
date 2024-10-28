"""
Some function to create observables.
"""

import qutip as qt
import numpy as np
from typing import Callable, NamedTuple


from q_channel_approx.pauli_strings import (
    k_random_pauli_strs,
    order_n_pauli_strs,
    all_pauli_strs,
    pauli_strs_2_ops,
    pauli_str_2_nonzero
)

class Observables(NamedTuple):
    Os: np.ndarray
    Os_names: np.ndarray
    Os_ids: int

    def __repr__(self) -> str:
        return f"Observables, containing pauli matrices {self.Os_names}."

    def __call__(self) -> np.ndarray:
        return self.Os


def k_random_observables(m: int, k: int, seed: int) -> list[qt.Qobj]:
    """Generate `k` random observables on `m` qubits.

    Args:
        m (int): _description_
        k (int): number of observables
        seed (int): seed used to generate the Pauli strings

    Returns:
        list[qt.Qobj]: list of the observables
    """

    pauli_strs = k_random_pauli_strs(m=m, k=k, seed=seed)
    return pauli_strs_2_ops(pauli_strs), pauli_strs, pauli_str_2_nonzero(pauli_strs)


def order_n_observables(m: int, n: int) -> list[qt.Qobj]:
    """Generate all observables on `m` qubits upto order `n`.

    Args:
        m (int): number of qubits.
        n (int): highest order Pauli strings included.

    Returns:
        list[qt.Qobj]: list of the observables.
    """

    pauli_strs = order_n_pauli_strs(m=m, n=n)
    return pauli_strs_2_ops(pauli_strs), pauli_strs, pauli_str_2_nonzero(pauli_strs)


def all_observables(m: int) -> list[qt.Qobj]:
    """All observables on `m` qubits.

    Args:
        m (int): number of qubits.

    Returns:
        list[qt.Qobj]: list of all observables.
    """

    pauli_strs = all_pauli_strs(m=m)
    return pauli_strs_2_ops(pauli_strs), pauli_strs, pauli_str_2_nonzero(pauli_strs)

def observables_fac(name, m):
    
    match name.split():
        case ["order", val]:
            Os, names, ids = order_n_observables(m=m, n= int(val))
        case ["random", val]:
            Os, names, ids = k_random_observables(m=m, k=int(val))
        case ["all"]:
            Os, names, ids = all_observables(m=m)
    return Observables(Os, names, ids)
