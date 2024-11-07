# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 14:45:25 2024

@author: 20168723
"""

import numpy as np
import qutip as qt
from typing import Callable, NamedTuple

from q_channel_approx.qubit_layouts import QubitLayout
from q_channel_approx.gate_operations import (
    driving_H_fac,
    control_H_fac
)


class PulseCircuit(NamedTuple):
    U: Callable[[np.ndarray], np.ndarray]
    U_full: Callable[[np.ndarray], np.ndarray]
    qubit_layout: QubitLayout
    Zdt: int
    t_max: float
    operations: tuple[list[qt.Qobj],list[qt.Qobj]]

    def __repr__(self) -> str:
        return f"Circuit; qubits layout: \n {self.qubit_layout} \n segments: {self.Zdt} \n Pulse time: {self.t_max} \n Operations {self.operations}"

    def __call__(self, theta: np.ndarray) -> np.ndarray:
        return self.U(theta)

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
        t_max: float,
        U_comp = None
        ) -> PulseCircuit:
    
    m = qubit_layout.m + qubit_layout.n_ancilla
    
    control_H = control_H_fac(m, control_H)
    driving_H = driving_H_fac(m, driving_H, qubit_layout)
    operations = (control_H, driving_H)

    if U_comp is None:
        U_comp = np.eye(2**m)
    elif U_comp.shape[0] != 2**m:
        n = U_comp.shape[0].bit_length()-1
        U_comp = np.kron(U_comp, np.eye(2**(m-n)))
    
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
        argsc = theta
        #argsc = np.reshape(theta,[m,Zdt,2])
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

        return U_comp @ U[-1].full()
    
    def unitary_full(theta: np.array):
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
        argsc = theta
        #argsc = np.reshape(theta,[m,Zdt,2])
        options = qt.Options()
        functions=np.ndarray([m,2,],dtype=object)
        for k in range(m):
            functions[k,0]=Function(k+1,1,t_max,argsc[k,:,:])
            functions[k,1]=Function(k+1,-1,t_max,argsc[k,:,:])
        H=[driving_H]
        for k in range(m):
            H.append([control_H[k,0],functions[k,0].f_t])
            H.append([control_H[k,1],functions[k,1].f_t])
        
        U_list = qt.propagator(H, t = np.linspace(0, t_max, len(argsc[0,:,0])+1), 
                          options=options,args = {"_step_func_coeff": True})

        return np.array([U_comp @ U.full() for U in U_list])
    
    return PulseCircuit(unitary, unitary_full, qubit_layout, Zdt, t_max, operations)