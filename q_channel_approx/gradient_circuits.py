# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 11:28:30 2024

@author: 20168723
"""

import numpy as np
import math
import qutip as qt

from typing import Callable, NamedTuple


from q_channel_approx.unitary_circuits_pulse import PulseCircuit
from q_channel_approx.qubit_layouts import QubitLayout
from q_channel_approx.training_data import TrainingData, measure_rhoss

class GradCircuit(NamedTuple):
    U: Callable[[np.ndarray], np.ndarray]
    U_full: Callable[[np.ndarray], np.ndarray]
    qubit_layout: QubitLayout
    Zdt: int
    t_max: float
    operations: tuple[list[qt.Qobj],list[qt.Qobj]]
    loss: list[Callable[[np.ndarray], np.ndarray]]
    grad: list[Callable[[np.ndarray], np.ndarray]]
    
    
    def __repr__(self) -> str:
        self.circuit.__repr__()
        
        
def evolve_rho(U: np.ndarray, training_data : TrainingData):
    
    N = training_data.N
    L = training_data.L
    dims_a = training_data.dims_A
    dims_b = U.shape[0]//dims_a
    
    rho_zero = training_data.rho0s
    rho_all = np.zeros([N+1, L, dims_a, dims_a], dtype = complex)
    rho_all[0] = rho_zero
    
    ancilla = np.zeros((dims_b, dims_b))
    ancilla[0, 0] = 1
    
    Udag = np.transpose(U.conj())
    
    for i in range(N):
        rho_ab = np.kron(rho_all[i], ancilla)
        UrhoU = np.einsum('ij, ajk, kl-> ail', U, rho_ab, Udag)
        rho = np.trace(UrhoU.reshape(L, dims_a, dims_b, dims_a, dims_b),axis1 = 2, axis2=4)
        rho_all[i+1] = rho
        
    return rho_all

def pulse_loss_fac(circuit: PulseCircuit, training_data, lambdapar):
    
    num_controls = circuit.operations[0].shape[0]
    Zdt = circuit.Zdt
    t_max = circuit.t_max
    
    def Znorm(Z,T):
        """
        Determines the L2 norm of the Z pulse

        Parameters
        ----------
        Z : np.ndarray complex, num_control x Zdt x 2
            Describes the (complex) pulse parameters of the system
        T : float
            Total evolution time.

        Returns
        -------
        float
            L2 norm of Z.

        """
        norm=0
        for t in range(len(Z[0,:,0])):
            norm += np.linalg.norm(Z[:,t,:], 'fro')**2
        return math.sqrt(1/len(Z[0,:,0])*norm*T)
    
    def loss(theta):
        return np.real(np.sum(lambdapar*Znorm(np.reshape(theta, (num_controls, Zdt, 2)), t_max)**2))
    
    def grad(theta):
        return lambdapar*theta
    
    return (loss, grad)

def pauli_loss_fac(circuit: PulseCircuit, training_data: TrainingData):
    """
    Creates loss function and gradient function for loss function
    Tr[sigma(rho_1 - rho_2)]^2

    Parameters
    ----------
    circuit : Circuit
        DESCRIPTION.
    training_data : TrainingData

    Returns
    -------
    loss : TYPE
        DESCRIPTION.
    grad : TYPE
        DESCRIPTION.

    """
    
    K = training_data.K # Observables
    L = training_data.L # Init states rho
    N = training_data.N # Repetitions (old num_t)
    m = training_data.m
    Zdt = circuit.Zdt # nr piecewise segments of pulse
    control_H = circuit.operations[0]
    num_controls = control_H.shape[0]
    
    dims_a = circuit.qubit_layout.dims_A
    dims_b = circuit.qubit_layout.dims_B
    dims_ab = circuit.qubit_layout.dims_AB
    
    state_00_B = np.zeros([dims_b, dims_b])
    state_00_B[0,0] = 1
    state_I00 = np.kron(np.eye(dims_a), state_00_B)

    
    
    def loss(theta):
        U = circuit(theta)
        rhos = evolve_rho(U, training_data)[1:]
        meas_rhos = measure_rhoss(rhos, training_data.Os)
        
        error = np.sum((meas_rhos - training_data.Esss[1:])**2)/(K*L*N)

        return np.real(error)
    
    def grad(theta):
        
        U_full = circuit.U_full(theta)
        U = U_full[-1]
        Udag = (np.transpose(U.conj()))
        
        rhos = evolve_rho(U, training_data)
        meas_rhos = measure_rhoss(rhos, training_data.Os)
        
        gradient=np.zeros([num_controls, len(theta[0,:,0]),2])
        
        eta_T = np.zeros([dims_ab, dims_ab], dtype = np.complex128)
        
        Os_U = np.zeros((N, K, dims_a, dims_a), dtype = np.csingle)
        Os_U[0] = training_data.Os
        for i in range(1,N):
            Os_temp = np.kron(Os_U[i-1], np.eye(dims_b))
            Os_temp = np.einsum('ij, ajk, kl, lm -> aim', Udag, Os_temp, U, state_I00, optimize='greedy')
            Os_temp = np.trace(Os_temp.reshape(K, dims_a, dims_b, dims_a, dims_b), axis1=2, axis2 = 4)
            Os_U[i] = Os_temp
            
        traces = meas_rhos - training_data.Esss
        
        # Calculate the product of trace * partial _ delta U matrix for all combinations
        Os_Uext = np.kron(Os_U, np.eye(dims_b))
        rhos_ext = np.kron(rhos, state_00_B)
        for n1 in range(1, N+1): #repetitions on U
            for n2 in range(N+1-n1): #index for rho_n
                matrices = np.einsum('kab, bc, lcd -> klad', Os_Uext[n1-1], U, rhos_ext[n2], optimize = 'greedy')
                #matrices = np.einsum('aij, jk, bkl -> abil', paulis_Uext[n1-1], U, rhos_approx_ext[n2], optimize = 'greedy')
                eta_T += np.einsum('kl, klij -> ij', traces[n1+n2], matrices, optimize= 'greedy' )
        
        eta_T = Udag @ eta_T/ (L* N* K)
    
        # Set the actual gradient based on eta
        eta_T_dag = np.conjugate(np.transpose(eta_T))
        for t in range(Zdt):
            for k in range(0,num_controls):
                updatepart = np.trace(-2j *control_H[k,1].full() @ (U_full[t] @ (eta_T - eta_T_dag) @np.transpose(U_full[t].conj()) ) )
                gradient[k,t,0] = - np.real(updatepart) #*(0.8+0.4*np.random.rand())
                gradient[k,t,1] = - np.imag(updatepart) #*(0.8+0.4*np.random.rand())
        
        return gradient
        
    return (loss, grad)

def gradCircuit_fac(
        circuit: PulseCircuit,
        training_data: TrainingData,
        loss_type: str,
        grad_type: str,
        lambdapar: float
        ):
    
    loss_l2, grad_l2 = pulse_loss_fac(circuit, training_data, lambdapar)
    
    match loss_type:
        case "paulis":
            loss, grad = pauli_loss_fac(circuit, training_data)
            return GradCircuit(**circuit._asdict(), loss=(loss, loss_l2), grad=(grad, grad_l2))
            #return GradCircuit(**circuit._asdict(), loss=(loss,), grad=(grad,))

        
        
        
if __name__ == "__main__":
    pass
            
    