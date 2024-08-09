# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 11:53:36 2023

@author: 20168723
"""


import json
import qutip as qt
import numpy as np

from q_channel_approx.unitary_circuits import circuit_fac
from q_channel_approx.qubit_layouts import qubitLayout_fac
from q_channel_approx.physics_defns.target_systems import DecaySystem
from q_channel_approx.training_data import ( 
    random_rho0s, 
    solve_lindblad_rho0s, 
    solve_lindblad_rho0, 
    mk_training_data, 
    measure_rhos
    )
from q_channel_approx.training_observables import ( 
    k_random_observables, 
    order_n_observables, 
    all_observables
    )
from q_channel_approx.optimizer import optimize, channel_fac

from q_channel_approx.physics_defns.initial_states import rho_rand_haar
from q_channel_approx.plotting.routines import plot_ess, compare_ess





#%% Setup circuit

# read circuit settings
circuit_file = "Input\\Circuit_pulse_v1.json"   
f = open(circuit_file)
circuit_pars = json.load(f)


qubits = qubitLayout_fac(**circuit_pars)
qubits.show_layout()

circuit = circuit_fac(qubits, **circuit_pars)

        
# =============================================================================
# m = circuit_pars["m"]
# layout = circuit_pars["layout"]
# cutoff = circuit_pars["cutoff"]
# distance = circuit_pars["distance"]
# observables = circuit_pars["observables"]  # K, L, N for total number of Tr[Q\rho]
# l = circuit_pars["n_rhos"]
# n = circuit_pars["n_depth"]
# seed = circuit_pars["seed"]
# =============================================================================

#set random seed
qt.rand_ket(N=2,seed = circuit_pars['seed'])
np.random.seed(circuit_pars['seed'])

#%% Setup target quantum channel parameters
channel_file="Input\\channel_v1.json"   
f = open(channel_file)
channel_pars = json.load(f)


ryd_interaction = channel_pars["ryd_interaction"]
omegas = channel_pars["omegas"]
gammas = channel_pars["gammas"]
m = channel_pars["m"]

l = circuit_pars["n_rhos"]
observables = circuit_pars["observables"]


system = DecaySystem(ryd_interaction=ryd_interaction, omegas=omegas, m=m, gammas=gammas)

rho0s = random_rho0s(m=m, L=l)
rhoss, ts = solve_lindblad_rho0s(rho0s=rho0s, delta_t=0.5, N=2, s=system)

match observables.split():
    case ["order", val]:
        Os = order_n_observables(m=m, n= int(val))
    case ["random", val]:
        Os = k_random_observables(m=m, k=int(val))
    case ["all"]:
        Os = all_observables(m=m)
        
training_data = mk_training_data(rhoss, Os)


#%% Initialize circuit




#%% Train

theta_opt, errors, thetas = optimize(circuit, training_data, max_count=100)


#%% Predict

rho0 = rho_rand_haar(1, 4)


def evolve_n_times(n: int, rho):
    rho_acc = rho
    rhos = [rho_acc]
    phi = channel_fac(circuit)(theta=theta_opt)
    for i in range(n):
        rho_acc = phi(rho=rho_acc)
        rhos.append(rho_acc)

    return np.array(rhos)


rhos = evolve_n_times(20, rho0)
#Os = create_readout_computational_basis(1)
ess = measure_rhos(rhos, Os)


rho_ref_s, ts = solve_lindblad_rho0(rho0, delta_t=0.5, N=20, s=system)
e_ref_ss = measure_rhos(rho_ref_s, Os)


compare_ess((ts, ess, "approx"), (ts, e_ref_ss, "ref"), labels=["1", "2", "3", "4"])
