# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 11:53:36 2023

@author: 20168723
"""


import json
import qutip as qt
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from datetime import datetime
import os

from q_channel_approx.qubit_layouts import qubitLayout_fac
from q_channel_approx.unitary_circuits import gate_circuit_fac
from q_channel_approx.unitary_circuits_pulse import unitary_pulsed_fac
from q_channel_approx.gradient_circuits import gradCircuit_fac

from q_channel_approx.physics_defns.target_systems import target_system_fac
from q_channel_approx.training_data import ( 
    random_rho0s,
    deterministic_rho0s,
    solve_lindblad_rho0s, 
    solve_lindblad_rho0, 
    mk_training_data, 
    measure_rhos,
    measure_rhoss
    )
from q_channel_approx.training_observables import observables_fac
from q_channel_approx.optimizer import optimize_pulse, channel_fac

from q_channel_approx.physics_defns.initial_states import rho_rand_haar
from q_channel_approx.plotting.routines import ( 
    compare_ess, 
    error_evolution,
    plot_pulses,
    save_figs,
    save_data,
    load_data,
    fancy_fig_1,
    fancy_fig_2)
from q_channel_approx.plotting.observables import create_observables_comp_basis


timestamp = "{:%Y-%m-%d_time_%H-%M-%S}".format(datetime.now()) # Filenames for saved figures. None for default timestamped
file_dir = os.getcwd()

#%% Files and parameters

save_results = False

#%% Load data

path1 = 'C:\\Users\\20168723\\OneDrive - TU Eindhoven\\TUe PhD\\Code\\Quantum_Channel_Approx_V2\\Code_examples\\Results'
#folder = "4level_long_2024-09-30_time_11-23-20"
folder = "1QubitDecayPM_long_2024-09-30_time_11-50-29"
#folder = "4level_test_2024-10-08_time_10-41-36"
#folder = "4level_basisrho0_reverse_2024-10-08_time_13-31-17"
#folder = "4level_basisrho0_2024-10-08_time_10-41-36"
path_loading = os.path.join(path1, folder, 'simulation_results.xlsx')
all_pars, theta_opt, errors = load_data(path_loading)
channel_pars = all_pars
circuit_pars = all_pars

results_path = os.path.join(path1, folder)


#%% Setup circuit


qubits = qubitLayout_fac(**circuit_pars)
m = qubits.m
qubits.show_layout()

circuit = unitary_pulsed_fac(qubits, circuit_pars["control_H"], circuit_pars["driving_H"], 
                             circuit_pars["Zdt"], circuit_pars["t_max"])


#set random seed
qt.rand_ket(N=2,seed = circuit_pars['seed'])

seed = 2
np.random.seed(seed)
#np.random.seed(circuit_pars['seed'])

#%% Setup target quantum channel parameters and observables

# Quantum channel
system = target_system_fac(**channel_pars)

# Generate and evolve density matrices
#rho0s = random_rho0s(m=m, L=channel_pars['n_rhos'])
rho0s = deterministic_rho0s(m=m, L=channel_pars['n_rhos'])
rhoss, ts = solve_lindblad_rho0s(rho0s=rho0s, delta_t=channel_pars['delta_t'], N=circuit_pars['n_depth'], s=system)

# Gather training data
Os = observables_fac(channel_pars["observables"], m)
training_data = mk_training_data(rhoss, Os)

# Determine circuit with loss and gradient function
gradcircuit = gradCircuit_fac(circuit, training_data, circuit_pars["loss_type"], "all", circuit_pars['lambdapar'])
    


#%% Predict

# nice seeds for 4level: 5, 9
time_reps = 20
prediction_seed = 5
rho0 = rho_rand_haar(m, prediction_seed)

# Evolution function rho
def evolve_n_times(n: int, rho):
    rho_acc = rho
    rhos = [rho_acc]
    phi = channel_fac(circuit)(theta=theta_opt)
    for i in range(n):
        rho_acc = phi(rho=rho_acc)
        rhos.append(rho_acc)

    return np.array(rhos)

# Data & fig for evolving pauli measurements
rhos = evolve_n_times(time_reps, rho0)
ess = measure_rhos(rhos, Os())

rho_ref_s, ts = solve_lindblad_rho0(rho0, delta_t=channel_pars['delta_t'], N=time_reps, s=system)
rho_ref_s = np.array([mat.full() for mat in rho_ref_s])
e_ref_ss = measure_rhos(rho_ref_s, Os())

labels = [f"{Os.Os_names[i]}" for i in range(4**m)]
comparison_fig = compare_ess(approx = (ts, ess, "approx"), ref= (ts, e_ref_ss, "ref"), labels=labels)

# Data & fig for basis measurements
Os_comp_basis = create_observables_comp_basis(m)
labels_basis = [format(i, f"0{m}b") for i in range(len(Os_comp_basis))]
labels_basis = [r'$0=|00\rangle$', r'$1=|01\rangle$', r'$3=|10\rangle$', r'$2=|11\rangle$']
#labels_basis = [r'$0=|0\rangle$', r'$1=|1\rangle$']
ess_basis = measure_rhos(rhos, Os_comp_basis)
e_ref_ss_basis = measure_rhos(rho_ref_s, Os_comp_basis)

basis_fig = compare_ess(approx = (ts, ess_basis, "approx"), ref = (ts, e_ref_ss_basis, "ref"), labels=labels_basis)
error_fig = error_evolution(errors)

fancy_fig = fancy_fig_2(approx = (ts, ess_basis, "approx"), ref = (ts, e_ref_ss_basis, "ref"), labels = labels_basis, error = errors, pulses = theta_opt)

if save_results:
    names = ["Pauli_evolution", "01_evolution", "Training error", "Combined_figure"]
    save_figs([comparison_fig, basis_fig, error_fig, fancy_fig], results_path, names)


#%% Predict on multiple rhos for Bures distance

# nice seeds for 4level: 5, 9
time_reps = 20
n_rhos = 10
prediction_seed = 5
rho0s = random_rho0s(m=m, L=n_rhos, seed = prediction_seed)

# Something wack is going on with solve_lindblad_rho0s, which is not happening
# with solve_lindblad_rho0
rho_ref_s, ts = solve_lindblad_rho0s(rho0s=rho0s, delta_t=channel_pars['delta_t'], N=time_reps, s=system)
e_ref_ss = measure_rhoss(rho_ref_s, Os())

def evolve_all_n_times(n: int, rho_input):
    rho_start = rho_input.copy()
    l = len(rho0s)
    d = rho0s[0].shape[0]
    rhos_ev = np.zeros((n+1,l,d,d), dtype = complex)
    rhos_ev[0] = rho_start
    phi = channel_fac(circuit)(theta=theta_opt)
    for i in range(n):
        for j, rho in enumerate(rhos_ev[i]):
            rhos_ev[i+1,j] = phi(rho=rhos_ev[i,j])
    return np.array(rhos_ev)

# Data for evolving pauli measurements
rhos = evolve_all_n_times(time_reps, rho0s)
ess = measure_rhoss(rhos, Os())


# Sanity check
rho_index = 2
comparison_fig = compare_ess(approx = (ts, ess[:,:,rho_index], "approx"), ref= (ts, e_ref_ss[:,:,rho_index], "ref"), labels=labels)


# Determine roots:
rhos_sqrt = np.zeros((time_reps+1, n_rhos, 2**m, 2**m), dtype = complex)
products = np.zeros((time_reps+1, n_rhos, 2**m, 2**m), dtype = complex)
bures = np.zeros((time_reps+1, n_rhos))
fidelity = np.zeros((time_reps+1, n_rhos))
fid_alt = np.zeros((time_reps+1, n_rhos))

for i in range(time_reps+1):
    for j in range(n_rhos):
        rhos_sqrt[i,j] = sc.linalg.sqrtm(rhos[i,j])
        #bures[i,j] = 2-2*np.abs(np.trace(sc.linalg.sqrtm(rhos_sqrt[i,j] @rho_ref_s[i,j] @rhos_sqrt[i,j])))
        products[i,j] = rhos_sqrt[i,j] @rho_ref_s[i,j] @rhos_sqrt[i,j]
        fidelity[i,j] = np.trace(sc.linalg.sqrtm(rhos_sqrt[i,j] @rho_ref_s[i,j] @rhos_sqrt[i,j]))**2
        fid_alt[i,j] = np.trace(rhos[i,j]@rho_ref_s[i,j]) + 2*np.sqrt(np.linalg.det(rhos[i,j])*np.linalg.det(rho_ref_s[i,j]))
        #fidelity[i,j] = np.trace(sc.linalg.sqrtm(np.einsum('ij,jk,kl->il', rhos_sqrt[i,j], rho_ref_s[i,j], rhos_sqrt[i,j])))**2
        bures[i,j] = 2-2*np.sqrt(fidelity[i,j])
bures_mean = np.mean(bures, axis = 1)

plt.figure()
plt.plot(ts, bures_mean)
plt.xlabel(r"Time $\tau$ [ms]")
plt.ylabel("Average Bures distance")



#%% Sanity check on training data

rho_index = 2
n_pred = 100

rhos = evolve_n_times(n_pred, training_data.rho0s[rho_index])
ess = measure_rhos(rhos, Os_comp_basis)


rho0 = qt.Qobj(training_data.rho0s[rho_index], dims = [[2]*m,[2]*m])
rho_ref_s, ts = solve_lindblad_rho0(rho0, delta_t = channel_pars['delta_t'], N=n_pred, s=system)
rho_ref_s = np.array([mat.full() for mat in rho_ref_s])
e_ref_ss = measure_rhos(rho_ref_s, Os_comp_basis)


comparison_fig = compare_ess(approx = (ts, ess, "approx"), ref = (ts, e_ref_ss, "ref"), labels=labels_basis)

    
#%% For pulses

pulse_fig = plot_pulses(theta_opt, circuit, circuit_pars['control_H'])

