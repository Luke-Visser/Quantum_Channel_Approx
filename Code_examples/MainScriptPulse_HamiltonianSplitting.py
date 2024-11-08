# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 11:53:36 2023

@author: 20168723

general todo:
    - comments
    - fix saving of optimal theta (save to correct control, give proper name?)
    - 
    
"""


import json
import qutip as qt
import numpy as np
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
    mk_training_data_comp,
    measure_rhos
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
#circuit_file = "Input\\Circuit_pulse_2qubit_v2.json" 
#channel_file = "Input\\channel_2decay_v1.json"

channel_file = "Input\\channel_1decay_v2.json"  
circuit_file = "Input\\Circuit_pulse_1qubit_v1.json"

# =============================================================================
# channel_file = "Input\\channel_2ising_v1.json"
# circuit_file = "Input\\Circuit_pulse_2qubit_v2.json"
# =============================================================================

training_iters = 1000

# read circuit settings
f = open(circuit_file)
circuit_pars = json.load(f)

# read channel settings
f = open(channel_file)
channel_pars = json.load(f)


save_results = True
if save_results:
    name_results = "{}Q_{}".format(channel_pars['m'], channel_pars['name']) + "Split" +timestamp
    #name_results = "1decay_ham" + timestamp
    results_path = os.path.join(file_dir, "Results", name_results)
    os.mkdir(results_path)



#%% Setup Qubits

qubits = qubitLayout_fac(**circuit_pars)
m = qubits.m
qubits.show_layout()

#set random seed
qt.rand_ket(N=2,seed = circuit_pars['seed'])

seed = 2
np.random.seed(seed)
#np.random.seed(circuit_pars['seed'])


#%% Setup training data

# Quantum channel
system = target_system_fac(**channel_pars)

# System without decay
ham_pars = {**channel_pars, 'gammas': [0 for _ in range(channel_pars['m'])]}
system_ham = target_system_fac(**ham_pars)

# Generate and evolve density matrices
#rho0s = random_rho0s(m=m, L=channel_pars['n_rhos'])
rho0s = deterministic_rho0s(m=m, L=channel_pars['n_rhos'])
rhoss, ts = solve_lindblad_rho0s(rho0s=rho0s, delta_t=channel_pars['delta_t'], N=circuit_pars['n_depth'], s=system)

# Gather training data
Os = observables_fac(channel_pars["observables"], m)
training_data = mk_training_data(rhoss, Os)
training_data_comp = mk_training_data_comp(rhoss, Os)


#%% Learn Hamiltonian behaviour

reduced_pars = {**circuit_pars, 'layout': 'comp_only', 'n_depth': 1, 'Zdt': 101, 't_max': 10, 'lambdapar': 0}

qubits_comp = qubitLayout_fac(**reduced_pars)
qubits_comp.show_layout()

circuit_comp = unitary_pulsed_fac(qubits_comp, reduced_pars["control_H"], reduced_pars["driving_H"], 
                             reduced_pars["Zdt"], reduced_pars["t_max"])
gradcircuit_comp = gradCircuit_fac(circuit_comp, training_data_comp, reduced_pars["loss_type"], "all", reduced_pars['lambdapar'])

theta_opt, errors, thetas = optimize_pulse(gradcircuit_comp, training_data_comp, max_count=100, verbose = True)

U_ham = circuit_comp(theta_opt)

#%% Check Hamiltonian estimate
pulse_fig = plot_pulses(theta_opt, circuit_comp, reduced_pars['control_H'])


# nice seeds for 4level: 5, 9
time_reps = 20
rho0 = rho_rand_haar(m, 5)
rho0 = rho0s[3]
def evolve_n_times_ham(n: int, rho):
    rho_acc = rho
    rhos = [rho_acc]
    for i in range(n):
        rho_acc = U_ham @ rho_acc @ np.transpose(U_ham.conj())
        rhos.append(rho_acc)

    return np.array(rhos)

# Data & fig for evolving pauli measurements
rhos = evolve_n_times_ham(time_reps, rho0.full())
ess = measure_rhos(rhos, Os())

rho_ref_s, ts = solve_lindblad_rho0(rho0, delta_t=channel_pars['delta_t'], N=time_reps, s=system_ham)
incr_res = 10
rho_ref_s, ts_ref = solve_lindblad_rho0(rho0, delta_t=channel_pars['delta_t']/incr_res, N=time_reps*incr_res, s=system_ham)
rho_ref_s = np.array([mat.full() for mat in rho_ref_s])
e_ref_ss = measure_rhos(rho_ref_s, Os())

labels = [f"{Os.Os_names[i]}" for i in range(4**m)]
comparison_fig = compare_ess(approx = (ts, ess, "approx"), ref =(ts_ref, e_ref_ss, "ref"), labels=labels)

# Data & fig for basis measurements
Os_comp_basis = create_observables_comp_basis(m)
labels_basis = [format(i, f"0{m}b") for i in range(len(Os_comp_basis))]
labels_basis = [r'$0=|00\rangle$', r'$1=|01\rangle$', r'$3=|10\rangle$', r'$2=|11\rangle$']
ess_basis = measure_rhos(rhos, Os_comp_basis)
e_ref_ss_basis = measure_rhos(rho_ref_s, Os_comp_basis)

basis_fig = compare_ess(approx = (ts, ess_basis, "approx"), ref = (ts_ref, e_ref_ss_basis, "ref"), labels=labels_basis)
error_fig = error_evolution(errors)
fancy_fig = fancy_fig_1(approx = (ts, ess_basis, "approx"), ref = (ts_ref, e_ref_ss_basis, "ref"), labels = labels_basis, error = errors, pulses = theta_opt)

if save_results:
    names = ["Pauli_evolution_ham", "01_evolution_ham", "Training error_ham", "fancy_fig_ham"]
    save_figs([comparison_fig, basis_fig, error_fig, fancy_fig], results_path, names)


#%% Learn dilation

circuit = unitary_pulsed_fac(qubits, circuit_pars["control_H"], circuit_pars["driving_H"], 
                             circuit_pars["Zdt"], circuit_pars["t_max"], U_comp = U_ham)

gradcircuit = gradCircuit_fac(circuit, training_data, circuit_pars["loss_type"], "all", circuit_pars['lambdapar'])


theta_opt, errors, thetas = optimize_pulse(gradcircuit, training_data, max_count=training_iters, verbose = True)


if save_results:
    save_data(results_path, theta_opt, errors, circuit_pars, channel_pars)


#%% Predict



# nice seeds for 4level: 5, 9
time_reps = 20
rho0 = rho_rand_haar(m, 5)
#rho0 = rho0s[6]

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
incr_res = 10
rho_ref_s, ts_ref = solve_lindblad_rho0(rho0, delta_t=channel_pars['delta_t']/incr_res, N=time_reps*incr_res, s=system)
rho_ref_s = np.array([mat.full() for mat in rho_ref_s])
e_ref_ss = measure_rhos(rho_ref_s, Os())

labels = [f"{Os.Os_names[i]}" for i in range(4**m)]
comparison_fig = compare_ess((ts, ess, "approx"), (ts_ref, e_ref_ss, "ref"), labels=labels)

# Data & fig for basis measurements
Os_comp_basis = create_observables_comp_basis(m)
labels_basis = [format(i, f"0{m}b") for i in range(len(Os_comp_basis))]
labels_basis = [r'$0=|00\rangle$', r'$1=|01\rangle$', r'$3=|10\rangle$', r'$2=|11\rangle$']
ess_basis = measure_rhos(rhos, Os_comp_basis)
e_ref_ss_basis = measure_rhos(rho_ref_s, Os_comp_basis)

basis_fig = compare_ess(approx = (ts, ess_basis, "approx"), ref = (ts_ref, e_ref_ss_basis, "ref"), labels=labels_basis)
error_fig = error_evolution(errors)
fancy_fig = fancy_fig_1(approx = (ts, ess_basis, "approx"), ref = (ts_ref, e_ref_ss_basis, "ref"), labels = labels_basis, error = errors, pulses = theta_opt)

if save_results:
    names = ["Pauli_evolution", "01_evolution", "Training error", "fancy_fig"]
    save_figs([comparison_fig, basis_fig, error_fig, fancy_fig], results_path, names)


#%% Sanity check on training data

rho_index = 2
time_reps = 10

rhos = evolve_n_times(time_reps, training_data.rho0s[rho_index])
ess = measure_rhos(rhos, Os_comp_basis)


rho0 = qt.Qobj(training_data.rho0s[rho_index], dims = [[2]*m,[2]*m])
rho_ref_s, ts = solve_lindblad_rho0(rho0, delta_t = 0.5, N=time_reps, s=system)
rho_ref_s = np.array([mat.full() for mat in rho_ref_s])
e_ref_ss = measure_rhos(rho_ref_s, Os_comp_basis)


comparison_fig = compare_ess((ts, ess, "approx"), (ts, e_ref_ss, "ref"), labels=labels_basis)

    
#%% For pulses

pulse_fig = plot_pulses(theta_opt, circuit, circuit_pars['control_H'])

