# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 11:53:36 2023

@author: 20168723
"""


import json
import qutip as qt
import numpy as np
import matplotlib.pyplot as plt

from q_channel_approx.qubit_layouts import qubitLayout_fac
from q_channel_approx.unitary_circuits import gate_circuit_fac
from q_channel_approx.unitary_circuits_pulse import unitary_pulsed_fac
from q_channel_approx.gradient_circuits import gradCircuit_fac

from q_channel_approx.physics_defns.target_systems import target_system_fac
from q_channel_approx.training_data import ( 
    random_rho0s, 
    solve_lindblad_rho0s, 
    solve_lindblad_rho0, 
    mk_training_data, 
    measure_rhos
    )
from q_channel_approx.training_observables import observables_fac
from q_channel_approx.optimizer import optimize_pulse, channel_fac

from q_channel_approx.physics_defns.initial_states import rho_rand_haar
from q_channel_approx.plotting.routines import plot_ess, compare_ess





#%% Setup circuit

# read circuit settings
circuit_file = "Input\\Circuit_pulse_v1.json"   
f = open(circuit_file)
circuit_pars = json.load(f)


qubits = qubitLayout_fac(**circuit_pars)
m = qubits.m
qubits.show_layout()

#circuit = gate_circuit_fac(qubits, **circuit_pars)
circuit = unitary_pulsed_fac(qubits, circuit_pars["control_H"], circuit_pars["driving_H"], 
                             circuit_pars["Zdt"], circuit_pars["t_max"])

        
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

# =============================================================================
# ryd_interaction = channel_pars["ryd_interaction"]
# omegas = channel_pars["omegas"]
# gammas = channel_pars["gammas"]
# 
# l = channel_pars["n_rhos"]
# observables = channel_pars["observables"]
# 
# system = DecaySystem(ryd_interaction=ryd_interaction, omegas=omegas, m=m, gammas=gammas)
# =============================================================================

system = target_system_fac(**channel_pars)

rho0s = random_rho0s(m=m, L=channel_pars['n_rhos'])
rhoss, ts = solve_lindblad_rho0s(rho0s=rho0s, delta_t=channel_pars['delta_t'], N=circuit_pars['n_depth'], s=system)

Os, Os_paulistrs = observables_fac(channel_pars["observables"], m)
        
training_data = mk_training_data(rhoss, Os)


#%% Initialize circuit

gradcircuit = gradCircuit_fac(circuit, training_data, "paulis", "all", circuit_pars['lambdapar'])



#%% Train

theta_opt, errors, thetas = optimize_pulse(gradcircuit, training_data, max_count=500, verbose = True)


#%% Predict

rho0 = rho_rand_haar(m, 4)


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
rho_ref_s = np.array([mat.full() for mat in rho_ref_s])
e_ref_ss = measure_rhos(rho_ref_s, Os)

labels = [f"{Os_paulistrs[i]}" for i in range(4**m)]
compare_ess((ts, ess, "approx"), (ts, e_ref_ss, "ref"), labels=labels)


#%% Verify decaying?

plt.figure()
for i in range(rhos[0].shape[0]):    
    plt.plot(ts, rhos[:,i,i], linestyle = ":")
plt.gca().set_prop_cycle(None)
for i in range(rhos[0].shape[0]):
    plt.plot(ts, rho_ref_s[:,i,i], linestyle = '-')


rho_index = 0

rhos = evolve_n_times(20, training_data.rho0s[rho_index])
ess = measure_rhos(rhos, Os)


rho0 = qt.Qobj(training_data.rho0s[rho_index], dims = [[2]*m,[2]*m])
rho_ref_s, ts = solve_lindblad_rho0(rho0, delta_t = 0.5, N=20, s=system)
rho_ref_s = np.array([mat.full() for mat in rho_ref_s])
e_ref_ss = measure_rhos(rho_ref_s, Os)

compare_ess((ts, ess, "approx"), (ts, e_ref_ss, "ref"), labels=labels)

plt.figure()
for i in range(rhos[0].shape[0]):    
    plt.plot(ts, rhos[:,i,i], linestyle = ":")
plt.gca().set_prop_cycle(None)
for i in range(rhos[0].shape[0]):
    plt.plot(ts, rho_ref_s[:,i,i], linestyle = '-')
    
#%% For pulses

from matplotlib.lines import Line2D
import matplotlib.axis as pltax
#plt.style.use('./Plot_styles/report_style.mplstyle')

colours = ['b', 'r', 'g', 'm', 'y', 'k']

plt.figure()
legend_elements = []

#legend_elements.append(Line2D([0],[0], color = 'k', ls = '-', lw = 2, label = 'Armijo descend'))

#legend_elements.append(Line2D([0],[0], color = colours[0], ls = '-', lw = 2, label = r'$q_{0}$'))
#legend_elements.append(Line2D([0],[0], color = colours[1], ls = '-', lw = 2, label = r'$q_{1}$ & $q_{2}$'))

x_range = np.linspace(0, circuit.t_max, circuit.Zdt)
control_H = circuit.operations[0]
for k in range(theta_opt.shape[0]):
    
    
    # real and imaginary
    plt.plot(x_range, theta_opt[k,:,0], '-', color = colours[k%6], label = 'qubit {}'.format(k))
    plt.plot(x_range, theta_opt[k,:,1], ':', color = colours[k%6])
    legend_elements.append(Line2D([0],[0], color = colours[k%6], ls = '-', lw = 2, label = 'qubit {}'.format(k)))
    if control_H.shape[0] == 2*(2*m+1):
        plt.plot(x_range, theta_opt[2*m+1+k,:,0], '--', color = colours[k%6])
        plt.plot(x_range, theta_opt[2*m+1+k,:,1], '-.', color = colours[k%6])
        
# =============================================================================
#     # Real only
#     plt.plot(x_range, theta_opt[k,:,0]+theta_opt[k,:,1], color = colours[k%6], label = f'q {k}')
#     legend_elements.append(Line2D([0],[0], color = colours[k%6], ls = '-', lw = 2, label = r'$q_{a}$'.format(a=k)))
#     if control_H.shape[0] == 2*(2*m+1):
#         plt.plot(x_range, theta_opt[2*m+1+k,:,0], '--', color = colours[k%6])
# =============================================================================
        

if circuit.operations[0].shape[0] == 2*(2*m+1):
# =============================================================================
#         legend_elements.append(Line2D([0],[0], color = 'gray', ls = '-', lw = 2, label = 'coupling - real'))
#         legend_elements.append(Line2D([0],[0], color = 'gray', ls = ':', lw = 2, label = 'coupling - imaginary'))
#         legend_elements.append(Line2D([0],[0], color = 'gray', ls = '--', lw = 2, label = 'detuning - real'))
#         legend_elements.append(Line2D([0],[0], color = 'gray', ls = '-.', lw = 2, label = 'detuning - imaginary'))
# =============================================================================
    
    legend_elements.append(Line2D([0],[0], color = 'k', ls = '-', lw = 2, label = 'coup'))
    legend_elements.append(Line2D([0],[0], color = 'k', ls = '--', lw = 2, label = 'det'))
else:
    legend_elements.append(Line2D([0],[0], color = 'gray', ls = '-', lw = 2, label = 'real'))
    legend_elements.append(Line2D([0],[0], color = 'gray', ls = ':', lw = 2, label = 'imaginary'))
plt.legend(handles = legend_elements, loc = (-0.3, 0.6))
#plt.legend(handles = legend_elements, loc = (-0.9,-0.2))
#plt.legend(handles = legend_elements, loc = (-1.0, 0.3))
#plt.legend(handles = legend_elements, loc = (-1.0, -0.3))

plt.xlabel(r'$\tau$ [ms]')
plt.xlim([0,circuit.t_max])
plt.ylabel(r'$z_r$ [kHz]')
plt.title("Final pulses")

