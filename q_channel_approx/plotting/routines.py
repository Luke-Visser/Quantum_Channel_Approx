"""
Provides some common plotting routines.
"""

import os
import itertools

from matplotlib.patches import Patch
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.axis as pltax

import numpy as np
import pandas as pd

# When using custom style
style = "report_style"
dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, f"plot_styles/{style}.mplstyle")
plt.style.use(os.path.join(dirname, filename))

# to use a predefined style
# plt.style.use("default")

#plt.style.use('./plot_styles/report_style.mplstyle')


def save_figs(figs, path, name):
    for i, fig in enumerate(figs):
        full_name = os.path.join(path, name[i])
        fig.savefig(full_name + ".pdf", bbox_inches = 'tight')
        
        
def save_data(results_path, theta, errors, circuit_pars, channel_pars):
    m = circuit_pars['m']
    settings_dict = {**circuit_pars, **channel_pars}
    *keys, = settings_dict
    vals = [settings_dict[key] for key in settings_dict]
    settings = pd.DataFrame.from_dict({'Parameters': keys, 'Values': vals})
    theta_dict = {}
    n_controls = len(theta)
    if '+11' in circuit_pars['control_H']:
        for i in range(n_controls//2):
            theta_dict.update({f'rotation control {i}, real': theta[i,:,0], 
                               f'rotation control {i}, imag': theta[i,:,1],
                               f'detuning control {i}, real': theta[m+i,:,0],
                               f'detuning control {i}, imag': theta[m+i,:,1],})
    else:
        for i in range(n_controls):
            theta_dict.update({f'rotation control {i}, real': theta[i,:,0], 
                               f'rotation control {i}, imag': theta[i,:,1]})
            
    assert len(theta_dict) == 2*n_controls
        
    theta = pd.DataFrame.from_dict(theta_dict)
    errors = pd.DataFrame(errors)
    
    with pd.ExcelWriter(os.path.join(results_path, 'simulation_results.xlsx'), engine = 'openpyxl') as writer:
        settings.to_excel(writer, sheet_name = 'Simulation settings',  \
                      float_format = "%.5f", index= False)
        theta.to_excel(writer, sheet_name = 'Optimal theta',  \
                      float_format = "%.5f", index= False)
        errors.to_excel(writer, sheet_name = 'Errors',  \
                       float_format = "%.10f", index= False)


def load_data(filename):
    file = pd.ExcelFile(filename)
    
    pars = file.parse('Simulation settings')
    keyval = pars.to_dict()
    keys = keyval['Parameters']
    vals = keyval['Values']
    pars = {keys[ind]:vals[ind] for ind in keys}
    
    for key in pars:
        if type(pars[key]) == str:
            if '[' in pars[key]:
                pars[key] = eval(pars[key])
        
    theta_df = file.parse('Optimal theta')
    N, dims = theta_df.shape
    dims = dims//2
    theta = np.zeros((dims, N, 2))
    if f'detuning control 0, real' in theta_df:
        m = dims//2
        for i in range(m):
            theta[i,:,0] = theta_df[f'rotation control {i}, real']
            theta[i,:,1] = theta_df[f'rotation control {i}, imag']
            theta[m+i,:,0] = theta_df[f'detuning control {i}, real']
            theta[m+i,:,1] = theta_df[f'detuning control {i}, imag']
    else:
        for i in range(dims):
            theta[i,:,0] = theta_df[f'rotation control {i}, real']
            theta[i,:,1] = theta_df[f'rotation control {i}, imag']
    
    errors_df = file.parse('Errors')
    errors = errors_df.to_numpy()
    
    return pars, theta, errors
    
def error_evolution(error: np.array):
    fig = plt.figure()
    ax = plt.subplot(111)
    plt.plot(range(1, len(error)), error[1:], color = 'gray', linewidth = 3)
    plt.yscale('log')
    plt.ylabel(r'$J_1(U)$ ')
    plt.xlabel('Iteration')
    plt.xlim(left = 0, right = len(error))
    
    return fig


def compare_ess(ref: tuple, approx: tuple, labels: list[str]):
    """ref is a tuple (ts, Ess, name),
    approx is similarly (ts, Ess, name)
    """
    ts_ref, Ess_ref, name_ref = ref
    ts_approx, Ess_approx, name_approx = approx

    fig, ax = plt.subplots()

    for k, Es in enumerate(Ess_approx.swapaxes(0,1)):
        ax.plot(ts_approx, Es, 'x', label=rf"{labels[k]}")
    plt.gca().set_prop_cycle(None)
    for k, Es in enumerate(Ess_ref.swapaxes(0,1)):
        ax.plot(ts_ref, Es, linestyle="-") # label=rf"{labels[k]}", 

    # some formatting to make plot look nice
    plt.ylabel("population")
    plt.xlabel("time")
    plt.suptitle("Evolution", weight="bold")
    plt.title(f"{name_approx}: crosses, {name_ref}: solid line")
    # plt.ylim(0, 1)
    plt.xlim(min(ts_ref), max(ts_ref))
    plt.legend(ncol = 2, loc = "upper left", bbox_to_anchor = (1,1))
    return fig

def fancy_fig_2(ref, approx, labels: list[str], error, pulses):
    """
    Figure from paper with (a) and (b) for 2 qubits.
    t_train has to be set manually and does not match the input data

    Parameters
    ----------
    ref : TYPE
        DESCRIPTION.
    approx : TYPE
        DESCRIPTION.
    labels : list[str]
        DESCRIPTION.
    error : TYPE
        DESCRIPTION.
    pulses : TYPE
        DESCRIPTION.

    Returns
    -------
    fig : TYPE
        DESCRIPTION.

    """
    
    colours = ['b', 'r', 'g', 'm', 'y', 'k']
    ts_ref, Ess_ref, name_ref = ref
    ts_approx, Ess_approx, name_approx = approx

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12, 4), width_ratios = (4,3))

    for k, Es in enumerate(Ess_approx.swapaxes(0,1)):
        ax1.plot(ts_approx, Es, 'x') 
    ax1.set_prop_cycle(None)
    for k, Es in enumerate(Ess_ref.swapaxes(0,1)):
        ax1.plot(ts_ref, Es, linestyle="-", label=rf"{labels[k]}")
    
    
    ax1.text(0.22, 0.80, 'Max population error:\n'+'{:.2E}'.format(np.max(Ess_approx - Ess_ref[::10])).rjust(15), transform=ax1.transAxes, fontsize = 12)

    
    # Create legend
    anchor = (0.5,1.0)
    #anchor = (0.75,0.7)
    anchor = (0.77,1.0)
    legend_elements = []
    legend_elements.append(Line2D([0],[0], color = 'gray', ls = '-', lw = 2, label = 'Exact'))
    legend_elements.append(Line2D([0],[0], color = 'gray', marker = 'x', lw = 0, label = 'Approx'))
    legend1 = ax1.legend(handles = legend_elements, loc = 'upper left', bbox_to_anchor = anchor)
    
    # Vertical line at training data with T_train
    ax1.axvline(x=2, ls = ':', color = 'dimgray', linewidth = 2)
    
    
    # some formatting to make plot look nice
    ax1.set_ylabel("Population")
    ax1.set_xlabel("Time")
    ax1.set_ylim(0, 1)
    ax1.set_xlim(min(ts_ref), max(ts_ref))
    ax1.legend(ncol = 1, loc = 'upper right', bbox_to_anchor = anchor)
    ax1.add_artist(legend1)
    
    # Adding t_train label
    labels = [item.get_text() for item in ax1.get_xticklabels()]
    labels[1] = r'$t_{train}$'
    ax1.set_xticklabels(labels)
    
    # Error figure
    #ax2.plot(range(1, len(error)), error[1:,0], color = 'gray', linewidth = 3)
    ax2.plot(range(1, len(error)), error[1:,1], "-", color = 'gray', linewidth = 3, label = "Direct")
    ax2.plot(range(1, len(error)), error[1:,0], "--", color = 'gray', linewidth = 3, label = "Split")
    legend2 = ax2.legend(loc = 'upper left', bbox_to_anchor = (0.15,1.0))
    ax2.set_yscale('log')
    ax2.set_ylabel(r'$J_1(U)$ ')
    ax2.set_xlabel('Iteration')
    ax2.set_xlim(left = 0, right = len(error))
    
    # Qubits with legend 
    geom_x = [0.50,0.65,0.425,0.575,0.575]
    geom_y = [0.55,0.55,0.37,0.37,0.73]
    pairs = [[0,1], [0,2], [0,3],[1,3],[1,4],[2,3],[0,4]]
    
    ax2.text(0.45, 0.77, 'Qubit layout', transform=ax2.transAxes, fontsize = 12)
    ax2.text(0.65, 0.40, r'$R=1.0\mu$m', transform=ax2.transAxes, fontsize = 12)
    
    dot_colours = colours
    legend_elements = []
    
    for pair1, pair2 in pairs:
        ax2.plot([geom_x[pair1], geom_x[pair2]],[geom_y[pair1],geom_y[pair2]], c='gray', transform = ax2.transAxes)
    
    for k in range(len(geom_x)):
        ax2.scatter(geom_x,geom_y, color = dot_colours[0:len(geom_x)], transform = ax2.transAxes, zorder = 2, s = 70)
        legend_elements.append(Line2D([0],[0], color = colours[k%6], ls = '-', lw = 2, label = rf'$q_{k}$'.format(k)))
    legend3 = ax2.legend(handles = legend_elements, loc = 'upper right', bbox_to_anchor = (0.95, 1.0))
    ax2.add_artist(legend2)
# =============================================================================
#     # Pulses figure as inset subplot
#     ax3 = plt.axes([0.75, 0.53, 0.20, 0.35])
#     m_all, Zdt, _ = pulses.shape
#     tmax = 15
#     m_all = m_all//2
#     x_range = np.linspace(0, tmax, Zdt)
#     
#     legend_elements = []
#     for k in range(m_all):
#         ax3.plot(x_range, pulses[k,:,0], '-', color = colours[k%6])
#         ax3.plot(x_range, pulses[m_all+k,:,0], '--', color = colours[k%6])
# 
#     ax3.set_xlabel(r'$\tau$ [ms]')
#     ax3.set_xlim([0,tmax])
#     ax3.set_ylabel(r'$z_r$ [kHz]')
#     
#     legend_elements = []
#     legend_elements.append(Line2D([0],[0], color = 'gray', ls = '-', lw = 2, label = 'coup'))
#     legend_elements.append(Line2D([0],[0], color = 'gray', ls = '--', lw = 2, label = 'det'))
#     ax2.legend(handles = legend_elements, loc = 'upper left', bbox_to_anchor = (0.25, 1.0))
#     ax2.add_artist(legend2)
# =============================================================================
    
    # Labels (a) and (b)
    ax1.text(-0.18, 0.95, '(b)', transform=ax1.transAxes, fontsize = 16)
    ax2.text(-0.18, 0.95, '(c)', transform=ax2.transAxes, fontsize = 16)
    
    # Keep distance between subplots
    fig.tight_layout()

    return fig

def fancy_fig_1(ref, approx, labels: list[str], error, pulses):
    """
    Figure for paper for 1 qubit setting.

    Parameters
    ----------
    ref : tuple
        tuple of arrays of time, measurement values, and the name used in the plots.
    approx : tuple
        tuple of arrays of time, measurement values, and the name used in the plots.
    labels : list[str]
        labels of the various measurements operators
    error : array
        training error.
    pulses : ndarray
        full pulse description.

    Returns
    -------
    fig : matplotlib.pyplot.Figure object
        Figure object.

    """
    
    colours = ['b', 'r', 'g', 'm', 'y', 'k']
    ts_ref, Ess_ref, name_ref = ref
    ts_approx, Ess_approx, name_approx = approx
    
    # Set the two main subfigures
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12, 4), width_ratios = (4,3))
    
    # Comparison measurement and actual evolution.
    for k, Es in enumerate(Ess_approx.swapaxes(0,1)):
        ax1.plot(ts_approx, Es, 'x') #, label=rf"{labels[k]}")
    ax1.set_prop_cycle(None)
    for k, Es in enumerate(Ess_ref.swapaxes(0,1)):
        ax1.plot(ts_ref, Es, linestyle="-", label=rf"{labels[k]}")
    
    # Create legend
    anchor = (0.75, 0.95)
    legend_elements = []
    legend_elements.append(Line2D([0],[0], color = 'gray', ls = '-', lw = 2, label = 'Exact'))
    legend_elements.append(Line2D([0],[0], color = 'gray', marker = 'x', lw = 0, label = 'Approx'))
    legend1 = ax1.legend(handles = legend_elements, loc = 'upper left', bbox_to_anchor = anchor)
    
    # Vertical line at training data with T_train
    ax1.axvline(x=2, ls = ':', color = 'dimgray', linewidth = 2)
    #ax1.text(0.05, -0.06, r'$t_{train}$', transform=ax1.transAxes, fontsize = 10)
    
    # some formatting to make plot look nice
    ax1.set_ylabel("Population")
    ax1.set_xlabel("Time")
    ax1.set_ylim(0, 1)
    ax1.set_xlim(min(ts_ref), max(ts_ref))
    ax1.legend(ncol = 1, loc = 'upper right', bbox_to_anchor = anchor)
    ax1.add_artist(legend1)
    
    # Add t_train label
    labels = [item.get_text() for item in ax1.get_xticklabels()]
    labels[1] = r'$t_{train}$'
    ax1.set_xticklabels(labels)
    
    # Prediction error figure
    #ax1sub = plt.axes([0.27, 0.25, 0.20, 0.20])
    ax1sub = plt.axes([0.32, 0.25, 0.20, 0.20])
    for k, Es in enumerate(Ess_approx.swapaxes(0,1)):    
        ax1sub.plot(ts_approx, Es - Ess_ref[::10,k], 'x')
    ax1sub.set_ylabel("Error")
    #ax1.text(0.50, 0.15, 'Max error of {:.2E}'.format(np.max(Ess_approx - Ess_ref)), transform=ax1.transAxes, fontsize = 12)

    
    # Training error figure
    ax2.plot(range(1, len(error)), error[1:,0], color = 'gray', linewidth = 3)
    ax2.set_yscale('log')
    ax2.set_ylabel(r'$J_1(U)$ ')
    ax2.set_xlabel('Iteration')
    ax2.set_xlim(left = 0, right = len(error))
    

    
    # Qubits with legend
    #geom_x = [0.80,0.90]
    #geom_y = [0.30,0.30]
    pairs = [[0,1],[1,2],[0,2]]
    
    geom_x = [0.10,0.25,0.175]
    geom_y = [0.25,0.25,0.07]
    ax2.text(0.25, 0.10, r'$R=1.0\mu$m', transform=ax2.transAxes, fontsize = 12)
    ax2.text(0.07, 0.32, 'Qubit layout', transform=ax2.transAxes, fontsize = 12)
    
    
    dot_colours = colours
    legend_elements = []
    
    for pair1, pair2 in pairs:
        ax2.plot([geom_x[pair1], geom_x[pair2]],[geom_y[pair1],geom_y[pair2]], c='gray', transform = ax2.transAxes)
    
    for k in range(len(geom_x)):
        ax2.scatter(geom_x,geom_y, color = dot_colours[0:len(geom_x)], transform = ax2.transAxes, zorder = 2, s = 70)
        legend_elements.append(Line2D([0],[0], color = colours[k%6], ls = '-', lw = 2, label = rf'$q_{k}$'.format(k)))
    legend_elements.append(Line2D([0],[0], color = 'gray', ls = '-', lw = 2, label = 'coup'))
    legend_elements.append(Line2D([0],[0], color = 'gray', ls = '--', lw = 2, label = 'det'))    

    ax2.legend(handles = legend_elements, loc = 'upper left', bbox_to_anchor = (0.05, 0.95))

    
    # Pulses figure as inset subplot
    #ax3 = plt.axes([0.75, 0.53, 0.20, 0.35])
    ax3 = plt.axes([0.80, 0.53, 0.15, 0.35])
    m_all, Zdt, _ = pulses.shape
    tmax = 15
    m_all = m_all//2
    x_range = np.linspace(0, tmax, Zdt)
    
    legend_elements = []
    for k in range(m_all):
        ax3.plot(x_range, pulses[k,:,0], '-', color = colours[k%6])
        ax3.plot(x_range, pulses[m_all+k,:,0], '--', color = colours[k%6])

    ax3.set_xlabel(r'$\tau$ [ms]')
    ax3.set_xlim([0,tmax])
    ax3.set_ylabel(r'$z_r$ [kHz]')
    
    # Labels (a) and (b) and qubit distance
    ax1.text(-0.18, 0.95, '(b)', transform=ax1.transAxes, fontsize = 16)
    ax2.text(-0.18, 0.95, '(c)', transform=ax2.transAxes, fontsize = 16)
    #ax2.text(0.80, 0.20, r'$R=1.0\mu$m', transform=ax2.transAxes, fontsize = 12)
    

    
    # Keep distance between subplots
    fig.tight_layout()

    return fig
    

def plot_evolution_basis_states(ts: np.ndarray, rhos1: np.ndarray, rhos2: np.ndarray):
    fig = plt.figure()
    for i in range(rhos1[0].shape[0]):    
        plt.plot(ts, rhos1[:,i,i], linestyle = ":")
    plt.gca().set_prop_cycle(None)
    for i in range(rhos2[0].shape[0]):
        plt.plot(ts, rhos2[:,i,i], linestyle = '-')
    return fig


def plot_evolution_computational_bs(
    ts: np.ndarray,
    Ess: list[np.ndarray],
) -> Axes:

    m = len(Ess).bit_length() - 1

    for i, Es in enumerate(Ess):
        plt.plot(
            ts,
            Es,
            label=rf"$|{format(i, f'0{m}b')}\rangle \langle{format(i, f'0{m}b')}|$",
        )

    # some formatting to make plot look nice
    plt.ylabel("Population")
    plt.xlabel("Time")
    plt.ylim(0, 1)
    plt.legend()

    return plt.gca()


def plot_evolution_individual_qs(ts: np.ndarray, Ess: list[np.ndarray]) -> Axes:
    """Plots the evolution of all rhos as a function of ts
    with some basic formatting.

    Args:
        ts (np.ndarray): times t_i
        rhoss (list[np.ndarray]): list of rho evolutions (for each rhos: rho_i at time t_i
    """

    fig, ax = plt.subplots()

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = itertools.cycle(prop_cycle.by_key()["color"])

    for i, Es in enumerate(Ess):
        state = i % 2
        linestyle = "-" if i % 2 == 0 else ":"

        if i % 2 == 0:
            color = next(colors)
        ax.plot(
            ts,
            Es,
            label=rf"$q_{i//2} : |{state}\rangle \langle{state}|$",
            linestyle=linestyle,
            color=color,
        )

    # some formatting to make plot look nice
    plt.ylabel("population")
    plt.xlabel("time")
    plt.ylim(0, 1)
    plt.legend()

    return ax

def plot_pulses(theta, circuit, type_H):
    colours = ['b', 'r', 'g', 'm', 'y', 'k']
    m = circuit.qubit_layout.m
    m_all = m + circuit.qubit_layout.n_ancilla
    
    fig = plt.figure()
    legend_elements = []

    #legend_elements.append(Line2D([0],[0], color = 'k', ls = '-', lw = 2, label = 'Armijo descend'))
    #legend_elements.append(Line2D([0],[0], color = colours[0], ls = '-', lw = 2, label = r'$q_{0}$'))
    #legend_elements.append(Line2D([0],[0], color = colours[1], ls = '-', lw = 2, label = r'$q_{1}$ & $q_{2}$'))

    x_range = np.linspace(0, circuit.t_max, circuit.Zdt)
    control_H = circuit.operations[0]
    
    match type_H:
        case 'rotations':
            for k in range(m_all):
                plt.plot(x_range, theta[k,:,0], '-', color = colours[k%6])
                plt.plot(x_range, theta[k,:,1], ':', color = colours[k%6])
                legend_elements.append(Line2D([0],[0], color = colours[k%6], ls = '-', lw = 2, label = 'qubit {}'.format(k)))
            legend_elements.append(Line2D([0],[0], color = 'gray', ls = '-', lw = 2, label = 'Re'))
            legend_elements.append(Line2D([0],[0], color = 'gray', ls = ':', lw = 2, label = 'Im'))
            plt.title("Final pulses, complex rotational control")
            
        case 'realrotations':
            for k in range(m_all):
                plt.plot(x_range, theta[k,:,0], '-', color = colours[k%6])
                legend_elements.append(Line2D([0],[0], color = colours[k%6], ls = '-', lw = 2, label = 'qubit {}'.format(k)))
            plt.title("Final pulses, real rotational control")
                
        case 'rotations+11':
            for k in range(m_all):
                plt.plot(x_range, theta[k,:,0], '-', color = colours[k%6])
                plt.plot(x_range, theta[k,:,1], ':', color = colours[k%6])
                plt.plot(x_range, theta[m_all+k,:,0], '--', color = colours[k%6])
                legend_elements.append(Line2D([0],[0], color = colours[k%6], ls = '-', lw = 2, label = 'qubit {}'.format(k)))
            legend_elements.append(Line2D([0],[0], color = 'gray', ls = '-', lw = 2, label = 'coup - Re'))
            legend_elements.append(Line2D([0],[0], color = 'gray', ls = ':', lw = 2, label = 'coup - Im'))
            legend_elements.append(Line2D([0],[0], color = 'gray', ls = '--', lw = 2, label = 'det'))
            plt.title("Final pulses, complex rotational and detuning control")
            
        case 'realrotations+11':
            for k in range(m_all):
                plt.plot(x_range, theta[k,:,0], '-', color = colours[k%6])
                plt.plot(x_range, theta[m_all+k,:,0], '--', color = colours[k%6])
                legend_elements.append(Line2D([0],[0], color = colours[k%6], ls = '-', lw = 2, label = 'qubit {}'.format(k)))
            legend_elements.append(Line2D([0],[0], color = 'gray', ls = '-', lw = 2, label = 'coup'))
            legend_elements.append(Line2D([0],[0], color = 'gray', ls = '--', lw = 2, label = 'det'))
            plt.title("Final pulses, real rotational and detuning control")
        
    plt.legend(handles = legend_elements, loc = (-0.3, 0.6))
    #plt.legend(handles = legend_elements, loc = (-0.9,-0.2))
    #plt.legend(handles = legend_elements, loc = (-1.0, 0.3))
    #plt.legend(handles = legend_elements, loc = (-1.0, -0.3))

    plt.xlabel(r'$\tau$ [ms]')
    plt.xlim([0,circuit.t_max])
    plt.ylabel(r'$z_r$ [kHz]')

    return fig

# New skool plotting, for report and presentation
def plot_in_computational_bs(
    ts: np.ndarray,
    Ess: list[np.ndarray],
    marker: str,
    linestyle: str,
    alpha: float,
) -> Axes:

    for Es in Ess:
        plt.plot(
            ts,
            Es,
            alpha=alpha,
            marker=marker,
            linestyle=linestyle,
        )

    return plt.gca()


def plot_approx(ts, Ess) -> Axes:
    plt.gca().set_prop_cycle(None)
    return plot_in_computational_bs(ts, Ess, marker="o", linestyle="none", alpha=0.6)


def plot_ref(ts, Ess, linestyle:str = ":") -> Axes:
    plt.gca().set_prop_cycle(None)
    return plot_in_computational_bs(ts, Ess, marker="none", linestyle=linestyle, alpha=1)


def legend_comp(m: int):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    legend_items = [
    Patch(
        color=colors[j],
        label=rf"$|{format(j, f'0{m}b')}\rangle \langle{format(j, f'0{m}b')}|$",
    )
    for j in range(2**m)
]
    return legend_items
