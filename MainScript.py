# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 11:53:36 2023

@author: 20168723
"""


import sys
import os
import time
from datetime import datetime


working_dir = os.getcwd()

from PythonScripts.readInput import read_input
from PythonScripts.QCApprox import qChanApprox, targetQChan

#Read parameters
input_path = os.path.join(working_dir, 'Input\\QComputer_pars.xlsx')
input_pars = read_input()
pars, sweep_pars = input_pars.read_xlsx(input_path)
sweep_pars_str = " ".join(sweep_pars)

#Create working directories for results
time_now = datetime.now()
result_dir_name = '{}-{}, h{}-{} sweep {}'.format(time_now.month, time_now.day,\
                                          time_now.hour, time_now.minute, sweep_pars_str)
result_dir = os.path.join(working_dir,'Results', result_dir_name)
os.mkdir(result_dir)
os.mkdir(os.path.join(result_dir, 'Figures'))
os.mkdir(os.path.join(result_dir, 'Input'))