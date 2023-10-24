# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 11:54:09 2023

@author: 20168723
"""

import pandas as pd
import numpy as np
import os

class read_input:
    
    def __init__(self):
        """
        

        Returns
        -------
        None.

        """
    
    def read_xlsx(self, path):
        all_input_pars = pd.read_excel(path, sheet_name="main", index_col=0)
        all_input_pars.dropna(axis=0, how='all', inplace=True)
        cutoff_index = all_input_pars.columns.get_loc("Units")
        all_input_pars = all_input_pars.get(all_input_pars.columns[range(cutoff_index)])
        pars_dict = all_input_pars.to_dict()['Value']
        
        varied_pars = []
        try:
            sweep_pars = pd.read_excel(path, sheet_name="Sweep_pars", index_col=0)
            sweep_pars.dropna(axis=0, how='all', inplace=True)
            sweep_pars = sweep_pars.to_dict(orient = 'index')
            for key in sweep_pars:
                if key in pars_dict:
                    pars_dict[key] = np.linspace(sweep_pars[key]['Start'], sweep_pars[key]['Stop'], sweep_pars[key]['Num'])
                    varied_pars.append(key)
                else:
                    print("{} not found in main parameter list, so not varied over.")
        except ValueError:
            pass
        
        return pars_dict, varied_pars
        

if __name__ == '__main__':
    test_case = read_input()
        