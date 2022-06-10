#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 15:00:04 2022

@author: bflucero
"""

import pandas as pd
import numpy as np

#%% SHECHTER PARAMETERS

paramsDF = pd.DataFrame(index=('phi_star', 'phi_plusmin', 'logphi_pm', 'M_star', 'M_plusmin', 'alpha', 'alpha_plusmin', 'z', 'M_UV_range'))

#Ono(2018)
paramsDF['Ono (2018)'] = [1.06e-3, np.array([0.13*10**-3, 0.11*10**-3]), np.nan, -20.96, np.array([0.06, 0.05]), -1.60, 
                          np.array([0.06, 0.11]), 5, np.array([-26, -14])] #[phi* (mpc-3) , M* (mag)]

#Bouwens (2021) - this includes the same dataset used to derive the 2015 parameters; just updated w more sources
paramsDF['Bouwens (2021a)'] = [0.79e-3, np.array([0.16e-3, 0.13e-3]), np.nan, -21.20, np.array([0.11, 0.11]), -1.74, 
                               np.array([0.06, 0.06]), 4.9, np.array([-21.85, -17.60])] #[phi* (mpc-3) , M* (mag)]

########################## NOTE the phi* and M* values have different units than the rest of the params ##################################################################
#Harikane 2021
def pw10_plus(upper, val):
    main = 10**val
    upper = 10**(val + upper)
    return(upper-main, upper)

def pw10_min(lower, val):
    main = 10**val
    lower = 10**(val - lower)
    return(main-lower, lower)

paramsDF['Harikane (2021) galaxy component (AGN_DPL + GAL_Schechter)'] = [10**(-3.16), [pw10_plus(0.03, -3.16)[0], pw10_min(0.03, -3.16)[0]], 
                                                                          [0.03, 0.03], -21.09, [0.04,0.03], -1.76, [0.04, 0.03], 5, [-29, -14]] #[phi* (mag) -NOTE: orig given in log(phi*), M* (mpc-3)]
paramsDF['Harikane (2021) Galaxy LF (Schechter)'] = [10**(-3.10), [pw10_plus(0.06, -3.10)[0], pw10_min(0.06, -3.10)[0]], 
                                                     [0.06, 0.06], -21.04, [0.08, 0.07], -1.76, [0.05, 0.05], 5,[-29, -14]]

# paramsDF.to_pickle("schechterparams.pkl")

#%% DOUBLE POWER LAW PARAMETERS 
DPLparams = pd.DataFrame(columns = [], index = ('phi_dpl', 'phi_dpl_pm', 'logphi_pm', 'M_dpl', 'M_dpl_pm', 'alpha_dpl', 'alpha_dpl_pm', 'beta', 'beta_pm', 'z', 'M_UV_range'))

#Ono 2018
DPLparams['Ono (2018) DPL'] = [0.36e-3, [0.05e-3, 0.05e-3], np.nan, -21.44, [0.07, 0.07], -1.88, [0.05, 0.04], -5.07, [0.17, 0.18], 5,[-26, -14]] #[phi* (mpc-3) , M* (mag)]
    
########################## NOTE the phi* and M* values have different units than the rest of the params ##################################################################
#Harikane 2021
DPLparams['Harikane (2021) AGN component (AGN_DPL + GAL_Schechter)'] = [10**(-8.71), [pw10_plus(0.89, -8.71)[0], pw10_min(0.67, -8.71)[0]], [0.89, 0.67], -27.67, [1.47, 0.88], -2.27, [0.48, 0.22], -5.92, [0.66, 1.14], 5, [-29, -14]]
DPLparams['Harikane (2021) AGN component (AGN_DPL + GAL_DPL)'] = [10**(-8.35), [pw10_plus(0.47, -8.35)[0], pw10_min(0.18, -8.35)[0]], [0.47, 0.18], -27.32, [0.76, 0.26], -1.92, [0.31,0.17], -4.77, [0.61,0.62], 5, [-29, -24]]
DPLparams['Harikane (2021) galaxy component (AGN_DPL + GAL_DPL)'] = [10**(-3.63), [pw10_plus(0.04, -3.63)[0], pw10_min(0.03, -3.63)[0]],[0.04, 0.03], -21.54, [0.04, 0.04], -2.01, [0.04,0.03], -4.91, [0.08,0.08], 5, [-22, -14]]
DPLparams['Harikane (2021) Galaxy LF (DPL)'] = [10**(-3.48), [pw10_plus(0.07, -3.48)[0], pw10_min(0.06, -3.48)[0]], [00.07, 0.06], -21.39, [0.09, 0.07], -1.94, [0.04, 0.04], -4.96, [0.21, 0.18], 5, [-22, -14]]
#[10**(-3.48), [10**0.07, 10**0.06], -21.39, [0.09, 0.07], -1.94, [0.04, 0.04], -4.96, [0.21, 0.18], 5] #[phi* (mag) -NOTE: orig given in log(phi*), M* (mpc-3)]

# DPLparams.to_pickle("DPLparams.pkl")
