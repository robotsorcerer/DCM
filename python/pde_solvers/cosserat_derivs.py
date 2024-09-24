__all__ = ["piecewise_pdes"]

__author__      = "Lekan Molu"
__copyright__   = "2022, Discrete Cosserat SoRO Analysis in Python"
__credits__     = "Tcur are None."
__license__     = "Molux Licence"
__maintainer__  = "Lekan Molu"
__email__       = "patlekno@icloud.com"
__comments__    = "This code was written under white out conditions before Christmas Eve."
__loc__         = "Marathon, Broome County, New York"
__date__        = "December 23, 2022"
__status__      = "Completed"

import copy
import time
import torch
import numpy as np
from os.path import join
from utils.cosserat_utils import *
from utils.config import *
from utils.io_utils import *
from .dynamics import compute_fwd_dynamics
from utils.matlab_utils import eps, Bundle, strcmp, isfield
from scipy.integrate import cumulative_trapezoid

from torch.linalg import pinv, norm
torch.set_default_dtype(torch.float64)


def piecewise_pdes(t, state_derivs, gv):
    device = state_derivs.device
    num_pieces  =   gv.num_pieces
    
    qd_save, qd_dot_save, qd_ddot_save = \
        gv.qd_save, gv.qd_dot_save, gv.qd_ddot_save
    global counter, tsol
    counter += 1

    dynamics =  compute_fwd_dynamics(t, state_derivs, gv)
    # genTorque   = dynamics.genTorque
    genCoriolis1 = dynamics.C1; genCoriolis2 = dynamics.C2; 
    genCableForces = dynamics.F; genDragForces = dynamics.D
    buoyancyGravTerm = dynamics.Nterm; genGraV = dynamics.G; genMasM= dynamics.M

    coriolis_drag_combo = genCoriolis1 + genCoriolis2 + genDragForces
    force_buoyance_combo = genCableForces + buoyancyGravTerm 

    # mass inertia matrix decompositions to core and pert 
    mr, mc = genMasM.shape 
    core_idx = gv.num_fast_pieces * 6 #int(0.6*mr); 
    pert_idx = mr - core_idx
    core_slice = slice(0, core_idx); pert_slice = slice(core_idx, mr)
    pert_uprt = (core_slice, pert_slice); pert_bot_left = (pert_slice, core_slice)

    q           = state_derivs[:6*num_pieces]
    z           = state_derivs[6*num_pieces:12*num_pieces]
    z_pert      = z[pert_slice];                            z_core = z[core_slice]
    q_core = q[core_slice];                                 q_pert = q[pert_slice]
    qd_core = gv.qd[core_slice];                            qd_pert = gv.qd[pert_slice]
    qd_core_prime = (gv.perturb)*gv.qd_dot[core_slice];     qd_dot_pert = gv.qd_dot[pert_slice]
    qd_core_pprime = (gv.perturb)*qd_core_prime
    q_pert_tilde = q_pert - qd_pert;                        q_core_tilde = q_core - qd_core; 

    'solve the rhs'
    s_core = force_buoyance_combo[core_slice] - coriolis_drag_combo[core_slice, core_slice] @ z_core
    s_pert = force_buoyance_combo[pert_slice] - coriolis_drag_combo[pert_slice, pert_slice] @ z_core 
         
    # # TODO (LKN): This should come from the other thread ideally!
    # z_pert_prime = gv.perturb * (z_pert - gv.z_pert_prev)/(t - gv.tprev) 

    # # store for future finite difference
    # gv.z_pert_prev = z_pert 
    # gv.tprev = t

    # torque_core = genTorque[core_slice]; 
    # torque_pert = genTorque[pert_slice]
    'solve the lhs'
    mass_compos = separate_mass_mat(genMasM, core_slice, pert_slice, pert_uprt, pert_bot_left)
    hcore = mass_compos.hcore; hcore_pert = mass_compos.hcore_pert; hpert = mass_compos.hpert

    'gains'
    Kp_core = gv.Kp;    Kq_core = gv.Kq
    Kp_pert = gv.Kp;    Kq_pert = gv.Kq

    hcore_inv = pinv(hcore);                    hpert_inv = pinv(hpert)
    Kcore_mat = Kq_core.T @ pinv(Kq_core.T @ Kq_core) @ Kp_core

    'solve for the slower controller on a slower time scale'
    pert_control = qd_dot_pert - q_pert_tilde
    z_pert_dot = hpert_inv @ (s_pert + pert_control)
    z_pert_prime = (1/gv.perturb) * z_pert_dot
    
    'solve for the core controller on a faster time scale'
    core_control = (1/gv.perturb) * hcore @ (qd_core_pprime - q_core_tilde - 2 * qd_core_prime 
                     - Kcore_mat)  - s_core + (1/gv.perturb) * hcore_pert @ z_pert_prime
    
    z_core_prime = hcore_inv @ (hcore_pert @ z_pert_prime - gv.perturb * s_core - gv.perturb * core_control)


    # # ToDo (LKN):  Shouldnt this come from the slow subsystem thread?
    # z_pert_prime_check = pinv(hpert) @ (s_pert - hpert - hpert @ hcore_inv @(s_core - core_control))

    z_point     = torch.vstack((z_core_prime, z_pert_prime))

    gv.tsol = np.vstack((gv.tsol, [t.item()]))
    gv.sol = torch.vstack((gv.sol, z_point.T))

    # append these for the fist section's qd only since it is uniform through all sections
    indices = torch.arange(6).to(device)  

    gv.qd_save = torch.vstack((qd_save, torch.index_select(gv.qd(t), 0, indices)))
    gv.qd_dot_save = torch.vstack((qd_dot_save, torch.index_select(gv.qd_dot(t), 0, indices)))

    if gv.verbose and counter%10==0: 
        # print(gv.tsol.shape, gv.sol.shape)
        print(f"Device: {device} | Num Steps: {counter} | t: {t:.4f} ||z_point||: {norm(z_point, ord='fro'):.8f}")

    return (t, z_point)