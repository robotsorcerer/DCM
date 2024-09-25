__all__ = ["piecewise_fast_pdes", "piecewise_slow_pdes"]

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

import os
import torch
import numpy as np
from os.path import join
from utils.cosserat_utils import *
from utils.config import *
from utils.io_utils import *
from .dynamics import compute_fwd_dynamics

from torch.linalg import pinv, norm
torch.set_default_dtype(torch.float64)

fcounter = 0


def piecewise_slow_pdes(t, state_derivs, gv):
    device = state_derivs.device
    num_pieces  =   gv.num_pieces
    
    # qd_save, qd_dot_save, qd_ddot_save = \
    #     gv.qd_save, gv.qd_dot_save, gv.qd_ddot_save
    global counter, tsol
    counter += 1

    # gv.num_pieces = gv.num_slow_pieces # for dynamics show
    gv.num_part_pieces = gv.num_slow_pieces
    # we compute the dynamics once for slow and fast:: fast reads this from HDD dump
    dynamics =  compute_fwd_dynamics(t, state_derivs, gv)
    # genTorque   = dynamics.genTorque
    genCoriolis1 = dynamics.C1; genCoriolis2 = dynamics.C2; 
    genCableForces = dynamics.F; genDragForces = dynamics.D
    buoyancyGravTerm = dynamics.Nterm; genMasM= dynamics.M

    coriolis_drag_combo = genCoriolis1 + genCoriolis2 + genDragForces
    force_buoyance_combo = genCableForces + buoyancyGravTerm 

    # mass inertia matrix decompositions to core and pert 
    mr, mc = genMasM.shape 
    core_idx = gv.num_fast_pieces * 6 #int(0.6*mr); 
    pert_idx = gv.num_slow_pieces * 6
    core_slice = slice(0, core_idx); 
    pert_slice = slice(core_idx, mr)
    pert_uprt  = (core_slice, pert_slice); pert_bot_left = (pert_slice, core_slice)

    q_pert      = state_derivs[:pert_idx]                      
    z_pert      = state_derivs[pert_idx:]                         
    qd_pert     = gv.qd_slow(t)    
    q_pert_tilde = q_pert - qd_pert; 
    qd_dot_pert = gv.qd_dot_slow(t)    

    'solve for the slower controller on a slower time scale'
    pert_control = qd_dot_pert - q_pert_tilde  

    'solve the rhs'
    s_pert = (force_buoyance_combo[pert_slice] - coriolis_drag_combo[pert_slice, pert_slice] @ z_pert).to(device)
         
    'solve the lhs'
    mass_compos = separate_mass_mat(genMasM, core_slice, pert_slice, pert_uprt, pert_bot_left)
    hpert = mass_compos.hpert.to(device);              hpert_inv = pinv(hpert).to(device)

    z_pert_dot = hpert_inv @ (s_pert + pert_control)

    z_point     = torch.vstack((z_pert, z_pert_dot))

    dumpee = {'s_pert': s_pert, 't': t, 'z_pert_dot': z_pert_dot, \
              'force_buoyance_combo': force_buoyance_combo, 'mass_compos': mass_compos, \
                'coriolis_drag_combo': coriolis_drag_combo}
    
    torch.save(dumpee, join(gv.save_dir, 'slow_dyna_dump.pt'))

    gv.tsol = np.vstack((gv.tsol, [t.item()]))
    gv.sol_slow = torch.vstack((gv.sol_slow, z_point.T))

    # # append these for the fist section's qd only since it is uniform through all sections
    # indices = torch.arange(6).to(device)  

    # gv.qd_save = torch.vstack((qd_save, torch.index_select(gv.qd(t), 0, indices)))
    # gv.qd_dot_save = torch.vstack((qd_dot_save, torch.index_select(gv.qd_dot(t), 0, indices)))

    if gv.verbose and counter%10==0: 
        print(f"[Slow PDEs]::Device: {device} | Num Steps: {counter} | t: {t:.4f} ||z_point||: {norm(z_point, ord='fro'):.8f}")

    return (t, z_point)



# def piecewise_slow_pdes(t, state_derivs, gv):
#     device = state_derivs.device
#     num_pieces  =   gv.num_pieces
    
#     # qd_save, qd_dot_save, qd_ddot_save = \
#     #     gv.qd_save, gv.qd_dot_save, gv.qd_ddot_save
#     global counter, tsol
#     counter += 1

#     # gv.num_pieces = gv.num_slow_pieces # for dynamics show
#     gv.num_part_pieces = gv.num_slow_pieces
#     # we compute the dynamics once for slow and fast:: fast reads this from HDD dump
#     dynamics =  compute_fwd_dynamics(t, state_derivs, gv)
#     # genTorque   = dynamics.genTorque
#     genCoriolis1 = dynamics.C1; genCoriolis2 = dynamics.C2; 
#     genCableForces = dynamics.F; genDragForces = dynamics.D
#     buoyancyGravTerm = dynamics.Nterm; genMasM= dynamics.M

#     coriolis_drag_combo = genCoriolis1 + genCoriolis2 + genDragForces
#     force_buoyance_combo = genCableForces + buoyancyGravTerm 

#     # mass inertia matrix decompositions to core and pert 
#     mr, mc = genMasM.shape 
#     core_idx = gv.num_fast_pieces * 6 #int(0.6*mr); 
#     pert_idx = gv.num_slow_pieces * 6
#     core_slice = slice(0, core_idx); 
#     pert_slice = slice(core_idx, mr)
#     pert_uprt  = (core_slice, pert_slice); pert_bot_left = (pert_slice, core_slice)

#     q_pert      = state_derivs[:pert_idx]                      
#     z_pert      = state_derivs[pert_idx:]                         
#     qd_pert     = gv.qd_slow(t)    
#     q_pert_tilde = q_pert - qd_pert; 
#     qd_dot_pert = gv.qd_dot_slow(t)    

#     'solve for the slower controller on a slower time scale'
#     pert_control = qd_dot_pert - q_pert_tilde  

#     'solve the rhs'
#     s_pert = force_buoyance_combo[pert_slice] - coriolis_drag_combo[pert_slice, pert_slice] @ z_pert
         
#     'solve the lhs'
#     mass_compos = separate_mass_mat(genMasM, core_slice, pert_slice, pert_uprt, pert_bot_left)
#     hpert = mass_compos.hpert;              hpert_inv = pinv(hpert)

#     z_pert_dot = hpert_inv @ (s_pert + pert_control)

#     z_point     = torch.vstack((z_pert, z_pert_dot))

#     dumpee = {'s_pert': s_pert, 't': t, 'z_pert_dot': z_pert_dot, \
#               'force_buoyance_combo': force_buoyance_combo, 'mass_compos': mass_compos, \
#                 'coriolis_drag_combo': coriolis_drag_combo}
    
#     torch.save(dumpee, join(gv.save_dir, 'slow_dyna_dump.pt'))

#     gv.tsol = np.vstack((gv.tsol, [t.item()]))
#     gv.sol_slow = torch.vstack((gv.sol_slow, z_point.T))

#     # # append these for the fist section's qd only since it is uniform through all sections
#     # indices = torch.arange(6).to(device)  

#     # gv.qd_save = torch.vstack((qd_save, torch.index_select(gv.qd(t), 0, indices)))
#     # gv.qd_dot_save = torch.vstack((qd_dot_save, torch.index_select(gv.qd_dot(t), 0, indices)))

#     if gv.verbose and counter%10==0: 
#         print(f"[Slow PDEs]::Device: {device} | Num Steps: {counter} | t: {t:.4f} ||z_point||: {norm(z_point, ord='fro'):.8f}")

#     return (t, z_point)

def piecewise_fast_pdes(t, state_derivs, gv):
    t = t/gv.perturb  # move to a faster time scale
    device = state_derivs.device
    num_pieces  =   gv.num_pieces
    
    # qd_save, qd_dot_save, qd_ddot_save = \
    #     gv.qd_save, gv.qd_dot_save, gv.qd_ddot_save
    global fcounter, tsol
    fcounter += 1
    
    gv.num_part_pieces = gv.num_fast_pieces
    # we compute the dynamics once for slow and fast:: fast reads this from HDD dump
    dynamics =  compute_fwd_dynamics(t, state_derivs, gv)
    # genTorque   = dynamics.genTorque
    genCoriolis1 = dynamics.C1; genCoriolis2 = dynamics.C2; 
    genCableForces = dynamics.F; genDragForces = dynamics.D
    buoyancyGravTerm = dynamics.Nterm; genMasM= dynamics.M

    coriolis_drag_combo = genCoriolis1 + genCoriolis2 + genDragForces
    force_buoyance_combo = genCableForces + buoyancyGravTerm 
    coriolis_drag_combo = genCoriolis1 + genCoriolis2 + genDragForces
    force_buoyance_combo = genCableForces + buoyancyGravTerm 
    # read dynamics at time t from HDD 
    obj_load = torch.load(join(gv.save_dir, 'slow_dyna_dump.pt'))
    z_pert_dot = obj_load['z_pert_dot']
    z_pert_prime = gv.perturb * z_pert_dot

    # force_buoyance_combo = obj_load['force_buoyance_combo']
    # coriolis_drag_combo = obj_load['coriolis_drag_combo']

    # mass inertia matrix decompositions to core and pert 
    mr, mc = genMasM.shape 
    core_idx = gv.num_fast_pieces * 6  
    core_slice = slice(0, core_idx); pert_slice = slice(core_idx, mr)
    pert_uprt = (core_slice, pert_slice); pert_bot_left = (pert_slice, core_slice)

    q_core      = state_derivs[:core_idx]                      
    z_core      = state_derivs[core_idx:]                         
    qd_core     = gv.qd_fast(t)    
    q_core_tilde = q_core - qd_core; 
    qd_prime_core = gv.perturb*gv.qd_dot_fast(t)    
    qd_pprime_core = gv.perturb**2*gv.qd_ddot_fast(t)   

    'solve the rhs of dynamics'
    s_core = force_buoyance_combo[core_slice] - coriolis_drag_combo[core_slice, core_slice] @ z_core

    'get hcore matrices'
    mass_compos = separate_mass_mat(genMasM, core_slice, pert_slice, pert_uprt, pert_bot_left)
    hcore = mass_compos.hcore; hcore_pert = mass_compos.hcore_pert; hcore_inv = pinv(hcore)

    'solve for control'
    gain_term = gv.Kq.T.matmul(pinv(gv.Kq.matmul(gv.Kq.T))).matmul(gv.Kp).matmul(q_core_tilde)
    u_core = (1/gv.perturb) * hcore_pert.matmul(z_pert_prime) - s_core 
    u_core += (1/gv.perturb) * hcore.matmul( qd_pprime_core - q_core_tilde - 2 * qd_prime_core - gain_term)
                
    z_core_prime = hcore_inv @ (gv.perturb * (s_core + u_core) - hcore_pert.matmul(z_pert_prime))


    z_point     = torch.vstack((z_core, z_core_prime))

    gv.tsol = np.vstack((gv.tsol, [t.item()]))
    gv.sol_fast = torch.vstack((gv.sol_fast, z_point.T))

    # append these for the fist section's qd only since it is uniform through all sections
    # indices = torch.arange(6).to(device)  

    # gv.qd_save = torch.vstack((qd_save, torch.index_select(gv.qd(t), 0, indices)))
    # gv.qd_dot_save = torch.vstack((qd_dot_save, torch.index_select(gv.qd_dot(t), 0, indices)))

    if gv.verbose and fcounter%10==0: 
        # print(gv.tsol.shape, gv.sol.shape)
        print(f"[Fast PDEs]::Device: {device} | Num Steps: {fcounter} | t: {t:.4f} ||z_point||: {norm(z_point, ord='fro'):.8f}")

    return (t, z_point)