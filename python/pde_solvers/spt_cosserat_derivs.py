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

    # mass inertia matrix decompositions to fast and slow 
    mr, mc = genMasM.shape 
    fast_idx = gv.num_fast_pieces * 6 #int(0.6*mr); 
    slow_idx = gv.num_slow_pieces * 6
    fast_slice = slice(0, fast_idx); 
    slow_slice = slice(fast_idx, mr)
    slow_uprt  = (fast_slice, slow_slice); slow_bot_left = (slow_slice, fast_slice)

    q_slow      = state_derivs[:slow_idx]                      
    z_slow      = state_derivs[slow_idx:]                         
    qd_slow     = gv.qd_slow(t)    
    q_slow_tilde = q_slow - qd_slow; 
    qd_dot_slow = gv.qd_dot_slow(t)    

    'solve for the slower controller on a slower time scale'
    slow_control = qd_dot_slow - q_slow_tilde  

    'solve the rhs'
    s_slow = (force_buoyance_combo[slow_slice] - coriolis_drag_combo[slow_slice, slow_slice] @ z_slow).to(device)
         
    'solve the lhs'
    mass_compos = separate_mass_mat(genMasM, fast_slice, slow_slice, slow_uprt, slow_bot_left)
    hslow = mass_compos.hslow.to(device);              hslow_inv = pinv(hslow).to(device)

    z_slow_dot = hslow_inv @ (s_slow + slow_control)

    z_point     = torch.vstack((z_slow, z_slow_dot))

    dumpee = {'s_slow': s_slow, 't': t, 'z_slow_dot': z_slow_dot, \
              'force_buoyance_combo': force_buoyance_combo, 'mass_compos': mass_compos, \
                'coriolis_drag_combo': coriolis_drag_combo}
    
    torch.save(dumpee, join(gv.save_dir, 'slow_dyna_dump.pt'))

    gv.tsol_slow = np.vstack((gv.tsol_slow, [t.item()]))
    gv.sol_slow = torch.vstack((gv.sol_slow, z_point.T))

    # # append these for the fist section's qd only since it is uniform through all sections
    # indices = torch.arange(6).to(device)  

    # gv.qd_save = torch.vstack((qd_save, torch.index_select(gv.qd(t), 0, indices)))
    # gv.qd_dot_save = torch.vstack((qd_dot_save, torch.index_select(gv.qd_dot(t), 0, indices)))

    if gv.verbose and counter%10==0: 
        print(f"[Slow PDEs]::Device: {device} | Num Steps: {counter} | t: {t:.4f} ||z_point||: {norm(z_point, ord='fro'):.8f}")

    return (t, z_point)


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
    z_slow_dot = obj_load['z_slow_dot'].to(device)
    z_slow_prime = gv.perturb * z_slow_dot

    # force_buoyance_combo = obj_load['force_buoyance_combo']
    # coriolis_drag_combo = obj_load['coriolis_drag_combo']

    # mass inertia matrix decompositions to fast and slow 
    mr, mc = genMasM.shape 
    fast_idx = gv.num_fast_pieces * 6  
    fast_slice = slice(0, fast_idx); slow_slice = slice(fast_idx, mr)
    slow_uprt = (fast_slice, slow_slice); slow_bot_left = (slow_slice, fast_slice)

    q_fast      = state_derivs[:fast_idx]                      
    z_fast      = state_derivs[fast_idx:]                         
    qd_fast     = gv.qd_fast(t)    
    q_fast_tilde = q_fast - qd_fast; 
    qd_prime_fast = gv.perturb*gv.qd_dot_fast(t)    
    q_prime_fast_tilde = z_fast - qd_prime_fast  
    qd_pprime_fast = gv.perturb**2*gv.qd_ddot_fast(t)   

    'solve the rhs of dynamics'
    s_fast = force_buoyance_combo[fast_slice] - coriolis_drag_combo[fast_slice, fast_slice] @ z_fast

    'get hfast matrices'
    mass_compos = separate_mass_mat(genMasM, fast_slice, slow_slice, slow_uprt, slow_bot_left)
    hfast = mass_compos.hfast; hfast_slow = mass_compos.hfast_slow; hfast_inv = pinv(hfast)

    'solve for control'
    gain_term = gv.Kq.T.matmul(pinv(gv.Kq.matmul(gv.Kq.T))).matmul(gv.Kp).matmul(q_fast_tilde)
    u_fast = (1/gv.perturb) * hfast_slow.matmul(z_slow_prime) - s_fast 
    # u_fast += (1/gv.perturb) * hfast.matmul( qd_pprime_fast - q_fast_tilde - 2 * q_prime_fast_tilde - gain_term)
    u_fast += (1/gv.perturb) * hfast.matmul( qd_pprime_fast - q_fast_tilde - 2 *( q_prime_fast_tilde-q_fast_tilde) - gain_term)
                
    z_fast_prime = hfast_inv @ (gv.perturb * (s_fast + u_fast) - hfast_slow.matmul(z_slow_prime))


    z_point     = torch.vstack((z_fast, z_fast_prime))

    gv.tsol_fast = np.vstack((gv.tsol_fast, [t.item()]))
    gv.sol_fast = torch.vstack((gv.sol_fast, z_point.T))

    # append these for the fist section's qd only since it is uniform through all sections
    # indices = torch.arange(6).to(device)  

    # gv.qd_save = torch.vstack((qd_save, torch.index_select(gv.qd(t), 0, indices)))
    # gv.qd_dot_save = torch.vstack((qd_dot_save, torch.index_select(gv.qd_dot(t), 0, indices)))

    if gv.verbose and fcounter%10==0: 
        # print(gv.tsol.shape, gv.sol.shape)
        print(f"[Fast PDEs]::Device: {device} | Num Steps: {fcounter} | t: {t:.4f} ||z_point||: {norm(z_point, ord='fro'):.8f}")

    return (t, z_point)