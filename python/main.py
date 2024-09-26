__author__      = "Lekan Molu"
__maintainer__  = "Lekan Molu"
__license__     = "Microsoft Licence"
__copyright__   = "2022, Discrete Cosserat SoRO Analysis in Python"
__credits__     = "There are None."
__email__       = "patlekno@icloud.com"
__date__        = "September 24, 2024"
__status__      = "Completed"

import os
import logging
import time, sys
import threading 
import numpy as np
from math import pi
from datetime import datetime
from torch.linalg import pinv, norm
import torch.multiprocessing as mp

import torch
torch.set_default_dtype(torch.float64)

from os.path import abspath, join, dirname
sys.path.append(dirname(dirname(abspath(__file__))))

from utils import *
from pde_solvers import *
from ode_solvers import *

from absl import app, flags
import matplotlib as mpl
import matplotlib.pyplot as plt

# parser = argparse.ArgumentParser('Cosserat Soft Arm Forward and Inverse Model')
flags.DEFINE_bool('verbose', default=True, help="run in verbose print mode.")
flags.DEFINE_integer('num_pieces', default=6, lower_bound=1, upper_bound=10, help="Number of DC PCS.")
flags.DEFINE_integer('num_slow_pieces', default=2, lower_bound=1, upper_bound=10, help="Number of DC PCS.")
flags.DEFINE_integer('num_fast_pieces', default=4, lower_bound=1, upper_bound=10, help="Number of DC PCS.")
flags.DEFINE_integer('t_time', default=30, lower_bound=10, upper_bound=40, help='length of time (X1000s) for simulation')
flags.DEFINE_bool('with_cable', True, help='control with cable-driven  dynamics')
flags.DEFINE_bool('with_drag', True, help= 'control underwater with drag forces?')
flags.DEFINE_string('resume', None, help= 'Resume from a previously checkpointed | Provide full path to model')
flags.DEFINE_bool('with_grav', True, help= 'Gravity compensation.')
flags.DEFINE_float('tip_load', 10, help="Tip load in Newtons")
flags.DEFINE_float('rtol', 1e-7, help="Relative tolerance for optimization/integrator.")
flags.DEFINE_float('atol', 1e-9, help="Absolute tolerance for optimization/integrator.")
flags.DEFINE_float('desired_strain', 0.5, help="Desired strain noticeable along Z")
flags.DEFINE_float('gain_prop', 4.5, help="Proportional gain for PD/PID controller")
flags.DEFINE_float('backstep_p', 4.3, help="Kp gain for backstep controller")
flags.DEFINE_float('backstep_d', 4.3, help="Kd gain for backstep controller")
flags.DEFINE_float('gain_deriv', 5.5, help="Derivative gain for PD/PID controller")
flags.DEFINE_float('gain_integ', 1.2, help="Integral gain for PID controller")
flags.DEFINE_float('perturb', 0.01, help="singularly perturbed parameter for fast time scale")
flags.DEFINE_string('controller', "spt", help="'spt | PD | PID'")
flags.DEFINE_string('reference', "setpoint", help="'setpoint or trajectory tracking?', 'setpoint | traktrack'")
flags.DEFINE_string('integrator', default="fehlberg2", help="'felhberg2 | dopri8 | dopri5 | euler | midpoint | rk4'")

FLAGS = flags.FLAGS

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logging.getLogger('matplotlib.font_manager').disabled = True # Turn off pyplot's spurious dumps on screen
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print()

def main(argv):
    del argv  

    #========================== Global Params ==============================================================#
    logger.info('>>====================================Time-advancing=====================================<<')
    logger.info(f'Num of sections: {num_sections}, Num of pieces: {FLAGS.num_pieces}')
    logger.info(f'Params:: Controller: {FLAGS.controller} | Underwater: {FLAGS.with_drag} | Cable-driven: {FLAGS.with_cable}')
    logger.info(f'Params:: Tip load: {FLAGS.tip_load}N | Kp: {FLAGS.gain_prop} | KD: {FLAGS.gain_deriv}')
    #========================================================================================================#

    #===================================== Save galleries===================================================#
    data_dir = join(f"/opt/SoRo{FLAGS.controller.upper()}") 
    os.makedirs(data_dir) if not os.path.exists(data_dir) else None 

    fname = datetime.strftime(datetime.now(), '%m%d%y_%H_%M_%S')
    if FLAGS.with_drag:
        fname = fname + "_drag"
    if FLAGS.with_cable:
        fname = fname + "_cable"
    if FLAGS.with_grav:
        fname = fname + "_grav"
    fname += f"_{FLAGS.num_pieces}pcs_{FLAGS.tip_load}N_{FLAGS.controller}.npz"

    logger.info(f"fname:  {join(data_dir, fname)}")
    #========================================================================================================#

    nsol        = int(FLAGS.t_time*10**3)             # t_time solution(s) every millisecond
    tspan       = torch.linspace(0,FLAGS.t_time,nsol).to(device)  # [s] time
    g           = torch.zeros((4*nsol,4*num_sections*FLAGS.num_pieces)).to(device)
    eta         = torch.zeros((6*nsol,num_sections*FLAGS.num_pieces)).to(device)

    nstep       = 1
    tic         = time.time()

    #-------------------------------------------------------------------------
    # Strain initial conditions
    xi_0          = torch.tensor(([[0., 0., 0., 1.0, 0., 0.]]), dtype=torch.float64) #.float()
    xidot_0       = torch.tensor(([[0., 0., 0., 1.0, 0., 0.]]), dtype=torch.float64) #.float()  #torch.zeros((1, 6))  

    state_derivs  = torch.hstack((torch.tile(xi_0,[1,FLAGS.num_pieces]), torch.tile(xidot_0,[1,FLAGS.num_pieces]) )).T.to(device)
    slow_state_derivs  = torch.hstack((torch.tile(xi_0,[1,FLAGS.num_slow_pieces]), torch.tile(xidot_0,[1,FLAGS.num_slow_pieces]) )).T #.to(device)
    fast_state_derivs  = torch.hstack((torch.tile(xi_0,[1,FLAGS.num_fast_pieces]), torch.tile(xidot_0,[1,FLAGS.num_fast_pieces]) )).T.to(device)

    #=================== Update global options ============================================================#
    gv_others     = {"tspan": tspan, "nsol": nsol, "nstep": nstep, "g": g, "eta": eta, "tic": tic,
                     "data_dir": data_dir, "fname": fname, "sol": torch.zeros([1, 6*2*FLAGS.num_pieces]).to(device),
                     "sol_slow": torch.zeros([1, 6*2*FLAGS.num_slow_pieces]), "save_dir": data_dir, 
                     "sol_fast": torch.zeros([1, 6*2*FLAGS.num_fast_pieces]).to(device),
                     "controller": FLAGS.controller, "with_cable": FLAGS.with_cable, "desired_strain": FLAGS.desired_strain, 
                     "num_fast_pieces": FLAGS.num_fast_pieces, "num_pieces": FLAGS.num_pieces, "num_slow_pieces": FLAGS.num_slow_pieces,
                     "with_drag": FLAGS.with_drag, "verbose": FLAGS.verbose, "state_derivs": state_derivs, 
                     "slow_state_derivs": slow_state_derivs, "fast_state_derivs": fast_state_derivs, 
                     "gain_deriv": FLAGS.gain_deriv, "gain_prop":  FLAGS.gain_prop, "tip_load": FLAGS.tip_load, 
                     "gain_integ":  FLAGS.gain_integ, "with_grav":  FLAGS.with_grav} 
    global gv
    gv.update(gv_others)
    gv = Bundle(gv)
    #=======================================================================================================#

    if os.path.exists(join(gv.save_dir, 'slow_dyna_dump.pt')):
        os.remove(join(gv.save_dir, 'slow_dyna_dump.pt'))
        
    qd            = torch.tensor([[0, 0, 0, 1, FLAGS.desired_strain, 0]])
    qd_dot        = torch.tensor([[0, 0, 0, 1, FLAGS.desired_strain, 0]]); 
    qd_ddot        = torch.tensor([[0, 0, 0, 1, FLAGS.desired_strain, 0]])

    gv.qd_slow = lambda t: torch.tile(qd.T, (FLAGS.num_slow_pieces, 1))
    gv.qd_dot_slow = lambda t: torch.tile(qd_dot.T, (FLAGS.num_slow_pieces, 1))
    gv.qd_ddot_slow = lambda t: torch.tile(qd_ddot.T, (FLAGS.num_slow_pieces, 1))

    gv.qd_fast = lambda t: torch.tile(qd.T.to(t.device), (FLAGS.num_fast_pieces, 1))
    gv.qd_dot_fast = lambda t: torch.tile(qd_dot.T.to(t.device), (FLAGS.num_fast_pieces, 1))
    gv.qd_ddot_fast = lambda t: torch.tile(qd_ddot.T.to(t.device), (FLAGS.num_fast_pieces, 1))
            
    if FLAGS.controller: 
        # assume setpoint by default 
        # track unit linear and angular strains that is constant in the +y direction in addition to a varying tip load        
        if strcmp(FLAGS.reference.lower(), 'trajtrack'):
            from math import sin, cos
            # track linear and angular strains that is sinusoidal in the +y direction in addition to a varying tip load throughout the soft material body
            gv.qd_slow = lambda t: torch.tile(torch.tensor([[0, 0, 0, 1,  sin(FLAGS.desired_strain*10*t), 0]]).T, (FLAGS.num_slow_pieces, 1))
            gv.qd_dot_slow = lambda t: torch.tile(torch.tensor([[0, 0, 0, 1, 10*cos(FLAGS.desired_strain*10*t), 0]]).T, (FLAGS.num_slow_pieces, 1))
            gv.qd_ddot_slow = lambda t: torch.tile(torch.tensor([[0, 0, 0, 1, -100*sin(FLAGS.desired_strain*10*t),0]]).T, (FLAGS.num_slow_pieces, 1))

            gv.qd_fast = lambda t: torch.tile(torch.tensor([[0, 0, 0, 1,  sin(FLAGS.desired_strain*10*t), 0]]).T.to(t.device), (FLAGS.num_fast_pieces, 1))
            gv.qd_dot_fast = lambda t: torch.tile(torch.tensor([[0, 0, 0, 1, 10*cos(FLAGS.desired_strain*10*t), 0]]).T.to(t.device), (FLAGS.num_fast_pieces, 1))
            gv.qd_ddot_fast = lambda t: torch.tile(torch.tensor([[0, 0, 0, 1, -100*sin(FLAGS.desired_strain*10*t),0]]).T.to(t.device), (FLAGS.num_fast_pieces, 1))

        # specify the controller gains if PD or PID
        if strcmp(FLAGS.controller.lower(), 'pd') or strcmp(FLAGS.controller.lower(), 'pid'):
            gv.Kp    = FLAGS.gain_prop*torch.eye(6*FLAGS.num_pieces).to(state_derivs.device)
            gv.Kd    = FLAGS.gain_deriv*torch.eye(6*FLAGS.num_pieces).to(state_derivs.device)
            gv.Ki    = FLAGS.gain_integ*torch.eye(6*FLAGS.num_pieces).to(state_derivs.device)
            
        elif strcmp(FLAGS.controller.lower(), "spt"):
            gv.num_fast_pieces = FLAGS.num_fast_pieces
            gv.num_slow_pieces = FLAGS.num_slow_pieces
            gv.perturb = FLAGS.perturb 

            gv.Kp    = FLAGS.backstep_p*torch.eye(6*FLAGS.num_fast_pieces) #.to(state_derivs.device)
            gv.Kq    = FLAGS.backstep_d*torch.eye(6*FLAGS.num_fast_pieces) #.to(state_derivs.device)        

            slow_cl_derivs = lambda t, state_derivs: piecewise_slow_pdes(t, state_derivs.cpu(), gv)
            # time.sleep(4) # wait to dump first z_pert_prime before running fast loop
            gv.Kp    = gv.Kp.to(tspan.device)
            gv.Kq    = gv.Kq.to(tspan.device)    
            fast_cl_derivs = lambda t, state_derivs: piecewise_fast_pdes(t/FLAGS.perturb, state_derivs, gv)

            tslow = tspan.cpu()            
            sol_slow = odeint(slow_cl_derivs, slow_state_derivs, tslow, method=FLAGS.integrator,rtol=FLAGS.rtol, atol=FLAGS.atol)
            # slow_thread = threading.Thread( target=lambda: odeint(slow_cl_derivs, slow_state_derivs, tslow, method=FLAGS.integrator,rtol=FLAGS.rtol, atol=FLAGS.atol) )
            # slow_thread.daemon = True
            # slow_thread.start()
            # fast_thread = mp.Process(target=lambda: odeint(fast_cl_derivs, fast_state_derivs, tspan, method=FLAGS.integrator,rtol=FLAGS.rtol, atol=FLAGS.atol) )
            # fast_thread.start()
            # sol_fast = fast_thread.join()
            sol_fast = odeint(fast_cl_derivs, fast_state_derivs, tspan, method=FLAGS.integrator,rtol=FLAGS.rtol, atol=FLAGS.atol)
    else:          
        raise ValueError("Unknown simulation type.")
        
    # sol_fast = fast_thread.join()
    if FLAGS.verbose:
        logger.info(f"slow dynamics solution for cur session: {sol_slow.shape}")
        logger.info(f"fast dynamics solution for cur session: {sol_fast.shape}")
        # logger.info(f"fast dynamics solution for cur session:  {fast_thread.join()}")

    toc =  time.time()
    fname_final = join(gv.data_dir, gv.fname.split(".npz")[0]+"_final.npz")
    print(f"Post-processing {fname_final} |  {(toc-tic):.4f} secs or {((toc-tic)/60):.4f} minutes")
    print("\n\n=======Starting New Session\n\n")

    toc = time.time()
    np.savez_compressed(fname_final, 
                    fname = gv.fname.split(".npz")[0]+"_final.npz",
                    slow_solution=sol_slow.detach().cpu().numpy(), 
                    fast_solution=sol_fast.detach().cpu().numpy(), 
                    soltime=gv.tsol,
                    runtime=toc-tic, 
                    with_drag=FLAGS.with_drag, 
                    with_cable=FLAGS.with_cable,  
                    gravity=FLAGS.with_grav, 
                    num_pieces=FLAGS.num_pieces,
                    num_slow_pieces=FLAGS.num_slow_pieces,
                    num_fast_pieces=FLAGS.num_fast_pieces,
                    num_sections=num_sections,
                    gain_prop=FLAGS.gain_prop, 
                    gain_deriv=FLAGS.gain_deriv, 
                    gain_integ=FLAGS.gain_integ,  
                    backstep_p=FLAGS.backstep_p, 
                    backstep_d=FLAGS.backstep_d, 
                    tip_load=FLAGS.tip_load, 
                    controller=FLAGS.controller, 
                    desired_strain=gv.desired_strain,
                    qd_slow=gv.qd_slow.detach().cpu().numpy(), 
                    qd_dot_slow=gv.qd_dot_slow.detach().cpu().numpy(), 
                    qd_ddot_slow=gv.qd_ddot_slow.detach().cpu().numpy(), 
                    qd_fast=gv.qd_fast.detach().cpu().numpy(), 
                    qd_dot_fast=gv.qd_dot_fast.detach().cpu().numpy(), 
                    qd_ddot_fast=gv.qd_ddot_fast.detach().cpu().numpy(), 
                    )

if __name__ == "__main__":
    app.run(main)