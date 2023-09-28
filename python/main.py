__author__      = "Lekan Molu"
__maintainer__  = "Lekan Molu"
__license__     = "Microsoft Licence"
__copyright__   = "2022, Discrete Cosserat SoRO Analysis in Python"
__credits__     = "There are None."
__email__       = "patlekno@icloud.com"
__comments__    = "This code was written under white-out conditions before Christmas Eve."
__loc__         = "Marathon, Broome County, New York"
__date__        = "December 23, 2022"
__status__      = "Completed"

import os
import logging
import time, sys
import numpy as np
from math import pi
from datetime import datetime

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
flags.DEFINE_float('backstep_prop', 4.3, help="Proportional gain for backstep controller")
flags.DEFINE_float('gain_deriv', 5.5, help="Derivative gain for PD/PID controller")
flags.DEFINE_float('gain_integ', 1.2, help="Integral gain for PID controller")
flags.DEFINE_string('controller', "pd", help="'spt | PD | PID'")
flags.DEFINE_string('reference', "setpoint", help="'setpoint or trajectory tracking?', 'setpoint | traktrack'")
flags.DEFINE_string('integrator', default="fehlberg2", help="'felhberg2 | dopri8 | dopri5 | euler | midpoint | rk4'")

FLAGS = flags.FLAGS

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logging.getLogger('matplotlib.font_manager').disabled = True # Turn off pyplot's spurious dumps on screen
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print()

def main(argv):
    '''Entry point for running one selfplay game.'''
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
    # configuration and strain field
    g           = torch.zeros((4*nsol,4*num_sections*FLAGS.num_pieces)).to(device)
    eta         = torch.zeros((6*nsol,num_sections*FLAGS.num_pieces)).to(device)
    nstep       = 1
    tic         = time.time()

    #-------------------------------------------------------------------------
    # Strain initial conditions
    xi_0          = torch.tensor(([[0, 0, 0, 1, 0, 0]])) 
    xidot_0       = torch.zeros((1, 6)) 
    state_derivs  = torch.hstack((torch.tile(xi_0,[1,FLAGS.num_pieces]), torch.tile(xidot_0,[1,FLAGS.num_pieces]) )).T.to(device)

    #=================== Update global options ============================================================#
    gv_others     = {"tspan": tspan, "nsol": nsol, "nstep": nstep, "g": g, "eta": eta, "tic": tic,
                     "data_dir": data_dir, "fname": fname, "state_derivs": state_derivs, "sol": torch.zeros([1, 6*2*FLAGS.num_pieces]).to(device),
                     "controller": FLAGS.controller, "with_cable": FLAGS.with_cable, "desired_strain": FLAGS.desired_strain,
                     "with_drag": FLAGS.with_drag, "verbose": FLAGS.verbose, "num_pieces": FLAGS.num_pieces,
                     "gain_deriv": FLAGS.gain_deriv, "gain_prop":  FLAGS.gain_prop, "tip_load": FLAGS.tip_load, 
                     "gain_integ":  FLAGS.gain_integ, "with_grav":  FLAGS.with_grav} 
    global gv
    gv.update(gv_others)
    gv = Bundle(gv)
    #=======================================================================================================#

    # if FLAGS.resume:
    #     checkpoint_dfname = FLAGS.resume 
    #     bundled = load_file(checkpoint_dfname)
    #     last_time, last_sol = bundled.tsol, bundled.sol 
        
    if FLAGS.controller: 
        # assume setpoint by default 
        # track a unit linear and angular strains that is constant in the +y direction in addition to a varying tip load
        gv.qd         = lambda t: torch.tile(torch.tensor([[0, 0, 0, 1, FLAGS.desired_strain, 0]]).T.to(t.device), (FLAGS.num_pieces, 1))
        gv.qd_dot     = lambda t: torch.tile(torch.tensor([[0, 0, 0, 1, FLAGS.desired_strain, 0]]).T.to(t.device), (FLAGS.num_pieces, 1))
        gv.qd_ddot    = lambda t: torch.tile(torch.tensor([[0, 0, 0, 1, FLAGS.desired_strain, 0]]).T.to(t.device), (FLAGS.num_pieces, 1)) 
        if strcmp(FLAGS.reference.lower(), 'trajtrack'):
            from math import sin, cos
            # track linear and angular strains that is sinusoidal in the +y direction in addition to a varying tip load throughout the soft material body
            gv.qd = lambda t: torch.tile(torch.tensor([[0, 0, 0, 1,  sin(FLAGS.desired_strain*10*t), 0]]).T.to(t.device), (FLAGS.num_pieces, 1))
            gv.qd_dot = lambda t: torch.tile(torch.tensor([[0, 0, 0, 1, 10*cos(FLAGS.desired_strain*10*t), 0]]).T.to(t.device), (FLAGS.num_pieces, 1))
            gv.qd_ddot = lambda t: torch.tile(torch.tensor([[0, 0, 0, 1, -100*sin(FLAGS.desired_strain*10*t),0]]).T.to(t.device), (FLAGS.num_pieces, 1))

        # specify the controller gains if PD or PID
        if strcmp(FLAGS.controller.lower(), 'pd') or strcmp(FLAGS.controller.lower(), 'pid'):
            gv.Kp    = FLAGS.gain_prop*torch.eye(6*FLAGS.num_pieces).to(state_derivs.device)
            gv.Kd    = FLAGS.gain_deriv*torch.eye(6*FLAGS.num_pieces).to(state_derivs.device)
            gv.Ki    = FLAGS.gain_integ*torch.eye(6*FLAGS.num_pieces).to(state_derivs.device)
        elif strcmp(FLAGS.controller.lower(), "spt"):
            gv.Kp   = FLAGS.backstep_prop*torch.eye(6*FLAGS.num_pieces).to(state_derivs.device)
        
        cl_derivs = lambda t, state_derivs: piecewise_pdes(t, state_derivs, gv)
        sol = odeint(cl_derivs, state_derivs, tspan, method=FLAGS.integrator,rtol=FLAGS.rtol, atol=FLAGS.atol)
    else:          
        fwd_dynamics = lambda t: piecewise_pdes(t, state_derivs, gv)
        sol = odeint(piecewise_pdes, state_derivs, tspan, method=FLAGS.integrator)
        
    if FLAGS.verbose:
        logger.info(f"solution for cur session: {sol}")

    toc =  time.time()
    print(f"Post-processing:  {(toc-tic):.4f} secs or {((toc-tic)/60):.4f} minutes")
    print("\n\n=======Starting New Session\n\n")
    toc = time.time()
    np.savez_compressed(join(gv.data_dir, gv.fname.split(".npz")[0]+"_final.npz"), 
                    solution=sol.cpu().numpy(), 
                    soltime=gv.tsol,
                    runtime=toc-tic, 
                    with_drag=FLAGS.with_drag, 
                    with_cable=FLAGS.with_cable,  
                    gravity=FLAGS.with_grav, 
                    num_pieces=FLAGS.num_pieces,
                    num_sections=num_sections,
                    gain_prop=FLAGS.gain_prop, 
                    gain_deriv=FLAGS.gain_deriv, 
                    gain_integ=FLAGS.gain_integ,  
                    tip_load=FLAGS.tip_load, 
                    controller=FLAGS.controller, 
                    desired_strain=gv.desired_strain,
                    qd=gv.qd_save.cpu().numpy(), 
                    qd_dot=gv.qd_dot_save.cpu().numpy(),
                    )

if __name__ == "__main__":
    app.run(main)