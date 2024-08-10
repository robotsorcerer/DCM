import copy
import time 
import argparse

import sys, os
from absl import app, flags

# from utils.viz import * 

import matplotlib as mpl
# mpl.use("PyQT5")
import matplotlib.pyplot as plt

from os.path import abspath, join, dirname
sys.path.append(dirname(dirname(abspath(os.getcwd() + "../__init__.py"))))

from utils import *
import matplotlib.pyplot as plt 
from os.path import join, expanduser

parser = argparse.ArgumentParser(description='recursive inverse dynamics for serial soro')
parser.add_argument('--maxIter', '-mi', type=int, default='20000',
						help='max num iterations' )
parser.add_argument('--randomSeed', '-rs', type=int, default='123',
						help='random seed to use for training/testing' )
parser.add_argument('--device', '-dv', type=int, default='0',
						help='random seed to use for training/testing' )
parser.add_argument('--silent', '-si', action='store_true', default=False,
						help='print all debug msgs to stdout' )
parser.add_argument('--use_gpu', '-ug', action='store_true', default=True,
						help='To use or not to use gpu' )
parser.add_argument('--save', '-sv', action='store_true', default=True,
						help='Save files to disk?' )
parser.add_argument('--disable-cuda',  '-dc', action='store_true',
					help='Disable CUDA')
parser.add_argument('--multiple_gpus', '-mg', action='store_true', default=False,
						help='use head dataset or prostate dataset')
args = parser.parse_args()

print(args)


flags.DEFINE_boolean('use_gpu', True, 'Use GPU or CPU?')
flags.DEFINE_boolean('silent', True, 'verbose or silent?')

FLAGS = flags.FLAGS

if args.use_gpu:
    import cupy as op
    from utils.config import *
    from utils.dynamics import one_step_dynamics
    from cupy.linalg import pinv, norm
    from cupyx.scipy.linalg import block_diag as cpblk_diag
else:
    import numpy as op
    from utils.config_cpu import *
    from numpy.linalg import pinv, norm
    from scipy.linalg import block_diag 
    from utils.dynamics_cpu import one_step_dynamics_cpu as one_step_dynamics

def get_reference_signals(tspan):
    sinusoid = 1 + op.sin(2*op.sin(tspan))
    
    u = op.linspace(0, 2.0 * op.pi, endpoint=True, num=50)
    v = op.linspace(-0.5, 0.5, endpoint=True, num=10)
    u, v = op.meshgrid(u, v)
    u, v = u.flatten(), v.flatten()

    # Mobius mapping: Taking a u, v pair and returning an x, y, z triple
    x = (1 + 0.5 * v * op.cos(u / 2.0)) * op.cos(u)
    y = (1 + 0.5 * v * op.cos(u / 2.0)) * op.sin(u)
    z = 0.5 * v * op.sin(u / 2.0)

    ref_sigs = Bundle(dict(sinusoid = sinusoid, mobius=(x,y,z)))

    return ref_sigs

def iterate_gain_kernel(Mn, C_1n, C_2n, Dn, tf=100):
    """
        Iterate the gain kernel to convergence using equation (26) in the paper.

        The gain kernel is given by:
            k_n(t,t) = -0.5 * M_n^{-1} \int_0^t { k_n(t,t) M_n k_n(t,t) + (C_{1n} + C_{2n} + D_n) } dt

        while the boundary controller is 
            U_n(t) = \int_0^t { q_n(t,t) + k_n(t,t) - M_n^{-1} [(C_{1n} + C_{2n} + D_n)] } dt

        Inputs:
            Mn: Mass density of the soft robot.
            Cn: Coriolis forces of the soft robot.
            Dn: Drag matrix of the robot.
            tf: final time for the simulation.

        Returns:
            k_n: All gains per time step per section per number of pieces for the soft robot.
    """
    kn          = op.zeros((tf,)+(C_2n.shape))    
    ScaledMnInv = -(1/2) * pinv(Mn)
    breve_Cn    = C_1n + C_2n

    for tp in range(1, tf):
        tpinv = 1/(tp)
        kn[tp] += ( tpinv * kn[tp-1] @ (Mn @ kn[tp-1]) + breve_Cn + Dn )
        kn[tp] *=  ScaledMnInv 

    return kn

def main(argv):

    FLAGS.use_gpu    = args.use_gpu
    FLAGS.silent     = args.silent 

    state_derivs     = init_cond
    num_sections     = gv.num_sections
    num_pieces       = gv.num_pieces

    Xci              = state_derivs[:6*num_pieces]
    Xcidot           = state_derivs[6*num_pieces:12*num_pieces]

    "Allocate memory for gain kernels"
    abscissa_len     = 6   
    tf               = 30
    gain             = op.zeros((len(tspan), num_pieces, tf)+(6,6)) 

    "Get reference Signals for Tracking"
    ref_sigs = get_reference_signals(tspan)
    sines    = ref_sigs.sinusoid
    mobius   = ref_sigs.mobius

    "save dir"
    save_dir = "data"

    for t in range(len(tspan)):        
        # get the one time step dynamics
        dynamics                = one_step_dynamics(state_derivs, t)
        M, C1, C2, D            = dynamics.M, dynamics.Co1, dynamics.Co2, dynamics.DMat
        gr, N, Gravity, Torque  = dynamics.g_r, dynamics.N, dynamics.Gravity, dynamics.Torque
        qdot, qddot             = dynamics.qdot, dynamics.qddot
        
        "Retrieve the sectional masses and control successively"
        # num_sections:  13 num_pieces 4
        for piece_num in range(num_pieces):
            mass_sec    = M[abscissa_len*piece_num:abscissa_len*piece_num+abscissa_len, abscissa_len*piece_num:abscissa_len*piece_num+abscissa_len] 
            co1_sec     = C1[abscissa_len*piece_num:abscissa_len*piece_num+abscissa_len, abscissa_len*piece_num:abscissa_len*piece_num+abscissa_len]
            co2_sec     = C2[abscissa_len*piece_num:abscissa_len*piece_num+abscissa_len, abscissa_len*piece_num:abscissa_len*piece_num+abscissa_len]
            torque_sec  = Torque[abscissa_len*piece_num:abscissa_len*piece_num+abscissa_len, :]
            NForces     = N[abscissa_len*piece_num:abscissa_len*piece_num+abscissa_len, :]
            Dsec        = D[abscissa_len*piece_num:abscissa_len*piece_num+abscissa_len, :]

            # get gain for section i of the robot
            gain[t, piece_num] = iterate_gain_kernel(mass_sec, co1_sec, co2_sec, Dsec, tf)

            print(rf"gain[t, piece_num] : {gain[t, piece_num].shape}")

        if FLAGS.silent:
            print("Mass M size:", M.shape)
            print("Coriolis Forces 1 C1 size:", C1.shape)
            print("Coriolis Forces 2 C2 size:", C2.shape)
            print("Drag Tip Forces: ", D.shape)
            print("Configuration at reference gr: ", gr.shape)
            print(rf"Internal elastic and actuation load (Torque), $\tau$: ", Torque.shape)
            print("Generalized forces, N: ", N.shape)
            print("Gravity: ", Gravity.shape)
            print() 
    
    # plot the gain for every section 
    
    # save the gain matrix
    if args.save:
        with open(join(save_dir, 'gain_matrix.npz'), 'wb') as foo:
            op.save(foo, gain)


if __name__ == "__main__":
    app.run(main)
