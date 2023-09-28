__all__ = ["gv", "num_sections", "counter"]

__author__      = "Lekan Molu"
__maintainer__  = "Lekan Molu"
__license__     = "Molux Licence"
__copyright__   = "2022, Discrete Cosserat SoRO Analysis in Python"
__credits__     = "There are None."
__email__       = "patlekno@icloud.com"
__date__        = "December 23, 2022"
__status__      = "Completed"

import torch 
import numpy as np
from math import pi 
from .matlab_utils import Bundle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

counter = 0

# Geometrical input of the silicone arm (section)
E        = 110e3                          # [Pa] Young's modulus 110e3 //
eta      = 3e3                            # [Pa*s] shear viscosity modulus 5e3 //
Poi      = 0                              # [-]Poisson modulus 0.5 //
G        = E/(2*(1+Poi))                  # [Pa] shear modulus //
R        = 100e-3  #10e-3                 # [m]  arm radius 10e-3 //
L        = 200e-3 #62.5e-3                # [m] Arm length 250e-3//
num_sections     = int(np.floor(L*2e2+1)) # one section by half a centimetre floor(L*2e2+1) //
X        = torch.asarray(torch.linspace(0,L,num_sections)).to(device)    # [m] curvilinear abscissa
A        = pi*R**2                        # [m^2]
J        = pi*R**4/4                      # [m^4]
I        = pi*R**4/2                      # [m^4]
#-------------------------------------------------------------------------
# initial configuration t= 0
xci_star    = torch.tensor(([[0, 0, 0, 1, 0, 0]])).squeeze().to(device)

#-------------------------------------------------------------------------
# Dynamics parameters

rho_arm     = 2000                         # [Kg/m^3] nominal density 1080
rho_water   = 997                          # water density kg/m^3
drag_coeff  = 0.82                         # We assuume robot is a long cylinder
Gra         = torch.zeros((6, 1)).to(device)                # [m/s^2] gravitational vector [000-9.8100]
Eps         = torch.asarray(torch.diagflat(torch.tensor([G*I, E*J, E*J, E*A ,G*A ,G*A]))).to(device)  # stifness matrix
Upsilon     = eta*torch.asarray(torch.diagflat(torch.tensor([I, 3*J, 3*J, 3*A, A, A]))).to(device)   # viscosity matrix eta*diag([I 3*J 3*J 3*A A A]) (see eq. 6)
M           = rho_arm*torch.asarray(torch.diagflat(torch.tensor([I, J, J, A, A, A]))).to(device)      # inertia matrix
# D           = 0.5 * rho_water*drag_coeff*A* # F_drag = 0.5 * C_D * A * rho_w * vel^2
D           = drag_coeff * torch.eye(6).to(device)

dX          = L/(num_sections-1)          # delta X

#-------------------------------------------------------------------------
# actuation load (body coordinates)
tact        = torch.tensor(1).to(device)                    # [s] torque time in z or y direction
trel        = torch.tensor(19.0).to(device)                 # [s] relaxation time

'below for lateral and vertical (+z) actuation loads'
"""
Fax         = lambda num_pieces: 0*torch.hstack((torch.zeros((1, num_pieces-1)), \
                                                torch.tensor(([[1]])) )).to(device)      # [N] contraction load
Fay         = lambda num_pieces: 0.1*torch.hstack((torch.zeros((1, num_pieces-1)), \
                                                torch.tensor(([[1]])) )).to(device)      # [N] lateral y load 0.1
Faz         = lambda num_pieces: 0.01*torch.hstack((torch.zeros((1, num_pieces-1)), \
                                                torch.tensor(([[1]])) )).to(device)      # [N] lateral z load 0.01
Famx        = lambda num_pieces: 0.001*torch.hstack((torch.zeros((1, num_pieces-1)), \
                                                torch.tensor(([[1]])) )).to(device)      # [Nm] torsion torque 0.001
Famy        = lambda num_pieces: 0.01*torch.hstack((torch.zeros((1, num_pieces-1)), \
                                                torch.tensor(([[1]])) )).to(device)      # [Nm] bending torque
Famz        = lambda num_pieces: 0.005*torch.hstack((torch.zeros((1, num_pieces-1)), \
                                                torch.tensor(([[1]])) )).to(device)      # [Nm] bending torque 0.005
"""
Fax         = lambda num_pieces: 0*torch.hstack((torch.zeros((1, num_pieces-1)), \
                                                torch.tensor(([[1]])) )).to(device)      # [N] contraction load
Fay         = lambda num_pieces: 0*torch.hstack((torch.zeros((1, num_pieces-1)), \
                                                torch.tensor(([[1]])) )).to(device)      # [N] lateral y load 0.1
Faz         = lambda num_pieces: 0*torch.hstack((torch.zeros((1, num_pieces-1)), \
                                                torch.tensor(([[1]])) )).to(device)      # [N] lateral z load 0.01
Famx        = lambda num_pieces: 0*torch.hstack((torch.zeros((1, num_pieces-1)), \
                                                torch.tensor(([[1]])) )).to(device)      # [Nm] torsion torque 0.001
Famy        = lambda num_pieces: 0*torch.hstack((torch.zeros((1, num_pieces-1)), \
                                                torch.tensor(([[1]])) )).to(device)      # [Nm] bending torque
Famz        = lambda num_pieces: 0*torch.hstack((torch.zeros((1, num_pieces-1)), \
                                                torch.tensor(([[1]])) )).to(device)      # [Nm] bending torque 0.005

#-------------------------------------------------------------------------
# external tip load (base (X= 0) coordinate)

Fpx         = lambda num_pieces: 0*torch.hstack((torch.zeros((1, num_pieces-1)), \
                                                torch.tensor(([[1]])) )).to(device)               # [N] contraction load
Fpy         = lambda num_pieces, tip_load: tip_load*torch.hstack((torch.zeros((1, num_pieces-1)), \
                                                torch.tensor(([[1]])) )).to(device) # [N] lateral y load
Fpz         = lambda num_pieces: 0*torch.hstack((torch.zeros((1, num_pieces-1)), \
                                                torch.tensor(([[1]])) )).to(device)               # [N] lateral z load
Fpmx        = lambda num_pieces: 0*torch.hstack((torch.zeros((1, num_pieces-1)), \
                                                torch.tensor(([[1]])) )).to(device)               # [Nm] torsion torque
Fpmy        = lambda num_pieces: 0*torch.hstack((torch.zeros((1, num_pieces-1)), \
                                                torch.tensor(([[1]])) )).to(device)               # [Nm] bending torque
Fpmz        = lambda num_pieces: 0*torch.hstack((torch.zeros((1, num_pieces-1)), \
                                                torch.tensor(([[1]])) )).to(device)               # [Nm] bending torque


qd_save = torch.zeros((6, 1)).to(device)
# global variable
gv = {"rho_arm": rho_arm, "rho_fluid": rho_water, "Gra": Gra,
        "L": L, "X": X, "R": R, "x": xci_star, "A": A,"Eps": Eps,
        "Upsilon": Upsilon,"M": M, "Drag": D, "tsol": np.array((0)),
        "num_sections": num_sections,"dX": dX, 
        "tact": tact,  "trel": trel, "Fax": Fax, "Fay": Fay, "Faz" : Faz,
        "Famx": Famx, "Famy": Famy,"Famz": Famz,  "Fpx" : Fpx,"Fpy": Fpy,
        "Fpz": Fpz, "Fpmx": Fpmx, "Fpmy": Fpmy, "Fpmz": Fpmz,
        "xci_star": xci_star, "qd_save": qd_save, "qd_dot_save": qd_save,
        "qd_ddot_save": qd_save}