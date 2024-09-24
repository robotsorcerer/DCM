__all__ = ["compute_fwd_dynamics"]

__author__      = "Lekan Molu"
__copyright__   = "2024, Discrete Cosserat SoRO Analysis in Python"
__credits__     = "Tcur are None."
__license__     = "Molux Licence"
__maintainer__  = "Lekan Molu"
__email__       = "patlekno@icloud.com"
__loc__         = "Marathon, Broome County, New York"
__date__        = "December 23, 2022"
__status__      = "Completed"

import copy
import torch
from utils.config import *
from utils.cosserat_utils import *
from collections import namedtuple
from utils.matlab_utils import isfield

from torch.linalg import pinv, norm
torch.set_default_dtype(torch.float64)

# M, C1, C2, D = 36 x 36
#  F, T, G, Nterm = 36 x 1
Dynamics = namedtuple('Dynamics', ('M', 'C1', 'C2', 'D', 'F', 'G', 'T', 'Nterm'))

# preallocations 
device = 'cuda:0'

"Mass matrix for the current configuration, parameterized by X"
MasX            = torch.zeros((6,6*num_sections)).to(device)
LMasX           = torch.zeros((6,6*num_sections)).to(device)
RMasX           = torch.zeros((6,6*num_sections)).to(device)
LRMasX          = torch.zeros((6,6*num_sections)).to(device)

"Coriolis forces 1"
Co1X            = torch.zeros((6,6*num_sections)).to(device)
LCo1X           = torch.zeros((6,6*num_sections)).to(device)
RCo1X           = torch.zeros((6,6*num_sections)).to(device)
LRCo1X          = torch.zeros((6,6*num_sections)).to(device)

"Coriolis forces 2"
Co2X            = torch.zeros((6,6*num_sections)).to(device)
LCo2X           = torch.zeros((6,6*num_sections)).to(device)

"Drag forces"
DragX           = torch.zeros((6,6*num_sections)).to(device)  
LDragX          = torch.zeros((6,6*num_sections)).to(device)  
RDragX          = torch.zeros((6,6*num_sections)).to(device)  
LRDragX         = torch.zeros((6,6*num_sections)).to(device)  

Mas_prev        = torch.zeros((6,6)).to(device)
LMas_prev       = torch.zeros((6,6)).to(device)
RMas_prev       = torch.zeros((6,6)).to(device)
LRMas_prev      = torch.zeros((6,6)).to(device)

Co1_prev        = torch.zeros((6,6)).to(device)
LCo1_prev       = torch.zeros((6,6)).to(device)
RCo1_prev       = torch.zeros((6,6)).to(device)
LRCo1_prev      = torch.zeros((6,6)).to(device)

Co2_prev        = torch.zeros((6,6)).to(device)
LCo2_prev       = torch.zeros((6,6)).to(device)

Drag_prev       = torch.zeros((6,6)).to(device)  
LDrag_prev      = torch.zeros((6,6)).to(device)  
RDrag_prev      = torch.zeros((6,6)).to(device)  
LRDrag_prev     = torch.zeros((6,6)).to(device)  

num_sections    = gv["num_sections"] 
# num_pieces      = gv["num_pieces"] 

# sectional matrices per discretization in each piece
MasX             = torch.zeros((6,6*num_sections)).to(device)
LMasX            = torch.zeros((6,6*num_sections)).to(device)
LRMasX           = torch.zeros((6,6*num_sections)).to(device)

DragX            = torch.zeros((6, 6*num_sections)).to(device)
LDragX            = torch.zeros((6, 6*num_sections)).to(device)
LRDragX          = torch.zeros((6,6*num_sections)).to(device)  

LRCo1X           = torch.zeros((6,6*num_sections)).to(device)

Mas_prev         = torch.zeros((6,6)).to(device)
LMas_prev        = torch.zeros((6,6)).to(device)
LRMas_prev       = torch.zeros((6,6)).to(device)
    
Drag_prev        = torch.zeros((6,6)).to(device)  
LDrag_prev       = torch.zeros((6,6)).to(device)  
LRDrag_prev      = torch.zeros((6,6)).to(device)  

LRCo1_prev       = torch.zeros((6,6)).to(device)

# Initialize previous kinematics
g_r              = torch.tensor([[0.0, -1.0, 0.0, 0.0],
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0]]).to(device)     # cantilever
g_prev           = torch.asarray(torch.diagflat((torch.ones((4)).to(device))))
eta_prev         = torch.zeros((6)).to(device)

def compute_fwd_dynamics(t, state_derivs, gv):
    """
        Returns the whole robot dynamics at time t
    """   
    device      = state_derivs.device
    num_pieces  = gv.num_pieces 

    Jaco_prev   = torch.diagflat(torch.cat((
                                torch.ones((1,6)).to(device),
                                torch.zeros((1, 6*(num_pieces-1))).to(device)
                            ), axis=1))

    # initialization of dynamics coefficients
    genMasM          = torch.zeros([6*num_pieces,6*num_pieces]).to(device)
    genDragForces    = torch.zeros([6*num_pieces,6*num_pieces]).to(device)
    genCoriolis1     = torch.zeros([6*num_pieces,6*num_pieces]).to(device)
    genCoriolis2     = torch.zeros([6*num_pieces,6*num_pieces]).to(device)
    genTorque        = torch.zeros([6*num_pieces,1]).to(device) # Generalized Forces F(q)
    genGraV          = torch.zeros([6*num_pieces,6]).to(device) # Generalized gravitational forces \mc{G}
    genCableForces   = torch.zeros([6*num_pieces,1]).to(device) # Drag load

    adetan_prev      = torch.zeros((6*num_pieces,6*num_pieces)).to(device)

    L           =   gv.L
    Eps         =   gv.Eps
    Upsilon     =   gv.Upsilon
    M           =   gv.M

    xci_star    =   gv.xci_star
    Gra         =   gv.Gra
    dX          =   gv.dX
    X           =   gv.X
    num_sections=   gv.num_sections   # sections in each piece
    num_pieces  =   gv.num_pieces
    tact        =   gv.tact
    trel        =   gv.trel
    Fax         =   gv.Fax(num_pieces)
    Fay         =   gv.Fay(num_pieces)
    Faz         =   gv.Faz(num_pieces)
    Famx        =   gv.Famx(num_pieces)
    Famy        =   gv.Famy(num_pieces)
    Famz        =   gv.Famz(num_pieces)
    Fpx         =   gv.Fpx(num_pieces)
    Fpy         =   gv.Fpy(num_pieces, gv.tip_load)  # we are varying the lateral tip loads in addition to doing desired joint trajectory tracking.
    Fpz         =   gv.Fpz(num_pieces)
    Fpmx        =   gv.Fpmx(num_pieces)
    Fpmy        =   gv.Fpmy(num_pieces)
    Fpmz        =   gv.Fpmz(num_pieces)
    D           =   gv.Drag if isfield(gv, "Drag") else None

    # Xci         = state_derivs[:6*num_pieces]
    # Xcidot      = state_derivs[6*num_pieces:12*num_pieces]
    state_len   = len(state_derivs)//2
    Xci         = state_derivs[:state_len]
    Xcidot      = state_derivs[state_len:,]

    #-------------------------------------------------------------------------
    # calculate the components of the dynamic coefficients

    # mass and coriolis 1 of the first section
    xci1 = Xci[:6,].squeeze();   xcidot1 = Xcidot[:6,].squeeze(); k1 = xci1[:3]; 
    theta1 = torch.sqrt(k1.T@k1) # angular strain
    # genMasM, genCoriolis1, genGraV, genTorque,genDragForces, genCableForces, genCoriolis2,  adetan_prev,Jaco_prev,\
    global MasX, LMasX, LRMasX, LRCo1X, DragX, LRDragX, RMasX,  \
         Co2X, DragX,LDragX, RDragX, RCo1X,  LCo2X, Co1X, LCo1X, Mas_prev, LMas_prev, \
         LRMas_prev, g_r, eta_prev,  g_prev, LRCo1_prev, Drag_prev, RMas_prev, \
         Co1_prev, LCo1_prev, RCo1X, RCo1_prev, LRCo1X, LCo2_prev, LDrag_prev, RDrag_prev, LRDragX, Co2_prev, LRDrag_prev
         

    for ii in range(num_sections):
        coAdjg1_cur                     = piecewise_coAdjoint(X[ii],theta1,xci1)        # because J^T = Ad_g^{-1}^T = coAd_g^{-1}
        inv_Adj_g1_cur                  = piecewise_inv_adj(X[ii],theta1,xci1)
        integ_tang_Adjg1_cur            = piecewise_tangop_expmap(X[ii],theta1,xci1)

        """
            Inertia Mass Operator for section n of the robot:

            (i) Project the continuous Cosserat model onto the material abscissa, L_{n-1};
            (ii) Then project the discretized Cosserat for section (n-1) onto the joint space -- with the
            sectional Jacobian S_n^T (see Section V.B, Second Pass in Renda's paper). Where we have used eq. 42.
            (iii) Last, integrate across the section of the material abscissa.

            \mc{M}_a = \int_{L_{n-1}}^{L_n} Ad_{g_n}^\star \mc{M} @ Ad_{g_n}^{-1}
        """
        Mas_cur                          = coAdjg1_cur.matmul(M).matmul(inv_Adj_g1_cur)
        trapz                            = dX * (Mas_prev + Mas_cur)/2
        "Mass of the Cantilever beam"
        MasX[:,6*ii:6*num_sections]      = MasX[:,6*ii:6*num_sections] + torch.tile(trapz,(1,num_sections-ii))
        Mas_prev                         = copy.copy(Mas_cur)

        """
            Left tangent operator of the exp. map for the mass matrix (Fig. 3 and eq. 31. Renda TRO18) -- ignoring the last term S_m
            M_{(n,m)} = \sum_{i=max(n,m)}^N \int_{{L_i -1}}^{L_i} S_n^T  M_a S_m dX
        """
        LMas_cur                         = integ_tang_Adjg1_cur.T @ Mas_cur
        trapz                            = dX * (LMas_prev + LMas_cur)/2
        "Add the Mass of the Cantilever beam"
        LMasX[:,6*ii:6*num_sections]     = LMasX[:,6*ii:6*num_sections]+torch.tile(trapz,(1, num_sections-ii))
        LMas_prev                        = copy.copy(LMas_cur)

        """
            Tangent operator of the exp. map for the mass matrix (Fig. 3 and eq. 31. Renda TRO18) -- including the last term S_m
            in equation (31) i.e.

            M_{(n,m)} = \sum_{i=max(n,m)}^N \int_{{L_i -1}}^{L_i} S_n^T  M_a S_m dX.
        """
        LRMas_cur                        = LMas_cur @ integ_tang_Adjg1_cur
        trapz                            = dX * (LRMas_prev + LRMas_cur)/2
        "Add the Mass of the Cantilever beam"
        LRMasX[:,6*ii:6*num_sections]    = LRMasX[:,6*ii:6*num_sections] + torch.tile(trapz,(1,num_sections-ii))
        LRMas_prev                       = copy.copy(LRMas_cur)

        # Coriolis
        LRCo1_cur                        = integ_tang_Adjg1_cur.T @ coadjoint_mat6x6(eta_prev + integ_tang_Adjg1_cur @ xcidot1.squeeze()) @ Mas_cur @ integ_tang_Adjg1_cur
        trapz                            = dX*(LRCo1_prev + LRCo1_cur)/2
        "Propagate the Coriolis of the Cantilever beam"
        LRCo1X[:,6*ii:6*num_sections]    += torch.tile(trapz,(1,num_sections-ii))
        LRCo1_prev                       = copy.copy(LRCo1_cur)

        # Drag
        Jq_dot                           = integ_tang_Adjg1_cur @ xcidot1.squeeze()
        # extract norm of linear strain component of Jqdot
        Jq_dot_linear                    = torch.norm(Jq_dot[:-3:], p=2)
        Drag_cur                         = integ_tang_Adjg1_cur.T @ D @ integ_tang_Adjg1_cur * Jq_dot_linear
        trapz                            = dX * (Drag_prev + Drag_cur)/2
        "Propagate this drag force across all sections"
        DragX[:,6*ii:6*num_sections]   += torch.tile(trapz, (1, num_sections-ii))
        Drag_prev                        = copy.copy(Drag_cur)

    "Remove the Cantilever mass and Coriolis forces."
    LMasX                               -= torch.tile(LMasX[:,:6],(1,num_sections))
    LRMasX                              -= torch.tile(LRMasX[:,:6],(1,num_sections))
    LRCo1X                              -= torch.tile(LRCo1X[:,:6],(1,num_sections))
    LRDragX                             -= torch.tile(LRDragX[:,:6], (1, num_sections))

    LMas                                 = copy.copy(LMasX[:,6*(num_sections-1):6*num_sections])
    LRMas                                = copy.copy(LRMasX[:,6*(num_sections-1):6*num_sections])
    LRCo1                                = copy.copy(LRCo1X[:,6*(num_sections-1):6*num_sections])
    LRDrag                               = copy.copy(LRDragX[:,6*(num_sections-1):6*num_sections])

    "Actuation load, internal load, and tip load of the first piece."
    Fa1 = torch.tensor(([Famx[0,0], Famy[0,0], Famz[0,0], Fax[0,0], Fay[0,0], Faz[0,0]])).T.to(device)
    if (t <= tact):                                      # tack
        Fa1 *=(t/tact)

    """Internal Forces:
        We adopt a constitutive Kelvin-Voight model for internal forces. Feel free to use
        any other linear model you may desire.

        Fi1: Constitutive forces.
        Fp: Tip point forces.
    """

    Fi1 = Eps @ (xci1 - xci_star) + Upsilon @ xcidot1
    # print(f"Fpmx: {Fpmx.shape} Fpmy : {Fpmy.shape}, Fpmz: {Fpmz.shape} Fpx: {Fpx.shape} Fpy: {Fpy.shape}, Fpz: {Fpz.shape}")
    Fp1 = torch.tensor(([[Fpmx[0,0], Fpmy[0,0], Fpmz[0,0], Fpx[0,0], Fpy[0,0], Fpz[0,0]]])).T.to(device)
    # next: Actuation load
    if num_pieces !=  1:
        Fa1_next = torch.tensor([[Famx[0, 1], Famy[0, 1], Famz[0, 1], Fax[0, 1], Fay[0, 1], Faz[0, 1]]]).T.to(device)
        if (t<= tact): # tack
            Fa1_next *= (t/tact)
    else:
        Fa1_next = torch.zeros((6, 1))

    "Update Newton-Euler dynamics coefficients."
    invAdjg1_last    = piecewise_inv_adj(X[num_sections-1],theta1,xci1)
    invAdjg1R_last   = torch.block_diag(invAdjg1_last[:3,:3],invAdjg1_last[3:6,3:6])
    intdAdjg1_last   = piecewise_tangop_expmap(X[num_sections-1],theta1,xci1)

    "Mass matrix for the entire robot as a block diagonal matrix"
    MasB             = torch.block_diag( LRMas, torch.zeros((6*(num_pieces-1), 6*(num_pieces-1))).to(device) )

    "Generalized mass matrix: equation (31), Renda TRO18"
    genMasM          += Jaco_prev.T @ MasB @ Jaco_prev

    "Generalized Coriolis 1 Forces: equation (32), Renda TRO18"
    Co1B             = torch.zeros((LRCo1.shape[0] + 6*(num_pieces-1), LRCo1.shape[1] + 6*(num_pieces-1))).to(device)
    Co1B[:LRCo1.shape[0], :LRCo1.shape[1]] = LRCo1
    genCoriolis1          +=  Jaco_prev.T @ Co1B @ Jaco_prev

    "Drag Forces as a block diagonal matrix \mathcal{D}"
    DragB           = torch.block_diag(LRDrag, torch.zeros((6*(num_pieces-1), 6*(num_pieces-1))).to(device) )
    "Generalized Drag forces, Roman{D}"
    genDragForces         += Jaco_prev.T @ DragB @ Jaco_prev 

    "Gravitational Forces N: equation (35), Renda TRO18"
    GraB             = torch.vstack((LMas, torch.zeros((6*(num_pieces-1),6)).to(device)))
    genGraV          += Jaco_prev.T @ GraB @ Adjoint_mat6x6(pinv(g_prev))

    "Torque"
    genTorque          += torch.vstack((L*(Fa1-Fi1).unsqueeze(1), torch.zeros((6*(num_pieces-1),1)).to(device))) # cavo tip2base
    CableForces             = torch.vstack(( (invAdjg1_last @ intdAdjg1_last).T @ (invAdjg1R_last @ Fp1), torch.zeros((6*(num_pieces-1),1)).to(device) ))
    genCableForces          += Jaco_prev.T @ CableForces

    """
        Now that we have genMasM, genCoriolis1, genDragForces, genGraV, genTorque, and genCableForces
        we must partition the matrices into fast and slow ones.
    """
    # recursive factors
    if num_pieces !=  1:
        temp_jaco  = invAdjg1_last @ intdAdjg1_last
        Jaco_prev  = torch.block_diag(temp_jaco, torch.zeros(( 6*(num_pieces-1),6*(num_pieces-1) )).to(device)) @Jaco_prev + \
                        torch.block_diag(torch.zeros((6,6)).to(device), torch.asarray(torch.diagflat(torch.ones((6)))).to(device), torch.zeros(( 6*(num_pieces-2),6*(num_pieces-2) )).to(device))

        g_prev     @= piecewise_expmap(X[num_sections-1], theta1, xci1)
        eta_prev   = invAdjg1_last @ (eta_prev + intdAdjg1_last @ xcidot1)

    num_part_pieces = gv.num_part_pieces
    #--------------------------------------------------------------------------
    # masses, Coriolis 1, Coriolis 2 from the second piece onwards
    for jj in range(1, num_part_pieces):
        xcin            = Xci[6*jj:6*jj+6].squeeze()
        xcidotn         = Xcidot[6*jj:6*jj+6].squeeze()
        kn              = xcin[:3].squeeze()
        thetan          = torch.sqrt(kn.T @ kn)

        for ii in range(num_sections):
            coAdjgn_cur = piecewise_coAdjoint(X[ii],thetan,xcin)
            invAdjgn_cur = piecewise_inv_adj(X[ii],thetan,xcin)
            intdAdjgn_cur = piecewise_tangop_expmap(X[ii],thetan,xcin)

            # Masses
            Mas_cur                         = coAdjgn_cur @ M @ invAdjgn_cur
            trapz                           = dX*(Mas_prev + Mas_cur)/2
            MasX[:,6*ii:6*num_sections]     += torch.tile(trapz,[1,num_sections-ii])
            Mas_prev                        = copy.copy(Mas_cur)

            LMas_cur                        = intdAdjgn_cur.T @ Mas_cur
            trapz                           = dX*(LMas_prev+LMas_cur)/2
            LMasX[:,6*ii:6*num_sections]    += torch.tile(trapz,[1,num_sections-ii])
            LMas_prev                       = copy.copy(LMas_cur)

            RMas_cur                        = Mas_cur @ intdAdjgn_cur
            trapz                           = dX*(RMas_prev + RMas_cur)/2
            RMasX[:,6*ii:6*num_sections]    +=torch.tile(trapz,[1,num_sections-ii])
            RMas_prev                       = copy.copy(RMas_cur)
           
            LRMas_cur                       = intdAdjgn_cur.T @ Mas_cur@intdAdjgn_cur 
            trapz                           = dX*(LRMas_prev+LRMas_cur)/2
            LRMasX[:,6*ii:6*num_sections]   +=torch.tile(trapz,[1,num_sections-ii])
            LRMas_prev                      = copy.copy(LRMas_cur)

            # Coriolis 1
            Co1_cur                         = coadjoint_mat6x6(eta_prev+intdAdjgn_cur@xcidotn)@Mas_cur
            trapz                           = dX*(Co1_prev+Co1_cur)/2
            Co1X[:,6*ii:6*num_sections]     +=torch.tile(trapz,[1,num_sections-ii])
            Co1_prev                        = copy.copy(Co1_cur)

            LCo1_cur                        = intdAdjgn_cur.T @ Co1_cur
            trapz                           = dX*(LCo1_prev + LCo1_cur)/2
            LCo1X[:,6*ii:6*num_sections]    +=torch.tile(trapz,[1,num_sections-ii])
            LCo1_prev                       = copy.copy(LCo1_cur)

            RCo1_cur                        = Co1_cur @ intdAdjgn_cur
            trapz                           = dX*(RCo1_prev + RCo1_cur)/2
            RCo1X[:,6*ii:6*num_sections]    +=torch.tile(trapz,[1,num_sections-ii])
            RCo1_prev                       = copy.copy(RCo1_cur)

            LRCo1_cur                       = intdAdjgn_cur.T @ Co1_cur @ intdAdjgn_cur
            trapz                           = dX*(LRCo1_prev+LRCo1_cur)/2
            LRCo1X[:,6*ii:6*num_sections]   +=torch.tile(trapz,[1,num_sections-ii])
            LRCo1_prev                      = copy.copy(LRCo1_cur)

            # Coriolis 2
            Co2_cur                         = Mas_cur@adjoint_mat6x6(intdAdjgn_cur@xcidotn)
            trapz                           = dX*(Co2_prev+Co2_cur)/2
            Co2X[:,6*ii:6*num_sections]     +=torch.tile(trapz,[1,num_sections-ii])
            Co2_prev                        = copy.copy(Co2_cur)

            LCo2_cur                        = intdAdjgn_cur.T @ Co2_cur
            trapz                           = dX*(LCo2_prev+LCo2_cur)/2
            LCo2X[:,6*ii:6*num_sections]    +=torch.tile(trapz,[1,num_sections-ii])
            LCo2_prev                       = copy.copy(LCo2_cur)

            # Drag forces
            Jq_dot                           = intdAdjgn_cur @ xcidotn.squeeze()
            # extract norm of linear strain component of Jqdot
            Jq_dot_linear                    = torch.norm(Jq_dot[:-3:], p=2)
            Drag_cur                         = intdAdjgn_cur.T @ D @ intdAdjgn_cur * Jq_dot_linear
            trapz                            = dX * (Drag_prev + Drag_cur)/2
            "Propagate this drag force across all sections"
            DragX[:,6*ii:6*num_sections]    += torch.tile(trapz, (1, num_sections-ii))
            Drag_prev                        = copy.copy(Drag_cur)

            LDrag_cur                        = intdAdjgn_cur.T @ Drag_cur
            trapz                            = dX*(LDrag_prev + LDrag_cur)/2
            LDragX[:,6*ii:6*num_sections]   += torch.tile(trapz, [1,num_sections-ii])
            LDrag_prev                       = copy.copy(LDrag_cur)

            RDrag_cur                       = Drag_cur @ intdAdjgn_cur * Jq_dot_linear
            trapz                           = dX * (RDrag_prev + RDrag_cur)/2
            RDragX[:,6*ii:6*num_sections]  += torch.tile(trapz, [1, num_sections-ii])
            RDrag_prev                      = copy.copy(RDrag_cur)

            LRDrag_cur                      = intdAdjgn_cur.T @ Drag_cur @ intdAdjgn_cur * Jq_dot_linear
            trapz                           = dX * (LRDrag_prev + LRDrag_cur)/2
            LRDragX[:,6*ii:6*num_sections] += torch.tile(trapz, [1, num_sections-ii])
            LRDrag_prev                      = copy.copy(RDrag_cur)

        MasX            -= torch.tile(MasX[:,:6],[1,num_sections])
        LMasX           -= torch.tile(LMasX[:,:6],[1,num_sections])
        RMasX           -= torch.tile(RMasX[:,:6],[1,num_sections])
        LRMasX          -= torch.tile(LRMasX[:,:6],[1,num_sections])

        Co1X            -= torch.tile(Co1X[:,:6],[1,num_sections])
        LCo1X           -= torch.tile(LCo1X[:,:6],[1,num_sections])
        RCo1X           -= torch.tile(RCo1X[:,:6],[1,num_sections])
        LRCo1X          -= torch.tile(LRCo1X[:,:6],[1,num_sections])

        Co2X            -= torch.tile(Co2X[:,:6],[1,num_sections])
        LCo2X           -= torch.tile(Co2X[:,:6],[1,num_sections])

        DragX          -= torch.tile(DragX[:,:6], [1, num_sections])
        LDragX         -= torch.tile(LDragX[:,:6], [1, num_sections])
        RDragX         -= torch.tile(RDragX[:,:6], [1, num_sections])
        LRDragX        -= torch.tile(LRDragX[:,:6], [1, num_sections])

        Mas             = MasX[:,6*(num_sections-1):6*num_sections]
        LMas            = LMasX[:,6*(num_sections-1):6*num_sections]
        RMas            = RMasX[:,6*(num_sections-1):6*num_sections]
        LRMas           = LRMasX[:,6*(num_sections-1):6*num_sections]

        Co1             = Co1X[:,6*(num_sections-1):6*num_sections]
        LCo1            = LCo1X[:,6*(num_sections-1):6*num_sections]
        RCo1            = RCo1X[:,6*(num_sections-1):6*num_sections]
        LRCo1           = LRCo1X[:,6*(num_sections-1):6*num_sections]

        Co2             = Co2X[:,6*(num_sections-1):6*num_sections]
        LCo2            = LCo2X[:,6*(num_sections-1):6*num_sections]

        Drag            = DragX[:,6*(num_sections-1):6*num_sections]
        LDrag           = LDragX[:,6*(num_sections-1):6*num_sections]
        RDrag           = RDragX[:,6*(num_sections-1):6*num_sections]
        LRDrag          = LRDragX[:,6*(num_sections-1):6*num_sections]


        # Actuation and internal load
        Fan         = torch.tensor(([[Famx[0,jj], Famy[0,jj], Famz[0,jj], Fax[0,jj], Fay[0,jj], Faz[0,jj]]])).T.to(device)
        if t<= tact:                                      # tack
            Fan    *= (t/tact)
        Fin             = Eps@(xcin-xci_star)+Upsilon@xcidotn
        Fpn             = torch.tensor([[Fpmx[0,jj], Fpmy[0,jj], Fpmz[0,jj], Fpx[0,jj], Fpy[0,jj], Fpz[0,jj]]]).T.to(device)

        # Next actuation and internal load
        if jj!= num_pieces-1:
            # print(f"Fpmx: {Fpmx.shape} Fpmy : {Fpmy.shape}, Fpmz: {Fpmz.shape} Fpx: {Fpx.shape} Fpy: {Fpy.shape}, Fpz: {Fpz.shape}")
            Fan_suc     = torch.tensor(([[Famx[0,jj+1], Famy[0,jj+1], Famz[0,jj+1], Fax[0,jj+1], Fay[0,jj+1], Faz[0,jj+1]]])).T.to(device)
            if t<= tact:    # tack
                Fan_suc *=(t/tact)
        else:
            Fan_suc     = torch.zeros((6,1)).to(device)

        # update dynamics coefficients
        invAdjgn_last   = piecewise_inv_adj(X[num_sections-1], thetan, xcin)
        invAdj_gn_R_last  = torch.block_diag(invAdjgn_last[:3,:3], invAdjgn_last[3:6,3:6])

        invAdj_g_prev     = Adjoint_mat6x6(pinv(g_prev))
        invAdj_gprev_R    = torch.block_diag(invAdj_g_prev[:3,:3], invAdj_g_prev[3:6,3:6])
        int_dAdj_gn_last  = piecewise_tangop_expmap(X[num_sections-1], thetan, xcin)

        MasB            = torch.block_diag(torch.vstack((torch.hstack((torch.tile(Mas,[jj, jj]), torch.tile(RMas,[jj, 1]))),\
                                                torch.hstack((torch.tile(LMas,[1, jj]), LRMas)) \
                                                )),  torch.zeros((6*(num_pieces-jj-1), 6*(num_pieces-jj-1) )).to(device) \
                                                )
        genMasM        +=  Jaco_prev.T @ MasB @ Jaco_prev

        Co1B            = torch.block_diag(torch.vstack(( torch.hstack((torch.tile(Co1,[jj, jj]), torch.tile(RCo1,[jj, 1]) )), \
                                                    torch.hstack((torch.tile(LCo1,[1, jj]), LRCo1)) \
                                                )), torch.zeros((6*(num_pieces-jj-1),6*(num_pieces-jj-1))).to(device) \
                                    )
        genCoriolis1   += Jaco_prev.T @ Co1B @ Jaco_prev
        Co2B            = torch.block_diag( torch.vstack(( torch.hstack((torch.tile(Co2,[jj, jj]), torch.zeros((6*jj,6)).to(device) )), \
                                                    torch.hstack((torch.tile(LCo2,[1, jj]), torch.zeros((6,6)).to(device)  )) \
                                                    )), \
                                        torch.zeros((6*(num_pieces-jj-1),6*(num_pieces-jj-1))).to(device) ) + (MasB @ adetan_prev)
        genCoriolis2   +=  Jaco_prev.T @ Co2B @ Jaco_prev
        
        DragB          = torch.block_diag(torch.vstack((torch.hstack((torch.tile(Drag,[jj, jj]), torch.tile(RDrag,[jj, 1]))),\
                                                torch.hstack((torch.tile(LDrag,[1, jj]), LRDrag)) \
                                                )),  torch.zeros((6*(num_pieces-jj-1), 6*(num_pieces-jj-1) )).to(device) \
                                                )
        genDragForces  = Jaco_prev.T @ DragB @ Jaco_prev 
        
        GraB           = torch.vstack((torch.tile(Mas,[jj, 1]), LMas, torch.zeros((6*(num_pieces-jj-1),6)).to(device) ))

        genGraV        += Jaco_prev.T @ GraB @ Adjoint_mat6x6(pinv(g_prev))
        Torque          = torch.vstack(( torch.tile(invAdjgn_last.T@(Fan-Fan_suc),[jj, 1]),
                                        (invAdjgn_last@int_dAdj_gn_last).T@(Fan-Fan_suc),
                                        torch.zeros((6*(num_pieces-jj-1),1)).to(device) # actuation force block matrix
                                        ))
        'This below constitutes the torque.'
        genTorque       +=  Jaco_prev.T @ Torque - torch.vstack((torch.zeros((6*(jj),1)).to(device), L*Fin.unsqueeze(1), torch.zeros((6*(num_pieces-jj-1),1)).to(device) )) 
        "Tip Forces consisting of rotation components minus the rhs which is the tip load (see Sec IV.C, eq. 2"
        CableForces     = torch.vstack((torch.tile(invAdjgn_last.T @ (invAdj_gn_R_last@invAdj_gprev_R@Fpn),[jj, 1]), \
                                            (invAdjgn_last @ int_dAdj_gn_last).T @ (invAdj_gn_R_last @ invAdj_gprev_R @ Fpn), \
                                            torch.zeros((6*(num_pieces-jj-1),1)).to(device) \
                                            ))
        genCableForces  += Jaco_prev.T @ CableForces # this is F(q) in the generalized NE equation.

        # assemble the core matrices decompositions

        # recursive factors
        prev_2_prev      = copy.copy(invAdjgn_last)
        for ii in range(1,jj):
            prev_2_prev  = torch.block_diag(prev_2_prev,invAdjgn_last)

        if jj == num_pieces - 1:
            Jaco_prev    = torch.block_diag(prev_2_prev,invAdjgn_last@int_dAdj_gn_last,torch.zeros((6*(num_pieces-jj-1),6*(num_pieces-jj-1))).to(device))@Jaco_prev
        else:
           Jaco_prev     = torch.block_diag(prev_2_prev,invAdjgn_last@int_dAdj_gn_last,torch.zeros((6*(num_pieces-jj-1),6*(num_pieces-jj-1))).to(device))@Jaco_prev+\
                                        torch.block_diag(torch.zeros((6*(jj+1),6*(jj+1))).to(device),\
                                                    torch.tile(torch.eye(6).to(device),[1, 1]),\
                                                    torch.zeros((6*(num_pieces-jj-2),6*(num_pieces-jj-2))).to(device)
                                                    )
        
        g_prev          @=  piecewise_expmap(X[num_sections-1],thetan,xcin)
        ADxin           = int_dAdj_gn_last @ xcidotn
        eta_prev        = invAdjgn_last @ (eta_prev+ADxin)

        prev_2_prev       = copy.copy(invAdjgn_last)
        prev_2_prev_inv   = pinv(invAdjgn_last)
        for zz in range(num_pieces):
                if (1 + zz == jj):
                    adetan_prev[0:6,6*(zz):6*(zz)+6]  = adjoint_mat6x6(ADxin)

        for ii in range(1, num_pieces):
            prev_2_prev   = torch.block_diag(prev_2_prev,invAdjgn_last)
            prev_2_prev_inv = torch.block_diag(prev_2_prev_inv,pinv(invAdjgn_last))
            for zz in range(num_pieces):
                if (ii + zz +1 == jj):
                    adetan_prev[6*(ii):6*(ii)+6,6*(zz):6*(zz)+6]  = adjoint_mat6x6(ADxin)

        adetan_prev     = prev_2_prev @ adetan_prev @ prev_2_prev_inv

    'Buoyancy-Gravity Term: N Ad_{g_r}^{-1} \mathcal{G} where N = (1-rho_f/rho)* \int{J^T M Ad_g^{-1} dX}'
    buoyancyGravTerm = (1-gv.rho_fluid/gv.rho_arm) * (genGraV @ Adjoint_mat6x6(pinv(g_r)) @ Gra)

    args = (genMasM, genCoriolis1, genCoriolis2, genDragForces, genCableForces, genGraV, genTorque, buoyancyGravTerm)

    return Dynamics(*args)
