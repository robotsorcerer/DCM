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
from utils.matlab_utils import eps, Bundle, strcmp, isfield
from scipy.integrate import cumulative_trapezoid

from torch.linalg import pinv, norm
torch.set_default_dtype(torch.float64)


def piecewise_pdes(t, state_derivs, gv):
    device = state_derivs.device
    qd_save, qd_dot_save, qd_ddot_save = \
        gv.qd_save, gv.qd_dot_save, gv.qd_ddot_save
    global counter, tsol
    counter += 1


    L           =   gv.L
    Eps         =   gv.Eps
    Upsilon     =   gv.Upsilon
    M           =   gv.M
    xci_star    =   gv.xci_star
    Gra         =   gv.Gra
    dX          =   gv.dX
    X           =   gv.X
    num_sections=   gv.num_sections
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

    #-------------------------------------------------------------------------
    # actual solution xci xcidot

    Xci              = state_derivs[:6*num_pieces]
    Xcidot           = state_derivs[6*num_pieces:12*num_pieces]

    # initialization of dynamics coefficients
    genMasM          = torch.zeros([6*num_pieces,6*num_pieces]).to(device)
    genDragForces    = torch.zeros([6*num_pieces,6*num_pieces]).to(device)
    genCoriolis1     = torch.zeros([6*num_pieces,6*num_pieces]).to(device)
    genCoriolis2     = torch.zeros([6*num_pieces,6*num_pieces]).to(device)
    genTorque        = torch.zeros([6*num_pieces,1]).to(device) # Generalized Forces F(q)
    genGraV          = torch.zeros([6*num_pieces,6]).to(device) # Generalized gravitational forces \mc{G}
    genCableForces          = torch.zeros([6*num_pieces,1]).to(device) # Drag load

    # Initialize previous kinematics
    g_r              = torch.tensor([[0.0, -1.0, 0.0, 0.0],
                                    [1.0, 0.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0, 0.0],
                                    [0.0, 0.0, 0.0, 1.0]]).to(device)     # cantilever
    Jaco_prev        = torch.diagflat(torch.cat((
                                        torch.ones((1,6)).to(device),
                                        torch.zeros((1, 6*(num_pieces-1))).to(device)
                                    ), axis=1))
    g_prev           = torch.asarray(torch.diagflat((torch.ones((4)).to(device))))
    eta_prev         = torch.zeros((6)).to(device)
    adetan_prev      = torch.zeros((6*num_pieces,6*num_pieces)).to(device)

    #-------------------------------------------------------------------------
    # calculate the components of the dynamic coefficients

    # mass and coriolis 1 of the first section
    xci1             = Xci[:6,].squeeze()
    xcidot1          = Xcidot[:6,].squeeze()
    k1               = xci1[:3]
    theta1           = torch.sqrt(k1.T@k1) # angular strain

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
        Mas_cur                          = coAdjg1_cur @ M @ inv_Adj_g1_cur
        trapz                            = dX * (Mas_prev + Mas_cur)/2
        "Mass of the Cantilever beam"
        MasX[:,6*ii:6*num_sections]      = MasX[:,6*ii:6*num_sections] + torch.tile(trapz,(1,num_sections-ii))
        Mas_prev                         = copy.copy(Mas_cur)

        """
            Left tangent operator of the exp. map for the mass matrix (eq. 20. Renda TRO18) -- ignoring the last term S_m
            M_{(n,m)} = \sum_{i=max(n,m)}^N \int_{{L_i -1}}^{L_i} S_n^T  M_a S_m dX

        """
        LMas_cur                         = integ_tang_Adjg1_cur.T @ Mas_cur
        trapz                            = dX * (LMas_prev + LMas_cur)/2
        "Add the Mass of the Cantilever beam"
        LMasX[:,6*ii:6*num_sections]     = LMasX[:,6*ii:6*num_sections]+torch.tile(trapz,(1, num_sections-ii))
        LMas_prev                        = copy.copy(LMas_cur)

        """
            Tangent operator of the exp. map for the mass matrix (eq. 20. Renda TRO18) -- including the last term S_m
            in equation (31) i.e.

            M_{(n,m)} = \sum_{i=max(n,m)}^N \int_{{L_i -1}}^{L_i} S_n^T  M_a S_m dX.
        """
        LRMas_cur                        = integ_tang_Adjg1_cur.T @ Mas_cur @ integ_tang_Adjg1_cur
        trapz                            = dX * (LRMas_prev + LRMas_cur)/2
        "Add the Mass of the Cantilever beam"
        LRMasX[:,6*ii:6*num_sections]  = LRMasX[:,6*ii:6*num_sections] + torch.tile(trapz,(1,num_sections-ii))
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

    # recursive factors
    if num_pieces !=  1:
        temp_jaco  = invAdjg1_last @ intdAdjg1_last
        Jaco_prev  = torch.block_diag(temp_jaco, torch.zeros(( 6*(num_pieces-1),6*(num_pieces-1) )).to(device)) @Jaco_prev + \
                        torch.block_diag(torch.zeros((6,6)).to(device), torch.asarray(torch.diagflat(torch.ones((6)))).to(device), torch.zeros(( 6*(num_pieces-2),6*(num_pieces-2) )).to(device))

        g_prev     @= piecewise_expmap(X[num_sections-1], theta1, xci1)
        eta_prev   = invAdjg1_last @ (eta_prev + intdAdjg1_last @ xcidot1)

    #--------------------------------------------------------------------------
    # masses, Coriolis 1, Coriolis 2 from the second piece onwards
    for jj in range(1, num_pieces):
        xcin            = Xci[6*jj:6*jj+6].squeeze()
        xcidotn         = Xcidot[6*jj:6*jj+6].squeeze()
        kn              = xcin[:3].squeeze()
        thetan          = torch.sqrt(kn.T @ kn)

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

    q_dot = Xcidot
    if gv.controller:
        q_tilde = Xci - gv.qd(t)

        if strcmp(gv.controller.lower(), 'pd'): # no gravity compensation is default
            'u = -K_p \tilde{q} - K_D \dot{q}'
            u = -gv.Kp @ q_tilde - gv.Kd @ q_dot 
            # if gv.feedforward: # compensate for feedforward gain # see pg 191 Murray and Sastry and Li
            #     u += (genCoriolis1 + genCoriolis2)@q_dot 
            'Eq. 28\'s rhs in paper  -- fluidic-driven. No Cable Forces. NB: q_ddot is really M \times q_ddot actually.'
            Mq_ddot = (u  - (genCoriolis1 + genCoriolis2) @ q_dot)
            
            if gv.with_cable:
                'u = -K_p \tilde{q} - K_D \dot{q} - F(q)'
                u -= genCableForces  # u = -K_p \tilde{q} - K_D \dot{q} - F(q)
                'Eq. 28\'s rhs in paper -- with cable forces'
                Mq_ddot +=  genCableForces

            'under water.' 
            if gv.with_drag:                
                Mq_ddot   -= (genDragForces @ q_dot)
            
            'Now account for gravity in u and q_ddot'
            if gv.with_grav:
                'Eqs. 36 & 40.'
                'if we are running with gravity compensation'
                if gv.with_drag: 
                    'if water density applies::in case this is being called for the octopus robot under water'
                    u -= buoyancyGravTerm
                    Mq_ddot += buoyancyGravTerm
                else:
                    'terrestrial operation.'
                    u -= (genGraV @ Adjoint_mat6x6(pinv(g_r)) @ Gra)
                    Mq_ddot += (genGraV @ Adjoint_mat6x6(pinv(g_r)) @ Gra)
                # Mq_ddot += buoyancyGravTerm

            q_ddot = pinv(genMasM) @ Mq_ddot

        elif strcmp(gv.controller.lower(), 'pid'): 
            'u = -{K_p} \tilde{q} - K_D \dot{q} -  K_I \int_0^T{q_dot_tilde}'
            q_tilde_np = q_tilde.cpu().numpy()
            integ_term = torch.asarray(cumulative_trapezoid(q_tilde_np.flatten(), initial=0).reshape(-1,1)).to(device)
            # print(integ_term.shape, q_tilde_np.shape, gv.Ki.shape)
            u = -(gv.Kp ) @ q_tilde - gv.Kd @ q_dot  -  gv.Ki @ integ_term
            'Eq. 27\'s rhs in paper  -- fluidic-driven. No Cable Forces. NB: q_ddot is really M \times q_ddot actually.'
            Mq_ddot = (u  - (genCoriolis1 + genCoriolis2) @ q_dot)
            
            if gv.with_cable:
                'u = -K_p \tilde{q} - K_D \dot{q} - F(q)'
                u -= genCableForces  # u = -K_p \tilde{q} - K_D \dot{q} - F(q)
                'Eq. 27\'s rhs in paper -- with cable forces'
                Mq_ddot +=  genCableForces

            'under water.' 
            if gv.with_drag:                
                Mq_ddot   -= (genDragForces @ q_dot)
            
            'Now account for gravity in u and q_ddot'
            if gv.with_grav:
                'Eqs. 34 & 38.'
                Mq_ddot += buoyancyGravTerm
                u -= buoyancyGravTerm  

            q_ddot = pinv(genMasM) @ Mq_ddot

        elif strcmp(gv.controller.lower(), 'spt'):  
            e1 = Xci - gv.qd(t)
            e2 = Xcidot - (gv.qd_dot(t) - e1)
            'Equation 21 in SoRoSPT Paper'
            tau = genMasM@(gv.qd_ddot(t)-2*e2 + e1) + (genCoriolis1+genCoriolis2)@(gv.qd(t)-e1) - gv.Kp@e1
            if gv.with_drag:
                'under water, fluid actuation, and no gravity compensation'
                tau += genDragForces @ q_dot
            if gv.with_cable:
                'subtract cable forces at the midpoint of the cable per section'
                tau -= genCableForces 
            if gv.with_grav: 
                'if we are running with gravity compensation'
                if gv.with_drag: 
                    'if water density applies::in case this is being called for the octopus robot under water'
                    tau -= buoyancyGravTerm
                else:
                    'terrestrial operation.'
                    tau -= (genGraV @ Adjoint_mat6x6(pinv(g_r)) @ Gra)
            
            'Implements eq. 8 in the SPT+backstepping paper'
            q_ddot =  pinv(genMasM) @ (tau - (genCoriolis1-genCoriolis2 + genDragForces) @ Xcidot + genCableForces + buoyancyGravTerm) 
        else:
            raise NotImplementedError              
    else:
        'forward dynamics simulation.'
        q_ddot    = pinv(genMasM) @ (genTorque + genGraV @ Adjoint_mat6x6(pinv(g_r)) @ Gra + genCableForces - (genCoriolis1 - genCoriolis2) @ q_dot)

    z_point         = torch.vstack((q_dot, q_ddot))

    'Stack up for saves'
    gv.tsol = np.vstack((gv.tsol, [t.item()]))
    gv.sol = torch.vstack((gv.sol, z_point.T))

    # append these for the fist section's qd only since it is uniform through all sections
    indices = torch.arange(6).to(device)  

    gv.qd_save = torch.vstack((qd_save, torch.index_select(gv.qd(t), 0, indices)))
    gv.qd_dot_save = torch.vstack((qd_dot_save, torch.index_select(gv.qd_dot(t), 0, indices)))

    # if counter % 500 == 0:
    #     toc = time.time()
    #     np.savez_compressed(join(gv.data_dir, gv.fname), 
    #                     solution=gv.sol.cpu().numpy(), 
    #                     soltime=np.asarray(gv.tsol),
    #                     runtime=toc-gv.tic, 
    #                     with_drag=gv.with_drag, 
    #                     with_cable=gv.with_cable,  
    #                     gravity=gv.with_grav, 
    #                     num_pieces=num_pieces,
    #                     num_sections=num_sections,
    #                     gain_prop=gv.gain_prop, 
    #                     gain_deriv=gv.gain_deriv, 
    #                     gain_integ=gv.gain_integ,  
    #                     tip_load=gv.tip_load, 
    #                     controller=gv.controller, 
    #                     desired_strain=gv.desired_strain,
    #                     qd=gv.qd_save.cpu().numpy(), 
    #                     qd_dot=gv.qd_dot_save.cpu().numpy(),
    #                     # qd_ddot=qd_ddot_save.cpu().numpy()
    #                     )

    if gv.verbose and counter%10==0: 
        # print(gv.tsol.shape, gv.sol.shape)
        print(f"Device: {device} | Num Steps: {counter} | t: {t:.4f} ||z_point||: {norm(z_point, ord='fro'):.8f}")

    return (t, z_point)
