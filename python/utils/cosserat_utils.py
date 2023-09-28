__all__ = ["skew_sym", "adjoint_mat6x6", "coadjoint_mat6x6", "Adjoint_mat6x6",
            "coAdjoint_mat6x6", "lie_group", "piecewise_tangop_expmap", "piecewise_liegroup_adj",
            "piecewise_coAdjoint", "piecewise_expmap", "piecewise_inv_adj",
            "piecewise_observables", "RK4_integ"]

__author__ 		= "Lekan Molu"
__copyright__ 	= "2022, Discrete Cosserat SoRO Analysis in Python"
__credits__  	= "There are none."
__license__ 	= "Molux Licence"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__comments__    = "This code was written under white out conditions before Christmas Eve."
__loc__         = "Marathon, Broome County, New York"
__date__ 		= "December 23, 2022"
__status__ 		= "Completed"

from absl import flags

import numpy as np
import torch
torch.set_default_dtype(torch.float64)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def RK4_integ(OdeFun, x, m, l):
    """
        RK4 integrator for a time-invariant damped pendulum.

        See https://lpsa.swarthmore.edu/NumInt/NumIntFourth.html

        This function must be called within a loop for a total of N
        steps of integration.
        Obviously, the smallet the value of T, the better

        Inputs:
            OdeFun: Right Hand Side of Ode function to be integrated
            x: State, must be a list, initial condition
            m: mass of the pendulum
            l: finite length of the pendulum

            Author: Lekan Molu, August 09, 2021
    """
    M = 4 # RK4 steps per interval
    h = 0.2 # time step

    X = np.array(x)
    T = 100
    for j in range(M):
        k1 = OdeFun(X, m, l)
        k2 = OdeFun(X + h/2 * k1, m, l)
        k3 = OdeFun(X + h/2 * k2, m, l)
        k4 = OdeFun(X + h * k3, m, l)
        X  = X+(h/6)*(k1 +2*k2 +2*k3 +k4)

    return X

def skew_sym(vec):
    """
        Convert a 3-vector to a skew symmetric matrix.
    """

    assert vec.size(0) == 3, "vec must be a 3-sized vector."
    if vec.ndim>1: vec = vec.squeeze()
    skew = torch.tensor(([ [ 0, -vec[2].item(), vec[1].item() ],
            [ vec[2].item(), 0, -vec[0].item() ],
            [-vec[1].item(), vec[0].item(), 0 ]
        ])).to(vec.device)

    return skew

def adjoint_mat6x6(screw):
    """
        Computes the adjoint of an SE(3) motion parameterized by a screw

        Input
        ------
            .screw: A 1 x 6 vector of position and rotation components.

        Output
        -------
            .adj = (\hat{\gamma} \hat{\omega}               0)
                    (\hat{\varepsilon}\hat{\nu}  \hat{\gamma} \hat{\omega}  )
    """
    assert screw.size(0)==6, "input must be a six-vector."

    adj         = torch.zeros((6,6)).to(screw.device)
    adj[:3, :3] = skew_sym(screw[:3])
    adj[3:, :3] = skew_sym(screw[3:])
    adj[3:, 3:] = skew_sym(screw[:3])

    return adj

def coadjoint_mat6x6(screw):
    """
        Computes the co-adjoint of an SE(3) motion parameterized by a screw

        Input
        ------
            .screw: A 1 x 6 vector of position and rotation components.

        Output
        -------
            .coadj = adj* = (\hat{\gamma} \hat{\omega}   \hat{\varepsilon}\hat{\nu}  \hat{\gamma})
                            (             0               \hat{\omega}  )
    """
    assert screw.size(0)==6, "input must be a six-vector."
    device = screw.device
    coadj        = torch.zeros((6,6)).to(device)
    coadj[:3,:3] = skew_sym(screw[:3])
    coadj[:3,3:] = skew_sym(screw[3:6])
    coadj[3:,3:] = skew_sym(screw[:3])

    return coadj

def Adjoint_mat6x6(g_conf):
    """
        Computes the Adjoint of the configuration matrix of a microsolid
        of the Cosserat continuum

        Input
        ------
            .g_conf: A 1 x 6 vector of position and rotation components for the conf mat in SE(3).

        Output
        -------
            .Adj(R             0    )
                ( \hat{p}R     R    )
    """
    assert g_conf.shape==(4,4), "input must be shaped 4 x 4."

    Adj         = torch.zeros((6,6)).to(g_conf.device)
    Adj[:3,:3]  = g_conf[:3,:3]
    Adj[3:,:3]  = skew_sym(g_conf[:3,3])@g_conf[:3,:3]
    Adj[3:,3:]  = g_conf[:3,:3]

    return Adj

def coAdjoint_mat6x6(g_conf):
    """
        Computes the Co-Adjoint of the configuration matrix of a microsolid
        of the Cosserat continuum

        Inputs
        ------
            .g_conf: A 1 x 6 vector of position and rotation components for the conf mat in SE(3).

        Output
        -------
            .Adj* = (R      \hat{p}R  )
                    (0           R    )
    """
    assert g_conf.shape==(4,4), "input must be a 6x6 matrix."

    coAdj         = torch.zeros((6,6)).to(g_conf.device)

    coAdj[:3,:3]  = g_conf[:3,:3]
    coAdj[:3,:6]  = skew_sym(g_conf[:3,3])@g_conf[:3,:3]
    coAdj[3:, 3:] = g_conf[:3,:3]

    return coAdj

def lie_group(screw):
    """
        Computes the Lie Group from the screw displacement of a point.
        Essentially the isomorphism from the Lie algebra in R^6 to the
        Lie group representation in SE(3).

        Inputs
        ------
            .screw: 6-D  vector of position and orientation.

        Output
        ------
            .group: 4 x 4 Lie group matrix representation.
    """
    group  = torch.zeros((4,4)).to(screw.device)
    group[:3, :3] = skew_sym(screw[:3])
    group[:3, 3]  = screw[3:]

    return group

def  piecewise_tangop_expmap(x,theta,xci):
    """
        Tangent operator of the exponential map.
        Equation (14) in
            @article{RendaTRO18,
              title={Discrete cosserat approach for multisection soft manipulator dynamics},
              author={Renda, Federico and Boyer, Fr{\'e}d{\'e}ric and Dias, Jorge and Seneviratne, Lakmal},
              journal={IEEE Transactions on Robotics},
              volume={34},
              number={6},
              pages={1518--1533},
              year={2018},
              publisher={IEEE}
            };
    """
    device = x.device

    adjxci       =  adjoint_mat6x6(xci)

    if theta==0:
        Adj_g =   x*torch.asarray(torch.diagflat(torch.ones((6)))).to(device)+((x**2)/2)*adjxci
    else:
        Adj_g =   x*torch.asarray(torch.diagflat(torch.ones((6)))).to(device)+((4-4*torch.cos(x*theta)-x*theta*torch.sin(x*theta))/(2*theta**2))*adjxci+\
                     ((4*x*theta-5*torch.sin(x*theta)+x*theta*torch.cos(x*theta))/(2*theta**3))*torch.linalg.matrix_power(adjxci, 2)+\
                     ((2-2*torch.cos(x*theta)-x*theta*torch.sin(x*theta))/(2*theta**4))*torch.linalg.matrix_power(adjxci, 3)+\
                     ((2*x*theta-3*torch.sin(x*theta)+x*theta*torch.cos(x*theta))/(2*theta**5))*torch.linalg.matrix_power(adjxci, 4)

    return Adj_g

def piecewise_liegroup_adj(x, theta, xci):
    """
        The exponential function in Eq (12) of
            @article{RendaTRO18,
              title={Discrete cosserat approach for multisection soft manipulator dynamics},
              author={Renda, Federico and Boyer, Fr{\'e}d{\'e}ric and Dias, Jorge and Seneviratne, Lakmal},
              journal={IEEE Transactions on Robotics},
              volume={34},
              number={6},
              pages={1518--1533},
              year={2018},
              publisher={IEEE}
    It is essentially the adjoint representation of the Lie group transformation g_n(X).

    This function is not used throughout the paper.
    """
    device = x.device

    adjxci       = adjoint_mat6x6(xci)

    if theta==0:
        Adjg        = torch.asarray(torch.diagflat(torch.ones((6)))).to(device)+x * adjxci
    else:
        Adjg        = torch.asarray(torch.diagflat(torch.ones((6)))).to(device)+((3*torch.sin(x*theta)-x*theta*torch.cos(x*theta))/(2*theta))*adjxci+\
                     ((4-4*torch.cos(x*theta)-x*theta*torch.sin(x*theta))/(2*theta**2))*torch.linalg.matrix_power(adjxci, 2)+\
                     ((torch.sin(x*theta)-x*theta*torch.cos(x*theta))/(2*theta**3))*torch.linalg.matrix_power(adjxci, 3)+\
                     ((2-2*torch.cos(x*theta)-x*theta*torch.sin(x*theta))/(2*theta**4))*torch.linalg.matrix_power(adjxci, 4)

    return Adjg

def piecewise_coAdjoint(x,theta,xci):
    device = x.device

    coadj_xci       = coadjoint_mat6x6(xci)

    if theta==0:
        coAdjg        = torch.asarray(torch.diagflat(torch.ones((6)))).to(device)+x*coadj_xci
    else:
        coAdjg        = torch.asarray(torch.diagflat(torch.ones((6)))).to(device)+((3*torch.sin(x*theta)-x*theta*torch.cos(x*theta))/(2*theta))*coadj_xci+\
                     ((4-4*torch.cos(x*theta)-x*theta*torch.sin(x*theta))/(2*theta**2))*torch.linalg.matrix_power(coadj_xci, 2)+\
                     ((torch.sin(x*theta)-x*theta*torch.cos(x*theta))/(2*theta**3))*torch.linalg.matrix_power(coadj_xci, 3)+\
                     ((2-2*torch.cos(x*theta)-x*theta*torch.sin(x*theta))/(2*theta**4))*torch.linalg.matrix_power(coadj_xci, 4)

    return coAdjg

def piecewise_expmap(x, theta, xci):
    """
        Returns the exponential map for a microsolid
    """
    device = x.device
    xci_hat = lie_group(xci)

    if theta ==0:
        g = torch.asarray(torch.diagflat(torch.ones((4)))).to(device)+x*xci_hat
    else:
        g = torch.asarray(torch.diagflat(torch.tensor([1, 1, 1, 1]))).to(device)+x*xci_hat + \
                 ((1-torch.cos(x*theta))/(theta**2))*torch.linalg.matrix_power(xci_hat, 2)+\
                 ((x*theta-torch.sin(x*theta))/(theta**3))*torch.linalg.matrix_power(xci_hat, 3)

    return g

def piecewise_inv_adj(x, theta, xci):
    """
        Returns the inverse adjoint map for a microsolid.
    """
    device = x.device

    adj_xci = adjoint_mat6x6(xci)

    if theta ==0:
        g = torch.asarray(torch.diagflat(torch.ones((6)))).to(device)-x*adj_xci
    else:
        # Fix: change adj_xci**n to matrix power torch.linalg.matrix_power(adj_xci, n)
        g = torch.asarray(torch.diagflat(torch.ones((6)))).to(device)-((3*torch.sin(x*theta)-x*theta*torch.cos(x*theta))/(2*theta))*adj_xci+\
                    ((4-4*torch.cos(x*theta)-x*theta*torch.sin(x*theta))/(2*theta**2))*torch.linalg.matrix_power(adj_xci, 2)-\
                    ((torch.sin(x*theta)-x*theta*torch.cos(x*theta))/(2*theta**3))*torch.linalg.matrix_power(adj_xci, 3)+\
                    ((2-2*torch.cos(x*theta)-x*theta*torch.sin(x*theta))/(2*theta**4))*torch.linalg.matrix_power(adj_xci, 4)
    return g

def piecewise_observables(t, z, gv):
    """
    Verify the iterates in the for loops here later on.
    """
    device = z.device

    X           = gv.X
    num_pieces  = gv.num_pieces
    num_sections= gv.num_sections

    #----------------------------------------------------------------------
    # observables: position (g), velocity (eta)

    for zz in range(len(t)):
        g                = gv.g
        eta              = gv.eta
        nstep            = gv.nstep

        Xci              = z[:6*num_pieces,zz]
        Xcidot           = z[6*num_pieces:12*num_pieces,zz]
        g_r              = torch.tensor(([[0, -1, 0, 0, 1],
                                      [1, 0, 0, 0],\
                                      [0, 0, 1,  0],\
                                      [0,  0, 0,  1]])).to(device)    # cantilever
        g_prev           = torch.asarray(torch.diagflat((torch.ones((4))))).to(device)
        eta_prev         = torch.zeros((6,1)).to(device)

        for jj in range(num_pieces):
            xcin         = Xci[6*(jj):6*(jj)+6,:]
            xcidotn      = Xcidot[6*(jj):6*(jj)+6,:]
            kn           = xcin[:3]
            thetan       = torch.sqrt(kn.T@kn)

            # kinematics
            for ii in range(num_sections):
                invAdjgn_cur    = piecewise_inv_adj(X[ii], thetan, xcin)
                intdAdjgn_cur   = piecewise_tangop_expmap(X[ii], thetan, xcin)
                g[4*(nstep-1):4*(nstep-1)+4,4*(jj)*num_sections+4*(ii):4*(jj)*num_sections+4*(ii)+4]=\
                                g_r@g_prev@piecewise_expmap(X[ii],thetan,xcin)
                eta[6*(nstep-1):6*(nstep-1)+6,(jj)*num_sections+ii] = invAdjgn_cur@(eta_prev+intdAdjgn_cur@xcidotn)

            # recursive factors
            invAdjgn_last   = piecewise_inv_adj(X[num_sections],thetan,xcin)
            intdAdjgn_last  = piecewise_tangop_expmap(X[num_sections],thetan,xcin)
            g_prev          = g_prev@piecewise_expmap(X[num_sections],thetan,xcin)
            ADxin           = intdAdjgn_last@xcidotn
            eta_prev        = invAdjgn_last@(eta_prev+ADxin)

        gv.g             = g
        gv.eta           = eta
        gv.nstep         = nstep
        gv.nstep         = nstep+1
        gv.status = 0

    return gv
