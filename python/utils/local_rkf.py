__all__ = ["runge_kutta_fehlberg"]

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

import time
import copy
import torch
import warnings
import numpy as np
from utils.cosserat_utils import *
from utils.matlab_utils import eps, Bundle
from torch.linalg import pinv, norm

torch.set_default_dtype(torch.float64)



def interp_split(time_interp, t, y, h, f1, f3, f4, f5, f6, f7):
    """
        Employs the data computed in the Runge-Kutta-Fehlberg algorithm to approximate the
        solution at time time_interp.

        .time_interp: scalar or row vector of interpolated times.
    """
    device = y.device

    bi12 = -183./64; bi13 = 37./12; bi14 = -145./128
    bi32 = 1500./371; bi33 = -1000./159; bi34 = 1000./371
    bi42 = -125./32;   bi43 = 125./12;    bi44 = -375./64
    bi52 = 9477./3392; bi53 = -729./106;  bi54 = 25515./6784
    bi62 = -11./7;     bi63 = 11./3;      bi64 = -55./28
    bi72 = 3./2;       bi73 = -4.;        bi74 = 5./2

    s = torch.tensor(([time_interp - t])).to(device)/h

    y_interp = torch.zeros(((y.shape[0], s.shape[-1]))).to(device)

    for jj in range(s.shape[-1]):
        sj  = s[jj]
        sj2 = sj * sj
        bs1 = (sj + sj2*(bi12 + sj*(bi13 + bi14*sj)))
        bs3 = (     sj2*(bi32 + sj*(bi33 + bi34*sj)))
        bs4 = (     sj2*(bi42 + sj*(bi43 + bi44*sj)))
        bs5 = (     sj2*(bi52 + sj*(bi53 + bi54*sj)))
        bs6 = (     sj2*(bi62 + sj*(bi63 + bi64*sj)))
        bs7 = (     sj2*(bi72 + sj*(bi73 + bi74*sj)))

        rhs =  y + h*(f1*bs1 + f3*bs3 + f4*bs4 + f5*bs5 + f6*bs6 + f7*bs7)
        y_interp[:,jj] = rhs.squeeze()

    return y_interp

def runge_kutta_fehlberg(OdeFun, x, tspan, rtol=1e-3, approx_scheme="fixed", output_sol=False):
    """
    The Runge-Kutta-Fehlberg Method
    -------------------------------

    This is the Runge-Kuhta-Fehlberg (ode45) method. It guarantees solution to the
    IVP by solving the problem twice using two different step sizes and compares answers at
    the mesh points corresponding to the larger step size. If the two answers are in close agreement,
    the approximation is accepted. If the two answers do not agree to a specified accuracy, the step
    size is reduced. If the answers agree to more significant digits than required, the step size is increased.

    Comments: .rk45_fehlberg is an implementation of the explicit Runge-Kutta (4,5) pair of
    Dormand and Prince called variously RK5(4)7FM, DOPRI5, DP(4,5) and DP54. It uses a "free" interpolant of
    order 4 (communicated privately by Dormand and Prince).  Local extrapolation is done.

    Inputs:
        OdeFun: RHS of diff. eq. to be integrated.
        x (Cupy array): State, an initial condition.
        rtol (float): Tolerance of the relative error from iterate to iterate for solution
             to be accepted.
        tspan: A Numpy list [start, end] that specifies over what time horizon to integrate the dynamics.
        approx_scheme: "fixed" or "adaptive".

    Reference: https://maths.cnam.fr/IMG/pdf/RungeKuttaFehlbergProof.pdf

    Author: Lekan Molu, Jan 16, 2023
    """
    device = x.device
    start_time = time.time()

    solver_name = 'ode45'
    output_ty  = not output_sol 
    nsteps, nfevals, nfailed = 0, 0, 0
    
    # record time to save the data
    t_last_saved = 0.0
    t_save_interval = 1e-4

    f3d = []
    sol = Bundle({});     

    if output_sol:
        sol.solver = solver_name 

    y0 = x
    t0 = tspan[0]
    ntspan = len(tspan)
    tfinal = tspan[-1]
    neq = x.shape[0]
    outputs = torch.arange(0, neq)
    tdir = int(torch.sign(tfinal-t0))

    nxt = 1   # next entry in tspan 
    hmax = 2
    normy = 0

    nfevals += 1

    threshold = 0.01
    normcontrol = False
    htspan = abs(tspan[1] - tspan[0])

    refine = 4
    if ntspan > 2:
        outputAt = 1 # output only at tspan points
    elif refine <= 1:
        outputAt = 2  # computed points, no refinement
    else:
        outputAt = 3  # computed points, with refinement
        S = torch.asarray(torch.linspace(0, refine-1) / refine).to(device)

    f0 = OdeFun(t0, y0)
    if isinstance(f0, tuple):
        t0, f0 = f0[0], f0[-1]

    t, y = copy.copy(t0), copy.copy(y0)

    t_record, y_record = t.unsqueeze(0), y
    runtime_record = [0.0]

    # Allocate memory if we're generating output.qa
    nout = 0
    tout, yout = [], []

    if output_sol:
        chunk = int(min(max(100, 50*refine), refine+np.floor((2**11)/neq)))
        tout = torch.zeros((1, chunk), dtype =float).to(device)
        yout = torch.zeros((neq, chunk), dtype =float).to(device)
        f3d  = torch.zeros((neq, 7, chunk), dtype =float).to(device)
    else: # output only at tspan points
        if ntspan > 2:
            tout = torch.zeros((1, ntspan), dtype=float).to(device)
            yout = torch.zeros((neq, ntspan), dtype=float).to(device)
        else:
            chunk = min(max(100, 50*refine), refine+np.floor((2**13)/neq))
            tout = torch.zeros((1, chunk), dtype=float).to(device)
            yout = torch.zeros((neq, chunk), dtype=float).to(device)

    nout = 1
    tout[0, nout-1] = t 
    yout[:,nout-1] = y.squeeze()

    # Initialize method parameters.
    power = 1./5
    A = torch.asarray(([[1./5, 3./10, 4./5, 8./9, 1., 1.]])).to(device) # Still used by restarting criteria
    """
    B = [
        1/5         3/40    44/45   19372/6561      9017/3168       35/384
        0           9/40    -56/15  -25360/2187     -355/33         0
        0           0       32/9    64448/6561      46732/5247      500/1113
        0           0       0       -212/729        49/176          125/192
        0           0       0       0               -5103/18656     -2187/6784
        0           0       0       0               0               11/84
        0           0       0       0               0               0
        ];
    E = [71/57600; 0; -71/16695; 71/1920; -17253/339200; 22/525; -1/40];
    """

    # Same values as above extracted as scalars (1 and 0 values ommitted)
    a2 = 1.0/5.0
    a3 = 3.0/10.0
    a4 = 4.0/5.0
    a5 = 8.0/9.0

    b11 = 1.0/5.0
    b21 = 3.0/40.0
    b31 = 44.0/45.0
    b41 = 19372.0/6561.0
    b51 = 9017.0/3168.0
    b61 = 35.0/384.0
    b22 = 9.0/40.0
    b32 = -56.0/15.0
    b42 = -25360.0/2187.0
    b52 = -355.0/33.0
    b33 = 32.0/9.0
    b43 = 64448.0/6561.0
    b53 = 46732.0/5247.0
    b63 = 500.0/1113.0
    b44 = -212.0/729.0
    b54 = 49.0/176.0
    b64 = 125.0/192.0
    b55 = -5103.0/18656.0
    b65 = -2187.0/6784.0
    b66 = 11.0/84.0

    e1 = 71.0/57600.0
    e3 = -71.0/16695.0
    e4 = 71.0/1920.0
    e5 = -17253.0/339200.0
    e6 = 22.0/525.0
    e7 = -1.0/40.0
    hmin = 16*eps 

    # Compute an initial step size h using y'(t).
    absh = min(hmax, htspan)
    if normcontrol:
        rh = (norm(f0, ord=2) / max(normy,threshold)) / (0.8 * rtol**power)
    else:
        rh = norm(torch.divide(f0, torch.maximum(abs(y),torch.tensor(threshold).to(device))), ord = float('inf')) / (0.8 * rtol**power)
    
    if absh * rh > 1.0:
        absh = 1.0 / rh
    absh = max(absh, hmin)

    f1 = f0

    # piecewise_observables([t, tfinal], y[outputs], gv)

    done = False
    while not done:
        """
        By default, hmin is a small number such that t+hmin is only slightly
        different than t.  It might be 0 if t is 0.
        """
        hmin   = 16*eps
        absh   = min(hmax, max(hmin, absh))    # couldn't limit absh until new hmin
        h      = tdir * absh

        # Stretch the step if within 10# of tfinal-t.
        if 1.1*absh >= abs(tfinal - t):
            h = tfinal - t
            absh = abs(h)
            done = True

        # LOOP FOR ADVANCING ONE STEP.
        Failed = True                      # no failed attempts
        while True:
            y2 = y + h * (b11*f1 )
            t2 = t + h * a2
            f2 = OdeFun(t2, y2)[1]

            y3 = y + h * (b21*f1 + b22*f2 )
            t3 = t + h * a3
            f3 = OdeFun(t3, y3)[1]

            y4 = y + h * (b31*f1 + b32*f2 + b33*f3 )
            t4 = t + h * a4
            f4 = OdeFun(t4, y4)[1]

            y5 = y + h * (b41*f1 + b42*f2 + b43*f3 + b44*f4 )
            t5 = t + h * a5
            f5 = OdeFun(t5, y5)[1]

            y6 = y + h * (b51*f1 + b52*f2 + b53*f3 + b54*f4 + b55*f5 )
            t6 = t + h
            f6 = OdeFun(t6, y6)[1]

            tnew = t + h
            if done:
                tnew = copy.copy(tfinal)
            h = tnew - t

            ynew = y + h *  ( b61 * f1 + b63 * f3 + b64 * f4 + b65 * f5 + b66 * f6 )
            f7 = OdeFun(tnew, ynew)[1]

            nfevals += 6

            # Estimate the error.
            NNrejectStep = False
            fE = f1*e1 + f3*e3 + f4*e4 + f5*e5 + f6*e6 + f7*e7

            err = absh * norm(torch.divide(fE, torch.maximum(torch.maximum(torch.abs(y),torch.abs(ynew)), torch.tensor(threshold).to(device))), ord=float('inf'))

            """
                Accept the solution only if the weighted error is no more than the
                tolerance rtol.  Estimate an h that will yield an error of rtol on
                the next step or the next try when taking this step, as the case may
                be, and use 0.8 of this value to avoid failures.
            """
            if err > rtol:
                nfailed += 1
                if absh <= hmin:
                    warnings.warn(f'Integration Tolerance Not Met:: t: {t}, hmin: {hmin}')                    
                    sol_out = Bundle(dict(nsteps=nsteps, nfailed=nfailed, 
                                          nfevals =nfevals, t=tout, z=yout))
                    return sol_out #piecewise_observables(t, z, gv)

                if Failed:
                    Failed = False
                    if NNrejectStep:
                        absh = max(hmin, 0.5*absh)
                    else:
                        absh = max(hmin, absh * max(0.1, 0.8*(rtol/err)**power))
                else:
                    absh = max(hmin, 0.5 * absh)
                h = tdir * absh
                done = False
            else:                                # Successful step
                NNreset_f7 = False
                break

        nsteps += 1

        if output_sol:
            nout += 1
            # tout has shape (1, tspan)
            if nout > tout.shape[1]:
                tout = torch.hstack((tout, torch.zeros((1, chunk).to(device), dtype=float) ))
                yout = torch.hstack((yout, torch.zeros((neq, chunk).to(device), dtype=float) ))
                f3d  = torch.cat((f3d, torch.zeros((neq, 7, chunk).to(device), dtype=float)), axis=2)

            tout[0, nout] = tnew
            yout[:,nout] = ynew.squeeze()
            f3d[:,:,nout] = torch.hstack((f1, f2, f3, f4, f5, f6, f7))

        if output_ty:
            if outputAt==1:  # output only at tspan point
                nout_new =  0
                tout_new = torch.zeros((1, 1)).to(device)
                yout_new = torch.zeros((neq, 1)).to(device)

                while nxt <= ntspan-1:
                    if tdir * (tnew - tspan[nxt]) < 0:
                        break
                    nout_new += 1
                    tout_new = torch.hstack((tout_new, tspan[nxt].reshape(1,1)))
                    if tspan[nxt] == tnew:
                        yout_new = torch.hstack((yout_new, ynew))
                    else:
                        y_interp = interp_split(tspan[nxt],t,y,h,f1,f3,f4,f5,f6,f7)
                        yout_new = torch.hstack(([yout_new, y_interp]))
                    nxt += 1
            
                # remove the inf terms at the 0-th indices for the two variables
                if tout_new.shape[1]>1 or yout_new.shape[1]>1:
                    tout_new = tout_new[:,1:]
                    yout_new = yout_new[:,1:]

            if nout_new > 0:
                if output_ty:
                    oldnout = nout 
                    nout += nout_new 
                    if nout > tout.shape[1]:
                        tout = torch.hstack((tout, torch.zeros((1, chunk)).to(device)))
                        yout = torch.hstack((yout, torch.zeros((neq, chunk)).to(device)))
                    idx = np.arange(oldnout+1, nout+1)
                    if idx.shape[0] > 1:
                        warnings.warn('Size of idx is greater than 1.')

                    tout[0, idx-1] = tout_new.squeeze()
                    yout[:,idx-1] = yout_new

        if done:
            break

        # If there were no failures, compute a new h
        if Failed:
            # Note that absh may shrink by 0.8, and that err may be 0
            temp = 1.25 * (err/rtol)**power

            if temp > 0.2:
                absh /= temp
            else:
                absh *= 5.0 

        # Advance the integration one step
        t = copy.copy(tnew)
        y = copy.copy(ynew)

        # if NNreset_f7:
        #     """ 
        #         Used f7 for unperturbed solution to interpolate.  
        #         Now reset f7 to move along constraint. 
        #     """
        #     f7 = OdeFun(tnew, ynew)
        #     nfevals += 1

        f1 = copy.copy(f7)  # Already have f(tnew,ynew)

        if t - t_last_saved > t_save_interval:
            t_last_saved = t

            t_record = torch.cat((t_record, t.unsqueeze(0)))
            y_record = torch.hstack((y_record, y))

            runtime_record.append(time.time() - start_time)

            sol_out = Bundle(dict(nsteps=nsteps, nfailed=nfailed, nfevals =nfevals, \
                           nout=nout, t=t_record, z=y_record, runtime = runtime_record))
                
    sol_out = Bundle(dict(nsteps=nsteps, nfailed=nfailed, nfevals =nfevals, \
                           nout=nout, t=tout, z=yout, runtime = time.time() - start_time))

    return sol_out

