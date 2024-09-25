__all__ = ["load_file", "do_ordered_blkdiag", "separate_mass_mat", "ThreadReturnClass"]

__author__      = "Lekan Molu"
__maintainer__  = "Lekan Molu"
__license__     = "Molux Licence"
__copyright__   = "2022, Discrete Cosserat SoRO Analysis in Python"
__credits__     = "There are None."
__email__       = "patlekno@icloud.com"
__date__        = "December 23, 2022"
__status__      = "Completed"

import torch 
from threading import Thread 
import numpy as np
from os.path import join 
from utils import strcmp, Bundle, isfield

class ThreadReturnClass(Thread):
    
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return
    
def do_ordered_blkdiag(M: torch.tensor) -> torch.tensor:
    """
        Given a square matrix, M, compute the ordered eigen values, 
        sort the eigenvectors based on the ordered eigen values (=P), 
        and return the block diagonal similarity transformed matrix
        by computing 

        Mnew = P^{-1} * M * P
    """

    mvals, mvecs  = torch.linalg.eig(M)
    vals_sort_idx = mvals.real.argsort().flip(dims=[0]) 
    mvecs_sorted = mvecs[:, vals_sort_idx].squeeze().real

    M = torch.inverse(mvecs_sorted).mul(M).mul(mvecs_sorted)

    return M


def separate_mass_mat(massmat: torch.Tensor, core_slice: slice, \
                      pert_slice: slice, pert_uprt: tuple, pert_bot_left: tuple)->Bundle:
    """
        Separate the mass inertia tensor into a core and perturbation 
        part based on supplied slice indices, core_slice.

        Inputs: 
            Massmat: Diagonalized massmat that sorts the mass of the sections in 
            order from most weighty sections to least weighty section according to 
            the eigenvalues of the mass matrix.

            core_slice: A slice of the core mass matrix that allows us to index Mp
            and Mc.

            pert_slice: slices of the perturbed matrix == subblock of Mp

            pert_uprt: indices for H_pert^core 

            pert_bot_left: indices for H_core^pert

        Returns:
            A Bundle of Mc and Mp, Hcore, Hpert, H^core_pert and H^pert_core
            based on equation (11) in the paper.
    """
    # Mc = torch.zeros_like(massmat); 
    # massmat = do_ordered_blkdiag(massmat)

    Mp = torch.zeros_like(massmat)
    Mp_mask = torch.ones_like(massmat, dtype=bool); Mp_mask[core_slice, core_slice] = False 
    Mc = massmat[core_slice, core_slice];   Mp[Mp_mask==True] = massmat[Mp_mask==True]
    
    hpert = Mp[pert_slice, pert_slice]
    hcore = Mc[core_slice, core_slice]; 
    hcore_pert = Mp[pert_uprt[0], pert_uprt[1]]
    hpert_core = Mp[pert_bot_left[0], pert_bot_left[1]]
    
    
    hmat = Bundle(dict(Mc = Mc, Mp=Mp, hcore=hcore, hpert=hpert, \
                       hcore_pert=hcore_pert, hpert_core=hpert_core))
    
    return hmat


def load_file(fname, data_dir="/opt/SoRoPD", verbose=True):

    bundle   = Bundle(np.load(join(data_dir, fname)))
    # if verbose:
    #     print(f"fname: {fname}, strain_goal(q^d): {qd}")
    #     print(f"num_pieces: {num_pieces} num_sections: {num_sections}, drag: {drag}, cable: {cable}")
    #     print(f"runtime: {runtime/60:.4f} mins or  {runtime/3600:.4f} hours.")
    #     if strcmp(controller, 'PD'):        
    #         print(f"controller: {controller} | Kp: {Kp} | Kd: {Kd} | tip_load: {tip_load}")
    #     elif strcmp(controller, 'PID'):
    #         print(f"controller: {controller} | Kp: {Kp} | Kd: {Kd} | Ki: {Ki} | tip_load: {tip_load}")
    #     elif strcmp(controller, 'Backstep'):
    #         print(f"controller: {controller} | Kp: {Kp} | tip_load: {tip_load}")

    # solution = bundle.solution 
    if isfield(bundle, 'solution'): # and len(solution) == 1: # non-spt
        solution = bundle.solution
        qslc = slice(0, bundle.num_pieces*6, 1); qdotslc = slice(bundle.num_pieces*6, 2*bundle.num_pieces*6, 1)
        if len(solution.shape)<3:
            qbatch  = solution[:, qslc]; qdbatch = solution[:, qdotslc]
        else:
            qbatch  = solution[:, qslc, 0]; qdbatch = solution[:, qdotslc, 0]

        sec_slices  = [(idx, slice(i, i+6, 1)) for (idx, i) in enumerate(range(0, bundle.num_pieces*6, 6))]
        qsecs       = {f"qsec{sec_slice[0]+1}": qbatch[:, sec_slice[1]] for sec_slice in sec_slices}
        qdsecs      = {f"qdsec{sec_slice[0]+1}": qdbatch[:, sec_slice[1]] for sec_slice in sec_slices}
        others      = dict(qbatch=qbatch, qdbatch=qdbatch)
        qsecs.update(qdsecs)
        qsecs.update(others) 
    else: # fast and slow solutions 
        sol_slow = bundle.slow_solution; sol_fast = bundle.fast_solution
        qslc_slow = slice(0, bundle.num_slow_pieces*6, 1); qdotslc_slow = slice(bundle.num_slow_pieces*6, 2*bundle.num_slow_pieces*6, 1)
        qslc_fast = slice(0, bundle.num_fast_pieces*6, 1); qdotslc_fast = slice(bundle.num_fast_pieces*6, 2*bundle.num_fast_pieces*6, 1)

        if len(sol_slow.shape)<3:
            qbatch_slow  = sol_slow[:, qslc_slow]; qdbatch_slow = sol_slow[:, qdotslc_slow]
            qbatch_fast  = sol_fast[:, qslc_fast]; qbatch_fast  = sol_fast[:, qdotslc_fast]
        else:
            qbatch_slow  = sol_slow[:, qslc_slow, 0]; qdbatch_slow = sol_slow[:, qdotslc_slow, 0]
            qbatch_fast  = sol_fast[:, qslc_fast, 0]; qdbatch_fast = sol_fast[:, qdotslc_fast, 0]

        sec_slow_slice   = [(idx, slice(i, i+6, 1)) for (idx, i) in enumerate(range(0, bundle.num_slow_pieces*6, 6))]
        sec_fast_slice   = [(idx, slice(i, i+6, 1)) for (idx, i) in enumerate(range(0, bundle.num_fast_pieces*6, 6))]

        qsecs = {}

        qsecs_slow       = {f"qsec_slow{sec_slow_slice[0]+1}": qbatch_slow[:, sec_slow_slice[1]] for sec_slow_slice in sec_slow_slice}
        qsecs_fast       = {f"qsec_fast{sec_fast_slice[0]+1}": qbatch_fast[:, sec_fast_slice[1]] for sec_fast_slice in sec_fast_slice}

        qdsecs_slow      = {f"qdsec_slow{sec_slow_slice[0]+1}": qdbatch_slow[:, sec_slow_slice[1]] for sec_slow_slice in sec_slow_slice}
        qdsecs_fast      = {f"qdsec_fast{sec_fast_slice[0]+1}": qdbatch_fast[:, sec_fast_slice[1]] for sec_fast_slice in sec_fast_slice}

        others           = dict(qbatch_slow=qbatch_slow, qdbatch_slow=qdbatch_slow, 
                                qbatch_fast=qbatch_fast, qdbatch_fast=qdbatch_fast)
        
        qsecs.update(qsecs_slow); qsecs.update(qsecs_fast); qsecs.update(qdsecs_slow); qsecs.update(qdsecs_fast); qsecs.update(others);   
        bundle.qsecs = Bundle(qsecs)

    return bundle 

def joint_screws_to_confs(batch_screw):
    """
        Convert a joint space screw corrdinate system for a robot's single section
        configuration to a curve parameterized by a rotation matrix R \in SO(3)
        and a translation vector T \in R^3 so that (T, R) \in SE(3).

        Inputs: 
            joint_screw: screw coordinate system for a robot section parameterized by 
            angular strains w(t) and linear strains k(t). In essence, if the joint space 
            is denoted by q(t), then we must have q(t) = [w(t), k(t)]

            The joint screw is [num_iter x 6] in dimensions based on the optimization process
            used in computing q_init to reach q_d.

    """
    assert len(batch_screw.shape) == 2, "screw coordinate must include batch dim"
    assert batch_screw.shape[1] % 6==0, "joint space screw coordinate system must be in R^6"
    assert isinstance(batch_screw, np.ndarray), "joint space screw system must be in ndarray format"

    screw_len = 6
    num_secs = batch_screw.shape[-1]//screw_len
    def local_skew_sym(vec):
        """
            Convert a 3-vector  to a skew symmetric matrix.
        """

        if vec.ndim>1: vec = vec.squeeze()
        skew = np.array(([ [ 0, -vec[2].item(), vec[1].item() ],
                [ vec[2].item(), 0, -vec[0].item() ],
                [-vec[1].item(), vec[0].item(), 0 ]
            ])) 

        return skew
        
    def local_lie_group(screw):
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
        group  = np.zeros((4,4)) 
        group[:3, :3] = local_skew_sym(screw[:3])
        group[:3, 3]  = screw[3:]

        return group

    if num_secs>1:
        gsec_conf = np.zeros((len(batch_screw), num_secs, 4, 4))
    else:
        gsec_conf = np.zeros((len(batch_screw), 4, 4))
        

    # get the Lie group transformation now
    for t in range(len(batch_screw)):
        if num_secs>1:
            sec_idx = 0
            for sec in range(0, batch_screw.shape[-1], screw_len):
                gsec_conf[t,sec_idx,:,:] = local_lie_group(batch_screw[t, sec:sec+screw_len])
                sec_idx += 1
        else:
            gsec_conf[t,:,:] = local_lie_group(batch_screw[t])

    return gsec_conf    

def test_all():
    a = torch.arange(1, 26).reshape(5, 5).float()
    # print(a)
    diaged = do_ordered_blkdiag(a)
    print('Sorted Mass')
    print(diaged)

    hmat = separate_mass_mat(diaged, core_slice=slice(0, 3), pert_slice=slice(3, 5), 
                                    pert_uprt=(slice(0, 3), slice(3, 5)), pert_bot_left=(slice(3, 5), slice(0, 3)))
    print('Mc')
    print(hmat.Mc); print('Mp'); print(hmat.Mp)

    print('hcore:'); print(); print(hmat.hcore); print('hpert'); print(hmat.hpert)
    print('hcp'); print(); print(hmat.hcore_pert)
    print('hpc'); print(); print(hmat.hpert_core)    