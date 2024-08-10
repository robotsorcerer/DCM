__all__ = ["load_file"]

__author__      = "Lekan Molu"
__maintainer__  = "Lekan Molu"
__license__     = "Molux Licence"
__copyright__   = "2022, Discrete Cosserat SoRO Analysis in Python"
__credits__     = "There are None."
__email__       = "patlekno@icloud.com"
__date__        = "December 23, 2022"
__status__      = "Completed"

import numpy as np
from os.path import join 
from utils import strcmp, Bundle 

def load_file(fname, data_dir="/opt/SoRoPD", verbose=True):

    first_trained   = np.load(join(data_dir, fname))
    controller      = first_trained["controller"]
    runtime         = first_trained['runtime']
    drag            = first_trained['with_drag']
    cable           = first_trained["with_cable"]
    num_pieces      = first_trained["num_pieces"]
    Kp              = first_trained["gain_prop"]
    Kd              = first_trained["gain_deriv"]
    Ki              = first_trained["gain_integ"] if "gain_integ" in first_trained.keys() else None 
    gravity         = first_trained["gravity"] if "gravity" in first_trained.keys() else None 
    num_sections    = first_trained["num_sections"]
    tip_load        = first_trained["tip_load"]
    solution        = first_trained["solution"]
    qd              =  first_trained['desired_strain'] if 'desired_strain' in first_trained.keys() else None 
    soltime         =  first_trained['soltime'] if 'soltime' in first_trained.keys() else None 

    if verbose:
        print(f"fname: {fname}, strain_goal(q^d): {qd}")
        print(f"num_pieces: {num_pieces} num_sections: {num_sections}, drag: {drag}, cable: {cable}")
        print(f"runtime: {runtime/60:.4f} mins or  {runtime/3600:.4f} hours.")
        if strcmp(controller, 'PD'):        
            print(f"controller: {controller} | Kp: {Kp} | Kd: {Kd} | tip_load: {tip_load}")
        elif strcmp(controller, 'PID'):
            print(f"controller: {controller} | Kp: {Kp} | Kd: {Kd} | Ki: {Ki} | tip_load: {tip_load}")
        elif strcmp(controller, 'Backstep'):
            print(f"controller: {controller} | Kp: {Kp} | tip_load: {tip_load}")

    qslc = slice(0, num_pieces*6, 1); qdotslc = slice(num_pieces*6, 2*num_pieces*6, 1)
    if len(solution.shape)<3:
        qbatch  = solution[:, qslc]; qdbatch = solution[:, qdotslc]
    else:
        qbatch  = solution[:, qslc, 0]; qdbatch = solution[:, qdotslc, 0]

    sec_slices  = [(idx, slice(i, i+6, 1)) for (idx, i) in enumerate(range(0, num_pieces*6, 6))]
    qsecs       = {f"qsec{sec_slice[0]+1}": qbatch[:, sec_slice[1]] for sec_slice in sec_slices}
    qdsecs      = {f"qdsec{sec_slice[0]+1}": qdbatch[:, sec_slice[1]] for sec_slice in sec_slices}
    others      = dict(qbatch=qbatch, qdbatch=qdbatch, Kp=Kp, Kd=Kd, Ki=Ki, pieces=num_pieces, \
                    cable=cable, drag=drag, controller=controller, sections=num_sections, qd=qd, \
                    tip_load=tip_load, runtime=runtime, gravity=gravity, fname=fname, tsol=soltime)
    qsecs.update(qdsecs)
    qsecs.update(others)    

    return Bundle(qsecs)

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