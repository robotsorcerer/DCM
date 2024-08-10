#!/usr/bin/env python
# coding: utf-8

# In[1]:


import copy
import time 
import torch
import argparse
import sys, os
import numpy as np
import numpy.linalg as LA
from os.path import abspath, join, dirname, expanduser

# sys.path.append(join(os.getcwd(), ".."))
# sys.path.append(join(os.getcwd(), "../.."))
from utils import *
from pde_solvers import *
from visuals import *
from utils.rotation import *

import matplotlib 
import matplotlib.pyplot as plt 

from datetime import datetime 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser('Cosserat Soft Arm Visualization')
parser.add_argument('--fname', '-fn', type=str, 
                    default="072223_19_25_01_4_pieces_tipload.10N_PD_Control.npz",
                    help="one of npz files in /opt/SoRoPD")
args = parser.parse_args()
print()
print('args ', args)


# In[2]:


controller="PD" 
data_dir = join("/opt/SoRoPD") #, f"{controller}_controller_"+  datetime.strftime(datetime.now(), '%m%d%y_%H_%M'))
fname = args.fname


first_trained = np.load(join(data_dir, fname))
runtime=first_trained['runtime']
drag = first_trained['with_drag']
cable=first_trained["with_cable"]
num_pieces= first_trained["num_pieces"]
Kp = first_trained["gain_prop"]
Kd = first_trained["gain_deriv"]
num_sections= first_trained["num_sections"]
tip_load = first_trained["tip_load"]
controller = first_trained["controller"]
solution = first_trained["solution"]

print(f"num_pieces: {num_pieces} num_sections: {num_sections}, , drag: {drag}, cable: {cable}")
print(f"runtime: {runtime/60:.4f} mins or  {runtime/3600:.4f} hours")
print(f"controller: {controller} Kp: {Kp} Kd: {Kd}, tip_load: {tip_load}")


def get_q_secs(solution):
    # take the joint position and velocity slices
    qslc = slice(0, 24, 1)
    qdotslc = slice(24, 48, 1)

    qbatch = solution[:, qslc, 0]
    qdbatch = solution[:, qdotslc, 0]

    # sectional slices 
    sec1 = slice(0, 6)
    sec2 = slice(6, 12)
    sec3 = slice(12, 18)
    sec4 = slice(18, 24)

    # retrieve the sectional batches
    qsec1 = qbatch[:, sec1]
    qsec2 = qbatch[:, sec2]
    qsec3 = qbatch[:, sec3]
    qsec4 = qbatch[:, sec4]

    return qsec1, qsec2, qsec3, qsec4 

def show_joint_trajs():
    fig = plt.figure(figsize=(16,9))
    labelsize=18
    fontdict = {'fontsize':20, 'fontweight':'bold'}

    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)


    colors = iter(plt.cm.inferno_r(np.linspace(.25, 1, 6)))
    qsec1, qsec2, qsec3, qsec4 = get_q_secs(solution)

    qsec1_norm = LA.norm(qsec1, axis=1)
    qsec2_norm = LA.norm(qsec2, axis=1)
    qsec3_norm = LA.norm(qsec3, axis=1)
    qsec4_norm = LA.norm(qsec4, axis=1)


    def joint_space_plotter(ax1, qsec_norm, sec="1", labelsize=18, color="b", linewidth=6, fontdict = {'fontsize':20, 'fontweight':'bold'}):
        t =np.linspace(0, qsec1.shape[0], qsec1.shape[0])

        ax1.plot(t, qsec_norm, linewidth=linewidth, color=color, label=f'Sec. {sec}')
        ax1.set_ylabel(rf'$\|\xi_{sec}(t)\|$', fontdict)

        if strcmp(sec, "1") or strcmp(sec, "2"):
            ax1.set_title(f'Strain Field', fontdict)
        if strcmp(sec, "3") or strcmp(sec, "4"):
            ax1.set_xlabel('Length of iterations ', fontdict)
            ax1.xaxis.set_tick_params(labelsize=labelsize+4)
            
        ax1.legend(loc='best', fontsize=23)
        ax1.grid("on")
        ax1.yaxis.set_tick_params(labelsize=labelsize+4)

    joint_space_plotter(ax1, qsec1_norm, sec="1", color=next(colors), labelsize=labelsize, linewidth=6)
    joint_space_plotter(ax2, qsec2_norm, sec="2", color=next(colors), labelsize=labelsize, linewidth=6)
    joint_space_plotter(ax3, qsec3_norm, sec="3", color=next(colors), labelsize=labelsize, linewidth=6)
    joint_space_plotter(ax4, qsec4_norm, sec="4", color=next(colors), labelsize=labelsize, linewidth=6)
    fig.tight_layout()

    # plt.suptitle("PD Controller on Fluid-actuated 4-section manipulators, each with 41 DCM pieces") #, fontdict=fontdict)
    # plt.savefig(join(expanduser("~"), "Documents/Papers/Pubs23/SoRoBC/figures/joint_space_4secs.jpg"), bbox_inches='tight',facecolor='None')
    # plt.show()
    return qsec1, qsec2, qsec3, qsec4 


def joint_screws_to_confs(joint_screw):
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
    assert len(joint_screw.shape) == 2, "joint screw coordinate must include batch dim"
    assert joint_screw.shape[1] == 6, "joint space screw coordinate system must be in R^6"
    assert isinstance(joint_screw, np.ndarray), "joint space screw system must be in ndarray format"


    def local_skew_sym(vec):
        """
            Convert a 3-vector  to a skew symmetric matrix.
        """

        # assert vec.size(0) == 3, "vec must be a 3-sized vector."
        if vec.ndim>1: vec = vec.squeeze()
        skew = np.array(([ [ 0, -vec[2].item(), vec[1].item() ],
                [ vec[2].item(), 0, -vec[0].item() ],
                [-vec[1].item(), vec[0].item(), 0 ]
            ])) #.to(vec.device)

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

    gsec_conf = np.zeros((len(joint_screw), 4, 4))

    # get the Lie group transformation now
    for t in range(len(joint_screw)):
        gsec_conf[t,:,:] = local_lie_group(joint_screw[t])

    return gsec_conf


qsec1, qsec2, qsec3, qsec4  = show_joint_trajs()
gsec1_confs = joint_screws_to_confs(qsec1)
gsec2_confs = joint_screws_to_confs(qsec2)
gsec3_confs = joint_screws_to_confs(qsec3)
gsec4_confs = joint_screws_to_confs(qsec4)


num_indices = 29

def curved_tube(cx, cy, radius,height, angle):
    z = np.linspace(0, height, num_indices)
    theta = np.linspace(0, 2*angle, num_indices)
    
    theta_grid, z_grid=np.meshgrid(theta, z)

    x_grid = radius*np.cos(theta_grid) + cx
    y_grid = radius*np.sin(theta_grid) + cy

    yshift = np.sin(z_grid) * radius/1.5
    y_grid += yshift  

    return x_grid,y_grid,z_grid

def midpoints(x):
    sl = ()
    for _ in range(x.ndim):
        x = (x[sl + np.index_exp[:-1]] + x[sl + np.index_exp[1:]]) / 2.0
        sl += np.index_exp[:]
    return x


def plot_tube_sec(xc, yc, zc, fill_color, face_color, lw=0.5, save=True, savename="I", ax=None):
    """ 
        Do a voxel-filled plot of the manipulator section, savename.

        Inputs:
        ======
        xc: x-coordinates;
        yc: y-coordinates;
        zc: z-coordinates;
        fill_color = color with which to fill the interior of the tube;
        face_color: facecolor of the outer shell;
        lw: line width for the voxel sections;
        save: whether to save the section to disk;
        savename: name of file to save;
        ax: axis handle for pyplot's 3D projection plotter

        Lekan Molu, 07/31/2023
    """

    assert xc.ndim == 2, "Dimension of x coordinates cannot be more than 2"
    assert yc.ndim == 2, "Dimension of y coordinates cannot be more than 2"
    assert zc.ndim == 2, "Dimension of z coordinates cannot be more than 2"

    _fontdict = {"size": 16, 'weight': 'bold'}
    
    if ax is None:
        ax = plt.figure(figsize=(16,7)).add_subplot(projection='3d')

    faces = ax.voxels(xc, yc, zc, filled=fill_color,
                facecolors= face_color,
                edgecolors= np.clip(face_color, 0.15, .85),  
                linewidth=0.5)
          
    ax.view_init(elev=5.0, azim=10)

    ax.set_aspect('auto')
    ax.grid('off')

    ax.set_xticks([])
    ax.set_yticks([])
    # see: https://stackoverflow.com/questions/49027061/matplotlib-3d-remove-axis-ticks-draw-upper-edge-border
    ax.zaxis.set_ticklabels([])

    ax.tick_params(axis='both', which='major', labelsize=28)
    ax.set_title(f"Deformable Manipulator: Section {savename}", fontdict=_fontdict)

    if save:
        plt.savefig(join(expanduser("~"),"Documents/Papers/Pubs23/SoRoBC", f"figures/tube_{savename}.jpg"),                 format='jpg', facecolor='gray', edgecolor='m', dpi=78)

    # plt.show()
    plt.tight_layout()

    return faces 


def iter_plot_tube_sec(gsec_confs, prev_pts, filled, fc, radius = 0.5, lw=0.5,                         save=False, savename="I", ax=None):
    """ 
        Do a voxel plot of the tranformed tube under the control law iteratively

        Params
        =======
        gsec_confs: Tranformation matrices across all time optimized indices.
        prev_pts:   3D coordinates of the center and height of the previously 
                    rendered cylindrical tube.
        *args: see function plot_tube_sec documentation.
    """
    
    for t_idx in range(len(gsec_confs)):
        xyz_pyts = gsec_confs[t_idx, :3, 3] + prev_pts

        rot = Rotation.from_dcm( gsec_confs[t_idx, :3,:3] )
        angle = rot.as_euler('zyx', degrees=True)[-1]
        
        x, y, z    = xyz_pyts
        xg, yg, zg = curved_tube(x, y, radius, z, angle)

        plt.cla()
        ax.set_ylabel(f"Time: {t_idx}", fontdict=fontdict)

        faces = plot_tube_sec(xg, yg, zg, filled, fc, lw, save, savename, ax)
        prev_pts   = xyz_pyts 


        f = plt.gcf()
        f.canvas.draw()
        f.canvas.flush_events()

        time.sleep(0.5)



ax = plt.figure(figsize=(16,9)).add_subplot(projection='3d')
radius, cz = 0.25, 8.0
cx, cy = 0.5, 0.5 
xc, yc, zc = curved_tube(cx, cy, radius, cz, angle=np.pi)
 
r, g, b = np.indices((num_indices, num_indices, num_indices)) / (num_indices-1.0)
rc = midpoints(r); gc = midpoints(g); bc = midpoints(b)
filled = (2*np.pi* (rc) *  cz ).astype(np.bool_)

# combine the color components
colors = np.zeros(filled.shape + (3,))
colors[..., 0] = rc
colors[..., 1] = gc
colors[..., 2] = bc

# plot initial tube 
faces = plot_tube_sec(xc, yc, zc, filled, colors, 0.5, True, "I", ax)
prev_pts = np.array([cx, cy, cz])

# plot subsequent tubes 
iter_plot_tube_sec(gsec1_confs, prev_pts, filled, colors, radius, ax = ax)


