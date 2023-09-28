import copy
import time
import torch
import sys, os
import argparse
import numpy as np
import numpy.linalg as LA
from datetime import datetime
from os.path import abspath, join, dirname, expanduser

sys.path.append("..")
sys.path.append("../..")
from utils import *
from visuals import *
from pde_solvers import *
from itertools import cycle

sys.path.append(join(os.getcwd(), ".."))

import matplotlib
# matplotlib.backend("QtAgg5")
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as Gridspec
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

# import matplotlib.pyplot as plt

parser = argparse.ArgumentParser('Cosserat Soft Arm Visualization')
parser.add_argument('--verbose', '-vb', action='store_true', default=False)
parser.add_argument('--elevation', '-el', type=int, default=12, help="elevation angle of plot display")
parser.add_argument('--azimuth', '-az', type=int, default=5, help="azimuth angle of plot display")
parser.add_argument('--pause_time', '-pt', type=float, default=0.02, help="time between plot updates")
parser.add_argument('--show_ss', '-ss', action='store_false', default=False, help="show steady state convergence")
parser.add_argument('--show_tube', '-st', action='store_false', default=False, help="show voxelized tube")
parser.add_argument('--plot_3d', '-pd', action='store_true', default=True, help="show 3d points of the discretixed soro over time.")
parser.add_argument('--interactive', '-it', action='store_true', default=False, help="plot evolution of soro points interactively?")
args = parser.parse_args()
print()
print('args ', args)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_file(fname, data_dir = join("/opt/SoRoPD"), verbose=True):

    first_trained = np.load(join(data_dir, fname))
    controller= first_trained["controller"]
    runtime=first_trained['runtime']
    drag = first_trained['with_drag']
    cable=first_trained["with_cable"]
    num_pieces= first_trained["num_pieces"]
    Kp = first_trained["gain_prop"]
    Kd = first_trained["gain_deriv"]
    Ki = first_trained["gain_integ"] if "gain_integ" in first_trained.keys() else None 
    num_sections= first_trained["num_sections"]
    tip_load = first_trained["tip_load"]
    solution = first_trained["solution"]

    if verbose:
        print(f"num_pieces: {num_pieces} num_sections: {num_sections}, , drag: {drag}, cable: {cable}")
        print(f"runtime: {runtime/60:.4f} mins or  {runtime/3600:.4f} hours")
        if strcmp(controller, 'PD'):        
            print(f"controller: {controller} | Kp: {Kp} | Kd: {Kd} | tip_load: {tip_load}")
        elif strcmp(controller, 'PID'):
            print(f"controller: {controller} | Kp: {Kp} | Kd: {Kd} | Ki: {Ki} | tip_load: {tip_load}")
        elif strcmp(controller, 'Backstep'):
            print(f"controller: {controller} | Kp: {Kp} | tip_load: {tip_load}")

    # batch position and velocity slices
    qslc = slice(0, 24, 1)
    qdotslc = slice(24, 48, 1)

    qbatch = solution[:, qslc, 0]
    qdbatch = solution[:, qdotslc, 0]

    # sectional slices 
    sec_slices = [(idx, slice(i, i+6, 1)) for (idx, i) in enumerate(range(0, num_pieces*6, 6))]
    qsecs = {f"qsec{sec_slice[0]+1}": qbatch[:, sec_slice[1]] for sec_slice in sec_slices}
    qdsecs = {f"qdsec{sec_slice[0]+1}": qdbatch[:, sec_slice[1]] for sec_slice in sec_slices}
    others = dict(qbatch=qbatch, qdbatch=qdbatch, Kp=Kp, Kd=Kd, Ki=Ki, pieces=num_pieces,
                        sections=num_sections)
    qsecs.update(qdsecs)
    qsecs.update(others)

    return Bundle(qsecs)


def joint_space_plotter(bundle, labelsize=18, linewidth=6, fontdict = {'fontsize':20, 'fontweight':'bold'},
						fid="unknown", save=True):
	"""
		Plot the joint coordinates for all pieces.

		Params:
			qbundle: strain and strain twist joint coordinates for the entire manipulator;
			sec: section label;
			labelsize: plot label size;
			linewidth: width of the plot;
			fontdict: font dictionary;
			fid: unique fid to save to disk
	"""
	qsec1, qsec2, qsec3, qsec4      = bundle.qsec1, bundle.qsec2, bundle.qsec3, bundle.qsec4
	qdsec1, qdsec2, qdsec3, qdsec4  = bundle.qdsec1, bundle.qdsec2, bundle.qdsec3, bundle.qdsec4
	Kp, Kd = bundle.Kp, bundle.Kd

	def single_plotter(ax, qsec, sec="I", title="Strain", color='b', deriv=False):
		t = np.linspace(0, qsec.shape[0], qsec.shape[0])/100

		qsec_norm = LA.norm(qsec, axis=1)

		ax.plot(t, qsec_norm, linewidth=linewidth, color=color, label=f'Sec. {sec}: Kp: {Kp}, KD: {Kd}.')
		if deriv:
			ax.set_ylabel(rf'$\|\eta_{sec}(t)\|$', fontdict)
		else:
			ax.set_ylabel(rf'$\|\xi_{sec}(t)\|$', fontdict)



		ax.yaxis.set_tick_params(labelsize=labelsize+4)
		if strcmp(sec, "1") or strcmp(sec, "2"):
			ax.set_title(f'{title} Field', fontdict)
		if strcmp(sec, "3") or strcmp(sec, "4"):
			ax.set_xlabel(rf'Iterations ($\times 100$)', fontdict)
			ax.xaxis.set_tick_params(labelsize=labelsize+4)
		if strcmp(sec, "2") or strcmp(sec, "4"):
			ax.axes.get_yaxis().set_ticks([])

		ax.legend(loc='upper right', fontsize=20)
		ax.grid("on")

		# ax.set_xlim([-50, 10001])
		# ax.set_ylim([qsec_norm.min()-0.05, qsec_norm.max()+.1])
		ax.set_aspect("auto")

	save_dir = join(expanduser("~"), "Documents/Papers/Pubs23/SoRoBC/figures")
	fig = plt.figure(figsize=(16,9)); fig.tight_layout()
	ax1 = fig.add_subplot(2, 2, 1);   ax2 = fig.add_subplot(2, 2, 2)
	ax3 = fig.add_subplot(2, 2, 3);   ax4 = fig.add_subplot(2, 2, 4)

	idx = 0
	colors = iter(plt.cm.inferno_r(np.linspace(.25, 1, 6)))
	for ax, qsec, lab in zip([ax1, ax2, ax3, ax4], [qsec1, qsec2, qsec3, qsec4], ["1", "2", "3", "4"]):
		single_plotter(ax, qsec, lab, color=next(colors))
		idx += 1

	fig2 = plt.figure(figsize=(16,9)); fig2.tight_layout()
	ax1 = fig2.add_subplot(2, 2, 1);   ax2 = fig2.add_subplot(2, 2, 2)
	ax3 = fig2.add_subplot(2, 2, 3);   ax4 = fig2.add_subplot(2, 2, 4)
	idx, colors = 0, iter(plt.cm.inferno_r(np.linspace(0.5, 1, 4)))
	for ax, qsec, lab in zip([ax1, ax2, ax3, ax4], [qdsec1, qdsec2, qdsec3, qdsec4], ["1", "2", "3", "4"]):
		single_plotter(ax, qsec, lab, title="Strain Twist", color=next(colors), deriv=True)
		idx += 1

	if save:
		fig.savefig(join(save_dir, f"qsec_{fid}.jpg"), bbox_inches='tight',  facecolor='w')
		fig2.savefig(join(save_dir, f"qdsec_{fid}.jpg"), bbox_inches='tight',  facecolor='w')
	plt.show()

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

def curved_tube(cx, cy, radius,height, angle, num_indices = 29):
	z = np.linspace(0, height, num_indices)
	theta = np.linspace(0, 2*angle, num_indices)

	theta_grid, z_grid=np.meshgrid(theta, z)

	x_grid = radius*np.cos(theta_grid) + cx
	y_grid = radius*np.sin(theta_grid) + cy

	yshift = np.sin(z_grid) * radius * 4
	y_grid += yshift

	return x_grid,y_grid,z_grid

def midpoints(x):
	sl = ()
	for _ in range(x.ndim):
		x = (x[sl + np.index_exp[:-1]] + x[sl + np.index_exp[1:]]) / 2.0
		sl += np.index_exp[:]
	return x

def sec_to_sec(sec_prev_center, sec_next_confs, radius=0.5):
	"""
		Retrieve the grid points as we apply the configuration transformations from
		section to section.

		Params
		======
		sec_prev_center: center points used in calculating the tube coordinates for the previous section
		sec_next_confs: time-indexed transformnation matrices used in moving to the next configuration.
	"""
	new_grid_points_3d = np.zeros((sec_next_confs.shape[0], ) + (num_indices, num_indices, num_indices))

	for t_idx in range(len(sec_next_confs)):
		new_grid_points_3d[t_idx] = sec_next_confs[t_idx, :3, 3] + sec_next_confs[t_idx, :3,:3] @ sec_prev_center
		# xyz_pyts = sec_next_confs[t_idx, :3, 3] + sec_prev_center

		# rot = Rotation.from_dcm( sec_next_confs[t_idx, :3,:3] )
		# angle = rot.as_euler('zyx', degrees=True)[-1]

		# x, y, z    = xyz_pyts
		# new_grid_points_3d[t_idx] = curved_tube(x, y, radius, z, angle)

	return new_grid_points_3d

class TubePlotter():
	def __init__(self, xc, yc, zc, fill_color, face_color='w', lw=0.5, \
					save=True, savename="I", labelsize=28, fig=None):
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

		plt.ion()

		if fig is None:
			self._fig = plt.figure(figsize=(16,9))
		else:
			self._fig = fig

		self._gs  = Gridspec.GridSpec(1, 1,self._fig)
		self._ax  = plt.subplot(self._gs[0], projection='3d')
		# self._ax = self._fig.add_subplot(projection='3d')

		self._fontdict = {"size": 16, 'weight': 'bold'}
		self.fc=face_color
		self.lw=lw
		self.save=save
		self.savename=savename
		self.fill_color=fill_color
		self.labelsize = labelsize

		self.init(xc, yc, zc)
		self._init = True

	def init(self, xc, yc, zc):

		self._ax.voxels(xc, yc, zc, filled=self.fill_color,
					facecolors= self.fc,
					edgecolors= np.clip(self.fc, 0.15, .85),
					linewidth=self.lw)

		self.set_axis_properties()
		self.draw()

	def update_tube(self, xc, yc, zc, time_step, delete_last_plot=False):


		faces = self._ax.voxels(xc, yc, zc, filled=self.fill_color,
				facecolors= self.fc,
				edgecolors= np.clip(self.fc, 0.15, .85),
				linewidth=self.lw)

		if delete_last_plot:
			plt.cla()

		self.set_axis_properties()

		if self.save:
			self._fig.set_size_inches(32, 24)

			plt.savefig(join(expanduser("~"),"Documents/Papers/Pubs23/SoRoBC", f"figures/tube_{time_step}.jpg"),                 format='jpg', facecolor=None, edgecolor=None, dpi=98, transparent=True, bbox_inches='tight')

		self.draw()
		time.sleep(args.pause_time)

		return faces

	def set_axis_properties(self, title=f"Deformable Manipulator"):

		self._ax.grid('off')
		self._ax.set_aspect('auto')
		self._fig.set_size_inches(32, 24)
		self._ax.view_init(elev=args.elevation, azim=args.azimuth)

		self._ax.axes.get_xaxis().set_ticks([])
		self._ax.axes.get_yaxis().set_ticks([])
		self._ax.tick_params(axis='both', which='major', labelsize=self.labelsize)
		self._ax.set_title(title, fontdict=self._fontdict) #  : Section {savename}

		# self._ax.set_xticks([])
		# self._ax.set_yticks([])
		# self._ax.get_xaxis().set_xticks([])
		# self._ax.get_yaxis().set_yticks([])
		# see: https://stackoverflow.com/questions/49027061/matplotlib-3d-remove-axis-ticks-draw-upper-edge-border
		self._ax.zaxis.set_ticklabels([])



		# Hide the right and top spines
		self._ax.spines[['right', 'top', 'left', 'bottom']].set_visible(False)
		plt.style.use('Solarize_Light2')
		# plt.tight_layout()

	def draw(self):
		self._fig.canvas.draw()
		self._fig.canvas.flush_events()

def iter_plot_tube_sec(gsec_confs, prev_pts, filled, fc, radius = 0.5, lw=0.5,
						   save=False, savename="I", ax=None):
	"""
		Do a voxel plot of the tranformed tube under the control law iteratively

		Params
		=======
		gsec_confs: Tranformation matrices across all time optimized indices.
		prev_pts:   3D coordinates of the center and height of the previously
					rendered cylindrical tube.
		*args: see function plot_tube_sec documentation.
	"""

	_fontdict = {"size": 16, 'weight': 'bold'}

	new_grid_points_3d, xyz_pyts = sec_to_sec(prev_pts, gsec1_confs)
	for t_idx in range(len(new_grid_points_3d)):

		xg, yg, zg = new_grid_points_3d[t_idx]

		plt.cla()
		ax.set_ylabel(f"Time: {t_idx}", fontdict=_fontdict)

		faces = plot_tube_sec(xg, yg, zg, filled, fc, lw, save, savename, ax)

		f = plt.gcf()
		f.canvas.draw()
		f.canvas.flush_events()

		time.sleep(0.5)

		if t_idx > 5:
			break

def plot_tube_sec(xc, yc, zc, fill_color, face_color, lw=0.5, save=True, savename="I", ax=None):
	"""
		Do a voxel-filled plot of the manipulator section, savename.

		Inputs:
		======
		xc: x-coordinates;
		yc: y-coordinates;
		zc: z-coordinates;

		fill_color = color with which to fill the interior of the tube;
		lw: line width for the voxel sections;
		face_color: facecolor of the outer shell;
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

	ax.view_init(elev=12.0, azim=5)

	ax.set_aspect('auto')
	ax.grid('off')

	ax.set_xticks([])
	ax.set_yticks([])
	# see: https://stackoverflow.com/questions/49027061/matplotlib-3d-remove-axis-ticks-draw-upper-edge-border
	ax.zaxis.set_ticklabels([])

	ax.tick_params(axis='both', which='major', labelsize=28)
	ax.set_title(f"Deformable Manipulator", fontdict=_fontdict) #  : Section {savename}

	# Hide the right and top spines
	ax.spines[['right', 'top', 'left', 'bottom']].set_visible(False)
	plt.style.use('Solarize_Light2')


	if save:
		figure = plt.gcf()  # get current figure
		figure.set_size_inches(32, 24)

		plt.savefig(join(expanduser("~"),"Documents/Papers/Pubs23/SoRoBC", f"figures/tube.jpg"),                 format='jpg', facecolor=None, edgecolor=None, dpi=98, transparent=True, bbox_inches='tight')

	plt.show()
	plt.tight_layout()

	return faces

def bundle_to_points(bundle):
	qbatch      = bundle.qbatch
	qdbatch     = bundle.qdbatch

	xslc        = slice(0, 24, 6)
	yslc        = slice(1, 24, 6)
	zslc        = slice(2, 24, 6)
	roll_slc    = slice(3, 24, 6)
	pitch_slc   = slice(4, 24, 6)
	yaw_slc     = slice(5, 24, 6)

	# print(f"qb: {qbatch.shape}, qdbatch: {qdbatch.shape}")

	qbatch_x, qdbatch_x  = qbatch[:,xslc], qdbatch[:,xslc]
	qbatch_y, qdbatch_y  = qbatch[:,yslc], qdbatch[:,yslc]
	qbatch_z, qdbatch_z  = qbatch[:,zslc], qdbatch[:,zslc]

	qbatch_roll, qdbatch_roll    = qbatch[:,roll_slc], qdbatch[:,roll_slc]
	qbatch_pitch, qdbatch_pitch  = qbatch[:,pitch_slc], qdbatch[:,pitch_slc]
	qbatch_yaw, qdbatch_yaw      = qbatch[:,yaw_slc], qdbatch[:,yaw_slc]
	# print(f"qb_x: {qbatch_x.shape}, qb_y: {qbatch_y.shape} , qb_z: {qbatch_z.shape}")

	return Bundle(dict(x=qbatch_x, y=qbatch_y, z=qbatch_z, roll=qbatch_roll,
			      pitch=qbatch_pitch, yaw=qbatch_yaw,
				  xd=qdbatch_x, yd=qdbatch_y, zd=qdbatch_z,
				  rolld=qdbatch_roll, pitchd=qdbatch_pitch,
				  yawd=qdbatch_yaw,))

if __name__== "__main__":
	controller="PD"
	data_dir = join("/opt/SoRoPD")
	_fontdict = {'fontsize':35, 'fontweight':'bold'}

	sorted(os.listdir(data_dir))

	files = sorted(os.listdir(data_dir))
	bundles = [np.nan for _ in range(len(files))]
	for idx in range(len(files)):
		bundles[idx] = load_file(files[idx])

	cur_bundle = bundles[-1]
	if args.plot_3d:
		bundle_pts = bundle_to_points(bundles[-1])

		fig = plt.figure(figsize=(16,9))
		fig.set_size_inches(32, 24)
		# surface plot of the transformed points
		ax = fig.subplots(1, 1, subplot_kw={"projection": "3d"})
		colors = cycle(plt.cm.coolwarm(np.linspace(.25, 1, 10)))

		if args.interactive:
			num_secs = cur_bundle.qbatch.shape[-1]//6
			g_confs = joint_screws_to_confs(cur_bundle.qbatch)

			cz = 12
			r, g, b = np.indices((num_secs, num_secs, num_secs)) / (num_secs-1.0)
			rc = midpoints(r); gc = midpoints(g); bc = midpoints(b)
			filled = (2*np.pi* (rc) *  cz ).astype(np.bool_)

			# combine the color components
			fc = np.zeros(filled.shape + (3,))
			fc[..., 0] = rc
			fc[..., 1] = gc
			fc[..., 2] = bc

			plt.ion()
			for t_idx in range(1, len(g_confs)):
				# Plot the surface.
				pts = np.zeros((num_secs, 3))
				for sec in range(num_secs):
					cur_rot      = g_confs[t_idx,sec,:,:].squeeze()[:3,3]
					cur_trans    = g_confs[t_idx,sec,:,:].squeeze()[:3,3]

					prev_trans    = g_confs[t_idx-1,sec,:,:].squeeze()[:3,3]
					pts[sec] = cur_rot @ cur_trans + prev_trans

				# plot all sectional points together
				surf = ax.plot3D(pts[:,0], pts[:,1], pts[:,2], color="blue", linewidth=4,
				 					linestyle="-.", antialiased=True, label=f"{t_idx}/{len(g_confs)}")

				ax.set_zlim(pts[:,2].min(), pts[:,2].max())
				ax.zaxis.set_major_locator(LinearLocator(10))
				# A StrMethodFormatter is used automatically
				ax.zaxis.set_major_formatter('{x:.02f}')

				# Add a color bar which maps values to colors.
				# fig.colorbar(surf, shrink=1.0, aspect=5)

				ax.view_init(elev=args.elevation, azim=args.azimuth)

				ax.set_aspect('auto')
				ax.grid('off')

				ax.set_xticks([])
				ax.set_yticks([])
				# see: https://stackoverflow.com/questions/49027061/matplotlib-3d-remove-axis-ticks-draw-upper-edge-border
				ax.zaxis.set_ticklabels([])
				ax.legend(loc='lower right')

				# ax.tick_params(axis='both', which='major', labelsize=28)
				ax.set_title(f"{num_secs} section manipulator", fontdict=_fontdict) #  : Section {savename}

				# Hide the right and top spines
				ax.spines[['right', 'top', 'left', 'bottom']].set_visible(False)
				plt.style.use('Solarize_Light2')

				# update the tube
				# xx, yy, zz = np.meshgrid(pts[:,0], pts[:,1], pts[:,2])
				# plot_tube_sec(xx[...,0], yy[...,0], zz[...,0], fill_color=filled, face_color=fc, lw=0.5, save=False, ax=ax[1])
				#
				# ax.draw_artist(ax.patch)
				fig.canvas.flush_events()

				plt.pause(args.pause_time)
				plt.cla()

			plt.ioff()

	if args.show_ss:
		for bundle in bundles:
			joint_space_plotter(bundle,  labelsize=18, linewidth=6, fid=f"{idx:0>3}")

	if args.show_tube:
		radius, cz = 0.25, 12.0
		cx, cy = 0.5, 0.5

		joints = bundle[-1]
		gsec1_confs = joint_screws_to_confs(joints.qsec1)
		gsec2_confs = joint_screws_to_confs(joints.qsec2)
		gsec3_confs = joint_screws_to_confs(joints.qsec3)
		gsec4_confs = joint_screws_to_confs(joints.qsec4)

		# cx, cy = gsec1_confs[0,:2,3]
		xc, yc, zc = curved_tube(cx, cy, radius, cz, angle=np.pi)

		num_indices = 29
		r, g, b = np.indices((num_indices, num_indices, num_indices)) / (num_indices-1.0)
		rc = midpoints(r); gc = midpoints(g); bc = midpoints(b)
		filled = (2*np.pi* (rc) *  cz ).astype(np.bool_)

		# combine the color components
		colors = np.zeros(filled.shape + (3,))
		colors[..., 0] = rc
		colors[..., 1] = gc
		colors[..., 2] = bc

		# plot_tube_sec(xc, yc, zc, filled, colors, lw=0.5, save=True, savename="I", ax=None)
		# fig = plt.figure(figsize=(16,9)) #.add_subplot(projection='3d')
		tube_plotter = TubePlotter(xc, yc, zc, fill_color=filled, \
									 face_color=colors, lw=0.5, \
									 save=False, fig=None)
		# plt.show() # this and ioff in init gets it done
		# t_idx = 0
		# while True:
		#     tube_plotter.update_tube(xc, yc, zc, t_idx, delete_last_plot=True)
		#     t_idx += 1
