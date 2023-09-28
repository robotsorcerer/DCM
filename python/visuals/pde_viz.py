__all__ = ["PDEVisualizer"]

import os
import time
import numpy as np
from os.path import join, expanduser
from skimage import measure
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class PDEVisualizer(object):
    def __init__(self, fig=None, ax=None, winsize=None,
                labelsize=18, fontdict=None):

        if winsize is None:
            self.winsize =(16, 9)
        if (fig and ax) is None:
            self._fig = plt.figure(figsize=winsize)
            self._fig.tight_layout()

        self._labelsize = labelsize
        self._projtype = 'rectilinear'
        if fontdict is None:
            self._fontdict = {'fontsize':14, 'fontweight':'bold'}

        self.draw()

    def visualize(self, control, target, pod):
        #  visualize our work
        NNN = max(control.T.shape)
        targ_xyz_eff = (target.xyz_eff_all - target.xyz_eff_init)*1000
        ax1 = self._fig.add_subplot(2, 4, 1)
        for i in range(3):  ax1.plot(control.T[0], targ_xyz_eff[i,:])
        ax1.set_ylabel(' Tumor Trans. (mm)', self._fontdict);
        # ax1.set_xlabel('Time (sec)', self._fontdict);
        ax1.set_xlim(left=0, right=max(control.T[0]));
        # ax1.grid('on');
        ax1.set_ylim(-2.7, 2.5);ax1.grid('on');
        ax1.legend(['X','Y','Z'], loc='upper left')
        ax1.set_title('Tumor (End Eff Frame) ', self._fontdict)
        ax1.xaxis.set_tick_params(labelsize=self._labelsize)
        ax1.yaxis.set_tick_params(labelsize=self._labelsize)
        del targ_xyz_eff

        ax6 = self._fig.add_subplot(2, 4, 5)
        targ_psi_eff = (target.psi_eff_all - target.psi_eff_init)*180/math.pi
        targ_psi_eff -= mat_round(targ_psi_eff/180)*180
        for i in [1, 0, 2]:
            ax6.plot(control.T[0], targ_psi_eff[i,:])
        ax6.set_ylabel(' Tumor Rotation(deg)', self._fontdict);
        ax6.set_xlabel('Time (sec)', self._fontdict);
        ax6.set_xlim(left=0, right=max(control.T[0]));
        ax6.grid('on');
        ax6.legend(['Roll','Pitch','Yaw'], loc='best')
        # ax6.set_title('(End eff frame)', self._fontdict)
        ax6.xaxis.set_tick_params(labelsize=self._labelsize)
        ax6.yaxis.set_tick_params(labelsize=self._labelsize)
        del targ_psi_eff

        # column II
        ax2 = self._fig.add_subplot(2, 4, 2)
        pod_xyz = pod.eff.xyz_all*1000
        pod_xyz = pod_xyz - expand(pod_xyz[:, 0], 1)
        for i in range(3):
            ax2.plot(control.T[0], pod_xyz[i,:])
        ax2.set_ylabel('End-Eff Trans. (mm)', self._fontdict);
        ax2.set_xlim(left=0, right=max(control.T[0]));
        ax2.grid('on');
        ax2.legend(['X','Y','Z'], loc='best')
        ax2.set_title('End-Eff Trans. (LINAC Frame)', self._fontdict)
        ax2.xaxis.set_tick_params(labelsize=self._labelsize)
        ax2.yaxis.set_tick_params(labelsize=self._labelsize)

        ax6 = self._fig.add_subplot(2, 4, 6)
        pod_rpy = pod.eff.psi_all*180/math.pi
        pod_rpy -= np.tile(expand(pod_rpy[:,0], 1), [NNN])  # I do not understand why you do this, Xinmin
        # pod_rpy -= expand(pod_rpy[:,0], 1)  # I do not understand why you do this, Xinmin
        pod_rpy -= mat_round(pod_rpy/360)*360
        for i in [1, 0, 2]:
            ax6.plot(control.T[0], pod_rpy[i,:])
        ax6.set_ylabel('End-Eff Rot. (deg)', self._fontdict);
        ax6.set_xlabel('Time (sec)', self._fontdict);
        ax6.set_xlim(left=0, right=max(control.T[0]));
        ax6.grid('on');
        ax6.legend(['Roll','Pitch','Yaw'], loc='best')
        ax6.xaxis.set_tick_params(labelsize=self._labelsize)
        ax6.yaxis.set_tick_params(labelsize=self._labelsize)

        del pod_rpy
        ax3 = self._fig.add_subplot(2, 4, 3)
        targ_xyz = target.xyz_all*1000
        for i in range(3):
            ax3.plot(control.T[0], targ_xyz[i,:])
        ax3.set_ylabel('Tumor Trans. (mm)', self._fontdict);
        ax3.set_xlim(left=0, right=max(control.T[0]));
        ax3.grid('on');
        ax3.legend(['X','Y','Z'], loc='upper right')
        ax3.set_title('Tumor (LINAC Frame)', self._fontdict)
        ax3.xaxis.set_tick_params(labelsize=self._labelsize)
        ax3.yaxis.set_tick_params(labelsize=self._labelsize)

        ax7 = self._fig.add_subplot(2, 4, 7)
        targ_rpy = target.psi_all*180/math.pi
        for i in [1, 0, 2]:
            ax7.plot(control.T[0], targ_rpy[i,:])
        ax7.set_ylabel('Tumor Rot. (deg)', self._fontdict);
        ax7.set_xlabel('Time (sec)', self._fontdict);
        ax7.set_xlim(left=0, right=max(control.T[0]));
        ax7.grid('on');
        ax7.legend(['Roll','Pitch','Yaw'], loc='best')
        ax7.xaxis.set_tick_params(labelsize=self._labelsize)
        ax7.yaxis.set_tick_params(labelsize=self._labelsize)

        ax4 = self._fig.add_subplot(2, 4, 4)
        norm_targ_xyz = expand(np.sqrt(np.sum(targ_xyz*targ_xyz, axis=0)), 0)
        norm_targ_rpy = expand(np.sqrt(np.sum(targ_rpy*targ_rpy, axis=0)), 0)
        norm_targ = np.r_[norm_targ_xyz, norm_targ_rpy, norm_targ_xyz+norm_targ_rpy]
        for i in range(3):
            ax4.plot(control.T[0], norm_targ[i,:])
        ax4.set_ylabel('MSE (mm-deg)', self._fontdict);
        ax4.set_xlim(left=0, right=max(control.T[0]));
        # ax4.set_ylim(-0.5, 7);
        ax4.grid('on');
        ax4.legend(['Trans.','Rot.','Trans+Rot'], loc='upper right')
        ax4.set_title('Tumor Pose MSE', self._fontdict)
        ax4.xaxis.set_tick_params(labelsize=self._labelsize)
        ax4.yaxis.set_tick_params(labelsize=self._labelsize)

        ax9 = self._fig.add_subplot(2, 4, 8)
        pod_legs = pod.leg_length_all*1000
        for i in range(min(pod_legs.shape)):
            ax9.plot(control.T[0], pod_legs[i,:])
        ax9.set_ylabel('Leg Length (mm)', self._fontdict);
        ax9.set_xlabel('Time (sec)', self._fontdict);
        ax9.set_xlim(left=0, right=max(control.T[0]));
        ax9.grid('on');
        ax9.legend(['Leg I','Leg II','Leg III',
                    'Leg IV', 'Leg V', 'Leg VI'], loc='best')
        ax9.set_title('Pod Legs Displacement', self._fontdict)
        ax9.xaxis.set_tick_params(labelsize=self._labelsize)
        ax9.yaxis.set_tick_params(labelsize=self._labelsize)

    def update_tube(self, amesh, ls_mesh, pgd_mesh, time_step, delete_last_plot=False):
        """
            Inputs:
                data - BRS/BRT data.
                amesh - zero-level set mesh of the analytic TTR mesh.
                ls_mesh - zero-level set mesh of the levelset tb TTR mesh.
                pgd_mesh - zero-level set mesh of the pgd TTR mesh.
                time_step - The time step at which we solved  this BRS/BRT.
                delete_last_plot - Whether to clear scene before updating th plot.
        """
        self._ax[1].grid('on')

        self._ax[1].axes.get_xaxis().set_ticks([])
        self._ax[1].axes.get_yaxis().set_ticks([])


        if self.grid.dim==3:
            self._ax[1].add_collection3d(amesh.mesh)

            self._ax[1].add_collection3d(amesh.mesh)
            self._ax[1].view_init(elev=self.params.elevation, azim=self.params.azimuth)

            xlim = (amesh.verts[:, 0].min(), amesh.verts[:,0].max())
            ylim = (amesh.verts[:, 1].min(), amesh.verts[:,1].max())
            zlim = (amesh.verts[:, 2].min(), amesh.verts[:,2].max())

            self._ax[1].set_xlim3d(*xlim)
            self._ax[1].set_ylim3d(*ylim)
            self._ax[1].set_zlim3d(*zlim)

            self._ax[1].set_xlabel("X", fontdict = self.params.fontdict.__dict__)
            self._ax[1].set_title(f'BRT at {time_step}.', fontweight=self.params.fontdict.fontweight)

        elif self.grid.dim==2:
            self._ax[0].cla() if delete_last_plot else self._ax[0].cla()
            CS1 = self._ax[0].contour(self.grid.xs[0], self.grid.xs[1], amesh, linewidths=3,  colors='red')
            self._ax[0].grid('on')
            self._ax[0].set_xlabel(rf'$x_1$', fontdict=self.params.fontdict.__dict__)
            self._ax[0].set_ylabel(rf'$x_2$', fontdict=self.params.fontdict.__dict__)
            # self._ax[0].set_title(f'Analytic and Numerical TTR @ {time_step} secs.', fontdict =self.params.fontdict.__dict__)
            self._ax[0].set_title(f'Analytic TTR@{time_step} secs.', fontdict =self.params.fontdict.__dict__)

            self._ax[0].tick_params(axis='both', which='major', labelsize=28)
            self._ax[0].tick_params(axis='both', which='minor', labelsize=18)
            self._ax[0].set_xlim([-1.02, 1.02])
            self._ax[0].set_ylim([-1.01, 1.01])
            self._ax[0].clabel(CS1, CS1.levels, inline=True, fmt=self.fmt, fontsize=self.params.fontdict.fontsize)

            self._ax[1].cla() if delete_last_plot else self._ax[1].cla()
            CS2 = self._ax[1].contour(self.g_rom.xs[0], self.g_rom.xs[1], pgd_mesh, linewidths=3, colors='magenta')
            self._ax[1].grid('on')
            self._ax[1].set_xlabel(rf'$x_1$', fontdict=self.params.fontdict.__dict__)
            self._ax[1].set_ylabel(rf'$x_2$', fontdict=self.params.fontdict.__dict__)
            self._ax[1].set_title(f'LF TTR@{time_step} secs.', fontdict=self.params.fontdict.__dict__)
            self._ax[1].tick_params(axis='both', which='major', labelsize=28)
            self._ax[1].tick_params(axis='both', which='minor', labelsize=18)
            self._ax[1].set_xlim([-1.02, 1.02])
            self._ax[1].set_ylim([-1.01, 1.01])
            self._ax[1].clabel(CS2, CS2.levels, inline=True, fmt=self.fmt, fontsize=self.params.fontdict.fontsize)

        plt.tight_layout()
        f = plt.gcf()
        f.savefig(join(expanduser("~"),"Documents/Papers/Safety/PGDReach", f"figures/dint_ttr_{time_step}.jpg"),
            bbox_inches='tight',facecolor='None')
        self.draw()
        time.sleep(self.params.pause_time)

    def fmt(self, x):
        s = f"{x:.2f}"
        if s.endswith("0"):
            s = f"{x:.0f}"
        return rf"{s} \s" if plt.rcParams["text.usetex"] else f"{s}"

    def add_legend(self, linestyle, marker, color, label):
        self._ax_legend.plot([], [], linestyle=linestyle, marker=marker,
                color=color, label=label)
        self._ax_legend.legend(ncol=2, mode='expand', fontsize=self.params.fontdict.fontsize)

    def draw(self, ax=None):
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()
