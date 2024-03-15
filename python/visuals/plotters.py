__all__ = ["plot_axis_strains", "joint_space_plotter"]

import random 
import numpy as np 
import scipy.linalg as LA
from os.path import join, expanduser 
from utils import strcmp, Bundle, isnumeric
import matplotlib.pyplot as plt 
import matplotlib as mpl 
mpl.use("QtAgg")

# avoid Type 3 fonts for paperplaza submissions ===> See http://phyletica.org/matplotlib-fonts/
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

def plot_axis_strains(ax, bundle, ref, _labelsize = 18, title=None, plot_type="pos",
                        ylabel=None, xlim=(-0.5, 10), ylim=[-0.02, 0.2], xy_pos=None,
                        _fontdict = {'fontsize':50, 'fontweight':'bold'}, \
                        plt_idx=4, lw=10, plt_len=None, save=False, annotate=True,  savename=None, \
                        save_dir = join(expanduser("~"), "Documents/Papers/MSRYears/SoRoBC/figures")):
   
    

    if strcmp(plot_type, "pos"):
        colors = iter(plt.cm.turbo_r(np.arange(0.15, bundle.pieces, bundle.pieces)))
        colors = iter(["red", "orange", "magenta", "olive", "purple", "pink", "green", "cyan", "black", "yellow"])
        # colors = iter(plt.cm.turbo_r(np.arange(0, bundle.pieces))) #*.25, bundle.pieces)))
    else:
        colors = iter(plt.cm.viridis_r(np.arange(0, bundle.pieces))) #*.25, bundle.pieces)))
        colors = iter(["red", "orange", "magenta",  "green", "cyan", "black", "yellow", "olive", "purple", "pink"])

    runtime, controller =  bundle.runtime, bundle.controller 
    qbatch = bundle.qbatch if strcmp(plot_type, "pos") else bundle.qdbatch
    if strcmp(controller, "backstep"):
        gains = rf'Kp: {bundle.Kp}, ${{\mathcal{{F}}}}_p^y$: {bundle.tip_load}N.'
    elif strcmp(controller, "PD"):
        gains = rf'Kp: {bundle.Kp}, KD: {bundle.Kd}, ${{\mathcal{{F}}}}_p^y$: {bundle.tip_load}N.'
    elif strcmp(controller, "PID"):
        gains = rf'Kp: {bundle.Kp}, KD: {bundle.Kd}, Ki: {bundle.Ki}, ${{\mathcal{{F}}}}_p^y$: {bundle.tip_load}N.' 

    tref = np.linspace(0, len(qbatch), len(qbatch))/1000
    plt_len = len(tref) if plt_len is None else plt_len

    # plot the linear and angular strains of the first piece
    if isnumeric(ref):
        ax.plot(tref[:plt_len], [ref]*plt_len, linewidth=lw, linestyle="solid", color='blue', label=f"Ref.")    
        lidx = 0
        for i in range(plt_idx, qbatch.shape[1], 6):
            ax.plot(tref[:plt_len], qbatch[:plt_len,i], linewidth=lw, color=next(colors), linestyle="--",label=f"Sec. {lidx+1}")
            lidx += 1 
    else:       
        ax.plot(tref[:plt_len], np.sin(10*tref[:plt_len]), linewidth=lw, linestyle="solid", color='b', label=f"Ref.")
        # plot the trajs
        lidx = 0
        for i in range(plt_idx, qbatch.shape[1], 6):
            ax.plot(tref[:plt_len], qbatch[:plt_len,i], linewidth=lw, color=next(colors), linestyle="--",label=f"Sec. {lidx+1}")
            lidx += 1

    if annotate:
        if xy_pos is None:
            xy_pos = (np.mean(xlim), np.mean(ylim)-.04) if "vary" in title.lower() else (np.mean(xlim), np.mean(ylim))
        fontsize=35 if "vary" in title.lower() else 40
        ax.annotate(gains, xy=xy_pos, xycoords="data", va="center", ha="center", fontsize=fontsize, 
                        fontweight='bold', bbox=dict(boxstyle="round", fc="w"))
        
    ax.set_xlim(xlim); ax.set_ylim(ylim); ax.grid('on')
    
    ax.set_ylabel(ylabel,  fontdict= {'fontsize':45, 'fontweight':'bold'})
    ax.set_xlabel(rf'Total RKF Iterations (X100)',  fontdict= _fontdict)
    ax.set_title(title, _fontdict)

    ax.tick_params(axis='both', which='major', labelsize=28)
    ax.tick_params(axis='both', which='minor', labelsize=28)
    
    ax.legend(loc='best', fontsize=25) #, prop=dict(weight='bold'))

    if save:
        fig_batch = plt.gcf()
        fig_batch.tight_layout()
        if not savename:
            fn = bundle.fname.split("_")
            savename = "_".join([x for x in (fn[0], *fn[4:-1])])
            # savename = f"{controller}"
            # if bundle.cable:
            #     savename += f"_cable"
            # if bundle.drag:
            #     savename += f"_drag"
            # if "vary" in title.lower():
            #     savename += f"_varying"
        print(f"savename: {savename}")
        fig_batch.savefig(join(save_dir, savename+".eps"), bbox_inches='tight',  \
            facecolor='w', format="eps", dpi=1000, transparent=True)

def joint_space_plotter(bundle, labelsize=18, linewidth=6, fontdict = {'fontsize':35, 'fontweight':'bold'}, 
                        fid="unknown", sections_plot=False, save=True, ax1=None, ax2=None):
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
    qbatch, qdbatch = bundle.qbatch, bundle.qdbatch
    Kp, Kd = bundle.Kp, bundle.Kd 

    def single_plotter(ax, qsec, sec="I", title="Strain", color='b', deriv=False, full_robot=False):
        t = np.linspace(0, qsec.shape[0], qsec.shape[0])/100

        qsec_norm = LA.norm(qsec, axis=1)


        ax.yaxis.set_tick_params(labelsize=labelsize+4)
        if not full_robot:
            _label = f'Tip load: {bundle.tip_load} | Sec. {sec}: | Kp: {Kp} | KD: {Kd}.' if strcmp(bundle.controller, 'PD') else \
                    f'Tip load: {bundle.tip_load} | Sec. {sec} | Kp: {Kp} | KD: {Kd} | Ki: {bundle.Ki}.'
            ax.plot(t, qsec_norm, linewidth=linewidth, color=color, label=_label)
            if strcmp(sec, "1") or strcmp(sec, "2"):
                ax.set_title(f'{title} Field',  fontdict= {'fontsize':25, 'fontweight':'bold'})
            if strcmp(sec, "3") or strcmp(sec, "4"):
                ax.set_xlabel(rf'Optim. steps ($\times 100$)',  fontdict= {'fontsize':35, 'fontweight':'bold'})
                ax.xaxis.set_tick_params(labelsize=labelsize+4)        
            if strcmp(sec, "2") or strcmp(sec, "4"):
                ax.axes.get_yaxis().set_ticks([]) 
            if deriv:
                ax.set_ylabel(rf'$\|{{\mathbf{{\eta_{sec} }} }}\|_2$',  fontdict= {'fontsize':35, 'fontweight':'bold'})
            else:
                ax.set_ylabel(rf'$\|{{\mathbf{{\xi_{sec} }} }}\|_2$',  fontdict= {'fontsize':35, 'fontweight':'bold'})
            ax.legend(loc='upper right', fontsize=20)
        else:
            _label = f'Tip load: {bundle.tip_load} | Kp: {Kp} | KD: {Kd}.' if strcmp(bundle.controller, 'PD') else \
                    f'Tip load: {bundle.tip_load} | Kp: {Kp} | KD: {Kd} | Ki: {bundle.Ki}.'
            ax.plot(t, qsec_norm, linewidth=linewidth, color=color, label=_label)
            if strcmp(title, "Strain Field"):
                ax.set_ylabel(rf"$\|{{\mathbf{{q(\xi)}} }}\|_2$",  fontdict= {'fontsize':40, 'fontweight':'bold'})
            else:
                ax.set_ylabel(rf"$\|\dot{{\mathbf{{q}} }}(\xi)\|_2$",  fontdict= {'fontsize':40, 'fontweight':'bold'})
            ax.xaxis.set_tick_params(labelsize=labelsize+4)   
            ax.set_xlabel(rf'Optim. steps ($\times 100$)', fontdict= {'fontsize':35, 'fontweight':'bold'}) 

            ax.legend(loc='best', fontsize=35)
            
        ax.set_aspect("auto")
    
    # save_dir = join(expanduser("~"), "Documents/Papers/Pubs23/SoRoBC/figures")
    save_dir = join(expanduser("~"), "Documents/Papers/MSRYears/SoRoBC/figures")

    colors = iter(plt.cm.inferno_r(np.linspace(.25, 1, 6)))
    if sections_plot:
        idx = 0
        fig = plt.figure(figsize=(16,9)); fig.tight_layout()
        ax1 = fig.add_subplot(2, 2, 1);   ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3);   ax4 = fig.add_subplot(2, 2, 4)
        for ax, qsec, lab in zip([ax1, ax2, ax3, ax4], [qsec1, qsec2, qsec3, qsec4], ["1", "2", "3", "4"]):
            single_plotter(ax, qsec, lab, color=next(colors))
            idx += 1

        fig2 = plt.figure(figsize=(16,9)); fig2.tight_layout()
        ax1 = fig2.add_subplot(2, 2, 1);   ax2 = fig2.add_subplot(2, 2, 2)
        ax3 = fig2.add_subplot(2, 2, 3);   ax4 = fig2.add_subplot(2, 2, 4)
        idx, colors = 0, iter(plt.cm.inferno_r(np.linspace(0.5, 2, 40)))
        for ax, qsec, lab in zip([ax1, ax2, ax3, ax4], [qdsec1, qdsec2, qdsec3, qdsec4], ["1", "2", "3", "4"]):
            single_plotter(ax, qsec, lab, title="Strain Twist", color=next(colors), deriv=True)
            idx += 1

    if not ax1 or ax2:
        fig_batch = plt.figure(figsize=(28,12)); 
        ax1 = fig_batch.add_subplot(1, 2, 1);   
        ax2 = fig_batch.add_subplot(1, 2, 2)
    single_plotter(ax1, qbatch, rf"$\|\mathbf{{q}}\|$", title="Strain Field", color=next(colors), full_robot=True)
    single_plotter(ax2, qdbatch, f"$\|\dot{{\mathbf{{q}} }}\|$", title="Strain Velocity Field", color=next(colors), full_robot=True)
    fig_batch.tight_layout()
    if save:
        fig_batch.savefig(join(save_dir, f"qbatch_{fid}.jpg"), bbox_inches='tight',  facecolor='w')
        if sections_plot:
            fig.savefig(join(save_dir, f"qsec_{fid}.jpg"), bbox_inches='tight',  facecolor='w')
            fig2.savefig(join(save_dir, f"qdsec_{fid}.jpg"), bbox_inches='tight',  facecolor='w')
    plt.show()
