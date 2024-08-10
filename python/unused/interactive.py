# this for generating norm of batched plots
# plot all in one screen shot 
ax1 = fig_batch.add_subplot(1, 2, 1);   
ax2 = fig_batch.add_subplot(1, 2, 2)

for idx in range(len(fnames)):
    bundle = load_file(fnames[idx], data_dir)
    joint_space_plotter(bundle,  labelsize=18, linewidth=6, fid="".join(fnames[0].split("_")[0:3]), \
        sections_plot=False, save=False)
    # plt.close()
    # break 
    print()
    

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
    
    save_dir = join(expanduser("~"), "Documents/Papers/Pubs23/SoRoBC/figures")

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

    x, y, z = bundle_pts.x, bundle_pts.y, bundle_pts.z
    roll, pitch, yaw = bundle_pts.roll, bundle_pts.pitch, bundle_pts.yaw
    if args.verbose:
        print(f"x: {x.shape}, y: {y.shape} z: {z.shape}")
    # colors = cycle(cm_colors)

    t_idx = 0
    cur_pts = np.array([x[t_idx], y[t_idx], z[t_idx]])

    for t_idx in range(1, len(x)):

        surf = ax.plot3D(x[t_idx], y[t_idx], z[t_idx], color="k",
                               linewidth=2, antialiased=True)

        # Customize the z axis.
        ax.set_zlim(z.min(), z.max())
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

        # ax.tick_params(axis='both', which='major', labelsize=28)
        ax.set_title(f"{bundles[-1].qbatch.shape[-1]//6} section manipulator", fontdict=_fontdict) #  : Section {savename}

        # Hide the right and top spines
        ax.spines[['right', 'top', 'left', 'bottom']].set_visible(False)
        plt.style.use('Solarize_Light2')


        ax.draw_artist(ax.patch)
        fig.canvas.flush_events()

        plt.pause(args.pause_time)
        plt.cla()
    # plt.ioff()
else:
