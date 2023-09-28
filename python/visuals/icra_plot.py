import copy
import time 
import torch
import sys, os
import numpy as np
import numpy.linalg as LA
from os.path import abspath, join, dirname, expanduser

sys.path.append(join(os.getcwd(), ".."))

from utils import *
from visuals import *
from pde_solvers import *

from scipy.io import savemat
from datetime import datetime 

import matplotlib 
import matplotlib.pyplot as plt 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_fontdict = {'fontsize':34, 'fontweight':'bold'}

def get_fig_ax():
    fig = plt.figure(figsize=(28,12)); 
    ax = fig.add_subplot(1, 1, 1)

    return ax 

def do_plot(data_dir="/opt/SoRoPD.bak/", _labelsize=18):
    plt.ion()

    ax = get_fig_ax()
    fname = "072423_10_30_23_with_cable_4_pieces_tipload.10N_PD_Control.npz"
    bundle = load_file(fname, data_dir)
    plot_axis_strains(ax, bundle, 0.0, _labelsize, plot_type="vel", \
                    title=f'{bundle.controller} Velocity Controller', lw=8, annotate=True, xy_pos=None, \
                    ylabel=r'${{ \dot {{\xi_{y} }} }}$',  ylim=(-0.1, 0.5), xlim=(0.03, 2.50), save=True)

    ax = get_fig_ax()
    fname = "072723_18_51_22_with_drag_4_pieces_tipload.10N_PD_Control.npz"
    bundle = load_file(fname, data_dir)
    plot_axis_strains(ax, bundle, 0.0, _labelsize, title=f'{bundle.controller} Velocity Controller', plot_type="vel", lw=8, \
                annotate=True, xy_pos=(5, 10), ylabel=r'${{ \dot {{\xi_{y} }} }}$', xlim=(0.01, 10.00), \
                ylim=[-0.03, 0.03], plt_len=10000, save=True)

    ax = get_fig_ax()
    fname = "072423_01_09_27_4_pieces_tipload.10N_PD_Control.npz"
    bundle = load_file(fname, data_dir)
    plot_axis_strains(ax, bundle, 0.0, _labelsize, title=f'{bundle.controller} Velocity Controller', plot_type="vel", annotate=True,
                 lw=8, ylabel=r'${{ \dot {{\xi_{y} }} }}$', xlim=(0.100, 10), ylim=[-0.022, 0.06], plt_len=10000, 
                 save=True, xy=(5.35, 0.03))
    
    ax = get_fig_ax()
    fname = "090623_11_02_36_with_cable_10_pieces_tipload.0.2N_PD_Control.npz"
    bundle = load_file(fname, data_dir)
    plot_axis_strains(ax, bundle, 0.0, _labelsize, title=f'{bundle.controller} Velocity Controller', plot_type="vel", annotate=True,
                 xy_pos=(5.5, -0.045), 
                plt_idx=3, ylabel=r'${{ \dot {{\xi_{y} }} }}$',  xlim=(-0.100, 10.000), ylim=[-0.052, 0.05], lw=6, 
                 _fontdict = {'fontsize':40, 'fontweight':'bold'}, save=True)

    ax = get_fig_ax()
    fname="091023_08_55_01_with_cable_10_pieces_tipload.1.0N_PD_Control.npz"
    bundle = load_file(fname, data_dir)
    qbatch, plt_idx = bundle.qbatch, 5
    plot_axis_strains(ax, bundle, 0.0, _labelsize, title=f'{bundle.controller} Velocity Controller', plot_type="vel", 
                plt_len=len(qbatch), plt_idx=plt_idx, ylabel=r'${{ \dot {{\xi_{y} }} }}$',  xlim=(-0.200, 10), annotate=False,
                ylim=[-.002, 0.03], lw=6, xy_pos=(25, 0.023), save=True)                 