import sys
import argparse
import numpy as np
import glob
import re
import random

from scipy.spatial.transform import Rotation as scipyR

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import pyplot

import viz_tools as mviz

sys.path.append('../my_utils')
import mf_manip as mfm
import mmisc
import my_eval_rpe as meval_rpe

from draw_pose_err_stats import get_stats_errs_file


if __name__ == '__main__':

    seqs = ['shapes_6dof', 'shapes_translation', 'poster_translation', 'hdr_poster',
            'boxes_6dof', 'boxes_translation', 'hdr_boxes', 'dynamic_6dof', 'dynamic_translation']

    base_path = '../../results/uslam_plots'

    plot_colors = ['darkorange', 'yellow', 'green', 'indigo', 'red', 'navy', 'black', 'teal',
                   'magenta', 'blue', 'darkolivegreen', 'purple']
    len_colors = len(plot_colors)

    plot_lstyles = ['--', '-']
    len_plot_lst = len(plot_lstyles)

    markers = ['o', 'v']
    len_markers = len(markers)

    eval_dists = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    uslam_files = []
    prop_files = []

    l_trans_info = []
    l_rot_info = []

    sel_seqs = []

    # retrieve plot files
    for i in range(0, len(seqs)):
        seq = seqs[i]

        trans_info = dict()
        rot_info = dict()

        uslam_file = glob.glob(base_path+'/'+seq+'_uslam_errs.txt')
        if len(uslam_file) == 1:
            uslam_files.append(uslam_file[0])
            get_stats_errs_file(uslam_file[0], trans_info, rot_info)
        else:
            print('Strange behavior in USLAM files for sequence: '+seq)
            continue

        prop_file = glob.glob(base_path+'/ef_ev_ethz_mono_ev_imu_'+seq+'*.txt')
        if len(prop_file) == 1:
            prop_files.append(prop_file[0])
            get_stats_errs_file(prop_file[0], trans_info, rot_info)
        else:
            print('Strange behavior in proposed files for sequence: ' + seq)

        sel_seqs.append(seq)
        l_trans_info.append(trans_info)
        l_rot_info.append(rot_info)

    fig_trans = plt.figure()
    ax_trans = fig_trans.gca()

    fig_rot = plt.figure()
    ax_rot = fig_rot.gca()

    l_trans_h = []
    l_rot_h = []
    l_method_h = []

    l_color_idx = random.sample(range(0, len_colors), len(sel_seqs))

    for i in range(0, len(l_trans_info)):

        seq = sel_seqs[i]

        trans_info = l_trans_info[i]
        rot_info = l_rot_info[i]

        curr_color = plot_colors[i] # l_color_idx[i]

        method_keys = list(trans_info.keys())

        if len(method_keys) != 2:
            print('sequence: '+seq+' has only one target')
            continue

        for j in range(0, len(method_keys)):

            method = method_keys[j]

            curr_trans = trans_info[method]
            curr_rot = rot_info[method]

            trans_x = []
            trans_y = []
            rot_x = []
            rot_y = []

            for trans_step in curr_trans.keys():
                trans_x.append(trans_step)
                trans_y.append(curr_trans[trans_step][2])

            for rot_step in curr_rot.keys():
                rot_x.append(rot_step)
                rot_y.append(curr_rot[rot_step][2])

            trans_h, = ax_trans.plot(trans_x, trans_y, plot_lstyles[j % len_plot_lst], color=curr_color)
            rot_h, = ax_rot.plot(rot_x, rot_y, plot_lstyles[j % len_plot_lst], color=curr_color)

            trans_sc_h = ax_trans.scatter(trans_x, trans_y, c=curr_color, marker=markers[j % len_markers])
            rot_sc_h = ax_rot.scatter(rot_x, rot_y, c=curr_color, marker=markers[j % len_markers])

            if j == 1:
                l_trans_h.append(trans_h)
                l_rot_h.append(rot_h)

            if curr_color == 'black':
                l_method_h.append((trans_h, trans_sc_h))

    legend1 = ax_trans.legend(l_method_h, method_keys, loc=2)
    ax_trans.legend(l_trans_h, sel_seqs, loc=9)
    # ax_rot.legend(list(rot_handles.values()), method_keys)
    ax_trans.add_artist(legend1)

    ytics_trans = np.linspace(0, 1.2, 7)
    ax_trans.set_ylim(0, 1.2)
    ax_trans.set_yticks(ytics_trans)

    ytics_rot = np.linspace(0, 10, 6)
    ax_rot.set_ylim(0, 10)
    ax_rot.set_yticks(ytics_rot)

    ax_trans.grid(True, linestyle='--')
    ax_rot.grid(True, linestyle='--')

    ax_trans.set_ylabel('Translation error [m]')
    # ax_trans.set_xlabel('Distance traveled [m]')
    ax_rot.set_ylabel('Yaw error [deg]')
    ax_rot.set_xlabel('Distance traveled [m]')

    # show plot
    # show plot
    plt.show()
