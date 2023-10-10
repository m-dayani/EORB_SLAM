import sys
import argparse
import numpy as np
import glob
import re

from scipy.spatial.transform import Rotation as scipyR

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import viz_tools as mviz

sys.path.append('../my_utils')
import mf_manip as mfm
import mmisc
import my_eval_rpe as meval_rpe


def get_stats_from_data(data):

    data_sorted = np.sort(data)
    n_data = len(data)
    stats = np.array([data_sorted[0], data_sorted[int(n_data * 0.25)], data_sorted[int(n_data * 0.5)],
                      data_sorted[int(n_data * 0.75)], data_sorted[-1]])

    return stats


def perform_test1():

    # Creating dataset
    np.random.seed(10)

    data = np.random.normal(100, 20, 200)
    data1 = np.random.normal(100, 20, 200)

    stats = get_stats_from_data(data)
    stats1 = get_stats_from_data(data1)

    pos_x = 1

    fig = plt.figure(figsize=(10, 7))

    # Creating plot
    # plt.boxplot(stats)

    box_color = 'blue'
    box_style = '--'

    mviz.draw_box_stats(stats, pos_x, box_color, box_style)
    mviz.draw_box_stats(stats1, pos_x + 0.6, 'red', '-')

    # show plot
    plt.show()


def get_stats_errs_file(file_name, trans_info, rot_info):

    with open(file_name, 'r') as stats_file:

        data = stats_file.read()
        lines = data.replace(",", " ").replace("\t", " ").split("\n")

        method_label = ''
        curr_info = dict()

        n_lines = len(lines)

        for i in range(0, n_lines):
            line = lines[i].strip()
            if len(line) <= 0:
                continue
            if line[0] == '#':
                if ':' in line:
                    header_parts = line[1:].split(':')
                    method_label = header_parts[0].strip()

                    if 'trans' in header_parts[1].strip():
                        trans_info[method_label] = dict()
                        curr_info = trans_info[method_label]

                    elif 'rot' in header_parts[1].strip():
                        rot_info[method_label] = dict()
                        curr_info = rot_info[method_label]

            else:
                curr_stat_list = [np.float64(v) for v in line.split(' ')]
                curr_info[curr_stat_list[0]] = curr_stat_list[1:]

    return trans_info, rot_info


def find_intersection(info_obj):

    res_list = []
    for method_key in list(info_obj.keys()):

        curr_method = info_obj[method_key]

        if len(res_list) <= 0:
            res_list = set(curr_method.keys())
        else:
            res_list &= set(curr_method.keys())

    return list(res_list)


if __name__ == '__main__':

    # parse command line
    parser = argparse.ArgumentParser(description='''
    Draw statistical graphs related to pose error stats stored in a specific format
    ''')
    parser.add_argument('base_path', help='base path of uslam error files e.g. ../../results/uslam_plots')
    parser.add_argument('seq_name', help='sequence name')

    parser.add_argument('--intersection', help='only show the intersection of distances, default: 0', default=0)

    args = parser.parse_args()

    base_path = args.base_path
    seq_name = args.seq_name

    intersect_keys = int(args.intersection)

    err_files = glob.glob(base_path+"/*" + seq_name + "*.txt")

    # Creating plot
    # plt.boxplot(stats)

    box_color = 'blue'
    box_style = '--'

    fig_styles = [['green', '-'], ['blue', '--'], ['black', '.-'], ['red', '.']]

    trans_info = dict()
    rot_info = dict()

    for err_file in reversed(err_files):
        trans_info, rot_info = get_stats_errs_file(err_file, trans_info, rot_info)

    method_keys = list(trans_info.keys())
    n_methods = len(method_keys)

    if n_methods != len(rot_info):
        exit(1)

    bx_width = 0.5
    dist_x = bx_width * 1.5 + 0.1

    fig, (ax_trans, ax_rot) = plt.subplots(2)
    fig.canvas.set_window_title(seq_name+" errors")

    trans_intersect = []
    rot_intersect = []
    if intersect_keys:
        trans_intersect = sorted(find_intersection(trans_info))
        rot_intersect = sorted(find_intersection(rot_info))

    trans_handles = dict()
    rot_handles = dict()

    for i in range(0, n_methods):

        method = method_keys[i]

        curr_style = fig_styles[int(i % 4)]

        trans_stats = trans_info[method]

        if len(trans_intersect) <= 0:
            trans_keys = list(trans_stats.keys())
        else:
            trans_keys = trans_intersect

        for trans_key in trans_keys:

            pos_x = trans_key - (dist_x / 2) * (2 * i - (n_methods - 1))
            stats = trans_stats[trans_key]
            new_stats = [stats[0], stats[3], stats[4], stats[5], stats[1]]

            h_box = mviz.draw_box_stats(ax_trans, new_stats, pos_x, method, curr_style[0], curr_style[1])
            trans_handles[method] = h_box

        rot_stats = rot_info[method]

        if len(rot_intersect) <= 0:
            rot_keys = list(rot_stats.keys())
        else:
            rot_keys = rot_intersect

        for rot_key in rot_keys:
            pos_x = rot_key - (dist_x / 2) * (2 * i - (n_methods - 1))
            stats = rot_stats[rot_key]
            new_stats = [stats[0], stats[3], stats[4], stats[5], stats[1]]

            h_box = mviz.draw_box_stats(ax_rot, new_stats, pos_x, method, curr_style[0], curr_style[1])
            rot_handles[method] = h_box

    ax_trans.legend(list(trans_handles.values()), method_keys)
    # ax_rot.legend(list(rot_handles.values()), method_keys)

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
    plt.show()
