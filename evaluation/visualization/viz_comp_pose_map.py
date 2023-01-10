import os.path
import sys
import argparse
import numpy as np
import glob
import re

import viz_tools as mviz

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.append('../')
import associate as assoc
from evaluate_ate_scale import align, plot_traj

sys.path.append('../my_utils')
import mf_manip as mfm
import mmisc
import my_eval_rpe as meval_rpe

from viz_pose_gt_est_ap import get_poses

if __name__ == '__main__':

    # parse command line
    parser = argparse.ArgumentParser(description='''
    This script plots piece-wise absolute ground-truth camera pose and aligned estimated pose. 
    ''')
    parser.add_argument('path_settings', help='settings.yaml (containing dataset path info)')
    parser.add_argument('-path_pose_est', nargs='*', help='estimated trajectory list (format: timestamp tx ty tz qx qy qz qw)',
                        default=[])
    parser.add_argument('-path_map_est', help='estimated trajectory list (format: x y z)',
                        default='')

    parser.add_argument('--max_ts', help='time offset added to the timestamps of the second file (default: -1.0, all)',
                        default=-1.0)

    parser.add_argument('--offset', help='time offset added to the timestamps of the second file (default: 0.0)',
                        default=0.0)
    parser.add_argument('--scale', help='scaling factor for the second trajectory (default: 1.0)', default=1.0)
    parser.add_argument('--max_difference',
                        help='maximally allowed time difference for matching entries (default: 0.01 s)', default=0.01)

    args = parser.parse_args()

    settings_file = args.path_settings
    list_est_pose = args.path_pose_est
    map_file = args.path_map_est

    max_ts = float(args.max_ts)

    gt_max_tdiff = float(args.max_difference)
    ts_offset = float(args.offset)
    ini_scale = float(args.scale)

    # Resolve GT path
    gtFile = mfm.parse_gt_path_cv(args.path_settings)
    print("> GT File: " + gtFile)
    gtList = np.array(mfm.read_file_list(gtFile))
    stamps_gt = gtList[:, 0]
    gtDict = mmisc.list2dict(gtList)

    # Resolve estimated map
    print("> EST Map File: " + map_file)
    mapData = []
    if os.path.isfile(map_file):
        mapData = np.loadtxt(map_file)

    list_aligned_pose = []
    list_aligned_gt = dict()

    # Resolve estimated pose paths path
    for est_pose in list_est_pose:

        if len(est_pose) <= 0 or not os.path.isfile(est_pose):
            print(est_pose + ' not found')
            continue

        print("> EST Pose File: " + est_pose)
        estPose = mfm.read_file_list(est_pose)

        if len(estPose) < 3:
            # For 2 poses, we can always match 2 line pieces exactly with 7DoF Sim3
            print("** ERROR: Not enough poses in current piece **")
            continue

        pose_list = estPose
        if max_ts > 0.0:
            pose_list = []
            first_ts = estPose[0][0]
            for pose in estPose:
                if abs(pose[0] - first_ts) < max_ts:
                    pose_list.append(pose)

        arrTransGt, arrTransEstAligned, matches = get_poses(pose_list, gtList, ts_offset, gt_max_tdiff)

        if len(arrTransGt) == 0:
            continue

        list_aligned_pose.append(arrTransEstAligned)
        list_aligned_gt = {**list_aligned_gt, **(dict((a, gtDict[a]) for a, b in matches if a in gtDict))}

    # merge and unify all gt trajectories
    xyz_gt_acc = []
    for ts in sorted(list_aligned_gt.keys()):
        xyz_gt_acc.append(list_aligned_gt[ts][0:3])
    xyz_gt_acc = np.array(xyz_gt_acc)

    ax = plt.figure().add_subplot(projection='3d')

    lcolors = ['g', 'r', 'b']

    if len(mapData) > 0:
        ax.scatter(mapData[:, 0], mapData[:, 1], mapData[:, 2], marker='.')

    ax.plot(xyz_gt_acc[:, 0], xyz_gt_acc[:, 1], xyz_gt_acc[:, 2], color=lcolors[0], label='GT')

    labels = ['E-I-C2 (Proposed)', 'E-I [10]']
    n_pose = len(list_aligned_pose)
    for i in range(0, n_pose):
        curr_pose = list_aligned_pose[i]
        ax.plot(curr_pose[:, 0], curr_pose[:, 1], curr_pose[:, 2], color=lcolors[(i+1) % len(lcolors)], label=labels[i % len(labels)])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.view_init(elev=-10., azim=10.)
    ax.legend()

    plt.show()

