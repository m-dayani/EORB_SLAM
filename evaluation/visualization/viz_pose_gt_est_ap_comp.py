import sys
import argparse
import numpy as np
import glob
import re

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import viz_tools as mviz

sys.path.append('../')
import associate as assoc
from evaluate_ate_scale import align, plot_traj

sys.path.append('../my_utils')
import mf_manip as mfm
import mmisc
import my_eval_rpe as meval_rpe

from viz_pose_gt_est_ap import get_poses


def filter_est_files(est_files_list, est_file, args):
    est_files_list = [f for f in est_files_list if re.match(est_file + r'_[0-9.]+\.txt', f)]
    if len(est_files_list) <= 0:
        est_file = args.path_pose_est
        est_files_list = glob.glob(est_file)
        if len(est_files_list) <= 0:
            # Abort
            print("** Neither settings path nor argument path: " + est_file + " can be found! **")
            exit(1)
    return est_files_list, est_file


if __name__ == '__main__':

    pathPoseEst = '../../ev_asynch_tracker_pose_chain.txt'
    pathPoseEst1 = '../../ev_asynch_tracker_pose_chain.txt'


    # parse command line
    parser = argparse.ArgumentParser(description='''
    This script plots piece-wise absolute ground-truth camera pose and aligned estimated pose. 
    ''')
    parser.add_argument('path_settings', help='settings.yaml (containing dataset path info)')
    parser.add_argument('-path_pose_est', help='estimated trajectory (format: timestamp tx ty tz qx qy qz qw)',
                        default=pathPoseEst)

    parser.add_argument('--est_im_idx', help='index of EST file being procecced: default: -1 (median over all)',
                        default=-1)
    parser.add_argument('--est_ev_im_idx', help='index of EST file being procecced: default: -1 (median over all)',
                        default=-1)
    parser.add_argument('--pose_config', help='pose config: 0: ORB Frame, 1: ORB KeyFrame (default), 2: EvKF, 3: EvImKF', default=1)

    parser.add_argument('--offset', help='time offset added to the timestamps of the second file (default: 0.0)',
                        default=0.0)
    parser.add_argument('--scale', help='scaling factor for the second trajectory (default: 1.0)', default=1.0)
    parser.add_argument('--max_difference',
                        help='maximally allowed time difference for matching entries (default: 0.01 s)', default=0.01)

    args = parser.parse_args()

    est_im_idx = int(args.est_im_idx)
    est_ev_im_idx = int(args.est_ev_im_idx)
    pose_config = int(args.pose_config)

    gt_max_tdiff = float(args.max_difference)
    ts_offset = float(args.offset)
    ini_scale = float(args.scale)

    # Resolve GT path
    gtFile = mfm.parse_gt_path_cv(args.path_settings)
    # Resolve EST paths
    estFile = mfm.parse_est_base_path_cv(args.path_settings, pose_config, "mono_im")
    estImFilesList = glob.glob(estFile + "*.txt")
    estImFilesList, estImFile = filter_est_files(estImFilesList, estFile, args)

    estFile = mfm.parse_est_base_path_cv(args.path_settings, pose_config, "mono_ev_im")
    estEvImFilesList = glob.glob(estFile + "*.txt")
    estEvImFilesList, estEvFile = filter_est_files(estEvImFilesList, estFile, args)

    # Load and refine GT data
    print("> GT File: " + gtFile)
    gtList = np.array(mfm.read_file_list(gtFile))
    stamps_gt = gtList[:, 0]

    # retrieve raw data
    if est_im_idx < 0:
        estImFile = estImFilesList[0]
    elif est_im_idx < len(estImFilesList):
        estImFile = estImFilesList[est_im_idx]
    else:
        print("** ERROR: Wrong EST file index: %d **" % est_im_idx)
        exit(1)

    if est_ev_im_idx < 0:
        estEvImFile = estEvImFilesList[0]
    elif est_ev_im_idx < len(estEvImFilesList):
        estEvImFile = estEvImFilesList[est_ev_im_idx]
    else:
        print("** ERROR: Wrong EST file index: %d **" % est_ev_im_idx)
        exit(1)

    print("> EST Image File: " + estImFile)
    estImList = mfm.read_file_list(estImFile)
    print("> EST Event-Image File: " + estEvImFile)
    estEvImList = mfm.read_file_list(estEvImFile)
    th_ts = 1e-12
    th_iden = 1e-3
    estImPosePieces = mmisc.break_pose_graph(np.array(estImList), th_ts, th_iden)
    estEvImPosePieces = mmisc.break_pose_graph(np.array(estEvImList), th_ts, th_iden)

    xyzImGt = []
    xyzEvImGt = []
    xyzImEst = []
    xyzEvImEst = []

    for posePiece in estImPosePieces:

        nPiece = len(posePiece)
        if nPiece < 3:
            # For 2 poses, we can always match 2 line pieces exactly with 7DoF Sim3
            print("** ERROR: Not enough poses in current piece: %d **s" % nPiece)
            continue

        else:
            arrTransGt, arrTransEstAligned = get_poses(posePiece, gtList, ts_offset, gt_max_tdiff)

            if len(arrTransGt) == 0:
                continue

            for i in range(len(arrTransGt)):
                xyzImEst.append(arrTransEstAligned[i])
                xyzImGt.append(arrTransGt[i])

    for posePiece in estEvImPosePieces:

        nPiece = len(posePiece)
        if nPiece < 3:
            # For 2 poses, we can always match 2 line pieces exactly with 7DoF Sim3
            print("** ERROR: Not enough poses in current piece: %d **s" % nPiece)
            continue

        else:
            arrTransGt, arrTransEstAligned = get_poses(posePiece, gtList, ts_offset, gt_max_tdiff)

            if len(arrTransGt) == 0:
                continue

            for i in range(len(arrTransGt)):
                xyzEvImEst.append(arrTransEstAligned[i])
                xyzEvImGt.append(arrTransGt[i])

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    mviz.plot_traj_3d(ax, np.array(xyzEvImGt), 'GT Traj.', ['y', '*', 'b'])

    mviz.plot_traj_3d(ax, np.array(xyzImEst), 'Est. I Traj.', ['c', '*', 'r'])

    mviz.plot_traj_3d(ax, np.array(xyzEvImEst), 'Est. EI Traj.', ['m', '*', 'g'])

    ax.legend()

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()
