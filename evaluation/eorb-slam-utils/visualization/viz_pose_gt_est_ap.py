import sys
import argparse
import numpy as np
import glob
import re

import viz_tools as mviz

sys.path.append('../')
import associate as assoc
from evaluate_ate_scale import align, plot_traj

sys.path.append('../my_utils')
import mf_manip as mfm
import mmisc
import my_eval_rpe as meval_rpe


def get_poses(posePiece, list_gt, ts_offset, gt_max_tdiff):

    first_list = mmisc.list2dict(list_gt)
    second_list = mmisc.list2dict(posePiece)

    matches = assoc.associate(first_list, second_list, ts_offset, gt_max_tdiff)

    if len(matches) < 2:
        return [], [], []

    iniScale = 1.0

    first_xyz = np.matrix([[float(value) for value in first_list[a][0:3]] for a, b in matches]).transpose()
    second_xyz = np.matrix(
        [[float(value) * float(iniScale) for value in second_list[b][0:3]] for a, b in matches]).transpose()

    rot, transGT, trans_errorGT, trans, trans_error, scale = align(second_xyz, first_xyz)

    second_xyz_aligned = scale * rot * second_xyz + transGT

    return np.asarray(first_xyz).transpose(), np.asarray(second_xyz_aligned).transpose(), matches


if __name__ == '__main__':

    pathPoseEst = '../../ev_asynch_tracker_pose_chain.txt'

    # parse command line
    parser = argparse.ArgumentParser(description='''
    This script plots piece-wise absolute ground-truth camera pose and aligned estimated pose. 
    ''')
    parser.add_argument('path_settings', help='settings.yaml (containing dataset path info)')
    parser.add_argument('-path_pose_est', help='estimated trajectory (format: timestamp tx ty tz qx qy qz qw)',
                        default=pathPoseEst)

    parser.add_argument('--est_idx', help='index of EST file being procecced: default: -1 (median over all)',
                        default=-1)
    parser.add_argument('--pose_config', help='pose config: 0: ORB Frame, 1: ORB KeyFrame (default), 2: EvKF, 3: EvImKF', default=1)

    parser.add_argument('--offset', help='time offset added to the timestamps of the second file (default: 0.0)',
                        default=0.0)
    parser.add_argument('--scale', help='scaling factor for the second trajectory (default: 1.0)', default=1.0)
    parser.add_argument('--max_difference',
                        help='maximally allowed time difference for matching entries (default: 0.01 s)', default=0.01)

    args = parser.parse_args()

    est_idx = int(args.est_idx)
    pose_config = int(args.pose_config)

    gt_max_tdiff = float(args.max_difference)
    ts_offset = float(args.offset)
    ini_scale = float(args.scale)

    # Resolve GT path
    gtFile = mfm.parse_gt_path_cv(args.path_settings)
    # Resolve EST path
    estFile = mfm.parse_est_base_path_cv(args.path_settings, pose_config)
    estFilesList = glob.glob(estFile + "*.txt")
    # Filter and refine the list
    estFilesList = [f for f in estFilesList if re.match(estFile + r'_[0-9.]+\.txt', f)]
    if len(estFilesList) <= 0:
        estFile = args.path_pose_est
        estFilesList = glob.glob(estFile)
        if len(estFilesList) <= 0:
            # Abort
            print("** Neither settings path nor argument path: " + estFile + " can be found! **")
            exit(1)

    # Load and refine GT data
    print("> GT File: " + gtFile)
    gtList = np.array(mfm.read_file_list(gtFile))
    stamps_gt = gtList[:, 0]

    # retrieve raw data
    if est_idx < 0:
        estFile = estFilesList[0]
    elif est_idx < len(estFilesList):
        estFile = estFilesList[est_idx]
    else:
        print("** ERROR: Wrong EST file index: %d **" % est_idx)
        exit(1)

    print("> EST File: " + estFile)
    estList = mfm.read_file_list(estFile)
    th_ts = 1e-12
    th_iden = 1e-3
    estPosePieces = mmisc.break_pose_graph(np.array(estList), th_ts, th_iden)

    xyzGt = []
    xyzEst = []

    for posePiece in estPosePieces:

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
                xyzEst.append(arrTransEstAligned[i])
                xyzGt.append(arrTransGt[i])

    mviz.draw_pose_gt_est(np.array(xyzGt), np.array(xyzEst))
