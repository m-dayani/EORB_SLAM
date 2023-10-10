import sys
import argparse
import numpy as np
import glob
import re

import viz_tools as mviz

sys.path.append('../my_utils')
import mf_manip as mfm
import mmisc
import my_eval_rpe as meval_rpe


def get_poses(posePiece, stamps_gt, list_gt, ts_offset, gt_max_tdiff, pose_dir=1):

    idx_gt_0, idx_gt_1, ts0, ts1 = meval_rpe.find_closest_gt(posePiece, stamps_gt, ts_offset, gt_max_tdiff)
    if idx_gt_0 < 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    c0_est = np.array(posePiece[0])
    c1_est = np.array(posePiece[1])

    Tc0w_est, Tc1w_est, Tc1c0_est = meval_rpe.transform44(c0_est, c1_est)
    Tc0w_gt, Tc1w_gt, Tc1c0_gt = meval_rpe.transform44(list_gt[idx_gt_0], list_gt[idx_gt_1])

    # Tc1c0_est is actually Twc1 here (T cam1 related in world coord. sys.)
    # Relative poses for concatenation here is reverse for computing error (my_eval_rpe)
    if pose_dir == 1:
        Tc1c0_est, relSc, gtDist, estDist = meval_rpe.scale(Tc1c0_est, Tc1c0_gt)
        Tc1w_est_p = np.matmul(Tc0w_gt, Tc1c0_est)
    else:
        # In case we already saved abs poses in world coord. sys.
        Tc1c0_est = np.linalg.inv(Tc1c0_est)
        Tc1c0_est, relSc, gtDist, estDist = meval_rpe.scale(Tc1c0_est, Tc1c0_gt)
        Tc1w_est_p = np.matmul(Tc0w_gt, Tc1c0_est)

    return Tc0w_gt, Tc1w_gt, Tc0w_est, Tc1w_est_p


if __name__ == '__main__':

    pathPoseEst = '../../ev_asynch_tracker_pose_chain.txt'

    # parse command line
    parser = argparse.ArgumentParser(description='''
    This script plots piece-wise relative ground-truth camera pose and aligned estimated pose. 
    ''')
    parser.add_argument('path_settings', help='settings.yaml (containing dataset path info)')
    parser.add_argument('-path_pose_est', help='estimated trajectory (format: timestamp tx ty tz qx qy qz qw)',
                        default=pathPoseEst)

    parser.add_argument('--est_idx', help='index of EST file being procecced: default: -1 (median over all)',
                        default=-1)
    parser.add_argument('--pose_config', help='pose config: 0: ORB Frame, 1: ORB KeyFrame (default), 2: EvKF, 3: EvImKF', default=1)

    parser.add_argument('--dir_pose', help='pose direction: 0: Twc (default), 1: Tcw', default=0)
    parser.add_argument('--dir_gt_qw', help='groundtruth quaternion direction: 0: qw last, 1: qw first (default)',
                        default=1)

    parser.add_argument('--all_pieces', help='compute error for all disconnected pieces in pose graph, default: 1',
                        default=1)

    parser.add_argument('--offset', help='time offset added to the timestamps of the second file (default: 0.0)',
                        default=0.0)
    parser.add_argument('--scale', help='scaling factor for the second trajectory (default: 1.0)', default=1.0)
    parser.add_argument('--max_difference',
                        help='maximally allowed time difference for matching entries (default: 0.01 s)', default=0.01)

    print("** WARNING: Choose the dir_gt_qw & dir_pose settings correctly! **")

    args = parser.parse_args()

    est_idx = int(args.est_idx)
    pose_config = int(args.pose_config)

    pose_dir = int(args.dir_pose)
    qw_dir = int(args.dir_gt_qw)

    all_pieces = True
    if int(args.all_pieces) == 0:
        all_pieces = False

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

    if gtList.shape[1] > 8:
        gtList = gtList[:, 0:8]

    if qw_dir == 1:
        gtList = mmisc.swap_qw(gtList)

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

    if not all_pieces:
        idx_max_piece = 0
        max_piece = 0
        i = 0
        for pose_piece in estPosePieces:
            if len(pose_piece) > max_piece:
                max_piece = len(pose_piece)
                idx_max_piece = i
            i += 1
        estPosePieces = [estPosePieces[idx_max_piece]]

    xyzGt = []
    xyzEst = []

    for posePiece in estPosePieces:

        nPiece = len(posePiece)
        if nPiece < 2:

            print("** ERROR: Not enough poses in current piece **")
            continue

        elif nPiece == 2:

            Tc0_gt, Tc1_gt, Tc0_est, Tc1_p = get_poses(posePiece, stamps_gt, gtList, ts_offset, gt_max_tdiff, pose_dir)

            if Tc0_gt.size == 0:
                continue

            xyzEst.append(Tc0_gt[0:3, 3])
            xyzEst.append(Tc1_p[0:3, 3])
            xyzGt.append(Tc0_gt[0:3, 3])
            xyzGt.append(Tc1_gt[0:3, 3])

        elif nPiece > 2:

            for i in range(len(posePiece)-1):

                Tc0_gt, Tc1_gt, Tc0_est, Tc1_p = get_poses(posePiece[i:i+2], stamps_gt, gtList, ts_offset, gt_max_tdiff, pose_dir)

                if Tc0_gt.size == 0:
                    continue

                xyzEst.append(Tc0_gt[0:3, 3])
                xyzEst.append(Tc1_p[0:3, 3])
                xyzGt.append(Tc0_gt[0:3, 3])
                xyzGt.append(Tc1_gt[0:3, 3])

        if not all_pieces:
            print('Relative pose: %f ~ %f' % (posePiece[0][0], posePiece[-1][0]))

    mviz.draw_pose_gt_est(np.array(xyzGt), np.array(xyzEst))


