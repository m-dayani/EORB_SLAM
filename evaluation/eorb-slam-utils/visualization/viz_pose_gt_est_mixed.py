import sys
import argparse
import numpy as np

import viz_tools as mviz
import viz_pose_gt_est_rp as viz_rp
import viz_pose_gt_est_ap as viz_ap

sys.path.append('../my_utils')
import mf_manip as mfm
import mmisc
import my_eval_rpe as meval_rpe


if __name__ == '__main__':

    pathPoseEst = '../../ev_asynch_tracker_pose_chain.txt'

    # parse command line
    parser = argparse.ArgumentParser(description='''
    This script plots piece-wise relative ground-truth camera pose and aligned estimated pose. 
    ''')
    parser.add_argument('path_settings', help='settings.yaml (containing dataset path info)')
    parser.add_argument('-path_pose_est', help='estimated trajectory (format: timestamp tx ty tz qx qy qz qw)',
                        default=pathPoseEst)
    parser.add_argument('-dir_gt_quat', help='groundtruth quaternion direction: 0: qw last, 1: qw first', default=1)

    args = parser.parse_args()

    # Construct gt path and data
    gtFile = mfm.parse_gt_path_cv(args.path_settings)
    poseFile = args.path_pose_est
    gtList = np.array(mfm.read_file_list(gtFile))
    stamps_gt = gtList[:, 0]

    if args.dir_gt_quat == 1:
        gtList = meval_rpe.swap_qw(gtList)

    # retrieve raw data
    estList = mfm.read_file_list(poseFile)
    th_ts = 1e-12
    th_iden = 1e-3
    estPosePieces = mmisc.break_pose_graph(np.array(estList), th_ts, th_iden)

    gt_max_tdiff = 0.01
    ts_offset = 0

    xyzGt = []
    xyzEst = []

    for posePiece in estPosePieces:

        nPiece = len(posePiece)
        if nPiece < 2:

            print("* ERROR: Not enough poses in current piece")
            continue

        elif nPiece == 2:

            Tc0_gt, Tc1_gt, Tc0_est, Tc1_p = viz_rp.get_poses(posePiece, stamps_gt, gtList, ts_offset, gt_max_tdiff)

            if Tc0_gt.size == 0:
                continue

            xyzEst.append(Tc0_gt[0:3, 3])
            xyzEst.append(Tc1_p[0:3, 3])
            xyzGt.append(Tc0_gt[0:3, 3])
            xyzGt.append(Tc1_gt[0:3, 3])

        elif nPiece > 2:

            arrPoseGt, arrPoseEst = viz_ap.get_poses(posePiece, gtList, ts_offset, gt_max_tdiff)

            if arrPoseGt.size == 0:
                continue

            for i in range(len(arrPoseGt)):
                xyzEst.append(arrPoseEst[i])
                xyzGt.append(arrPoseGt[i])

    mviz.draw_pose_gt_est(np.array(xyzGt), np.array(xyzEst))
