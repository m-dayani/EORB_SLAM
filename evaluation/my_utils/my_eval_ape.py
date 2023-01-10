import sys
import argparse
import numpy as np
import glob
import re

sys.path.append('../')
import associate as assoc
from evaluate_ate_scale import align, plot_traj

sys.path.append('../my_utils')
import mf_manip as mfm
import mmisc


def eval_est_poses(posePiece, gtList, trans_errors, trans_errorsGT, dtrans, dtimes, ts_offset, gt_max_tdiff, ini_scale):

    if len(posePiece) < 3:
        return trans_errors, trans_errorsGT, dtrans, dtimes

    first_list = mmisc.list2dict(gtList)
    second_list = mmisc.list2dict(posePiece)

    matches = assoc.associate(first_list, second_list, ts_offset, gt_max_tdiff)

    if len(matches) < 2:
        print("** Cannot find gt pose matches for piece: %f ~ %f **" % (posePiece[0][0], posePiece[-1][0]))
        return trans_errors, trans_errorsGT, dtrans, dtimes

    first_xyz = np.matrix([[float(value) for value in first_list[a][0:3]] for a, b in matches]).transpose()
    second_xyz = np.matrix(
        [[float(value) * float(ini_scale) for value in second_list[b][0:3]] for a, b in matches]).transpose()

    rot, transGT, trans_errorGT, trans, trans_error, scale = align(second_xyz, first_xyz)

    # second_xyz_aligned = scale * rot * second_xyz + transGT
    first_xyz_aligned = np.array(first_xyz).transpose()

    for i in range(len(trans_error)):
        trans_errors.append(trans_error[i])
        trans_errorsGT.append(trans_errorGT[i])
        if i > 0:
            dtrans.append(first_xyz_aligned[i] - first_xyz_aligned[i - 1])
            dtimes.append(matches[i][0] - matches[i-1][0])

    return trans_errors, trans_errorsGT, dtrans, dtimes


def eval_est_file(file_name, gtList, ts_offset, gt_max_tdiff, ini_scale):

    print("> EST File: " + file_name)

    # estList = mfm.read_file_list(file_name)
    estPosePieces = mfm.read_dosconn_graph_list(file_name)

    # th_ts = 1e-12
    # th_iden = 1e-3
    # if sort_ts:
    #     estList = mmisc.sort_pose_list(estList)

    # estPosePieces = mmisc.break_pose_graph(np.array(estList), th_ts, th_iden)

    trans_errors = []
    trans_errorsGT = []
    dtrans = []
    dtimes = []

    for posePiece in estPosePieces:

        trans_errors, trans_errorsGT, dtrans, dtimes = eval_est_poses(
            posePiece, gtList, trans_errors, trans_errorsGT, dtrans, dtimes, ts_offset, gt_max_tdiff, ini_scale)

    n_compared = len(trans_errors)
    traj_dur = 0
    traj_len = 0
    traj_ape = 0
    traj_ape_gt = 0

    if n_compared > 0:
        trajDists = np.array([np.linalg.norm(trans) for trans in dtrans])
        traj_dur = np.sum(dtimes) / time_scale
        traj_len = np.sum(trajDists)
        traj_ape = np.sqrt(np.dot(trans_errors, trans_errors) / len(trans_errors))
        traj_ape_gt = np.sqrt(np.dot(trans_errorsGT, trans_errorsGT) / len(trans_errorsGT))

    return n_compared, traj_dur, traj_len, traj_ape, traj_ape_gt


if __name__ == '__main__':

    pathPoseEst = '../../ev_asynch_tracker_pose_chain.txt'

    # parse command line
    parser = argparse.ArgumentParser(description='''
    This script plots piece-wise ground-truth camera pose and aligned estimated pose. 
    ''')
    parser.add_argument('path_settings', help='settings.yaml (containing dataset path info)')
    parser.add_argument('-path_pose_est', help='estimated trajectory (format: timestamp tx ty tz qx qy qz qw)',
                        default=pathPoseEst)

    parser.add_argument('--est_idx', help='index of EST file being procecced: default: -1 (median over all)',
                        default=-1)
    parser.add_argument('--pose_config', help='pose config: 0: ORB Frame, 1: ORB KeyFrame (default), 2: EvKF, 3: EvImKF', default=1)
    parser.add_argument('--sort_ts', help='0 (default): do not sort poses based on ts, 1: otherwise', default=0)

    parser.add_argument('--offset', help='time offset added to the timestamps of the second file (default: 0.0)',
                        default=0.0)
    parser.add_argument('--scale', help='scaling factor for the second trajectory (default: 1.0)', default=1.0)
    parser.add_argument('--max_difference',
                        help='maximally allowed time difference for matching entries (default: 0.01 s)', default=0.01)
    parser.add_argument('--time_scale',
                        help='time scale for dataset (default: 1 s)', default=1)

    args = parser.parse_args()

    est_idx = int(args.est_idx)
    pose_config = int(args.pose_config)
    sort_ts = False
    if int(args.sort_ts) == 1:
        sort_ts = True

    gt_max_tdiff = float(args.max_difference)
    ts_offset = float(args.offset)
    ini_scale = float(args.scale)
    # time_scale = float(args.time_scale)

    # Resolve GT path
    gtFile = mfm.parse_gt_path_cv(args.path_settings)
    # Resolve EST path
    estFile, time_scale, qw_dir, ds_config = mfm.parse_est_base_path_cv(args.path_settings, pose_config)
    estFilesList = glob.glob(estFile + "*.txt")
    # Filter and refine the list
    estFilesList = [f for f in estFilesList if re.match(estFile+r'_[0-9._]+\.txt', f)]
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

    # Process each EST file
    nMatchedPairs = []
    trackingTimes = []
    trajDuration = []
    trajLength = []
    trajApe = []
    trajApeGt = []

    if est_idx < 0:
        # median over all runs
        for idx in range(len(estFilesList)):

            currFile = estFilesList[idx]
            avg_ttime = re.findall(r'\d+\.\d+', currFile)
            nmatched, duration, trajlen, err_ape, ape_gt = eval_est_file(
                currFile, gtList, ts_offset, gt_max_tdiff, ini_scale)

            if nmatched > 0:
                nMatchedPairs.append(nmatched)
                if len(avg_ttime) > 0:
                    trackingTimes.append(float(avg_ttime[0]))
                else:
                    trackingTimes.append(0)
                trajDuration.append(duration)
                trajLength.append(trajlen)
                trajApe.append(err_ape)
                trajApeGt.append(ape_gt)

    elif est_idx < len(estFilesList):

        currFile = estFilesList[est_idx]
        avg_ttime = re.findall(r'\d+\.\d+', currFile)
        nmatched, duration, trajlen, err_ape, ape_gt = eval_est_file(
            currFile, gtList, ts_offset, gt_max_tdiff, ini_scale)

        if nmatched > 0:
            nMatchedPairs.append(nmatched)
            if len(avg_ttime) > 0:
                trackingTimes.append(float(avg_ttime[0]))
            else:
                trackingTimes.append(0)
            trajDuration.append(duration)
            trajLength.append(trajlen)
            trajApe.append(err_ape)
            trajApeGt.append(ape_gt)

    if len(nMatchedPairs) > 0:

        trajDuration = np.sort(trajDuration)
        trajLength = np.sort(trajLength)
        trajApe = np.sort(trajApe)
        trajApeGt = np.sort(trajApeGt)

        print("- Average tracking time: %f s" % trackingTimes[int(len(trackingTimes) / 2)])
        print("- Total traveled time: %f s" % trajDuration[int(len(trajDuration) / 2)])
        print("- Total traveled distance: %f m" % trajLength[int(len(trajLength) / 2)])
        print("- RMS APE: %f m" % trajApe[int(len(trajApe) / 2)])
        print("- RMS APE GT: %f m" % trajApeGt[int(len(trajApeGt) / 2)])
    else:
        print("** Cannot find any estimated trajectory match! **")
