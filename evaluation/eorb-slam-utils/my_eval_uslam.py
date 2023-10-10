import sys
import argparse
import numpy as np
import glob
import re
import os
from scipy.spatial.transform import Rotation as scipyR

sys.path.append('../')
import associate as assoc
from evaluate_ate_scale import align, plot_traj

sys.path.append('../my_utils')
import mf_manip as mfm
import mmisc

sys.path.append('../tum_tools')
import evaluate_rpe as eval_rpe

'''
    A few important notes:
    1. For accurate rotation error, provide accurate directions (especially qw_dir) for both gt and est. data
    2. Translation error is not sensitive to qw_dir (though saved pose dir might still be important)
    3. The wider the matching time range, the closer the results to the original evaluate_ate_scale.py
    4. For EuRoC dataset, the gt provided in ORB-SLAM3, estimated_gt0, and leica0 result in different errors 
        (the error of estimated_gt0 is almost double the error of ORB-SLAM file)!
'''


def eval_est_poses(posePiece, gtList, eval_times, eval_trans, eval_rots,
                   time_scale, gt_max_tdiff, tmatch_range=(3, 8), ini_scale=1.0):

    dtrans_aligned = []
    dtimes = []

    n_compared = 0
    traj_dur = 0
    traj_len = 0
    traj_ape = 0
    traj_ape_gt = 0
    rot_error_avg = 0
    yaw_error_avg = 0
    ts_offset = 0.0

    deval_step = 1.0 / 4

    if len(posePiece) < 3:
        return eval_trans, eval_rots, n_compared, traj_dur, traj_len, traj_ape, traj_ape_gt, rot_error_avg, yaw_error_avg

    s_time = tmatch_range[0]
    e_time = tmatch_range[1]
    first_time = min(gtList[0][0], posePiece[0][0])
    first_time /= time_scale

    first_list = mmisc.list2dict(gtList)
    second_list = mmisc.list2dict(posePiece)

    matches = assoc.associate(first_list, second_list, ts_offset, gt_max_tdiff)

    if len(matches) < 2:
        print("** Cannot find gt pose matches for piece: %f ~ %f **" % (posePiece[0][0], posePiece[-1][0]))
        return eval_trans, eval_rots, n_compared, traj_dur, traj_len, traj_ape, traj_ape_gt, rot_error_avg, yaw_error_avg

    # Align using a subset of all matched poses
    first_xyz = np.matrix([[float(value) for value in first_list[a][0:3]]
                           for a, b in matches if s_time < a/time_scale - first_time < e_time]).transpose()
    second_xyz = np.matrix([[float(value) * float(ini_scale) for value in second_list[b][0:3]]
                            for a, b in matches if s_time < a/time_scale - first_time < e_time]).transpose()

    if len(first_xyz) == 0 or len(second_xyz) == 0:
        print("** Cannot find gt pose matches for selected ts range: (%f, %f) **" % (match_ts_s, match_ts_e))
        return eval_trans, eval_rots, n_compared, traj_dur, traj_len, traj_ape, traj_ape_gt, rot_error_avg, yaw_error_avg

    rot, transGT, trans_errorGT, trans, trans_error, scale = align(second_xyz, first_xyz)

    align_rot_obj = scipyR.from_matrix(rot)

    # Retrieve all data and align the estimation
    first_xyz_all = np.matrix([[float(value) for value in first_list[a][0:3]] for a, b in matches])
    second_xyz_all = np.matrix([[float(value) * float(ini_scale) for value in second_list[b][0:3]] for a, b in matches]).transpose()

    first_quat_all = np.matrix([[float(value) for value in first_list[a][3:]] for a, b in matches])
    second_quat_all = np.matrix([[float(value) for value in second_list[b][3:]] for a, b in matches])

    second_xyz_aligned_gt = scale * rot * second_xyz_all + transGT
    second_xyz_aligned_gt = np.array(second_xyz_aligned_gt).transpose()

    second_xyz_aligned = rot * second_xyz_all + trans
    second_xyz_aligned = np.array(second_xyz_aligned).transpose()

    trans_errorGT = []
    trans_error = []
    rot_error = []
    yaw_error = []

    dist_traveled = 0

    for i in range(len(second_xyz_aligned_gt)):

        curr_ts = matches[i][0] / time_scale

        curr_err_gt = first_xyz_all[i] - second_xyz_aligned_gt[i]
        curr_err_gt = np.sqrt(np.dot(curr_err_gt.A[0], curr_err_gt.A[0]))
        trans_errorGT.append(curr_err_gt)

        curr_err = first_xyz_all[i] - second_xyz_aligned[i]
        curr_err = np.sqrt(np.dot(curr_err.A[0], curr_err.A[0]))
        trans_error.append(curr_err)

        est_rot_obj = scipyR.from_quat(second_quat_all[i])
        est_rot_aligned = align_rot_obj * est_rot_obj
        gt_rot_obj = scipyR.from_quat(first_quat_all[i])
        rot_err_obj = est_rot_aligned * gt_rot_obj.inv()
        rot_err = eval_rpe.compute_angle(rot_err_obj.as_matrix().squeeze())
        rot_error.append(rot_err / np.pi * 180.0)
        rot_euler = rot_err_obj.as_euler('zyx', degrees=True).squeeze()
        curr_yaw_err = rot_euler[0]
        yaw_error.append(curr_yaw_err)

        if i > 0:
            curr_dtrans = first_xyz_all[i] - first_xyz_all[i - 1]
            dtrans_aligned.append(curr_dtrans)
            dist_traveled += np.linalg.norm(curr_dtrans)

        eval_metric = dist_traveled # curr_ts - first_time
        for eval_ts in eval_times:
            if eval_ts - deval_step < eval_metric < eval_ts + deval_step:
                eval_trans[eval_ts].append(curr_err)
                eval_rots[eval_ts].append(curr_yaw_err)

    dtimes.append(matches[-1][1] - matches[0][1])

    n_compared = len(trans_error)

    if n_compared > 0:
        trajDists = np.array([np.linalg.norm(trans) for trans in dtrans_aligned])
        traj_dur = np.sum(dtimes)
        traj_len = np.sum(trajDists)
        traj_ape = np.sqrt(np.dot(trans_error, trans_error) / len(trans_error))
        traj_ape_gt = np.sqrt(np.dot(trans_errorGT, trans_errorGT) / len(trans_errorGT))
        rot_error_avg = np.sqrt(np.dot(rot_error, rot_error)) / len(rot_error)
        yaw_error_avg = np.sqrt(np.dot(yaw_error, yaw_error)) / len(yaw_error)

    return eval_trans, eval_rots, n_compared, traj_dur, traj_len, traj_ape, traj_ape_gt, rot_error_avg, yaw_error_avg


def eval_est_file(file_name, gtList, eval_times, eval_trans, eval_rots,
                  time_scale, gt_max_tdiff, tmatch_range=(3, 8), ini_scale=1.0):

    print("> EST File: " + file_name)

    # estList = mfm.read_file_list(file_name)
    estPosePieces = mfm.read_dosconn_graph_list(file_name)

    # th_ts = 1e-12
    # th_iden = 1e-3
    # if sort_ts:
    #     estList = sort_pose_list(estList)

    # Disconnected graph calcs are not supported here
    # estPosePieces = mmisc.break_pose_graph(np.array(estList), th_ts, th_iden)
    # bestEstPosePiece = mmisc.select_best_pose_piece(estPosePieces)

    tot_n_comp = 0
    tot_traj_dur = 0
    tot_traj_len = 0

    l_traj_ape = []
    l_traj_ape_gt = []
    l_rot_err = []
    l_yaw_err = []

    for posePiece in estPosePieces:

        piece_dur = posePiece[-1][0] - posePiece[0][0]
        t0_mch = posePiece[0][0] + piece_dur / 20
        t1_mch = t0_mch + piece_dur / 12
        tmatch_range = (t0_mch, t1_mch)

        eval_trans, eval_rots, n_compared, traj_dur, traj_len, traj_ape, traj_ape_gt, rot_error_avg, yaw_error_avg = \
            eval_est_poses(posePiece, gtList, eval_times, eval_trans, eval_rots,
                           time_scale, gt_max_tdiff, tmatch_range, ini_scale)

        tot_n_comp += n_compared
        tot_traj_dur += traj_dur
        tot_traj_len += traj_len

        l_traj_ape.append(traj_ape)
        if not np.isnan(traj_ape_gt):
            l_traj_ape_gt.append(traj_ape_gt)
        l_rot_err.append(rot_error_avg)
        l_yaw_err.append(yaw_error_avg)

    avg_traj_ape = np.mean(l_traj_ape)
    avg_traj_ape_gt = np.mean(l_traj_ape_gt)
    avg_rot_err = np.mean(l_rot_err)
    avg_yaw_err = np.mean(l_rot_err)

    print("\t\tError stat: %.3f, %.3f, %.3f" % (avg_traj_ape, avg_traj_ape_gt, avg_yaw_err))

    return eval_trans, eval_rots, tot_n_comp, tot_traj_dur, tot_traj_len, avg_traj_ape, avg_traj_ape_gt, avg_rot_err, avg_yaw_err


if __name__ == '__main__':

    pathPoseEst = '../../ev_asynch_tracker_pose_chain.txt'

    eval_times = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    # parse command line
    parser = argparse.ArgumentParser(description='''
    This script plots piece-wise ground-truth camera pose and aligned estimated pose. 
    Note: ORB-SLAM saves qw last, but in euroc ground-truth it's first.
    ''')
    parser.add_argument('path_settings', help='settings.yaml (containing dataset path info)')
    parser.add_argument('-path_pose_est', help='estimated trajectory (format: timestamp tx ty tz qx qy qz qw)',
                        default=pathPoseEst)

    parser.add_argument('--pose_config', help='pose config: 0: ORB Frame, 1: ORB KeyFrame (default), 2: EvKF, 3: EvImKF', default=1)

    parser.add_argument('--dir_pose', help='saved est. pose direction: 0: Twc (default), 1: Tcw', default=0)
    parser.add_argument('--dir_gt_qw', help='groundtruth quaternion direction: 0: qw last, 1: qw first (EuRoC, default)')

    parser.add_argument('--time_scale', help='time scale for dataset (default: 1 s)')

    parser.add_argument('--max_difference',
                        help='maximally allowed time difference for matching entries (default: 0.01 s)', default=0.01)

    parser.add_argument('--scale', help='scaling factor for the second trajectory (default: 1.0)', default=1.0)

    parser.add_argument('--sort_ts', help='0 (default): do not sort poses based on ts, 1: otherwise', default=0)

    parser.add_argument('--match_ts_s', help='0 (default): do not sort poses based on ts, 1: otherwise', default=3)
    parser.add_argument('--match_ts_e', help='0 (default): do not sort poses based on ts, 1: otherwise', default=8)

    parser.add_argument('--est_idx', help='index of EST file being procecced: default: -1 (median over all)',
                        default=-1)

    args = parser.parse_args()

    est_idx = int(args.est_idx)
    pose_config = int(args.pose_config)
    sort_ts = False
    if int(args.sort_ts) == 1:
        sort_ts = True

    pose_dir = int(args.dir_pose)

    ini_scale = float(args.scale)

    gt_max_tdiff = float(args.max_difference)
    match_ts_s = float(args.match_ts_s)
    match_ts_e = float(args.match_ts_e)
    tmatch_range = (match_ts_s, match_ts_e)

    # Resolve GT path
    gtFile = mfm.parse_gt_path_cv(args.path_settings)

    # Resolve EST path
    estFile, time_scale, qw_dir, ds_config = mfm.parse_est_base_path_cv(args.path_settings, pose_config)
    defEstFile = args.path_pose_est

    if args.time_scale is not None:
        time_scale = float(args.time_scale)

    if args.dir_gt_qw is not None:
        qw_dir = int(args.dir_gt_qw)

    estFilesList = glob.glob(estFile + "*.txt")
    # Filter and refine the list
    estFilesList = [f for f in estFilesList if re.match(estFile + r'_[0-9._]+\.txt', f)]
    defEstFilesList = glob.glob(defEstFile + "*.txt")

    if len(defEstFilesList) > 0:
        estFile = defEstFile
        estFilesList = defEstFilesList

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

    # Process each EST file
    nMatchedPairs = []
    trajDuration = []
    trajLength = []

    trajApe = []
    trajApeGt = []
    rotErr = []
    yawErr = []

    evalApeInfo = dict()
    evalYawErrInfo = dict()
    for ts in eval_times:
        evalApeInfo[ts] = []
        evalYawErrInfo[ts] = []

    if est_idx < 0:
        for idx in range(len(estFilesList)):

            currFile = estFilesList[idx]

            evalApeInfo, evalYawErrInfo, nmatched, duration, trajlen, err_ape, ape_gt, rot_err, yaw_err = eval_est_file(
                currFile, gtList, eval_times, evalApeInfo, evalYawErrInfo,
                time_scale, gt_max_tdiff, tmatch_range, ini_scale)

            if nmatched > 0:
                nMatchedPairs.append(nmatched)
                trajDuration.append(duration)
                trajLength.append(trajlen)
                trajApe.append(err_ape)
                trajApeGt.append(ape_gt)
                rotErr.append(rot_err)
                yawErr.append(yaw_err)

    elif est_idx < len(estFilesList):

        currFile = estFilesList[est_idx]

        evalApeInfo, evalYawErrInfo, nmatched, duration, trajlen, err_ape, ape_gt, rot_err, yaw_err = eval_est_file(
            currFile, gtList, eval_times, evalApeInfo, evalYawErrInfo,
            time_scale, gt_max_tdiff, tmatch_range, ini_scale)

        if nmatched > 0:
            nMatchedPairs.append(nmatched)
            trajDuration.append(duration)
            trajLength.append(trajlen)
            trajApe.append(err_ape)
            trajApeGt.append(ape_gt)
            rotErr.append(rot_err)
            yawErr.append(yaw_err)

    if len(nMatchedPairs) > 0:

        trajDuration = np.sort(trajDuration)
        trajLength = np.sort(trajLength)
        trajApe = np.sort(trajApe)
        trajApeGt = np.sort(trajApeGt)

        # print("- Average tracking time: %f s" % trackingTimes[int(len(trackingTimes) / 2)])
        print("- Total traveled time: %f s" % (trajDuration[int(len(trajDuration) / 2)] / time_scale))
        print("- Total traveled distance: %f m" % trajLength[int(len(trajLength) / 2)])
        print("- RMS APE: %f m" % trajApe[int(len(trajApe) / 2)])
        print("- RMS APE GT: %f m" % trajApeGt[int(len(trajApeGt) / 2)])
        print("- AVG Rotation Error: %f degrees" % rotErr[int(len(rotErr) / 2)])
        print("- AVG Yaw Error: %f degrees" % yawErr[int(len(yawErr) / 2)])

        # Save errors info for later drawing
        err_info_file_name = "../../results/uslam_plots/" + os.path.basename(estFile) + "_uslam_errs.txt"
        delim = " "
        with open(err_info_file_name, 'w') as err_file:
            err_file.write('# ts min max avg m1 m2 m3\n')

            err_file.write('# E-I-C2 (Proposed): translation-error\n')
            for ts in eval_times:
                err_seq = np.sort(np.array(evalApeInfo[ts]))
                n_err_seq = len(err_seq)
                if n_err_seq <= 0:
                    continue
                err_file.write(str(ts) + delim + str(np.min(err_seq)) + delim + str(np.max(err_seq)) + delim +
                               str(np.mean(err_seq)) + delim + str(err_seq[int(n_err_seq * 0.25)]) + delim +
                               str(err_seq[int(n_err_seq * 0.5)]) + delim + str(err_seq[int(n_err_seq * 0.75)]) + '\n')

            err_file.write('# E-I-C2 (Proposed): rotation-error\n')
            for ts in eval_times:
                err_seq = np.sort(np.abs(np.array(evalYawErrInfo[ts])))
                n_err_seq = len(err_seq)
                if n_err_seq <= 0:
                    continue
                err_file.write(str(ts) + delim + str(np.min(err_seq)) + delim + str(np.max(err_seq)) + delim +
                               str(np.mean(err_seq)) + delim + str(err_seq[int(n_err_seq * 0.25)]) + delim +
                               str(err_seq[int(n_err_seq * 0.5)]) + delim + str(err_seq[int(n_err_seq * 0.75)]) + '\n')
        print("- Error info writen to: " + err_info_file_name)
    else:
        print("** Cannot find any estimated trajectory match! **")
