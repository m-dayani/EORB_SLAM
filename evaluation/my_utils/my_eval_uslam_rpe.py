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


def get_last_frame(idx0, acc_dists, dist):
    for k in range(idx0, len(acc_dists), 1):
        if acc_dists[k] > acc_dists[idx0] + dist:
            return k
    return -1


def eval_est_poses(posePiece, gtList, eval_times, eval_trans, eval_rots,
                   time_scale, gt_max_tdiff, tmatch_range=(3, 8), ini_scale=1.0):

    n_compared = 0
    traj_dur = 0
    traj_len = 0
    traj_ape = 0
    traj_ape_gt = 0
    rot_error_avg = 0
    yaw_error_avg = 0
    ts_offset = 0.0

    # This is 1 for EvETHZ, 2 for EuRoC, and 2 for EvMVSEC -> various datasets define body frame differently
    yaw_idx = 1
    step_size = 1
    n_errs_def = 60

    eval_keys = list(eval_trans.keys())
    n_eval_keys = len(eval_keys)

    if len(posePiece) < 3:
        return eval_trans, eval_rots, n_compared, traj_dur, traj_len, traj_ape, traj_ape_gt, rot_error_avg, yaw_error_avg

    s_time = tmatch_range[0]
    e_time = tmatch_range[1]
    first_time = min(gtList[0][0], posePiece[0][0])
    first_time /= time_scale

    first_list = mmisc.list2dict(gtList)
    second_list = mmisc.list2dict(posePiece)

    matches = assoc.associate(first_list, second_list, ts_offset, gt_max_tdiff)
    n_matches = len(matches)

    if n_matches < 2:
        print("** Cannot find gt pose matches for piece: %f ~ %f **" % (posePiece[0][0], posePiece[-1][0]))
        return eval_trans, eval_rots, n_compared, traj_dur, traj_len, traj_ape, traj_ape_gt, rot_error_avg, yaw_error_avg

    # Align using a subset of all matched poses
    first_xyz = np.matrix([[float(value) for value in first_list[a][0:3]]
                           for a, b in matches if s_time < a/time_scale < e_time]).transpose()
    second_xyz = np.matrix([[float(value) * float(ini_scale) for value in second_list[b][0:3]]
                            for a, b in matches if s_time < a/time_scale < e_time]).transpose()

    if len(first_xyz) == 0 or len(second_xyz) == 0:
        print("** Cannot find gt pose matches for selected ts range: (%f, %f) **" % (match_ts_s, match_ts_e))
        return eval_trans, eval_rots, n_compared, traj_dur, traj_len, traj_ape, traj_ape_gt, rot_error_avg, yaw_error_avg

    rot, transGT, trans_errorGT, trans, trans_error, scale = align(second_xyz, first_xyz)

    rot_mat = np.eye(4)
    rot_mat[0:3, 0:3] = rot

    gt_dists = [0]
    for i in range(0, n_matches-1):
        t0 = matches[i][0]
        t1 = matches[i+1][0]
        gt_dists.append(gt_dists[i]+np.linalg.norm(first_list[t0][0:3]-first_list[t1][0:3]))

    trans_errorGT = []
    trans_error = []
    rot_error = []
    yaw_error = []
    dtimes = []
    euler_idx = []

    for idx0 in range(0, n_matches, step_size):
        for i in range(n_eval_keys):

            eval_k = eval_keys[i]
            n_comp = len(eval_trans[eval_k])

            if n_comp >= n_errs_def:
                continue

            idx1 = get_last_frame(idx0, gt_dists, eval_k)

            if idx1 < 0:
                continue

            # calculate relative error
            ts0_gt, ts0_est = matches[idx0]
            ts1_gt, ts1_est = matches[idx1]

            T0_est = eval_rpe.transform44(np.insert(second_list[ts0_est], 0, ts0_est))
            T0_est = np.matmul(rot_mat, T0_est)
            T1_est = eval_rpe.transform44(np.insert(second_list[ts1_est], 0, ts1_est))
            T1_est = np.matmul(rot_mat, T1_est)
            T0_gt = eval_rpe.transform44(np.insert(first_list[ts0_gt], 0, ts0_gt))
            T1_gt = eval_rpe.transform44(np.insert(first_list[ts1_gt], 0, ts1_gt))

            dT_est = eval_rpe.ominus(T0_est, T1_est)
            dT_est_scaled = eval_rpe.scale(dT_est, scale)
            dT_gt = eval_rpe.ominus(T0_gt, T1_gt)

            E = eval_rpe.ominus(dT_est, dT_gt)
            E_gt = eval_rpe.ominus(dT_est_scaled, dT_gt)

            err_t = eval_rpe.compute_distance(E)
            err_t_gt = eval_rpe.compute_distance(E_gt)
            err_rot = eval_rpe.compute_angle(E)

            obj_rot_err = scipyR.from_matrix(E[0:3, 0:3])
            rot_euler = obj_rot_err.as_euler('zyx', degrees=True).squeeze()
            err_yaw = rot_euler[yaw_idx]

            # append to error stats
            eval_trans[eval_k].append(err_t)
            eval_rots[eval_k].append(err_yaw)

            trans_error.append(err_t)
            trans_errorGT.append(err_t_gt)
            rot_error.append(err_rot)
            yaw_error.append(err_yaw)

            euler_idx.append(np.argmax(abs(rot_euler)))

    unique_euidx, counts_euidx = np.unique(euler_idx, return_counts=True)

    dtimes.append(matches[-1][0] - matches[0][0])

    n_compared = len(trans_error)

    if n_compared > 0:
        traj_dur = np.sum(dtimes)
        traj_len = gt_dists[-1]     # np.sum(trajDists)
        traj_ape = np.sqrt(np.dot(trans_error, trans_error) / len(trans_error))
        traj_ape_gt = np.sqrt(np.dot(trans_errorGT, trans_errorGT) / len(trans_errorGT))
        rot_error_avg = np.sqrt(np.dot(rot_error, rot_error)) / len(rot_error) * (180.0 / np.pi)
        yaw_error_avg = np.sqrt(np.dot(yaw_error, yaw_error)) / len(yaw_error)

    return eval_trans, eval_rots, n_compared, traj_dur, traj_len, traj_ape, traj_ape_gt, rot_error_avg, yaw_error_avg


def eval_est_file(file_name, gtList, eval_times, eval_trans, eval_rots,
                  sort_ts, time_scale, gt_max_tdiff, tmatch_range=(3, 8), ini_scale=1.0):

    print("> EST File: " + file_name)

    # estList = mfm.read_file_list(file_name)
    estPosePieces = mfm.read_dosconn_graph_list(file_name)

    # th_ts = 1e-12
    # th_iden = 1e-3
    # if sort_ts:
    #     estList = sort_pose_list(estList)
    #
    # # Disconnected graph calcs are not supported here
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

        if n_compared <= 0:
            continue

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

    return eval_trans, eval_rots, tot_n_comp, tot_traj_dur, tot_traj_len, avg_traj_ape, avg_traj_ape_gt, avg_rot_err, avg_yaw_err


if __name__ == '__main__':

    eval_times = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    # parse command line
    parser = argparse.ArgumentParser(description='''
    This script plots piece-wise ground-truth camera pose and aligned estimated pose. 
    Note: ORB-SLAM saves qw last, but in euroc ground-truth it's first.
    ''')
    parser.add_argument('path_settings', help='settings.yaml (containing dataset path info)')
    parser.add_argument('-path_pose_est', help='estimated trajectory (format: timestamp tx ty tz qx qy qz qw)',
                        default='')

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

    # parser.add_argument('--est_idx', help='index of EST file being procecced: default: -1 (median over all)',
    #                     default=-1)

    args = parser.parse_args()

    # est_idx = int(args.est_idx)
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
    if len(defEstFile) > 0:
        estFile = defEstFile

    if args.time_scale is not None:
        time_scale = float(args.time_scale)

    if args.dir_gt_qw is not None:
        qw_dir = int(args.dir_gt_qw)

    estFilesList = glob.glob(estFile + "*.txt")
    # Filter and refine the list
    estFilesList = [f for f in estFilesList if re.match(estFile + r'_[0-9._]+\.txt', f)]

    if len(estFilesList) <= 0:
        # Abort
        print("** Neither settings path nor argument path: " + estFile + " can be found! **")
        exit(1)

    # Resolve est_idx
    est_idx_prompt = "Enter est_file idx: {\n"
    for i in range(0, len(estFilesList)):
        currFile = estFilesList[i]
        avg_ttime = re.findall(r'\d+\.\d+', currFile)
        if len(avg_ttime) == 0:
            avg_ttime_str = '?'
        else:
            avg_ttime_str = avg_ttime[0]
        est_idx_prompt += '\t\t' + str(i) + ': ' + avg_ttime_str + '\n'
    est_idx_prompt += "\t\t-1: all (default)}\n? "
    est_idx = int(input(est_idx_prompt) or '-1')

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
                sort_ts, time_scale, gt_max_tdiff, tmatch_range, ini_scale)

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
            sort_ts, time_scale, gt_max_tdiff, tmatch_range, ini_scale)

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
        method = 'USLAM [10]'
        delim = " "
        with open(err_info_file_name, 'w') as err_file:
            err_file.write('# ts min max avg m1 m2 m3\n')

            err_file.write('# '+method+': translation-error\n')
            for ts in eval_times:

                curr_seq = np.array(evalApeInfo[ts])
                n_err_seq = len(curr_seq)
                if n_err_seq <= 0:
                    continue

                err_seq = np.sort(curr_seq)     #np.random.choice(curr_seq, 100))
                # n_err_seq = len(err_seq)

                err_file.write(str(ts) + delim + str(np.min(err_seq)) + delim + str(np.max(err_seq)) + delim +
                               str(np.mean(err_seq)) + delim + str(err_seq[int(n_err_seq * 0.25)]) + delim +
                               str(err_seq[int(n_err_seq * 0.5)]) + delim + str(err_seq[int(n_err_seq * 0.75)]) + '\n')

            err_file.write('# '+method+': rotation-error\n')
            for ts in eval_times:

                curr_seq = np.abs(np.array(evalYawErrInfo[ts]))
                n_err_seq = len(curr_seq)
                if n_err_seq <= 0:
                    continue

                err_seq = np.sort(curr_seq)     #np.random.choice(curr_seq, 100))
                # n_err_seq = len(err_seq)

                err_file.write(str(ts) + delim + str(np.min(err_seq)) + delim + str(np.max(err_seq)) + delim +
                               str(np.mean(err_seq)) + delim + str(err_seq[int(n_err_seq * 0.25)]) + delim +
                               str(err_seq[int(n_err_seq * 0.5)]) + delim + str(err_seq[int(n_err_seq * 0.75)]) + '\n')
        print("- Error info writen to: " + err_info_file_name)
    else:
        print("** Cannot find any estimated trajectory match! **")
