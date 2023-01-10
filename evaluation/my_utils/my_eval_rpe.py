import sys
import argparse
import numpy as np
import glob
import re

import mf_manip as mfm
import mmisc

from scipy.spatial.transform import Rotation as scipyR

sys.path.append('../')
import associate as assoc
from evaluate_ate_scale import align, plot_traj

sys.path.append('../tum_tools')
import evaluate_rpe as eval_rpe

sys.path.append('../visualization')
import viz_tools as mviz


def find_closest_gt(poses, stamps_gt, time_scale=1.0, param_offset=0.0, gt_max_time_difference=0.01):
    """
    Find the 2 closest gtPoses to estimated poses based on closest ts
    :param poses: list[pose0, pose1] in format: [ts x y z qx qy qz qw]
    :param stamps_gt: list of all gt timestamps
    :param param_offset:
    :param gt_max_time_difference:
    :return: if successful: gtIdx0, gtIdx1, gtTs0, gtTs1 else -1, -1, 0.0, 0.0
    """
    assert len(poses) == 2, "wrong estimated poses length: %d" % (len(poses))

    stamp_est_0 = poses[0][0]/time_scale
    stamp_est_1 = poses[1][0]/time_scale

    idx_gt_0 = eval_rpe.find_closest_index(stamps_gt, stamp_est_0 + param_offset)
    idx_gt_1 = eval_rpe.find_closest_index(stamps_gt, stamp_est_1 + param_offset)

    stamp_gt_0 = stamps_gt[idx_gt_0]
    stamp_gt_1 = stamps_gt[idx_gt_1]

    if (abs(stamp_gt_0 - (stamp_est_0 + param_offset)) > gt_max_time_difference or
            abs(stamp_gt_1 - (stamp_est_1 + param_offset)) > gt_max_time_difference):
        return -1, -1, 0.0, 0.0

    return idx_gt_0, idx_gt_1, stamp_gt_0, stamp_gt_1


def transform44(pose0, pose1, pose_dir=0):
    """
    Converts poses like [ts x y z qx qy qz qw] to Tcw4x4 and calculated relative cam pose
    :param pose0: Twc0 in format [ts x y z qx qy qz qw]
    :param pose1: Twc1 in format [ts x y z qx qy qz qw]
    :param pose_dir: poses are represented as 0: Twc, 1: Tcw
    :return: Twc0, Twc1, Tc1c0
    """
    if pose_dir == 1:
        pose_c0w = eval_rpe.transform44(pose0)
        pose_c1w = eval_rpe.transform44(pose1)
        pose_wc0 = np.linalg.inv(pose_c0w)
        pose_wc1 = np.linalg.inv(pose_c1w)
    else:
        pose_wc0 = eval_rpe.transform44(pose0)
        pose_wc1 = eval_rpe.transform44(pose1)

    pose_c1c0 = eval_rpe.ominus(pose_wc1, pose_wc0)

    return pose_wc0, pose_wc1, pose_c1c0


def scale(rp_est, rp_gt):
    """
    Scales relative translations so that they are comparable
    :param rp_est: estimated rel. trans 4x4
    :param rp_gt: gt rel. trans 4x4
    :return: [Rest, test * (abs(tgt) / abs(test))] trans 4x4, rel. scale, gt scale, est. scale
    """
    fx_sc = eval_rpe.compute_distance(rp_gt)
    curr_sc = eval_rpe.compute_distance(rp_est)
    rel_sc = fx_sc / curr_sc

    rp_est = eval_rpe.scale(rp_est, rel_sc)

    return rp_est, rel_sc, fx_sc, curr_sc


def compute_rel_err_dpose(dpose_est, dpose_gt):
    """
    Computes translational and angular error between two relative poses
    :param dpose_est: Tc1c0_est (4x4)
    :param dpose_gt: Tc1c0_gt (4x4)
    :return: trans_err, rot_err, error44
    """
    error44 = eval_rpe.ominus(dpose_est, dpose_gt)

    trans = eval_rpe.compute_distance(error44)
    rot = eval_rpe.compute_angle(error44)

    return trans, rot, error44


def compute_rel_err(piece_est, stamps_gt, list_gt, nn_elems, time_scale=1.0, ts_offset=0.0, gt_max_tdiff=0.01,
                    is_scaled=True, pose_dir=0):
    """
    Calculate RPE between two estimated pose and two gt pose
    :param piece_est:
    :param stamps_gt:
    :param list_gt:
    :param ts_offset:
    :param gt_max_tdiff:
    :param pose_dir: 1: Estimated poses are saved as Tcw, 0: Twc
    :return:
    """
    assert len(piece_est) == 2, "* ERR: Wrong pose piece length: %d" % (len(piece_est))

    c0_est = np.array(piece_est[0])
    c1_est = np.array(piece_est[1])

    idx_gt_0, idx_gt_1, ts0, ts1 = find_closest_gt(piece_est, stamps_gt, time_scale, ts_offset, gt_max_tdiff)
    if idx_gt_0 < 0:
        return 0, 0, -1, nn_elems

    # Order of estimated data is reverse because we save estimated poses in camera coord. sys.
    # Both relative poses must be related in camera coord. sys.
    Tc1_est, Tc0_est, Tc1c0_est = transform44(c1_est, c0_est, pose_dir)
    Tc0_gt, Tc1_gt, Tc1c0_gt = transform44(list_gt[idx_gt_0], list_gt[idx_gt_1], pose_dir)

    gtDist = eval_rpe.compute_distance(Tc1c0_gt)
    estDist = eval_rpe.compute_distance(Tc1c0_est)

    if is_scaled:
        Tc1c0_est, relSc, gtDist, estDist = scale(Tc1c0_est, Tc1c0_gt)

    trans, rot, err44 = compute_rel_err_dpose(Tc1c0_est, Tc1c0_gt)

    nn_elems.append((Tc1c0_gt[0, 1] + Tc1c0_est[0, 1], Tc1c0_gt[1, 2] + Tc1c0_est[1, 2], Tc1c0_gt[0, 2] + Tc1c0_est[0, 2]))

    return trans, rot, gtDist, nn_elems


def compute_rel_err_scale(piece_est, stamps_gt, list_gt, nn_elems, scale, time_scale=1.0, ts_offset=0.0,
                          gt_max_tdiff=0.01, is_scaled=True, pose_dir=0):
    """
    Calculate RPE between two estimated pose and two gt pose
    :param piece_est:
    :param stamps_gt:
    :param list_gt:
    :param ts_offset:
    :param gt_max_tdiff:
    :param pose_dir: 1: Estimated poses are saved as Tcw, 0: Twc
    :return:
    """
    assert len(piece_est) == 2, "* ERR: Wrong pose piece length: %d" % (len(piece_est))

    c0_est = np.array(piece_est[0])
    c1_est = np.array(piece_est[1])

    idx_gt_0, idx_gt_1, ts0, ts1 = find_closest_gt(piece_est, stamps_gt, time_scale, ts_offset, gt_max_tdiff)
    if idx_gt_0 < 0:
        return 0, 0, -1, nn_elems

    # Order of estimated data is reverse because we save estimated poses in camera coord. sys.
    # Both relative poses must be related in camera coord. sys.
    Tc1_est, Tc0_est, Tc1c0_est = transform44(c1_est, c0_est, pose_dir)
    Tc0_gt, Tc1_gt, Tc1c0_gt = transform44(list_gt[idx_gt_0], list_gt[idx_gt_1], pose_dir)

    if is_scaled:
        Tc1c0_est = eval_rpe.scale(Tc1c0_est, scale)

    gtDist = eval_rpe.compute_distance(Tc1c0_gt)
    estDist = eval_rpe.compute_distance(Tc1c0_est)

    trans, rot, err44 = compute_rel_err_dpose(Tc1c0_est, Tc1c0_gt)

    nn_elems.append((Tc1c0_gt[0, 1] + Tc1c0_est[0, 1], Tc1c0_gt[1, 2] + Tc1c0_est[1, 2], Tc1c0_gt[0, 2] + Tc1c0_est[0, 2]))

    return trans, rot, gtDist, nn_elems


def eval_est_poses(posePiece, stamps_gt, gtList, time_scale, ts_offset, gt_max_tdiff, is_scaled):

    first_list = mmisc.list2dict(gtList)

    n_pairs = 0
    nn_elems = []

    cur_err_trans = []
    cur_err_rot = []
    cur_dp = []
    cur_dts = []

    nPiece = len(posePiece)
    if nPiece < 2:

        print("** ERROR: Not enough poses in current piece **")
        return n_pairs, cur_err_trans, cur_err_rot, cur_dp, cur_dts

    elif nPiece == 2:

        trans, rot, dtrans, nn_elems = compute_rel_err(posePiece, stamps_gt, gtList, nn_elems, time_scale,
                                                                ts_offset, gt_max_tdiff, is_scaled)

        if dtrans < 0:
            return n_pairs, cur_err_trans, cur_err_rot, cur_dp, cur_dts

        cur_err_trans.append(trans)
        cur_err_rot.append(rot)
        cur_dp.append(dtrans)
        cur_dts.append(posePiece[1][0] - posePiece[0][0])

        n_pairs += 1

    elif nPiece > 2:

        # compute relative scale:
        second_list = mmisc.list2dict(posePiece)
        matches = assoc.associate(first_list, second_list, ts_offset, gt_max_tdiff)

        if len(matches) < 2:
            print("** Cannot find gt pose matches for piece: %f ~ %f **" % (posePiece[0][0], posePiece[-1][0]))
            return n_pairs, cur_err_trans, cur_err_rot, cur_dp, cur_dts

        first_xyz = np.matrix([[float(value) for value in first_list[a][0:3]] for a, b in matches]).transpose()
        second_xyz = np.matrix(
            [[float(value) * float(ini_scale) for value in second_list[b][0:3]] for a, b in matches]).transpose()

        rot, transGT, trans_errorGT, trans, trans_error, scale = align(second_xyz, first_xyz)

        for i in range(len(posePiece) - 1):

            trans, rot, dtrans, nn_elems = compute_rel_err_scale(posePiece[i:i + 2], stamps_gt, gtList,
                                                    nn_elems, scale, time_scale, ts_offset, gt_max_tdiff, is_scaled)

            if dtrans < 0:
                return n_pairs, cur_err_trans, cur_err_rot, cur_dp, cur_dts

            cur_err_trans.append(trans)
            cur_err_rot.append(rot)
            cur_dp.append(dtrans)
            cur_dts.append(posePiece[i + 1][0] - posePiece[i][0])

            n_pairs += 1

    # mviz.plot_nn_elements(nn_elems)

    return n_pairs, cur_err_trans, cur_err_rot, cur_dp, cur_dts


def eval_est_file(file_name, stamps_gt, gtList, time_scale, ts_offset, gt_max_tdiff, is_scaled, all_pieces, rpe_mode):

    print("> EST File: " + file_name)

    estPosePieces = mfm.read_dosconn_graph_list(file_name)

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

    errTrans = []
    errRotDeg = []
    deltaPs = []
    deltaTs = []
    tot_n_pairs = 0

    for posePiece in estPosePieces:

        n_pairs, cur_err_trans, cur_err_rot, cur_dp, cur_dts = eval_est_poses(
            posePiece, stamps_gt, gtList, time_scale, ts_offset, gt_max_tdiff, is_scaled)

        if len(cur_err_trans) <= 0:
            continue

        tot_n_pairs += n_pairs
        tot_trans = np.sum(cur_dp)
        deltaPs.append(tot_trans)
        deltaTs.append(np.sum(cur_dts))

        if rpe_mode == 1:
            # In this mode, calculate the rmse of errors and normalize with the length of piece
            errTrans.append(np.sqrt(np.dot(cur_err_trans, cur_err_trans) / len(cur_err_trans)) / tot_trans)
            errRotDeg.append(np.sqrt(np.dot(cur_err_rot, cur_err_rot) / len(cur_err_rot)) / tot_trans)
        else:
            # Append all and calculate rmse as before
            errTrans += cur_err_trans
            errRotDeg += cur_err_rot

    rot_sc = 180.0 / np.pi
    traj_dur = 0
    traj_len = 0
    rpe_trans = 0
    rpe_rot = 0

    if tot_n_pairs > 0:
        traj_dur = np.sum(deltaTs)
        traj_len = np.sum(deltaPs)

        if rpe_mode == 1:
            rpe_trans = np.mean(errTrans)
            rpe_rot = np.mean(errRotDeg)
        else:
            rpe_trans = np.sqrt(np.dot(errTrans, errTrans) / len(errTrans))
            rpe_rot = np.sqrt(np.dot(errRotDeg, errRotDeg) / len(errRotDeg)) * rot_sc

    return tot_n_pairs, traj_dur, traj_len, rpe_trans, rpe_rot, errTrans, errRotDeg


# WARNING: Choose the dir_gt_qw & dir_pose settings correctly!
if __name__ == '__main__':

    # parse command line
    parser = argparse.ArgumentParser(description='''
    This script plots ground-truth camera pose and aligned estimated pose. 
    ''')
    parser.add_argument('path_settings', help='settings.yaml (containing dataset path info)')
    parser.add_argument('-path_pose_est', help='estimated trajectory (format: timestamp tx ty tz qx qy qz qw)',
                        default='')

    parser.add_argument('--est_idx', help='index of EST file being procecced: default: -1 (median over all)',
                        default=-1)
    parser.add_argument('--pose_config',
                        help='pose config: 0: ORB Frame, 1: ORB KeyFrame (default), 2: EvKF, 3: EvImKF', default=1)

    parser.add_argument('--dir_pose', help='pose direction: 0: Twc (default), 1: Tcw', default=0)
    parser.add_argument('--dir_gt_qw', help='groundtruth quaternion direction: 0: qw last (EvETHZ), 1: qw first (EuRoC, default)',
                        default=1)

    parser.add_argument('--all_pieces', help='compute error for all disconnected pieces in pose graph, default: 1',
                        default=1)
    parser.add_argument('--rpe_mode', help='how to compute RPE: 0: RMS error (default), 1: normalized with rel. length',
                        default=0)

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
    rpe_mode = int(args.rpe_mode)

    gt_max_tdiff = float(args.max_difference)
    ts_offset = float(args.offset)
    ini_scale = float(args.scale)

    # Resolve GT path
    gtFile = mfm.parse_gt_path_cv(args.path_settings)
    # Resolve EST path
    pathPoseEst = args.path_pose_est
    pathEstOther, time_scale, qw_dir, ds_config = mfm.parse_est_base_path_cv(args.path_settings, pose_config)
    if len(pathPoseEst) <= 0:
        pathPoseEst = pathEstOther
    estFilesList = glob.glob(pathPoseEst + "*.txt")
    # Filter and refine the list
    estFilesList = [f for f in estFilesList if re.match(pathPoseEst + r'_[0-9._]+\.txt', f)]
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
    stamps_gt = gtList[:, 0]/time_scale

    if gtList.shape[1] > 8:
        gtList = gtList[:, 0:8]

    if qw_dir == 1:
        gtList = mmisc.swap_qw(gtList)

    is_scaled = 'mono' in ds_config and 'imu' not in ds_config

    # Process estimated data:
    nMatchedPairs = []
    trajDuration = []
    trajLength = []
    trajRpeTrans = []
    trajRpeRot = []
    trajStats = dict()

    ttime_stats = dict()
    ttime_stats['times_cnt'] = dict()

    if est_idx < 0:
        for idx in range(len(estFilesList)):

            currFile = estFilesList[idx]

            ttime_stats = mfm.read_timing_hears(currFile, ttime_stats)

            nmatched, duration, trajlen, rpe_tran, rpe_rot, errTrans, errRot = eval_est_file(
                currFile, stamps_gt, gtList, time_scale, ts_offset, gt_max_tdiff, is_scaled, all_pieces, rpe_mode)

            if nmatched > 0:
                nMatchedPairs.append(nmatched)
                trajDuration.append(duration)
                trajLength.append(trajlen)
                trajRpeTrans.append(rpe_tran)
                trajRpeRot.append(rpe_rot)

                trajStats[rpe_tran] = [nmatched, duration, trajlen, rpe_rot]

    elif est_idx < len(estFilesList):

        currFile = estFilesList[est_idx]

        ttime_stats = mfm.read_timing_hears(currFile, ttime_stats)

        nmatched, duration, trajlen, rpe_tran, rpe_rot, errTrans, errRot = eval_est_file(
            currFile, stamps_gt, gtList, time_scale, ts_offset, gt_max_tdiff, is_scaled, all_pieces, rpe_mode)

        if nmatched > 0:
            nMatchedPairs.append(nmatched)
            trajDuration.append(duration)
            trajLength.append(trajlen)
            trajRpeTrans.append(rpe_tran)
            trajRpeRot.append(rpe_rot)

            trajStats[rpe_tran] = [nmatched, duration, trajlen, rpe_rot]

    if len(nMatchedPairs) > 0:

        # trajDuration = np.sort(trajDuration)
        # trajLength = np.sort(trajLength)
        # trajRpeTrans = np.sort(trajRpeTrans)
        # trajRpeRot = np.sort(trajRpeRot)
        trajStatsSorted = sorted(trajStats)
        trajRpeTrans = trajStatsSorted[int(len(trajStatsSorted) / 2)]

        if rpe_mode == 1:
            trans_unit = ""
            rot_unit = "deg/m"
        else:
            trans_unit = "m"
            rot_unit = "deg"

        # print("- Average tracking time: %f s" % trackingTimes[int(len(trackingTimes) / 2)])
        print("- Average tracking time stats ((min, max, med, avg) (s)):")
        for ttime_key in list(ttime_stats.keys()):
            if ttime_key == 'times_cnt':
                continue
            ttime_cnt = ttime_stats['times_cnt'][ttime_key]
            ttsl = [v/ttime_cnt for v in ttime_stats[ttime_key]]
            print("\t-- "+ttime_key+": (%.6f, %.6f, %.6f, %.6f)" % (ttsl[0], ttsl[1], ttsl[2], ttsl[3]))

        print("- Total traveled time: %f s" % trajStats[trajRpeTrans][1])
        print("- Total traveled distance: %f m" % trajStats[trajRpeTrans][2])
        print("- RMS Translational RPE: %f " % trajRpeTrans + trans_unit)
        print("- RMS Rotational RPE: %f " % trajStats[trajRpeTrans][3] + rot_unit)
    else:
        print("** Cannot find any estimated trajectory match! **")
