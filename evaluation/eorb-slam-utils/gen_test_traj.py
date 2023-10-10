import sys
import argparse
import random
import numpy as np

import mf_manip as mfm
import mmisc

sys.path.append('../tum_tools')
import evaluate_rpe as eval_rpe


def euler_ang_to_rot(euler_angles):

    phi = euler_angles[0]
    Rx = np.array((
        (1.0, 0.0, 0.0),
        (0.0, np.cos(phi), -np.sin(phi)),
        (0.0, np.sin(phi), np.cos(phi))
    ), dtype=np.float64)
    theta = euler_angles[1]
    Ry = np.array((
        (np.cos(theta), 0.0, np.sin(theta)),
        (0.0, 1.0, 0.0),
        (-np.sin(theta), 0.0, np.cos(theta))
    ), dtype=np.float64)
    psi = euler_angles[2]
    Rz = np.array((
        (np.cos(psi), -np.sin(psi), 0.0),
        (np.sin(psi), np.cos(psi), 0.0),
        (0.0, 0.0, 1.0)
    ), dtype=np.float64)

    return np.matmul(Rz, np.matmul(Ry, Rx))


def gen_rand_trans(max_trans, max_angle, scale):

    t = np.random.rand(3) * max_trans
    euler_angles = np.random.rand(3) * (max_angle * np.pi / 180.0)
    R = euler_ang_to_rot(euler_angles)

    T = np.identity(4)
    T[0:3, 0:3] = scale * R
    T[0:3, 3] = t

    return T, R, t


def rot_to_quat(rot, dir_qw=1):
    """
    Converts a 3x3 rotation matrix to quaternion representation
    :param rot:
    :param dir_qw: 0: [qx qy qz qw], 1: [qw qx qy qz]
    :return:
    """

    qw = 0.5 * np.sqrt(1 + np.trace(rot))
    qw_inv = 1.0 / qw
    qx = 0.25 * qw_inv * (rot[2, 1] - rot[1, 2])
    qy = 0.25 * qw_inv * (rot[0, 2] - rot[2, 0])
    qz = 0.25 * qw_inv * (rot[1, 0] - rot[0, 1])

    if dir_qw == 0:
        return [qx, qy, qz, qw]
    else:
        return [qw, qx, qy, qz]


def trans_quat(pose44, dir_qw):

    q = rot_to_quat(pose44[0:3, 0:3], dir_qw)
    return [pose44[0, 3], pose44[1, 3], pose44[2, 3], q[0], q[1], q[2], q[3]]


if __name__ == '__main__':

    pathPoseEst = '../../eval_test_poses.txt'

    # parse command line
    parser = argparse.ArgumentParser(description='''
    Generating simulated trajectory based on ground-truth to test evaluation scripts. 
    ''')
    parser.add_argument('path_settings', help='settings.yaml (containing dataset path info)')
    parser.add_argument('-path_pose_est', help='estimated trajectory (format: timestamp tx ty tz qx qy qz qw)',
                        default=pathPoseEst)

    parser.add_argument('--dir_pose', help='pose direction: 0: Twc, 1: Tcw (default)', default=1)
    parser.add_argument('--dir_gt_quat', help='groundtruth quaternion direction: 0: qw last, 1: qw first (default)', default=1)

    parser.add_argument('--scale', help='scaling factor for the second trajectory (default: 1.0)', default=1.0)
    parser.add_argument('--max_trans', help='maximum translation in meter (default: 1.0)', default=1.0)
    parser.add_argument('--max_rot', help='maximum rotation angle in degrees (default: 180)', default=180.0)

    parser.add_argument('--max_n_samples', help='maximum number of transformed samples (default: len(gtList))', default=-1)
    parser.add_argument('--offset', help='time offset added to the timestamps of the second file (default: 0.0)',
                        default=0.0)
    parser.add_argument('--max_difference',
                        help='maximally allowed time difference for matching entries (default: 0.01 s)', default=0.01)

    args = parser.parse_args()

    pose_dir = int(args.dir_pose)
    qw_dir = int(args.dir_gt_quat)

    scale = float(args.scale)
    max_trans = float(args.max_trans)
    max_rot = float(args.max_rot)

    max_n_samples = int(args.max_n_samples)
    gt_max_tdiff = float(args.max_difference)
    ts_offset = float(args.offset)

    # Construct gt path and data
    gtFile = mfm.parse_gt_path_cv(args.path_settings)
    poseFile = args.path_pose_est
    gtList = np.array(mfm.read_file_list(gtFile))

    indices = range(len(gtList))
    if max_n_samples > 0:
        indices = random.sample(indices, max_n_samples)
        indices = np.sort(indices)

    if gtList.shape[1] > 8:
        gtList = gtList[:, 0:8]

    if qw_dir == 1:
        gtList = mmisc.swap_qw(gtList)

    # Generate a random transformation
    T44, R, t = gen_rand_trans(max_trans, max_rot, scale)

    # estPoseList = []
    with open(poseFile, 'w') as outFile:

        for idx in indices:
            # Convert pose to trans4x4
            gtPose44 = eval_rpe.transform44(gtList[idx])

            # Transform pose
            estPose44 = np.matmul(T44, gtPose44)

            # Convert back and save pose
            outData = trans_quat(estPose44, qw_dir)
            outFile.write(str(gtList[idx][0]) + " " + " ".join(map(str, outData)) + "\n")

    print("R = ")
    print(R)
    print("t = ")
    print(t)
    print("sc = %f" % scale)
