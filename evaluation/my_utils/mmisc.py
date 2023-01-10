import numpy as np
from scipy.spatial.transform import Rotation as scipyR
from scipy.spatial.transform import Slerp


def interp_pose(p0, p1, time_i, time_01):

    d_times = time_01[1] - time_01[0]
    d_ti = time_i - time_01[0]
    s = d_ti / d_times
    # p01 = np.matmul(p0, np.linalg.inv(p1))

    rot0 = p0[0:3, 0:3]
    rot1 = p1[0:3, 0:3]

    rots_obj = scipyR.from_matrix([rot0, rot1])
    slerp = Slerp(time_01, rots_obj)
    times = [time_01[0], time_i, time_01[1]]
    interp_rots = slerp(times)

    t0i = (1-s) * p0[0:3, 3] + s * p1[0:3, 3]
    p0i = np.identity(4, dtype=np.float32)
    p0i[0:3, 0:3] = interp_rots.as_matrix()[1]
    p0i[0:3, 3] = t0i

    # np.matmul(p0, p0i)
    return p0i


def calc_dist_identity_quat(curr_trans):
    """
    Calculates the distance of pose represented as [x y z qx qy qz qw] to identity transform
    :param curr_trans:
    :return:
    """
    return np.sum(np.abs(curr_trans - [0, 0, 0, 0, 0, 0, 1]))


def is_identity_quat(curr_trans, th=1e-6):
    """
    :param curr_trans: input transformation: [tx ty tz qx qy qz qw]
    :param th:
    :return: True: is identity transform (I4x4)
            False: is not identity
    """
    return calc_dist_identity_quat(curr_trans) < th


def swap_qw(pose_list):
    """
    Swaps qw in a list of poses like: l[ts x y z qw qx qy qz] to l[... z qx qy qz qw]
    :param pose_list:
    :return:
    """
    for gt in pose_list:
        sval = gt[4]
        gt[4:7] = gt[5:]
        gt[7] = sval

    return pose_list


def break_pose_graph(pose_list, th_ts=1e-12, th_iden=1e-3):
    """
    Breaks pose chain by equal timestamps or identity transform
    :param pose_list: [[ts0 x0 y0 z0 qx0 qy0 qz0 qw0] ...]
    :param th_ts:
    :param th_iden:
    :return: list of consecutive poses [[[x0 y0 ...],[x1 y1 ...]...]...]
    """
    posePieces = []
    nList = len(pose_list)

    if nList < 3:
        posePieces.append(pose_list)
        return posePieces

    lastPose = pose_list[0]
    currPiece = []
    currPiece.append(lastPose)

    for i in range(1, nList):
        currPose = pose_list[i]
        if abs(currPose[0]-lastPose[0]) < th_ts or is_identity_quat(currPose[1:], th_iden):

            posePieces.append(currPiece)
            lastPose = currPose
            currPiece = []
            currPiece.append(lastPose)
        else:
            currPiece.append(currPose)
            if i == nList-1:
                posePieces.append(currPiece)

    return posePieces


def select_best_pose_piece(posePieces):

    if len(posePieces) <= 0:
        return []

    max_len = 0
    max_pg = []

    for posePiece in posePieces:
        if max_len < len(posePiece):
            max_len = len(posePiece)
            max_pg = posePiece

    return max_pg


def list2dict(pose_list):
    """
    Convert a list of floats: [ts x y ...] to dict: [ts, [x y ...]]
    :param pose_list:
    :return:
    """
    bin_list = [[pose[0], pose[1:]] for pose in pose_list]
    return dict(bin_list)


def sort_pose_list(pose_list):
    pose_dict = list2dict(pose_list)
    pose_ts = list(pose_dict.keys())
    pose_ts.sort()
    sorted_pose = []
    for ts in pose_ts:
        curr_pose = []
        curr_pose.append(ts)
        for val in pose_dict[ts]:
            curr_pose.append(val)
        sorted_pose.append(curr_pose)
    return sorted_pose


if __name__ == "__main__":

    key_rots = scipyR.random(2, random_state=2342345)
    p0 = np.identity(4)
    p0[0:3, 0:3] = key_rots.as_matrix()[0]
    p0[0:3, 3] = np.array([-0.23, 0.55, 0.12]).transpose()
    p1 = np.identity(4)
    p1[0:3, 0:3] = key_rots.as_matrix()[1]
    p1[0:3, 3] = np.array([0.13, 0.055, -0.007]).transpose()

    pi = interp_pose(p0, p1, 0.35, [0.012, 1.108])

    print("Testing utils:")
    trans = np.array([-1.23448e-07, -1.33685e-07, -3.18998e-07, -2.8187e-05, -4.26486e-05, 8.75161e-05, 1])
    print("Transform: ")
    print(trans)
    if is_identity_quat(trans, 1e-3):
        print("Is identity")
    else:
        print("Is not identity")
    trans = np.array([-0.0823314, -0.0130145, -0.0204296, -0.00542771, -0.0044356, 0.00535609, 0.999961])
