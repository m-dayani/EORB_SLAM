import sys
import argparse
import numpy as np
import glob
import re
import collections
import heapq

from scipy.spatial.transform import Rotation as scipyR

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

sys.path.append('../tum_tools')
import evaluate_rpe as eval_rpe

sys.path.append('../my_utils')
import mf_manip as mfm
import mmisc
import my_eval_rpe as meval_rpe


def change_rot_dir_quat(pose):
    """

    :param qin: [qx qy qz qw]
    :return: qout: [qqx qqy qqz qqw]
    """
    rot = scipyR.from_quat(pose[4:8])
    rot_mat = np.asarray(rot.as_matrix())
    rot_mat = rot_mat.transpose()
    rot = scipyR.from_matrix(rot_mat)
    qinv = rot.as_quat()
    return np.array([pose[0], pose[1], pose[2], pose[3], qinv[0], qinv[1], qinv[2], qinv[3]], dtype=np.float64)


def change_pose_dir(pose):

    pose4x4 = eval_rpe.transform44(pose)
    pose4x4 = np.linalg.inv(pose4x4)

    rot = scipyR.from_matrix(pose4x4[0:3, 0:3])
    q = rot.as_quat()

    return np.array([pose[0], pose4x4[0, 3], pose4x4[1, 3], pose4x4[2, 3], q[0], q[1], q[2], q[3]], dtype=np.float64)


def pose2string(pose, tsc=1.0):

    out_str = "%.9f" % (tsc * pose[0])
    for i in range(len(pose)-1):
        out_str += " %.9f" % pose[i+1]

    return out_str


if __name__ == "__main__":

    root_path = '../../results/ev_ethz'

    allPoseTxt = glob.glob(root_path+'/ekf_ev_ethz_mono_ev_shapes_6dof_*.txt')

    for filePath in allPoseTxt:

        print(">> Processing file: "+filePath)

        poseList = mfm.read_file_list(filePath)

        with open(filePath, "w") as outFile:
            for pose in poseList:
                newpose = change_rot_dir_quat(pose)
                #newpose = change_pose_dir(pose)
                line = pose2string(newpose) + "\n"
                outFile.write(line)
