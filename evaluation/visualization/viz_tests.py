import sys
import argparse
import numpy as np
import glob
import re

from scipy.spatial.transform import Rotation as scipyR

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import viz_tools as mviz

sys.path.append('../my_utils')
import mf_manip as mfm
import mmisc
import my_eval_rpe as meval_rpe


if __name__ == '__main__':

    pathRootData = '../../../data'

    # parse command line
    parser = argparse.ArgumentParser(description='''
    This script shows test plots. 
    ''')
    parser.add_argument('--path_settings', help='settings.yaml (containing dataset path info)')

    args = parser.parse_args()

    # Test1: plot poses and interpolated pose
    key_rots = scipyR.random(2, random_state=2342345)
    p0 = np.identity(4)
    p0[0:3, 0:3] = key_rots.as_matrix()[0]
    p0[0:3, 3] = np.array([-0.23, 0.55, 0.12]).transpose()
    p1 = np.identity(4)
    p1[0:3, 0:3] = key_rots.as_matrix()[1]
    p1[0:3, 3] = np.array([0.13, 0.055, -0.007]).transpose()

    pi = mmisc.interp_pose(p0, p1, 0.1, [0.012, 1.108])
    pi0 = mmisc.interp_pose(p0, p1, 0.22, [0.012, 1.108])
    pi1 = mmisc.interp_pose(p0, p1, 0.35, [0.012, 1.108])
    pi2 = mmisc.interp_pose(p0, p1, 0.5, [0.012, 1.108])
    pi3 = mmisc.interp_pose(p0, p1, 0.75, [0.012, 1.108])
    pi4 = mmisc.interp_pose(p0, p1, 0.85, [0.012, 1.108])
    pi5 = mmisc.interp_pose(p0, p1, 0.94, [0.012, 1.108])

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Setting axis to equal is a disaster in matplotlib!
    # ax.set_aspect('equal')
    # axisEqual3D(ax)

    mviz.draw_coord_sys(ax, p0, 0.05, 'O0')
    mviz.draw_coord_sys(ax, pi, 0.05, 'Oi')
    mviz.draw_coord_sys(ax, pi0, 0.05, 'Oi0')
    mviz.draw_coord_sys(ax, pi1, 0.05, 'Oi1')
    mviz.draw_coord_sys(ax, pi2, 0.05, 'Oi2')
    mviz.draw_coord_sys(ax, pi3, 0.05, 'Oi3')
    mviz.draw_coord_sys(ax, pi4, 0.05, 'Oi4')
    mviz.draw_coord_sys(ax, pi5, 0.05, 'Oi5')
    mviz.draw_coord_sys(ax, p1, 0.05, 'O1')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

