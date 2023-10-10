import sys
import argparse
import numpy as np

import viz_tools as mviz


if __name__ == '__main__':
    defPathMap = '../../../data/orb_map.txt'
    defPathPose = '../../results/ev_ethz/okf_ev_ethz_mono_ev_im_slider_depth_etm1_ftdtm2_tTiFr_0.44483.txt'

    # parse command line
    parser = argparse.ArgumentParser(description='''
    This script plots estimated estimated camera pose and 3D world points. 
    ''')
    parser.add_argument('-path_pose', help='estimated trajectory (format: timestamp tx ty tz qx qy qz qw)',
                        default=defPathPose)
    parser.add_argument('-path_map', help='estimated map (format: x y z)', default=defPathMap)
    parser.add_argument('-xlim', help='x limits: [low up]', nargs=2, type=float, default=[])
    parser.add_argument('-ylim', help='y limits: [low up]', nargs=2, type=float, default=[])
    parser.add_argument('-zlim', help='z limits: [low up]', nargs=2, type=float, default=[])

    args = parser.parse_args()

    # Load Pose Data (ts1 x1 y1 z1 i1 j1 k1 w1\n...)
    poseData = np.loadtxt(args.path_pose)

    # Load Map Data (x1 y1 z1\nx2 y2 ...)
    mapData = np.loadtxt(args.path_map)

    mviz.draw_pose_map(poseData, mapData, args.xlim, args.ylim, args.zlim)
