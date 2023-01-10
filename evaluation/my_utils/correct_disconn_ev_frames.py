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

if __name__ == '__main__':

    pathPoseEst = '../../ev_asynch_tracker_pose_chain.txt'

    # parse command line
    parser = argparse.ArgumentParser(description='''
    This script plots piece-wise ground-truth camera pose and aligned estimated pose. 
    Note: ORB-SLAM saves qw last, but in euroc ground-truth it's first.
    ''')
    parser.add_argument('path_settings', help='settings.yaml (containing dataset path info)')
    parser.add_argument('-path_pose_est', help='estimated trajectory (format: timestamp tx ty tz qx qy qz qw)',
                        default=pathPoseEst)

    parser.add_argument('--est_idx', help='index of EST file being procecced: default: -1 (median over all)',
                        default=-1)

    args = parser.parse_args()

    est_idx = int(args.est_idx)

    # Resolve EST path
    estFile, time_scale, qw_dir, ds_config = mfm.parse_est_base_path_cv(args.path_settings)
    defEstFile = args.path_pose_est

    estFilesList = glob.glob(estFile + "*.txt")
    # Filter and refine the list
    estFilesList = [f for f in estFilesList if re.match(estFile + r'_[0-9._]+\.txt', f)]
    defEstFilesList = glob.glob(defEstFile + "*.txt")

    curr_path = os.path.dirname(estFilesList[0])
    print(curr_path)

    for i in range(0, len(estFilesList)):

        curr_file = estFilesList[i]

        curr_kf_name = os.path.basename(curr_file)
        curr_fr_name = curr_kf_name.replace('ekf', 'ef')

        kf_pieces = mfm.read_dosconn_graph_list(curr_file)
        fr_pieces = mfm.read_dosconn_graph_list(os.path.join(curr_path, curr_fr_name))[0]

        print(curr_fr_name)



