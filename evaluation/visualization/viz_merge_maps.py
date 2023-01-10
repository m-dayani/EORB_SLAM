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

import viz_tools as mviz

sys.path.append('../tum_tools')
import evaluate_rpe as eval_rpe

sys.path.append('../my_utils')
import mf_manip as mfm
import mmisc
import my_eval_rpe as meval_rpe


def get_all_files_dict(str_pattern, fileList):

    filteredFiles = [f for f in fileList if re.match(str_pattern, f)]
    refTsList = []

    for currFile in filteredFiles:
        refTs = re.findall(r'\d+\.\d+', currFile)
        refTsList.append([float(refTs[0]), currFile])

    return dict(refTsList)


def find_closest_ts(ts_list, ts_search, th_ts=0.01):

    n_list = len(ts_list)
    closest_idx = (np.abs(ts_list - ts_search)).argmin()
    closest_ts = ts_list[closest_idx]

    if np.abs(closest_ts - ts_search) < th_ts:
        return 1, closest_idx, -1

    next_idx = closest_idx+1
    if ts_search < closest_ts:
        next_idx = closest_idx
        closest_idx = next_idx-1

    if n_list > closest_idx >= 0 and n_list > next_idx >= 0:
        first_ts = ts_list[closest_idx]
        second_ts = ts_list[next_idx]

        if np.abs(first_ts-ts_search) < th_ts:
            return 1, closest_idx, next_idx
        if np.abs(second_ts-ts_search) < th_ts:
            return 1, next_idx, -1
        if first_ts <= ts_search < second_ts:
            return 2, closest_idx, next_idx

    return 0, -1, -1


def compute_scene_med_depth(map_data, pose_cw):
    depth_list = []

    for p3d in map_data:
        pw = np.array([p3d[0], p3d[1], p3d[2], 1])
        pc = np.matmul(pose_cw, pw.transpose())
        depth_list.append(pc[2])

    depth_list.sort()
    return depth_list[int(len(depth_list) / 2)]


def merge_worlds(ref_pose_list, ref_map, srch_pose_list, srch_map):

    ref_srch_ts = srch_pose_list[0, 0]
    ref_pose_ts = ref_pose_list[:, 0]
    stat, idx1, idx2 = find_closest_ts(ref_pose_ts, ref_srch_ts, 0.01)

    if stat == 1:
        Tcwr = eval_rpe.transform44(ref_pose_list[idx1])
    elif stat == 2:
        Tcwr = mmisc.interp_pose(eval_rpe.transform44(ref_pose_list[idx1]),
                                 eval_rpe.transform44(ref_pose_list[idx2]),
                                 ref_srch_ts, ref_pose_ts[idx1:idx2 + 1])
    else:
        return [], []

    Tcws = eval_rpe.transform44(srch_pose_list[0])
    refSc = compute_scene_med_depth(ref_map, Tcwr)
    srchSc = compute_scene_med_depth(srch_map, Tcws)
    relSc = refSc / srchSc
    Tcws_scaled = eval_rpe.scale(Tcws, relSc)

    Twrws = np.matmul(np.linalg.inv(Tcwr), Tcws_scaled)
    Twswr = np.linalg.inv(Twrws)

    # Transform all event poses to new ORB world
    srchPosInWr = []
    for pose in srch_pose_list:
        Tciws = eval_rpe.transform44(pose)
        Tciwr = np.matmul(eval_rpe.scale(Tciws, relSc), Twswr)
        srchPosInWr.append((Tciwr[0:3, 3]))

    srchMapInWr = []
    for p3d in srch_map:
        p3d = relSc * p3d
        piws = np.array([p3d[0], p3d[1], p3d[2], 1])
        piwr = np.matmul(Twrws, piws.transpose())
        srchMapInWr.append(piwr[0:3])

    return srchPosInWr, srchMapInWr


if __name__ == '__main__':

    pathRootData = '../../../data'

    # parse command line
    parser = argparse.ArgumentParser(description='''
    This script loads Pose-Map info for each ORB/Event map
    and then experience map transformations. 
    ''')
    parser.add_argument('--path_maps', help='root path where all pose-map info is stored, default: ../../../data',
                        default=pathRootData)

    args = parser.parse_args()

    data_root = args.path_maps

    # Load all maps
    # Retrieve all .txt files
    allTxtFiles = glob.glob(data_root + "/*.txt")

    # Retrieve all ORB poses
    orbPoseDict = get_all_files_dict(data_root + r'/orb_pose_[0-9.]+\.txt', allTxtFiles)
    # Retrieve all ORB maps
    orbMapDict = get_all_files_dict(data_root + r'/orb_map_[0-9.]+\.txt', allTxtFiles)
    orbTs = list(orbPoseDict.keys())
    orbTs.sort()
    if len(orbPoseDict) != len(orbMapDict):
        print("** ERROR: ORB Pose-Map mismatch!")
        exit(1)

    # Retrieve all ORB poses
    evPoseDict = get_all_files_dict(data_root + r'/ev_pose_[0-9.]+\.txt', allTxtFiles)
    # Retrieve all ORB maps
    evMapDict = get_all_files_dict(data_root + r'/ev_map_[0-9.]+\.txt', allTxtFiles)
    evTs = list(evPoseDict.keys())
    evTs.sort()
    if len(evPoseDict) != len(evMapDict):
        print("** ERROR: ORB Pose-Map mismatch!")
        exit(1)

    figDiscMaps = plt.figure()
    figMergedMaps = plt.figure()
    axDiscMaps = figDiscMaps.gca(projection='3d')
    axDiscMaps.set_title("Disconnected Graphs")
    axMergedMaps = figMergedMaps.gca(projection='3d')
    axMergedMaps.set_title("Merged Graphs")

    orbIdx = 0
    evIdx = 0
    while orbIdx < len(orbTs) and evIdx < len(evTs):

        refTsOrb = orbTs[orbIdx]
        refTsEv = evTs[evIdx]

        orbPoseFile = orbPoseDict[refTsOrb]
        orbPoseListQ = np.array(mfm.read_file_list(orbPoseFile))

        orbMapFile = orbMapDict[refTsOrb]
        orbMap = np.loadtxt(orbMapFile)

        evPoseFile = evPoseDict[refTsEv]
        evPoseListQ = np.array(mfm.read_file_list(evPoseFile))

        evMapFile = evMapDict[refTsEv]
        evMap = np.loadtxt(evMapFile)

        mviz.plot_pc_3d(axDiscMaps, orbMap, ['g', '.'])
        mviz.plot_traj_3d(axDiscMaps, orbPoseListQ[:, 1:4], 'ORB Disc.', ['y', '*', 'b'])

        mviz.plot_pc_3d(axDiscMaps, evMap, ['m', '.'])
        mviz.plot_traj_3d(axDiscMaps, evPoseListQ[:, 1:4], 'Ev Disc.', ['c', '*', 'r'])

        if refTsOrb <= refTsEv:
            refPoseList = orbPoseListQ[:, 1:4]
            refMap = orbMap
            # try to find evRef in orbGraph
            mergedPoseList, mergedMap = merge_worlds(orbPoseListQ, orbMap, evPoseListQ, evMap)
            orbIdx += 1

            if len(mergedPoseList) == 0:
                continue

        else:
            refPoseList = evPoseListQ[:, 1:4]
            refMap = evMap
            # try to find orbRef in evGraph
            mergedPoseList, mergedMap = merge_worlds(evPoseListQ, evMap, orbPoseListQ, orbMap)
            evIdx += 1

            if len(mergedPoseList) == 0:
                continue

        mviz.plot_pc_3d(axMergedMaps, refMap, ['g', '.'])
        mviz.plot_traj_3d(axMergedMaps, refPoseList, 'Ref.', ['y', '*', 'b'])

        mviz.plot_pc_3d(axMergedMaps, np.array(mergedMap), ['m', '.'])
        mviz.plot_traj_3d(axMergedMaps, np.array(mergedPoseList), 'Merged', ['c', '*', 'r'])

    axDiscMaps.set_xlabel('X')
    axDiscMaps.set_ylabel('Y')
    axDiscMaps.set_zlabel('Z')

    axMergedMaps.set_xlabel('X')
    axMergedMaps.set_ylabel('Y')
    axMergedMaps.set_zlabel('Z')

    plt.show()

