import sys
import os
import yaml
import cv2
import numpy as np
import re


def parse_gt_path_cv(settingsPath):
    fsSettings = cv2.FileStorage(settingsPath, cv2.FILE_STORAGE_READ)

    dsPathNode = fsSettings.getNode("Path.DS")
    dsRootNode = dsPathNode.getNode("root")
    dsGtNode = dsPathNode.getNode("gt")
    dsSeqIdx = fsSettings.getNode("DS.Seq.target")
    dsSeqNames = fsSettings.getNode("DS.Seq.names")

    seqName = dsSeqNames.at(int(dsSeqIdx.real())).string()

    seqFullPath = os.path.join(dsRootNode.string(), seqName, dsGtNode.string())
    return seqFullPath


def parse_est_base_path_cv(path_settings, pose_config=1, def_ds_config=""):
    fs_settings = cv2.FileStorage(path_settings, cv2.FILE_STORAGE_READ)

    # can retrieve some DS attributes based on dataset
    ts_scale = 1.0
    gt_qw_dir = 1

    # resolve ds name
    node_ds_name = fs_settings.getNode("DS.name")
    ds_name = node_ds_name.string()

    # resolve ds format
    node_ds_format = fs_settings.getNode("DS.format")
    ds_format = node_ds_format.string()

    # resolve timestamp scale and gt qw direction based on ds format
    if ds_format == "euroc":
        ts_scale = 1e9
        gt_qw_dir = 1   # qw first
    elif ds_format == "ev_ethz":
        ts_scale = 1.0
        gt_qw_dir = 0   # qw last

    # resolve sensor config
    if len(def_ds_config) > 0:
        ds_config = def_ds_config
    else:
        node_ds_config = fs_settings.getNode("DS.config")
        ds_config = node_ds_config.string()

    # resolve name prefix
    if ds_config == "mono_ev":
        fname_prefix = "ekf_"

    elif ds_config == "mono_ev_im":
        if pose_config == 0:
            fname_prefix = "of_"
        elif pose_config == 2:
            fname_prefix = "ekf_"
        elif pose_config == 3:
            fname_prefix = "mkf_"
        else:
            fname_prefix = "okf_"

    elif ds_config == "mono_ev_imu":
        if pose_config == 0:
            fname_prefix = "ef_"
        else:
            fname_prefix = "ekf_"
    else:
        if pose_config == 0:
            fname_prefix = "f_"
        else:
            fname_prefix = "kf_"

    # construct init. path
    base_path = "../../results/" + ds_name + "/" + fname_prefix + ds_format + '_' + ds_config + '_'

    node_seq_idx = fs_settings.getNode("DS.Seq.target")
    seq_idx = int(node_seq_idx.real())
    if seq_idx < 0:
        print("* WARNING: Sequence index set to spin mode (-1) *")
        return base_path, ts_scale, gt_qw_dir, ds_format

    node_seq_names = fs_settings.getNode("DS.Seq.names")
    seq_name = node_seq_names.at(seq_idx).string()

    base_path += seq_name

    node_mixed_fts = fs_settings.getNode("Features.mode")
    mixed_fts = int(node_mixed_fts.real())
    if mixed_fts != 0:
        base_path += "_mfts"

    if "ev" in ds_config:
        node_l2_tracking_mode = fs_settings.getNode("Event.l2TrackMode")
        l2_tracking_mode = int(node_l2_tracking_mode.real())
        base_path += "_etm" + str(l2_tracking_mode)

        node_ev_ftdt = fs_settings.getNode("Event.fts.detMode")
        ev_ftdt_mode = int(node_ev_ftdt.real())
        base_path += "_ftdtm" + str(ev_ftdt_mode)

        node_track_tiny_fr = fs_settings.getNode("Event.trackTinyFrames")
        ev_ttifr = node_track_tiny_fr.string()

        node_fixed_l1win = fs_settings.getNode("Event.data.l1FixedWin")
        fixed_l1win = node_fixed_l1win.string()

        node_cont_evt = fs_settings.getNode("Event.contTracking")
        cont_evt = node_cont_evt.string()

        if ev_ttifr == "true":
            base_path += "_tTiFr"

        if fixed_l1win == 'true':
            base_path += "_fxWin"

        if cont_evt == 'true':
            base_path += "_cont"

    return base_path, ts_scale, gt_qw_dir, ds_config


def parse_gt_path_yaml(settingsPath):
    with open(settingsPath, 'r') as settings:
        try:
            print(yaml.safe_load(settings))
        except yaml.YAMLError as exc:
            print(exc)


def read_file_list(filename):
    """
    Reads a trajectory from a text file.

    File format:
    The file format is "stamp d1 d2 d3 ...", where stamp denotes the time stamp (to be matched)
    and "d1 d2 d3.." is arbitary data (e.g., a 3D position and 3D orientation) associated to this timestamp.

    Input:
    filename -- File name

    Output:
    dict -- dictionary of (stamp,data) tuples
    """
    file = open(filename)
    data = file.read()
    lines = data.replace(",", " ").replace("\t", " ").split("\n")
    mylist = [[v.strip() for v in line.split(" ") if v.strip() != ""] for line in lines if
              len(line) > 0 and line[0] != "#"]
    mylist = [[np.float64(v) for v in l] for l in mylist if len(l) > 1]
    return mylist


def read_timing_hears(filename, times_list):
    file = open(filename)
    data = file.read()
    lines = data.split("\n")

    for i in range(0, len(lines)):
        line = lines[i].strip()
        if len(line) <= 0 or line[0] != '#':
            break

        line_parts = line.split(':')

        key = line_parts[0][1:].strip()

        times_str = re.findall(r'[\d.]+', line_parts[1])
        times = [np.float32(v) for v in times_str]

        if len(times_list.get(key, '')) <= 0 or len(times_list[key]) != len(times):
            times_list['times_cnt'][key] = 1
            times_list[key] = times
        else:
            times_list['times_cnt'][key] += 1
            times_list[key] = [sum(x) for x in zip(times_list[key], times)]

    return times_list


def read_dosconn_graph_list(filename, disconn_expr="# DISCONNECTED POSE GRAPH"):
    """
    Reads a disconnected trajectory from a text file.

    File format:
    The file format is "stamp d1 d2 d3 ...", where stamp denotes the time stamp (to be matched)
    and "d1 d2 d3.." is arbitary data (e.g., a 3D position and 3D orientation) associated to this timestamp.

    Input:
    filename -- File name
    disconn_expr -- Disconnection Expression, commented line that shows the location of discontinuity

    Output:
    dict -- dictionary of (stamp,data) tuples
    """
    file = open(filename)
    data = file.read()
    lines = data.replace(",", " ").replace("\t", " ").split("\n")

    pose_piece = []
    all_poses = []
    n_lines = len(lines)

    for i in range(0, n_lines):
        line = lines[i].strip()
        if "nan" in line:
            continue
        if len(line) <= 0 or (line[0] == "#" and line != disconn_expr):
            if i == n_lines-1 and len(pose_piece) > 0:
                all_poses.append(pose_piece)
            continue
        elif line == disconn_expr:
            all_poses.append(pose_piece)
            pose_piece = []
        else:
            pose_piece.append(np.array([np.float64(v.strip()) for v in line.split(" ") if v.strip() != ""]))

    return all_poses
