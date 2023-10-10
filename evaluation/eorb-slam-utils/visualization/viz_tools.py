import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def filter_map(mapData, xlim=None, ylim=None, zlim=None):

    nMapData = len(mapData)

    xMask = np.ones(nMapData, dtype=bool)
    if xlim is not None and len(xlim) == 2:
        xMask = np.logical_and(mapData[:, 0] >= xlim[0], mapData[:, 0] < xlim[1])
    yMask = np.ones(nMapData, dtype=bool)
    if ylim is not None and len(ylim) == 2:
        yMask = np.logical_and(mapData[:, 1] >= ylim[0], mapData[:, 1] < ylim[1])
    zMask = np.ones(nMapData, dtype=bool)
    if zlim is not None and len(zlim) == 2:
        zMask = np.logical_and(mapData[:, 2] >= zlim[0], mapData[:, 2] < zlim[1])

    mapMask = np.logical_and(np.logical_and(xMask, yMask), zMask)
    mapData = mapData[mapMask, :]

    return mapData, nMapData - np.sum(mapMask)


def plot_traj_3d(ax, traj_data, label_, style_):
    """
    style_: [marker color, marker sign, line color]
    """
    ax.scatter(traj_data[:, 0], traj_data[:, 1], traj_data[:, 2], c=style_[0], marker=style_[1])
    ax.plot(traj_data[:, 0], traj_data[:, 1], traj_data[:, 2], color=style_[2], label=label_)


def plot_pc_3d(ax, map_pc, style_):
    """
    style_: [marker color, marker sign]
    """
    ax.scatter(map_pc[:, 0], map_pc[:, 1], map_pc[:, 2], c=style_[0], marker=style_[1])


def draw_coord_sys(ax, trans, sc=1.0, label=''):

    coords = np.identity(4, dtype=np.float32)
    coords[3, 0:3] = np.array([1, 1, 1])

    new_coords = np.matmul(trans, coords)

    x = np.repeat(new_coords[0, 3], 3)
    y = np.repeat(new_coords[1, 3], 3)
    z = np.repeat(new_coords[2, 3], 3)

    u = new_coords[0, 0:3]
    v = new_coords[1, 0:3]
    w = new_coords[2, 0:3]

    cols = ['red', 'green', 'blue']

    for i in range(3):
        ax.quiver(x[i], y[i], z[i], u[i], v[i], w[i], length=sc, normalize=True, colors=cols[i])

    dtext = 0.08 * sc
    ax.text(x[0]-dtext, y[0]-dtext, z[0]-dtext, label, color='m')


def draw_pose_gt_est(gtData, estDataAligned):

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    plot_traj_3d(ax, gtData, 'gt traj.', ['y', '*', 'b'])

    plot_traj_3d(ax, estDataAligned, 'est. traj.', ['c', '*', 'r'])

    ax.legend()

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()


def draw_pose_map(poseData, mapData, xlim=None, ylim=None, zlim=None):

    # Filter data (fit to bounds)
    mapData, nOutliers = filter_map(mapData, xlim, ylim, zlim)
    print("Num. map point outliers: %d" % (nOutliers))

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Plot
    plot_pc_3d(ax, mapData, ['g', '.'])

    plot_traj_3d(ax, poseData[:, 1:4], 'pose', ['y', '*', 'b'])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # plt.legend()

    plt.show()


def plot_nn_elements(nn_elems):
    # matplotlib.use('Agg')
    # import matplotlib.pyplot as plt
    # import matplotlib.pylab as pylab

    idxs = np.array([i for i in range(0, len(nn_elems))])
    e1 = np.array([el1 for el1, el2, el3 in nn_elems])
    e2 = np.array([el2 for el1, el2, el3 in nn_elems])
    e3 = np.array([el3 for el1, el2, el3 in nn_elems])

    fig = plt.figure()
    ax = fig.gca()
    ax.plot(idxs, e1, '-', color="blue")
    ax.plot(idxs, e2, '-.', color="red")
    ax.plot(idxs, e3, '--', color="green")
    # ax.plot([t for t,e in err_rot],[e for t,e in err_rot],'-',color="red")
    ax.set_xlabel('time [s]')
    ax.set_ylabel('translational error [m]')
    plt.show()

    return 1


def draw_box_stats(ax, stats, pos_x, box_label, box_color='blue', box_style='-', box_width=0.5):

    box_hw = box_width / 2
    box_qw = box_hw / 2

    first_h, = ax.plot([pos_x - box_qw, pos_x + box_qw], [stats[0], stats[0]], label=box_label, color=box_color, linestyle=box_style)
    ax.plot([pos_x - box_hw, pos_x + box_hw], [stats[1], stats[1]], label=box_label, color=box_color, linestyle=box_style)
    ax.plot([pos_x - box_hw, pos_x + box_hw], [stats[2], stats[2]], label=box_label, color=box_color, linestyle=box_style)
    ax.plot([pos_x - box_hw, pos_x + box_hw], [stats[3], stats[3]], label=box_label, color=box_color, linestyle=box_style)
    ax.plot([pos_x - box_qw, pos_x + box_qw], [stats[-1], stats[-1]], label=box_label, color=box_color, linestyle=box_style)

    ax.plot([pos_x, pos_x], [stats[0], stats[1]], label=box_label, color=box_color, linestyle=box_style)
    ax.plot([pos_x - box_hw, pos_x - box_hw], [stats[1], stats[3]], label=box_label, color=box_color, linestyle=box_style)
    ax.plot([pos_x + box_hw, pos_x + box_hw], [stats[1], stats[3]], label=box_label, color=box_color, linestyle=box_style)
    ax.plot([pos_x, pos_x], [stats[3], stats[-1]], label=box_label, color=box_color, linestyle=box_style)

    return first_h


if __name__ == "__main__":

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # draw_coord_sys(ax, np.identity(4, dtype=np.float32), 0.05, 'O')
    plot_traj_3d(ax, np.array([[0, 0, 0], [1, 2, -1], [2, 1, 0]]), '2d-array', ['c', '*', 'r'])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()
