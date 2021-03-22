"""
@author "Laszlo Szoke (CC-AD/ENG1-Bp)"
"""
import copy
import glob
import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

"""
FL
0 - dx
1 - dv
FE
2 - dx
3 - dv
FR
4 - dx
5 - dv
RL
6 - dx
7 - dv
RE
8 - dx
9 - dv
RR
10 - dx
11 - dv
12 - EL
13 - ER
14 - speed
15 - lane_id
16 - des_speed
17 - heading
"""


def plot_episode_stat(file):
    with open(file, "br") as f:
        dict_ = pickle.load(f)
    s = np.asarray(dict_["state"][:-1])
    r = np.asarray(dict_["reward"])
    front_left_distance = s[:, 0] * 50
    front_ego_distance = s[:, 2] * 50
    front_right_distance = s[:, 4] * 50
    front_left_speeds = s[:, 1] * 50
    front_ego_speeds = s[:, 3] * 50
    front_right_speeds = s[:, 5] * 50
    rear_left_distances = s[:, 6] * 50
    rear_ego_distances = s[:, 8] * 50
    rear_right_distances = s[:, 10] * 50
    rear_left_speeds = s[:, 7] * 50
    rear_ego_speeds = s[:, 9] * 50
    rear_right_speeds = s[:, 11] * 50
    lanes = s[:, 15] * 2
    speeds = s[:, 14] * 50
    desired_speeds = s[:, 16] * 50

    front_visible = front_ego_distance < 48
    rear_visible = rear_ego_distances > -48
    time_ = np.asarray(list(range(len(lanes))))
    lanes = np.asarray(lanes)
    # # Plotting front distances relative to the ego based on the different lanes
    # plt.scatter(time_, front_right_distance, label="FR")
    # plt.scatter(time_, front_left_distance, label="FL")
    # plt.scatter(time_, front_ego_distance, label="FE")
    # plt.legend()
    # plt.xlabel("steps [-]")
    # plt.ylabel("Distance [m]")
    # plt.title("Front Distance")
    # plt.show()
    # # Plotting rear distances relative to the ego based on the different lanes
    # plt.scatter(time_, rear_right_distances, label="RR")
    # plt.scatter(time_, rear_left_distances, label="RL")
    # plt.scatter(time_, rear_ego_distances, label="RE")
    # plt.xlabel("steps [-]")
    # plt.ylabel("Distance [m]")
    # plt.legend()
    # plt.title("Rear Distance")
    # plt.show()
    # # Plotting speed of the ego, desired speed and front vehicle speed
    # plt.plot(speeds, label="ego")
    # plt.plot(desired_speeds, label="desired")
    # plt.plot(front_ego_speeds + speeds, label="front")
    # plt.legend()
    # plt.xlabel("steps [-]")
    # plt.ylabel("Speed [m/s2]")
    # plt.title("Speed values")
    # plt.show()
    # # Plotting the speed differences
    # plt.plot(desired_speeds - speeds, label="desired")
    # plt.plot(front_ego_speeds, label="front")
    # plt.xlabel("steps [-]")
    # plt.ylabel("Speed [m/s2]")
    # plt.legend()
    # plt.title("Speed Differences")
    # plt.show()
    # # Plotting speed - distance of rear vehicles
    # plt.scatter((rear_ego_speeds + speeds)[rear_visible], -1*rear_ego_distances[rear_visible])
    # plt.scatter((rear_right_speeds + speeds)[rear_visible], -1*rear_right_distances[rear_visible])
    # plt.scatter((rear_left_speeds + speeds)[rear_visible], -1*rear_left_distances[rear_visible])
    # plt.title("Rear time_ till collision")
    # plt.xlabel("Speed [m/s2]")
    # plt.ylabel("Distance [m]")
    # plt.show()
    # Plotting speed - distance of front vehicles
    # plt.scatter(speeds[front_visible],
    #             front_ego_distance[front_visible] / (front_ego_speeds[front_visible] + speeds[front_visible]),
    #             label="FE")
    # plt.scatter(speeds[front_visible],
    #              front_left_distance[front_visible] / (front_left_speeds[front_visible] + speeds[front_visible]),
    #              label="FL")
    # # plt.scatter(time_[front_visible], front_right_distance[front_visible]/(front_right_speeds[front_visible]+speeds[front_visible]), label="FR")
    # # plt.scatter(time_[front_visible], front_left_distance[front_visible]/-1/front_left_speeds[front_visible], label="RL")
    # plt.title("Time in-between front vehicle")
    # plt.xlabel("Distance [m]")
    # plt.ylabel("Time in-between vehicles [s]")
    # plt.show()
    # summing the correct situation when the ego is keeping right as much as it can.
    keep_right = (sum(lanes == 0) + sum(np.logical_and(lanes != 0, s[:, 13] == 1))) / len(lanes)
    lane_changes = lanes[1:] != lanes[:-1]
    distance_before_lane_change = front_ego_distance[:-1][np.logical_and(lane_changes, front_visible[:-1])]
    distance_after_lane_change = rear_ego_distances[1:][np.logical_and(lane_changes, rear_visible[:-1])]

    speed_diff_before_lane_change = (front_ego_speeds + speeds)[:-1][np.logical_and(lane_changes, front_visible[:-1])]
    speed_diff_after_lane_change = (rear_ego_speeds + speeds)[1:][np.logical_and(lane_changes, rear_visible[:-1])]

    # time_ of approach
    tiv_before_lane_change = distance_before_lane_change / speed_diff_before_lane_change
    tiv_after_lane_change = distance_after_lane_change / speed_diff_after_lane_change

    return {'ego_speed': speeds,
            "follow_distance": front_ego_distance[front_visible],
            "front_tiv": front_ego_distance[front_visible] / (front_ego_speeds[front_visible] + speeds[front_visible]),
            "rear_tiv": rear_ego_distances[rear_visible] / (rear_ego_speeds[rear_visible] + speeds[rear_visible]),
            "lane_changes": sum(lane_changes) / len(lane_changes),
            "desired_speed_difference": speeds - desired_speeds,
            "keeping_right": keep_right,
            "distance_before_lane_change": distance_before_lane_change,
            "distance_after_lane_change": distance_after_lane_change,
            "speed_before_lane_change": speed_diff_before_lane_change,
            "speed_after_lane_change": speed_diff_after_lane_change,
            "tiv_before_lane_change": tiv_before_lane_change,
            "tiv_after_lane_change": tiv_after_lane_change,
            }


def plot_evaluation_statistics(path_to_env_log, extention="*.pkl"):
    files = glob.glob(f'{os.path.join(path_to_env_log, extention)}')
    files.sort(key=os.path.getmtime)
    with open(f'{os.path.split(path_to_env_log)[0]}/args_eval.txt', 'r') as f:
        params = json.load(f)
    statistics_in_folder = []
    for filename in files:
        return_dict = plot_episode_stat(filename)
        return_dict["weights"] =  params.get("model", "") + " "+ decode_w_for_readable_names(params["w"])
        statistics_in_folder.append(return_dict)
    return statistics_in_folder

def decode_w_for_readable_names(w):

    # if w == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]:
    #     w_string = "safe"
    # elif w == [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]:
    #     w_string = "safe and speedy"
    # elif w == [1.0, 0.0, 1.0, 0.0, 0.0, 0.0]:
    #     w_string = "safe LC"
    # elif w == [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]:
    #     w_string = "safe right"
    # elif w == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]:
    #     w_string = "safe follow"
    # elif w == [1.0, 0.0, 0.0, 0.0, 0.0, 1.0]:
    #     w_string = "safe cut-in"
    # elif w == [1.0, 0.0, 0.0, 0.0, 1.0, 1.0]:
    #     w_string = "safe follow cut-in"
    # elif w == [1.0, 1.0, 0.0, 1.0, 1.0, 1.0]:
    #     w_string = "all but lc"
    # else:
    w_string = str(w)

    return  w_string


def eval_full_statistics(global_statistics, save_figures_path=None):
    eval_values = ["ego_speed", "desired_speed_difference", "follow_distance", "front_tiv",
                   "lane_changes", "keeping_right", "distance_before_lane_change", "distance_after_lane_change",
                   "tiv_before_lane_change", "tiv_after_lane_change", ]
    if save_figures_path is not None and not os.path.exists(save_figures_path):
        os.makedirs(save_figures_path)
    global_statsss = []
    global_names = []
    global_labels = []
    for name in eval_values:
        name_list = []
        name_stat = []
        for i, item in enumerate(global_statistics):
            episode_stat = []
            for episode in item:
                episode_stat.append(
                    copy.deepcopy(np.expand_dims(episode[name], -1) if episode[name].ndim == 0 else episode[name]))
            episode_stat = np.concatenate(episode_stat)
            name_list.append(episode["weights"])
            # plt.hist(episode_stat, bins=min(episode_stat.size//10, 50), histtype="barstacked", density=True, label=name_list[-1], stacked=True)
            name_stat.append(episode_stat)
        global_statsss.append(name_stat)
        global_names.append(name)
        global_labels.append(name_list)
        # fig_plot(data=name_stat, title=name, names=name_list)

        # fig, ax = plt.subplots(len(name_stat) // 2, 2, sharex=True, sharey=True)
        # plt.title(name)
        # for i, ax_i in enumerate(ax.flatten()):
        #     ax_i.hist(name_stat[i], bins=min(episode_stat.size // 10, 50), histtype="bar", density=True,
        #               label=i, stacked=False)
        #     ax_i.text(0.5, 0.5, str((2, len(name_stat) // 2, i)),
        #               fontsize=18, ha='center')
        # plt.legend(name_list)
        # plt.tight_layout()
        # plt.show()

        # if save_figures_path is not None:
        #     plt.savefig(f'{save_figures_path}/{name}_hist.jpg')
        #     plt.cla()
        #     plt.clf()
        # else:
        #     plt.show()

        # sns.boxplot(data=name_stat, fliersize=0)

    draw_boxplot(global_statsss, global_labels, global_names)
    if save_figures_path is not None:
        plt.savefig(f'{save_figures_path}/all_boxplot.jpg')
        plt.cla()
        plt.clf()
    else:
        plt.show()


def draw_boxplot(data, labels, names):
    fig, axes = plt.subplots(data.__len__() // 2, 2, sharex=False, sharey=True, figsize=(8, 12))
    # fig.suptitle("title")
    plt.autoscale()
    for i, ax in enumerate(axes.flatten()):
        ax.boxplot(data[i], labels=labels[i], autorange=True, showfliers=True,
                   notch=False, meanline=True, whis=[5, 95], sym="", vert=False)
        ax.set_title(names[i])
        # ax.annotate(names[i], (0.5, 0.9), xycoords='axes fraction', va='center', ha='center')

def fig_plot(data, title, names):
    fig, axes = plt.subplots(data.__len__() // 2, 2, sharex=True, sharey=True, figsize=(8, 12))
    fig.suptitle(title)
    plt.autoscale()
    for i, ax in enumerate(axes.flatten()):
        # Bulbasaur
        sns.histplot(data=data[i],
                     bins='auto',
                     kde=True,
                     ax=ax,
                     stat="probability",
                     common_norm=True,
                     common_bins=True,
                     multiple="layer",
                     label=names[i])
        ax.annotate(names[i], (0.5, 0.9), xycoords='axes fraction', va='center', ha='center')


if __name__ == "__main__":
    dir_of_eval = [
        "/cache/hdd/new_rewards/FastRLv1_SuMoGyM_discrete/20210130_163259",
        "/cache/hdd/new_rewards/Qnetwork_SimpleMLP_SuMoGyM_discrete/20210209_101340"]
    for run in dir_of_eval:
        global_stat = []
        eval_dirs = os.listdir(run)
        for dir_ in eval_dirs:
            if "eval" not in dir_:
                continue
            single_stat = plot_evaluation_statistics(os.path.join(run, dir_, "env"))
            global_stat.append(single_stat)
        eval_full_statistics(global_stat,
                             )  # save_figures_path=os.path.join(dir_of_eval, f"plots_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}"))

    print()
