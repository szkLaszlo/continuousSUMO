"""
@author "Laszlo Szoke (CC-AD/ENG1-Bp)"
"""
import copy
import glob
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np


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
    global time
    with open(file, "br") as f:
        dict_ = pickle.load(f)
    s = np.asarray(dict_["state"][:-1])
    r = np.asarray(dict_["reward"])
    front_left_distance=s[:,0] * 50
    front_ego_distance=s[:,2] * 50
    front_right_distance = s[:,4] * 50
    front_left_speeds = s[:,1] * 50
    front_ego_speeds = s[:,3] * 50
    front_right_speeds = s[:,5] * 50
    rear_left_distances = s[:,6] * 50
    rear_ego_distances = s[:,8] * 50
    rear_right_distances = s[:,10] * 50
    rear_left_speeds = s[:,7] * 50
    rear_ego_speeds = s[:,9] * 50
    rear_right_speeds = s[:,11] * 50
    lanes = s[:,15] * 2
    speeds = s[:,14] * 50
    desired_speeds = s[:,16] * 50

    front_visible = front_ego_distance<50
    rear_visible = rear_ego_distances>-50
    time = np.asarray(list(range(len(lanes))))
    lanes = np.asarray(lanes)
    # # Plotting front distances relative to the ego based on the different lanes
    # plt.scatter(time, front_right_distance, label="FR")
    # plt.scatter(time, front_left_distance, label="FL")
    # plt.scatter(time, front_ego_distance, label="FE")
    # plt.legend()
    # plt.xlabel("steps [-]")
    # plt.ylabel("Distance [m]")
    # plt.title("Front Distance")
    # plt.show()
    # # Plotting rear distances relative to the ego based on the different lanes
    # plt.scatter(time, rear_right_distances, label="RR")
    # plt.scatter(time, rear_left_distances, label="RL")
    # plt.scatter(time, rear_ego_distances, label="RE")
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
    # plt.title("Rear time till collision")
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
    # # plt.scatter(time[front_visible], front_right_distance[front_visible]/(front_right_speeds[front_visible]+speeds[front_visible]), label="FR")
    # # plt.scatter(time[front_visible], front_left_distance[front_visible]/-1/front_left_speeds[front_visible], label="RL")
    # plt.title("Time in-between front vehicle")
    # plt.xlabel("Distance [m]")
    # plt.ylabel("Time in-between vehicles [s]")
    # plt.show()
    # summing the correct situation when the ego is keeping right as much as it can.
    keep_right = (sum(lanes==0) + sum(np.logical_and(lanes!=0,s[:,13]==1)))/len(lanes)
    lane_changes = lanes[1:]!=lanes[:-1]
    distance_before_lane_change = front_ego_distance[:-1][np.logical_and(lane_changes, front_visible[:-1])]
    distance_after_lane_change = rear_ego_distances[1:][np.logical_and(lane_changes, rear_visible[:-1])]

    speed_diff_before_lane_change = (front_ego_speeds+speeds)[:-1][np.logical_and(lane_changes, front_visible[:-1])]
    speed_diff_after_lane_change = (rear_ego_speeds+speeds)[1:][np.logical_and(lane_changes, rear_visible[:-1])]

    #time of approach
    tiv_before_lane_change = distance_before_lane_change/speed_diff_before_lane_change
    tiv_after_lane_change = distance_after_lane_change/speed_diff_after_lane_change

    return {'ego_speed': speeds,
            "follow_distance": front_ego_distance[front_visible],
            "front_tiv": front_ego_distance[front_visible]/(front_ego_speeds[front_visible]+speeds[front_visible]),
            "rear_tiv": rear_ego_distances[rear_visible]/(rear_ego_speeds[rear_visible]+speeds[rear_visible]),
            "lane_changes": sum(lane_changes),
            "desired_speed_difference": desired_speeds-speeds,
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
    statistics_in_folder = []
    for filename in files:
        return_dict = plot_episode_stat(filename)
        statistics_in_folder.append(return_dict)
    return  statistics_in_folder

def eval_full_statistics(global_statistics):
    eval_values = ["ego_speed", "follow_distance", "front_tiv", "lane_changes", "keeping_right", "desired_speed_difference", "tiv_before_lane_change"]
    for name in eval_values:
        for i, item in enumerate(global_statistics):
            episode_stat = []
            for episode in item:
                episode_stat.append(copy.deepcopy(np.expand_dims(episode[name],-1) if episode[name].ndim == 0 else episode[name]))
            episode_stat = np.concatenate(episode_stat)
            plt.hist(episode_stat, bins=min(episode_stat.size//10, 100), histtype="step", density=True, label=str(i), stacked=True)
        plt.title(name)
        plt.legend()
        plt.show()
        print()

if __name__ == "__main__":
    dir_of_eval = ["/cache/hdd/new_rewards/FastRLv1_SuMoGyM_discrete/20210130_163259/eval_20210201_160724/env/",
                   "/cache/hdd/new_rewards/FastRLv1_SuMoGyM_discrete/20210130_163259/eval_20210201_161219/env/",
                   "/cache/hdd/new_rewards/FastRLv1_SuMoGyM_discrete/20210130_163259/eval_20210201_160542/env/",
                   "/cache/hdd/new_rewards/FastRLv1_SuMoGyM_discrete/20210130_163259/eval_20210201_161403/env/",
                   ]
    global_stat = []
    for dir_ in dir_of_eval:
        single_stat = plot_evaluation_statistics(dir_)
        global_stat.append(single_stat)
    eval_full_statistics(global_stat)

    print()
