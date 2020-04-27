import os,re
from matplotlib import pyplot as plt
import pandas as pd
experiment_dir = os.path.abspath("./experiments/CarRacing-v0")
experiment_dir = os.path.abspath("./experiments/9696-800avg/CarRacing-v0")
experiment1_dir = os.path.abspath("./experiments/9696_fullbrake/CarRacing-v0")
experiment_fix_dir = os.path.abspath("./experiments/96fix_2000epi/CarRacing-v0")

def read_result(experiment_dir):
    f = open(experiment_dir+"/episode_reward.txt", "r")
    rewards = []
    Lines = f.readlines()
    count = 0
    for line in Lines:
        string_list = line.split(' ')

        for str_float in string_list:
            if str_float:
                rewards.append(float(str_float))
    return rewards

# fig1 = plt.figure(figsize=(12, 9))
# plt.plot(rewards)
# plt.xlabel("Episode")
# plt.ylabel("Episode Reward")
# plt.title("Episode Reward")
# plt.show(fig1)
def plot_avg(rewards,smoothing_window =100):
    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(8, 6))
    rewards_smoothed = pd.Series(rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    plt.show(fig2)
    fig2.savefig("forreport.png")

def plot_avg_compare(dir1,dir2,smoothing_window =100):
    rewards1 = read_result(dir1)
    rewards2 = read_result(dir2)
    min_length = min(len(rewards1),len(rewards2))
    rewards1 = rewards1[1:min_length]
    rewards2 = rewards2[1:min_length]
    rewards1 = pd.Series(rewards1).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards2 = pd.Series(rewards2).rolling(smoothing_window, min_periods=smoothing_window).mean()

    fig = plt.figure(figsize=(8, 6))
    plt.plot(rewards1, label="with full brake")
    plt.plot(rewards2, label="without full brake")
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    plt.show(fig)
    fig.savefig("forreport.png")

if __name__=="__main__":
    # rewards = read_result(experiment1_dir)
    # plot_avg(rewards)
    #plot_avg_compare(experiment1_dir,experiment_dir)
    fix_reward = read_result(experiment_fix_dir)
    print(max(fix_reward))
    plot_avg(fix_reward)