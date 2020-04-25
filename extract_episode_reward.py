import os,re
from matplotlib import pyplot as plt
import pandas as pd
experiment_dir = os.path.abspath("./experiments/CarRacing-v0")

f = open(experiment_dir+"/episode_reward.txt", "r")
#print(f.read())
rewards = []
Lines = f.readlines()
count = 0
for line in Lines:
    #print(line)
    string_list = line.split(' ')

    for str_float in string_list:
        #print("/"+str_float+'/')
        if str_float:
            rewards.append(float(str_float))

    # reward = re.search('Episode Reward: (.*) ,',line)
    # if reward:
    #     #print(reward.group(1))
    #     rewards.append(float(reward.group(1)))

print(len(rewards))
# fig1 = plt.figure(figsize=(12, 9))
# plt.plot(rewards)
# plt.xlabel("Episode")
# plt.ylabel("Episode Reward")
# plt.title("Episode Reward")
# plt.show(fig1)

# Plot the episode reward over time
smoothing_window =50
fig2 = plt.figure(figsize=(12, 9))
rewards_smoothed = pd.Series(rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
plt.plot(rewards_smoothed)
plt.xlabel("Episode")
plt.ylabel("Episode Reward (Smoothed)")
plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
plt.show(fig2)