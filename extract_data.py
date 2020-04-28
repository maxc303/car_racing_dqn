import os,re
from matplotlib import pyplot as plt
import pandas as pd
experiment_dir = os.path.abspath("./experiments/Apr22_afternoon/CarRacing-v0")
import extract_episode_reward as eer
f = open(experiment_dir+"/result.txt", "r")
#print(f.read())
rewards = []
Lines = f.readlines()
count = 0
for line in Lines:

    reward = re.search('Episode Reward: (.*) ,',line)
    if reward:
        #print(reward.group(1))
        rewards.append(float(reward.group(1)))

print(len(rewards))
fig1 = plt.figure(figsize=(12, 9))
plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Episode Reward")
plt.title("Episode Reward")
plt.show(fig1)

# Plot the episode reward over time
smoothing_window =100
fig2 = plt.figure(figsize=(8, 6))
rewards_smoothed = pd.Series(rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
plt.plot(rewards_smoothed)
plt.xlabel("Episode")
plt.ylabel("Episode Reward (Smoothed)")
plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
plt.show(fig2)
fig2.savefig("apr22")

rewards1 = rewards
dir2 = os.path.abspath("./experiments/9696-800avg/CarRacing-v0")

rewards2 = eer.read_result(dir2)

min_length = min(len(rewards1), len(rewards2))
rewards1 = rewards1[1:min_length]
rewards2 = rewards2[1:min_length]
rewards1 = pd.Series(rewards1).rolling(smoothing_window, min_periods=smoothing_window).mean()
rewards2 = pd.Series(rewards2).rolling(smoothing_window, min_periods=smoothing_window).mean()

fig = plt.figure(figsize=(8, 6))
plt.plot(rewards1, label="Cropped to 84x84")
plt.plot(rewards2, label="96x96 ")
plt.legend()
plt.xlabel("Episode")
plt.ylabel("Episode Reward (Smoothed)")
plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
plt.show(fig)
fig.savefig("forreport.png")