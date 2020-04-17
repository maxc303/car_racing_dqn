import gym

if __name__=="__main__":
    render = True
    n_episodes = 1
    env = gym.make('CarRacing-v0')
    env.render()
    rewards = []
    for i_episode in range(n_episodes):
        observation = env.reset()
        sum_reward = 0
        for t in range(1000):
            if render:
                env.render()
            # [steering, gas, brake]
            action = env.action_space.sample()
            # observation is 96x96x3
            observation, reward, done, _ = env.step(action)
            print(len(observation))
            print(len(observation[0]))
            print(len(observation[0][0]))
            print(observation)
            # break
            sum_reward += reward
            if (t % 100 == 0):
                print(t)
            if done or t == 999:
                print("Episode {} finished after {} timesteps".format(i_episode, t + 1))
                print("Reward: {}".format(sum_reward))
                rewards.append(sum_reward)
            if done:
                break