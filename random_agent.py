import gym
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
import matplotlib.pyplot as plt

class CR_StateProcessor():
    def __init__(self):
    # Build the Tensorflow graph
        with tf.variable_scope("state_processor"):
            self.input_state = tf.placeholder(shape=[96, 96, 3], dtype=tf.uint8)
            self.output = tf.image.rgb_to_grayscale(self.input_state)
            #self.output = tf.image.crop_to_bounding_box(self.output, 0, 6, 84, 84)
#             self.output = tf.image.resize_images(
#                 self.output, [84, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            self.output = tf.squeeze(self.output)

    def process(self, sess, state):
        """
        Args:
            sess: A Tensorflow session object
            state: A [210, 160, 3] Atari RGB State

        Returns:
            A processed [84, 84] state representing grayscale values.
        """
        return sess.run(self.output, { self.input_state: state })

env = gym.make("CarRacing-v0")
sp = CR_StateProcessor()
observation = env.reset()

for t in range(100):
    action = env.action_space.sample()
    observation, reward, done, _ = env.step(action)
    env.render()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Example observation batch
    #observation = env.reset()

    observation_p = sp.process(sess, observation)
    print(observation_p)
    print(observation_p.shape)

    plt.imshow(observation_p)
    plt.savefig("test.jpeg")
env.close()
# if __name__=="__main__":
#     render = True
#     n_episodes = 1
#     env = gym.make('CarRacing-v0')
#     #env.render()
#     rewards = []
#     for i_episode in range(n_episodes):
#         observation = env.reset()
#         sum_reward = 0
#         for t in range(10000):
#             if render:
#                 env.render()
#             # [steering, gas, brake]
#             #action = env.action_space.sample()
#             action = [1,0.2,0.1]
#             # observation is 96x96x3
#             observation, reward, done, _ = env.step(action)
#
#             sum_reward += reward
#             if (t % 100 == 0):
#                 print(t)
#                 print(action)
#             if done or t == 10000:
#                 print("Episode {} finished after {} timesteps".format(i_episode, t + 1))
#                 print("Reward: {}".format(sum_reward))
#                 rewards.append(sum_reward)
#             if done:
#                 break