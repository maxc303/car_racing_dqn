import os
import numpy as np
import tensorflow as tf
import itertools
import gym
from matplotlib import pyplot as plt
from car_racing_dqn_train import Estimator, CarStateProcessor
from action_config import VALID_ACTIONS, idx2act

RENDER = True

def make_policy(estimator, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.

    Args:
        estimator: An estimator that returns q values for a given state
        nA: Number of actions in the environment.

    Returns:
        A function that takes the (sess, observation, epsilon) as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """
    def policy_fn(sess, observation):
        A = np.zeros(nA, dtype=float)
        q_values = estimator.predict(sess, np.expand_dims(observation, 0))[0]
        best_action = np.argmax(q_values)
        A[best_action] =1
        # From the course slide
        return best_action
    return policy_fn

def run_model(sess,state_processor,q_estimator):

    # Make Policy
    policy = make_policy(
        q_estimator,
        len(VALID_ACTIONS))

    state = env.reset()

    state = state_processor.process(sess, state)
    state = np.stack([state] * 4, axis=2)
    total_reward = 0
    for t in itertools.count():
        #print(t)
        # Take a step
        best_action = policy(sess, state)
        next_state, reward, done, _ = env.step(idx2act(VALID_ACTIONS[best_action]))
        total_reward += reward
        next_state = state_processor.process(sess, next_state)
        next_state = np.append(state[:, :, 1:], np.expand_dims(next_state, 2), axis=2)
        if RENDER:
            env.render()
        if done:
            break

        state = next_state
    print("Total Reward of this trial: ", total_reward)
    return total_reward


if __name__=="__main__":


    # Preprocess the Car_racing v0 state
    state_processor = CarStateProcessor()
    # Set Number of trials
    num_trials = 10

    exp_rewards = 0

    with tf.Session() as sess:
        env = gym.make('CarRacing-v0')

        # The time step of the environment can be overwritten here
        env._max_episode_steps = 1000
        experiment_dir = os.path.abspath("./experiments/{}".format(env.spec.id))

        # Init q_estimator
        q_estimator = Estimator(scope="q_estimator", summaries_dir=experiment_dir)
        print(experiment_dir)
        checkpoint_dir = os.path.join(experiment_dir, "checkpoints")

        saver = tf.train.Saver()
        # Load a previous checkpoint if we find one
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)


        if latest_checkpoint:
            print("Loading model checkpoint {}...\n".format(latest_checkpoint))
            saver.restore(sess, latest_checkpoint)

        for i in range(num_trials):
            print("Start the test #",i)
            reward = run_model(sess,state_processor,q_estimator)
            exp_rewards += reward
    env.close()


    print("Average reward in 100 runs: ", exp_rewards/num_trials)
