import os
import numpy as np
import tensorflow as tf
import itertools
import gym
from action_config import VALID_ACTIONS, idx2act
from matplotlib import pyplot as plt

class Estimator():
    """Q-Value Estimator neural network.

    This network is used for both the Q-Network and the Target Network.
    """

    def __init__(self, scope="estimator", summaries_dir=None):
        self.scope = scope
        # Writes Tensorboard summaries to disk
        self.summary_writer = None
        with tf.variable_scope(scope):
            # Build the graph
            self._build_model()
            if summaries_dir:
                summary_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))
                if not os.path.exists(summary_dir):
                    os.makedirs(summary_dir)
                self.summary_writer = tf.summary.FileWriter(summary_dir)

    def _build_model(self):
        """
        Builds the Tensorflow graph.
        """

        # Placeholders for our input
        # Our input are 4 grayscale frames of shape 84, 84 each
        self.X_pl = tf.placeholder(shape=[None, 96, 96, 4], dtype=tf.uint8, name="X")
        # The TD target value
        self.y_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        # Integer id of which action was selected
        self.actions_pl = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

        X = tf.to_float(self.X_pl) / 255.0
        batch_size = tf.shape(self.X_pl)[0]

        # Three convolutional layers
        conv1 = tf.contrib.layers.conv2d(
            X, 32, 8, 4, activation_fn=tf.nn.relu)
        conv2 = tf.contrib.layers.conv2d(
            conv1, 64, 4, 2, activation_fn=tf.nn.relu)
        conv3 = tf.contrib.layers.conv2d(
            conv2, 64, 3, 1, activation_fn=tf.nn.relu)

        # Fully connected layers
        flattened = tf.contrib.layers.flatten(conv3)
        fc1 = tf.contrib.layers.fully_connected(flattened, 512)
        self.predictions = tf.contrib.layers.fully_connected(fc1, len(VALID_ACTIONS))

        # Get the predictions for the chosen actions only
        gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_pl
        self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

        # Calculate the loss
        self.losses = tf.squared_difference(self.y_pl, self.action_predictions)
        self.loss = tf.reduce_mean(self.losses)

        # Optimizer Parameters from original paper
        self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
        self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

        # Summaries for Tensorboard
        self.summaries = tf.summary.merge([
            tf.summary.scalar("loss", self.loss),
            tf.summary.histogram("loss_hist", self.losses),
            tf.summary.histogram("q_values_hist", self.predictions),
            tf.summary.scalar("max_q_value", tf.reduce_max(self.predictions))
        ])

    def predict(self, sess, s):
        """
        Predicts action values.

        Args:
          sess: Tensorflow session
          s: State input of shape [batch_size, 4, 84, 84, 1]

        Returns:
          Tensor of shape [batch_size, NUM_VALID_ACTIONS] containing the estimated
          action values.
        """
        return sess.run(self.predictions, { self.X_pl: s })

    def update(self, sess, s, a, y):
        """
        Updates the estimator towards the given targets.

        Args:
          sess: Tensorflow session object
          s: State input of shape [batch_size, 4, 84, 84, 1]
          a: Chosen actions of shape [batch_size]
          y: Targets of shape [batch_size]

        Returns:
          The calculated loss on the batch.
        """
        feed_dict = { self.X_pl: s, self.y_pl: y, self.actions_pl: a }
        summaries, global_step, _, loss = sess.run(
            [self.summaries, tf.contrib.framework.get_global_step(), self.train_op, self.loss],
            feed_dict)
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)
        return loss

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

class CR_StateProcessor():
    def __init__(self):
    # Build the Tensorflow graph
        with tf.variable_scope("state_processor"):
            self.input_state = tf.placeholder(shape=[96, 96, 3], dtype=tf.uint8)
            self.output = tf.image.rgb_to_grayscale(self.input_state)
         #   self.output = tf.image.crop_to_bounding_box(self.output, 0, 6, 84, 84)
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
def run_model(sess,state_processor,q_estimator):


    experiment_dir = os.path.abspath("./experiments/{}".format(env.spec.id))

    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "model")
   ## bestcheckpoint_path =  os.path.join(checkpoint_dir, "best_model")


    saver = tf.train.Saver()
    # Load a previous checkpoint if we find one
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)

    if latest_checkpoint:
        print("Loading model checkpoint {}...\n".format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)

    policy = make_policy(
        q_estimator,
        len(VALID_ACTIONS))
    env.seed(0)

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
        env.render()
        if done:
            break

        state = next_state
    env.close()
    return total_reward
CUDA_VISIBLE_DEVICES=""
if __name__=="__main__":

    env = gym.make("CarRacing-v0")
    observation = env.reset()
    #sp = CR_StateProcessor()
    # for t in range(100):
    #     action = env.action_space.sample()
    #     observation, reward, done, _ = env.step(action)
    #     env.render()
    #
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #
    #     # Example observation batch
    #     #observation = env.reset()
    #
    #     observation_p = sp.process(sess, observation)
    #     print(observation_p)
    #     print(observation_p.shape)
    #
    #     plt.imshow(observation_p)
    #     plt.savefig("test.png")
    # env.close()


    env = gym.make('CarRacing-v0')
    env._max_episode_steps = 1000
    state_processor = CR_StateProcessor()
    exp_rewards = 0
    experiment_dir = os.path.abspath("./experiments/{}".format(env.spec.id))
    q_estimator = Estimator(scope="q_estimator", summaries_dir=experiment_dir)
    with tf.Session() as sess:
        for i in range(10):
            reward = run_model(sess,state_processor,q_estimator)
            exp_rewards += reward



    print("Average reward in 10 runs: ", exp_rewards/10)