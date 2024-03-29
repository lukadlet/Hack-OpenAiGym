'''
Luc Kadletz, 7/14/2019

'''

# Standard Libraries
import os
import shutil
# Third Party Imports
import tensorflow as tf
import numpy as np
import keras
# Local Imports


class Agent:

    _downsample_dimensions = [128, 128]
    _downsample_size = np.prod(_downsample_dimensions)

    def __init__(self, obs_size: [int, int], actions: [[bool]], loss_fn: callable):
        self.obs_size = obs_size
        self.action_size = len(actions)
        self.actions = actions
        self.loss_fn = loss_fn

        self.log_location = '.logs/dev'
        self.idle_count = 0
        self.step_count = 0
        self.next_action = actions[0]  # Hold right
        self.loss = 0.0
        self.last_observation = None

        self._build()

    def _build(self):
        with tf.name_scope("input_processing"):
            # screen_buffer is a raw vector of bits
            screen_buffer = tf.placeholder(tf.float32, name="screen_buffer")
            self.input = screen_buffer
            # screen_image is the image after it's been reshaped to a pixel tensor
            screen_dimensions = [-1, self.obs_size[0], self.obs_size[1], 3]
            screen_image = tf.reshape(screen_buffer, screen_dimensions)
            tf.summary.image('screen_image', screen_image)
            # screen_greyscale is the image after it's been transformed to greyscale
            screen_greyscale = tf.image.rgb_to_grayscale(screen_image)
            tf.summary.image('screen_greyscale', screen_greyscale)
            # screen_downsampled is a downscale for a standard / reasonable input size for RL
            screen_downsampled = tf.image.resize_bicubic(
                screen_greyscale, self._downsample_dimensions)
            tf.summary.image('screen_downsampled', screen_downsampled)
            # screen_flattened is a 1D arrangement of the downsample
            screen_flattened = tf.reshape(
                screen_downsampled, [-1, self._downsample_size])

        with tf.name_scope("reinforcement_learning"):
            # Just a layer of neurons to predict output
            weights = tf.Variable(tf.random_normal(
                [self._downsample_size, self.action_size]), name="weights")
            biases = tf.Variable(tf.random_normal(
                [self.action_size, ]), name="biases")
            layer = tf.matmul(screen_flattened, weights) + biases

            self.output = tf.nn.l2_normalize(layer)

        with tf.name_scope("loss"):
            # Create an input to get our actual loss at runtime
            self.loss_in = tf.placeholder(tf.float32, name="loss_actual")
            # Just a layer of neurons to predict loss
            weights = tf.Variable(tf.random_normal(
                [self.action_size, 99]), name="weights")
            biases = tf.Variable(tf.random_normal([99]), name="biases")
            self.loss_estimator = tf.matmul(self.output, weights) + biases
            # Register the loss with tf
            tf.summary.scalar('actual_loss', self.loss_in)
            # tf.summary.scalar('estimated_loss', self.loss_estimator)
            tf.losses.add_loss(self.loss_estimator)

        with tf.name_scope("error"):
            # I grabbed this one off a tutorial somewhere
            self.error = tf.nn.softmax_cross_entropy_with_logits(
                labels=self.loss_in, logits=self.loss_estimator)

        # off the shelf gradient descent optimization
        self.optimizer = tf.train.GradientDescentOptimizer(0.5)

        # saver is for saving / loading the model
        self.saver = tf.train.Saver()

    def start(self):
        self._clear_logs()
        self.writer = tf.summary.FileWriter(
            self.log_location, tf.get_default_graph())

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        self.summary = tf.summary.merge_all()

    def _clear_logs(self):
        if(os.path.exists(self.log_location)):
            for filename in os.listdir(self.log_location):
                filepath = os.path.join(self.log_location, filename)
                try:
                    shutil.rmtree(filepath)
                except OSError:
                    os.remove(filepath)

    def tick(self, observation, loss_info):
        self.idle_count = self.idle_count + 1
        self.step_count = self.step_count + 1

        # Add extra stuff to loss info
        loss_info["idle_count"] = self.idle_count
        loss_info["novelty"] = self._get_novelty(observation)
        self.loss = self.loss_fn(loss_info)

        feed_dict = {
            self.input: observation,
            self.loss_in: self.loss
        }

        # Note that we need to support not always getting info when
        #  evaluating, and only the reward instead

        summary_str, output_values = self.session.run(
            [self.summary, self.output], feed_dict)

        self.next_action = self.select_action(output_values)

        # Write to log file for debugging (is this too slow to do every frame?)
        self.writer.add_summary(summary_str, self.step_count)

    def optimize(self):
        self.optimizer.minimize(self.loss_estimator)

    def _get_novelty(self, observation):
        # Note that these numbers only work well for pokemon
        NOVELTY_MAX = 100
        NOVELTY_THRESHOLD = 10
        novelty = 0.0
        if(self.last_observation is not None):
            for i in range(len(observation)):
                delta = np.subtract(observation[i], self.last_observation[i])
                novelty = novelty + np.linalg.norm(delta)
            novelty = novelty / len(observation)
        self.last_observation = observation

        #if novelty is nonzero, reset step count
        if(novelty > NOVELTY_THRESHOLD):
            self.idle_count = 0
        if(novelty > NOVELTY_MAX):
            novelty = NOVELTY_MAX
        
        return novelty

    def select_action(self, action):
        index = np.argmax(action)
        return self.actions[index]

    def load(self, path):
        try:
            self.saver.restore(self.session, path)
            print("Loaded session: ", path)
        except Exception as ex:
            print(ex)
            print("Could not load session at ", path)

    def save(self, path):
        try:
            self.saver.save(self.session, path)
            print("Saved at: ", path)
        except Exception as ex:
            print(ex)
            print("Could not save at ", path)

    def __str__(self):
        return str.format("[Agent - Actions x {0},\n Next Action: {1}\n Loss: {2}]",
                          len(self.actions), self.next_action, self.loss)
