'''
Luc Kadletz, 7/14/2019

'''

# Standard Libraries

# Third Party Imports
import tensorflow as tf
import numpy as np
# Local Imports


class Agent:

    def __init__(self, obs_size: [int, int], actions: [[bool]], loss_fn: callable):
        self.obs_size = obs_size
        self.actions = actions
        self.loss_fn = loss_fn

        self.next_action = actions[0]  # Hold right

        self.build()

    _downsample_dimensions = [128, 128]
    _downsample_size = np.prod(_downsample_dimensions)

    def build(self):
        with tf.name_scope("Input Processing"):

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
                screen_greyscale, _downsample_dimensions)
            tf.summary.image('screen_downsampled', screen_downsampled)

            # screen_flattened is a 1D arrangement of the downsample
            screen_flattened = tf.reshape(
                screen_downsampled, [-1, _downsample_size])

        with tf.name_scope("Reinforcement Learning"):
            weights = tf.Variable(tf.random_normal(
                [_downsample_size, 8]), name="weights")
            biases = tf.Variable(tf.random_normal([8, ]), name="biases")

            self.output = tf.matmul(image, weights) + biases
            self.output = tf.nn.l2_normalize(self.output)

        with tf.name_scope("loss"):
            self.loss_in = tf.placeholder(tf.float32, name="loss_actual")

        self.optimizer = tf.train.GradientDescentOptimizer(0.5)

        self.saver = tf.train.Saver()

    def start(self):
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        self.summary = tf.summary.merge_all()

    def tick(self, observation, loss_info):
        feed_dict = {
            self.input: observation,
            self.loss_in: self.loss_fn(loss_info)
        }

        summary_str, output_values = self.session.run(
            [self.summary, self.output], feed_dict)

        self.next_action = self.actions[0]

    def optimize(self):
        self.optimizer.minimize(self.loss)

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

    def _compute_error(self):
        return None

    def __str__(self):
        return str.format("[Agent - Actions x {0}, Next Action: {1}]",
                          len(self.actions), self.next_action)
