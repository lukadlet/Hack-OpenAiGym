'''
Luc Kadletz, 7/14/2019

'''

# Standard Libraries

# Third Party Imports
import tensorflow as tf
# Local Imports


class Agent:

    def __init__(self, obs_size: [int, int], actions: [[bool]]):
        self.obs_size = obs_size
        self.actions = actions
        self.build()
        self.next_action = actions[0]  # Hold right

    def build(self):
        # Here be magic bullshit
        with tf.name_scope("inputs"):
            screen_buffer = tf.placeholder(tf.float32, name="screen_buffer")
            tf.summary.image('original', screen_buffer)
            self.input = screen_buffer

        with tf.name_scope("screen_preprocessing"):
            tensor_dim = [-1, self.obs_size[0], self.obs_size[1], 3]
            screen_image = tf.reshape(screen_buffer, tensor_dim)
            image = tf.image.rgb_to_grayscale(screen_buffer)
            tf.summary.image('greyscale', image)
            image = tf.image.resize_bicubic(image, [64, 64])
            tf.summary.image('downscaled', image)
            image = tf.reshape(image, [-1, 64 * 64])

        with tf.name_scope("AI"):
            weights = tf.Variable(tf.random_normal(
                [64 * 64, 8]), name="weights")
            biases = tf.Variable(tf.random_normal([8, ]), name="biases")
            output = tf.matmul(image, weights) + biases
            output = tf.nn.l2_normalize(output)

        with tf.name_scope("loss"):
            self.loss = tf.placeholder(tf.float32, name="loss_actual")

        self.saver = tf.train.Saver()
        self.optimizer = tf.train.GradientDescentOptimizer(0.5)

    def start(self):
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def load(self, path):
        self.saver.restore(self.session, path)
        pass

    def save(self, path):
        # If not session error
        self.saver.save(self.session, path)
        print("Saved {0} at {1}")

    def tick(self, screen, loss):
        feed_dict = {
            self.input: screen,
            self.loss: loss
        }
        self.next_action = self.actions[0]

    def optimize(self):
        self.optimizer.minimize(self.loss)

    def _compute_error(self):
        return None

    def __str__(self):
        return "Agent"
