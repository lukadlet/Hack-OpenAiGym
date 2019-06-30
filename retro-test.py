import tensorflow as tf
import numpy as np
from retro import make

def test_tensorflow():
    hello = tf.constant('Hello Hackathon!')
    sess = tf.Session()
    print(sess.run(hello))

def filter_screen(image, downsample_x, downsample_y):
    with tf.name_scope("filter_screen"):
        tf.summary.image('original', image)
        
        image = tf.image.rgb_to_grayscale(image)
        tf.summary.image('greyscale', image)

        image = tf.image.resize_bicubic(image, [downsample_x, downsample_y]) 
        tf.summary.image('downscaled', image)

        image = tf.reshape(image, [-1, downsample_x * downsample_y])
    return image

def filter_buttons(buttons):
    with tf.name_scope("filter_buttons"):
        buttons = tf.round(buttons)
        buttons = tf.reshape(buttons, shape=[-1,]) # Flatten to 1D array
        tf.summary.tensor_summary("buttons",buttons)
    return buttons

def choose_actions(buttons):
    # All necessary actions
    #"B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"
    actions = [
        [False, False, False, False, False, False, False, True, False, False, False, False],
        [False, False, False, False, False, False, True, False, False, False, False, False],
        [False, False, False, False, False, True, True, False, False, False, False, False],
        [False, False, False, False, False, True, False, True, False, False, False, False],
        [False, False, False, False, False, True, False, False, False, False, False, False],
        [True, False, False, False, False, True, False, False, False, False, False, False],
        [True, False, False, False, False, False, False, False, False, False, False, False],
        [False, False, False,False, False, False, False, False, False, False, False, False]
    ]

    index = np.argmax(buttons)
    return actions[index]


def estimate_loss(output):
    with tf.name_scope("loss_estimator"):
        weights = tf.Variable(tf.random_normal([8, 99]), name="Weights")
        biases = tf.Variable(tf.random_normal([99]),name="Biases" )
        loss_estimator = tf.matmul(output, weights) + biases

    return loss_estimator;

def compute_loss(info):
    return info["screen_x_end"] - info["x"]

def compute_error(estimated_loss, actual_loss):
    with tf.name_scope("error"):
        error = tf.nn.softmax_cross_entropy_with_logits(labels = actual_loss, logits = estimated_loss)
    return error

def setup_tensor_graph():
    # Create a placeholder to put the screen buffer
    with tf.name_scope("inputs"):
        screen_buffer = tf.placeholder(tf.float32, name="ScreenBuffer")
        screen_image = tf.reshape(screen_buffer, [-1, 224, 320, 3])

    # Greyscale and resize the screen so its a computable size
    screen_flattened = filter_screen(screen_image, 64, 64)

    # Actual AI
    with tf.name_scope("AI"):
        weights = tf.Variable(tf.random_normal([64 * 64, 8]), name = "Weights")
        biases = tf.Variable(tf.random_normal([8,]), name = "Biases")
        output = tf.matmul(screen_flattened, weights) + biases
        output = tf.nn.l2_normalize(output)

    # Turn output into button presses
    # noise = tf.random_normal([12,])
    buttons = filter_buttons(output)

    loss = estimate_loss(output)
    tf.losses.add_loss(loss)

    actual_loss = tf.placeholder(tf.float32, name="ActualLoss")
    error = compute_error(loss, actual_loss)

    # Write everything down    
    writer = tf.summary.FileWriter('./logs/dev/', tf.get_default_graph())
    return screen_buffer, output, actual_loss, error, writer
 

def capture_screenbuffer(writer, summary, step, capture_each = 30):
    if(step % capture_each == 0):
        writer.add_summary(summary, step)
    return step + 1

def main():
    screen_buffer, output, actual_loss, error, writer = setup_tensor_graph()
    env = make(game='SonicTheHedgehog-Genesis', state = 'GreenHillZone.Act1', record='.')
    obs = env.reset()

    button_presses = [0,0,0,0,0,0,0,1,0,0,0,0,]
    done = False

    summary = tf.summary.merge_all()
    optimizer = tf.train.GradientDescentOptimizer(0.5)

    saver = tf.train.Saver()

    while True:
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, 'models/model.checkpoint')
            step = 0
            while not done:
                obs, rew, done, info = env.step(button_presses)
                
                feed_dict = {
                    screen_buffer : obs,
                    actual_loss : compute_loss(info)
                }

                summary_str, output_values = sess.run([summary, output], feed_dict)

                button_presses = choose_actions(output_values[0])
                
                step = capture_screenbuffer(writer, summary_str, step)
                if(step % 60 == 0):
                    saver.save(sess, 'models/model.checkpoint');
                    print("checkpoint!")

                # Train network here
                optimizer.minimize(error)

                env.render()
        obs = env.reset()

if __name__ == '__main__':
    main()