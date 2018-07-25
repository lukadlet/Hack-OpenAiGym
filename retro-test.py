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

        tf.summary.tensor_summary("buttons", buttons)
    return buttons

def find_closest(buttons):
    # All necessary actions
    #"B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"
    actions = [
        [False, False, False, False, False, False, False, True, False, False, False, False],
        [False, False, False, False, False, False, True, False, False, False, False, False],
        [False, False, False, False, False, True, True, False, False, False, False, False],
        [False, False, False, False, False, True, False, True, False, False, False, False],
        [False, False, False, False, False, True, False, False, False, False, False, False],
        [True, False, False, False, False, True, False, False, False, False, False, False],
        [True, False, False, False, False, False, False, False, False, False, False, False]
    ]

    # Find closest action
    closestIndex = 0
    minDiff = 100000000000
    for actionIndex in range(6):
        difference = 0.0
        actionFloat = np.array(actions[actionIndex]).astype(float)

        for buttonIndex in range(11):
            difference = difference + abs(actionFloat[buttonIndex] - buttons[buttonIndex])

        if (difference < minDiff):
            minDiff = difference
            closestIndex = actionIndex
            print(minDiff)
            print(closestIndex)

    return actions[closestIndex]

def loss(info):
    return info.screen_max_x - info.screen_x

def setup_tensor_graph():
    # Create a placeholder to put the screen buffer
    with tf.name_scope("inputs"):
        screen_buffer = tf.placeholder(tf.float32, name="ScreenBuffer")
        screen_image = tf.reshape(screen_buffer, [-1, 224, 320, 3])

    # Greyscale and resize the screen so its a computable size
    screen_flattened = filter_screen(screen_image, 64, 64)

    # Actual AI
    with tf.name_scope("AI"):
        weights = tf.Variable(tf.random_normal([64 * 64, 12]), name = "Weights")
        biases = tf.Variable(tf.random_normal([12,]), name = "Biases")
        output = tf.matmul(screen_flattened, weights) + biases

    # Turn output into button presses
    buttons = filter_buttons(output)

    # Write everything down    
    writer = tf.summary.FileWriter('./logs/dev/', tf.get_default_graph())
    return screen_buffer, buttons, writer

def capture_screenbuffer(writer, summary, step, capture_each = 30):
    if(step % capture_each == 0):
        writer.add_summary(summary, step)
    return step + 1

def main():
    screen_buffer, buttons, writer = setup_tensor_graph()
    env = make(game='SonicTheHedgehog-Genesis', state = 'LabyrinthZone.Act1')
    obs = env.reset()

    button_presses = [12,]
    done = False

    summary = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        step = 0
        while not done:
            obs, rew, done, info = env.step(button_presses)
            
            feed_dict = {
                screen_buffer : obs
            }

            summary_str, button_presses = sess.run([summary, buttons], feed_dict)
            button_presses = find_closest(button_presses)
            step = capture_screenbuffer(writer, summary_str, step)

            # Train network here
            
            env.render()

        obs = env.reset()

if __name__ == '__main__':
    main()