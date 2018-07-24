import tensorflow as tf
import numpy as np
from retro import make

def test_tensorflow():
    hello = tf.constant('Hello Hackathon!')
    sess = tf.Session()
    print(sess.run(hello))

def filter_screen(image, downsample_x, downsample_y):
    with tf.name_scope("filter"):
        tf.summary.image('original', image)
        
        image = tf.image.rgb_to_grayscale(image)
        tf.summary.image('greyscale', image)

        image = tf.image.resize_bicubic(image, [downsample_x, downsample_y]) 
        tf.summary.image('downscaled', image)

        image = tf.reshape(image, [-1, downsample_x * downsample_y])
    
    return image

def setup_tensor_graph():

    # Create a placeholder to put the screen buffer
    screen_buffer = tf.placeholder(tf.float32, name="ScreenBuffer")
    screen_image = tf.reshape(screen_buffer, [-1, 224, 320, 3])

    # Greyscale and resize the screen so its a computable size
    screen_flattened = filter_screen(screen_image, 64, 64)

    # AI uses these things somewhere, right?
    weights = tf.Variable(tf.random_uniform([64 * 64, 12]), name = "Weights")
    biases = tf.Variable(tf.random_uniform([12,]), name = "Biases")

    # Uhh, just hit random buttons I guess
    noise =  tf.random_uniform([12,])

    output = tf.matmul(screen_flattened, weights)
    output = tf.reshape(tf.cast(tf.round(output), tf.bool), shape=[-1,])

    # Write everything down    
    writer = tf.summary.FileWriter('./logs/dev/', tf.get_default_graph())
    return screen_buffer, output, writer


def capture_screenbuffer(writer, summary, step, capture_each = 30):
    if(step % capture_each == 0):
        writer.add_summary(summary, step)
    return step + 1

def main():
    screen_buffer, output, writer = setup_tensor_graph()
    env = make(game='SonicTheHedgehog-Genesis', state = 'LabyrinthZone.Act1')
    obs = env.reset()

    button_pressses = [12,]
    done = False

    summary = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        step = 0
        while not done:
            print(button_pressses)
            obs, rew, done, info = env.step(button_pressses)
            
            feed_dict = {
                screen_buffer : obs
            }

            summary_str, button_pressses = sess.run([summary, output], feed_dict)
            
            step = capture_screenbuffer(writer, summary_str, step)

            # Train network here
            
            env.render()

        obs = env.reset()

if __name__ == '__main__':
    main()