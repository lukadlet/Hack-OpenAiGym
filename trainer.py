'''

Luc Kadletz, 7/14/2019

'''

# Standard Libraries

# Third Party Imports
import argparse
# Local Imports
from agent import Agent
from enviroment import Environment
from game import sonic


class Trainer:

    def __init__(self):
        self.done = False
        self.pause = False

    def _start_environment(self, environment):
        self.done = False

        environment.load()

    def train(self, agent, environment):
        print("Training", agent, " on ", environment)

        # This should be in a loop for x times, with y environments

        self._start_environment(environment)

        while not self.done:
            agent.tick(environment.screen, environment.loss)
            environment.step(agent.next_action)
            agent.optimize()
            # Don't render
            if environment.done:
                self.done = True

    def test(self, agent, environment):
        print("Testing", agent, " on ", environment)
        self._start_environment(environment)

        while not self.done:
            agent.tick(environment.screen, environment.info)
            environment.step(agent.next_action)
            environment.render()
            if environment.done:
                self.done = True

        print("Testing done in ", environment.step_count, "steps")

    def replay(self, environment, filename):
        print("Replaying", filename, " on ", environment)


def main(args):
    # By default, make a new agent, and train it on the first game, state listed
    trainer = Trainer()
    agent = Agent(sonic.obs_size, sonic.actions, sonic.loss_fn)
    enviroment = Environment(sonic.name, sonic.states[0])

    if(args.train_sequence != None):
        lines = [line.rstrip('\n') for line in args.train_sequence]
        print("I'm totally gonna train on ", lines)

    agent.start()
    if(args.model != ""):
        agent.load(args.model)

    trainer.test(agent, enviroment)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='', required=False,
                        help="Load a model to use for training/testing")
    parser.add_argument('-t', '--train_sequence', type=open, default=None,
                        help="A path to a text file containing the game to train, " +
                        "followed by each scene to train")

    args = parser.parse_args()

    main(args)
