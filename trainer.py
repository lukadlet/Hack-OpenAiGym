'''

Luc Kadletz, 7/14/2019

'''

# Standard Libraries

# Third Party Imports

# Local Imports
from agent import Agent
from enviroment import Environment
from game import Sonic


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
            agent.tick(environment.screen, environment.loss)
            environment.step(agent.next_action)
            environment.render()
            if environment.done:
                self.done = True

        print("Testing done in ", environment.step_count, "steps")

    def replay(self, environment, filename):
        print("Replaying", filename, " on ", environment)


def main():
    # By default, make a new agent, and train it on the first game, state listed
    trainer = Trainer()
    agent = Agent(Sonic.obs_size, Sonic.actions)
    enviroment = Environment(Sonic.name, Sonic.states[0])
    trainer.test(agent, enviroment)

if __name__ == '__main__':
    main()
