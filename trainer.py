'''

Luc Kadletz, 7/14/2019

'''

# Standard Libraries

# Third Party Imports
import argparse
# Local Imports
from enviroment import Environment


class Trainer:

    def __init__(self):
        self.done = False
        self.pause = False

    def _start_environment(self, environment):
        self.done = False
        environment.load()

    def train(self, agent, environment):
        print("Training", agent, " on ", environment)
   
        self._start_environment(environment)

        while not self.done:
            agent.tick(environment.screen, environment.info)
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

