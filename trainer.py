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
        self.headless = False
        self.verbose = False

    def _start_environment(self, environment):
        self.done = False
        environment.load()

    def train(self, agent, environment):
        print("Training", agent, " on ", environment)

        self._start_environment(environment)

        while not self.done:
            environment.step(agent.next_action)
            agent.tick(environment.screen, environment.info)
            if(self.verbose):
                print(environment)
                print(agent)
                print(environment.info)
            if(not self.headless):
                environment.render()
            if environment.done:
                self.done = True
            agent.optimize()

    def test(self, agent, environment):
        print("Testing", agent, " on ", environment)
        self._start_environment(environment)

        while not self.done:
            environment.step(agent.next_action)
            agent.tick(environment.screen, environment.info)
            if(self.verbose):
                pass
            if(not self.headless):
                environment.render()
            if environment.done:
                self.done = True

        print("Testing done in ", environment.step_count, "steps")

    def replay(self, environment, filename):
        print("Replaying", filename, " on ", environment)
