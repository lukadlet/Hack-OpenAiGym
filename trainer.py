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
        self.idle_kill_timer = None

    def _start_environment(self, environment):
        self.done = False
        environment.load()

    def train(self, agent, environment):
        print("Training", agent, " on ", environment)

        self._start_environment(environment)

        while not self.done:
            environment.step(agent.next_action)
            agent.tick(environment.screen, environment.info)

            self._verbose_info(agent, environment)
            self._check_idle_kill(agent, environment)

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

            self._verbose_info(agent, environment)
            self._check_idle_kill(agent, environment)

            if(not self.headless):
                environment.render()

            if environment.done:
                self.done = True

        print("Testing done in ", environment.step_count, "steps")

    def _verbose_info(self, agent, environment):
        if(self.verbose):
            print(environment)
            print(agent)

    def _check_idle_kill(self, agent, environment):
        if(self.idle_kill_timer is not None):
            if(agent.idle_count > self.idle_kill_timer):
                environment.done = True
                print("Idle counter exceeded ",
                      self.idle_kill_timer, ", ending test.")

    def replay(self, environment, filename):
        print("Replaying", filename, " on ", environment)
