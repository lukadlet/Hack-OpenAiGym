'''
Luc Kadletz, 7/14/2019

'''

# Standard Libraries

# Third Party Imports
import numpy as np
import retro
# Local Imports
import game


class Environment:

    def __init__(self, game: str, state: str):
        self.game = game
        self.state = state

        self.gym = None

        self.screen = None
        self.info = None
        self.reward = None

        self.record = False
        self.done = False
        self.step_count = 0  # Maybe this should be a property of the agent?

    def load(self):
        if(self.record):
            self.gym = retro.make(self.game, self.state)
        else:
            self.gym = retro.make(self.game, self.state, record=self.record)
        self.screen = self.gym.reset()

    def step(self, buttons):
        self.step_count = self.step_count + 1

        obs, rew, done, info = self.gym.step(buttons)
        self.screen = obs
        self.reward = rew
        self.done = done
        self.info = info

    def render(self):
        self.gym.render()

    def __str__(self):
        return str.format("[Game : {0}, State : {1}]", self.game, self.state)
