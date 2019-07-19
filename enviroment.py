'''
Luc Kadletz, 7/14/2019

'''

# Standard Libraries

# Third Party Imports
import numpy as np
import retro
# Local Imports

class Environment:

    GAMES = [
        'SonicTheHedgehog-Genesis'
    ]

    STATES = [
        'GreenHillZone.Act1'
    ]

    def __init__(self, game: str, state: str):
        self.game = game
        self.state = state

        self.gym = None
        self.done = False
        self.step_count = 0
        self.record = False
        
        self.loss = 0

    def load(self):
        if(self.record):
            self.gym = retro.make(self.game, self.state)
        else:
            self.gym = retro.make(self.game, self.state, record = self.record)
        self.screen = self.gym.reset()

    def step(self, buttons):
        obs, rew, done, info = self.gym.step(buttons)
        self.screen = obs
        self.loss = -rew # sure?????
        self.done = done
        self.step_count = self.step_count + 1

    def render(self):
        self.gym.render()

    def __str__(self):
        return str.format("[Game : {0}, State : {1}]", self.game, self.state)

'''
Find a way to define games with their states and inputs in a way that can be
adapted between agents. Do we want to include their loss function / any unique
preprocessing here? Or should the loss/preprocessing get its own layer
class Game:
    def __init__(self, states, obs_size, buttons):
        self.states = states
        self.obs_size = obs_size
        self.buttons = buttons
'''
