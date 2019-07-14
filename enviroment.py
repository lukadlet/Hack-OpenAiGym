'''
Luc Kadletz, 7/14/2019

'''

# Standard Libraries

# Third Party Imports
import numpy as np
from retro import make

# Local Imports

class Environment:

    GAMES = [
        'SonicTheHedgehog-Genesis'
    ]

    STATES = [
        'GreenHillZone.Act1'
    ]

    def __init__(self, game, state):
        pass

    def load(self, filename):
        pass

    def tick(self, buttons):
        pass

    def render(self):
        pass

class Game:
    def __init__(self, states, buttons):
        pass

