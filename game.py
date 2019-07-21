'''
Luc Kadletz 7/19/2019
'''

# Standard Libraries

# Third Party Imports

# Local Imports
import enviroment


class Game:
    def __init__(self, name, states, obs_size, actions):
        self.name = name
        self.states = states
        self.obs_size = obs_size
        self.actions = actions


sonic = Game(
    name='SonicTheHedgehog-Genesis',
    states=[
        'GreenHillZone.Act1'
    ],
    obs_size=[244, 320],
    actions=[
        # [ B, A, MODE, START, UP, DOWN, LEFT, RIGHT, C, Y, X, Z ]
        # RIGHT
        [False, False, False, False, False, False,
         False, True, False, False, False, False],
        # LEFT
        [False, False, False, False, False, False,
         True, False, False, False, False, False],
        # DOWN + LEFT
        [False, False, False, False, False, True,
         True, False, False, False, False, False],
        # DOWN + RIGHT
        [False, False, False, False, False, True,
         False, True, False, False, False, False],
        # DOWN
        [False, False, False, False, False, True,
         False, False, False, False, False, False],
        # JUMP + DOWN
        [True, False, False, False, False, True,
         False, False, False, False, False, False],
        # JUMP
        [True, False, False, False, False, False,
         False, False, False, False, False, False],
        # NOTHING
        [False, False, False, False, False, False,
         False, False, False, False, False, False]
    ]
)

'''
This is just a guess, I have not imported anything yet
'''
pokemon = Game(
    name='PokemonRed-Gameboy',
    states=[
        'Rival.1.A',
        'Rival.1.B',
        'Rival.1.C',
    ],
    obs_size=[160, 144],
    actions=[
        # [ B, A, START, SELECT, UP, DOWN, LEFT, RIGHT ]
        # B
        [True, False, False, False, False, False, False, False],
        # A
        [False, True, False, False, False, False, False, False],
        # UP
        [False, False, False, False, True, False, False, False],
        # DOWN
        [False, False, False, False, False, True, False, False],
        # LEFT
        [False, False, False, False, False, False, True, False],
        # RIGHT
        [False, False, False, False, False, False, False, True],
        # NOTHING
        [False, False, False, False, False, False, False, False]
    ]
)
