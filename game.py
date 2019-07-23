'''
Luc Kadletz 7/19/2019
'''

# Standard Libraries

# Third Party Imports

# Local Imports
import enviroment


class Game:
    def __init__(self,
                 name: str,
                 states: [str],
                 obs_size: [int, int],
                 actions: [[bool]],
                 loss_fn: callable):
        self.name = name
        self.states = states
        self.obs_size = obs_size
        self.actions = actions
        self.loss_fn = loss_fn


def sonic_loss(info):
    return info["screen_x_end"] - info["x"] - (info["novelty"] * 100)


sonic = Game(
    name='SonicTheHedgehog-Genesis',
    states=[
        'GreenHillZone.Act1'
    ],
    obs_size=[224, 320],
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
    ],
    loss_fn=sonic_loss
)


def pokemon_loss(info):
    step_penalty = info["idle_count"]
    agent_total_hp = (info["poke1HP"] + info["poke2HP"] + info["poke3HP"] +
        info["poke4HP"] + info["poke5HP"] + info["poke6HP"]) * 10
    enemy_total_hp = info["enemy1HP"] * 100
    novelty = -info["novelty"] * 500

    return (novelty) + step_penalty + enemy_total_hp - agent_total_hp


pokemon = Game(
    name='PokemonRed-GameBoy',
    states=[
        'bulbasaur_vs_charmander'
    ],
    obs_size=[144, 160],
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
    ],
    loss_fn=pokemon_loss
)
