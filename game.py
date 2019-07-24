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

def pokemon_loss(info):
    step_penalty = info["step_count"]
    agent_total_hp = (info["poke1HP"] + info["poke2HP"] + info["poke3HP"] +
        info["poke4HP"] + info["poke5HP"] + info["poke6HP"])
    enemy_total_hp = info["enemyHP"]
    return step_penalty + enemy_total_hp - agent_total_hp


pokemon = Game(
    name='PokemonRed-GameBoy',
    states=[
        'bulbasaur_vs_charmander'
    ],
    obs_size=[144, 160],
    actions=[
        # ["B", None, "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A"]
        [False, False, False, False, False, False, False, False, True]
    ],
    loss_fn=pokemon_loss
)
