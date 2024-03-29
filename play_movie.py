import retro

movie = retro.Movie('SonicTheHedgehog-Genesis-GreenHillZone.Act1-0000.bk2')
movie.step()

env = retro.make(game=movie.get_game(), state= None, use_restricted_actions=retro.ACTIONS_ALL)
env.initial_state = movie.get_state()
env.reset()

while movie.step():
    keys = []
    for i in range(env.NUM_BUTTONS):
        keys.append(movie.get_key(i))
    _obs, _rew, _done, _info = env.step(keys)