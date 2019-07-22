'''
Luc Kadletz, 7/22/2019
Microsoft Hackathon
'''

# Standard Libraries

# Third Party Imports
import argparse
# Local Imports
from trainer import Trainer
from game import sonic, pokemon
from agent import Agent
from enviroment import Environment


class Sequence:
    def __init__(self, game, states):
        self.game = game
        self.states = states
        self.num_states = len(states)


def main():
    args = parse_args()

    agent = make_agent(args.game)
    sequence = make_training_sequence(args.train_sequence, args.game)

    agent.start()
    # TODO load here if a model was provided

    trainer = Trainer()

    if(args.train_sequence == None):
        environment = Environment(sequence.game, sequence.states[0])
        trainer.test(agent, environment)
    else:
        for state in sequence.states:
            environment = Environment(sequence.game, state)
            trainer.train(agent, environment)
        # TODO Save completed sequence somewhere


def make_training_sequence(sequence_file, game):
    if(sequence_file is None):
        return get_default_sequence(game)
    else:
        lines = [line.rstrip('\n') for line in sequence_file]
        return Sequence(lines[0], lines[1:])


def get_default_sequence(game):
    if(game == 'pokemon'):
        return Sequence(pokemon.name, pokemon.states)
    elif(game == 'sonic'):
        return Sequence(sonic.name, sonic.states)
    else:
        print("Unknown game " + game)
        return None


def make_agent(game):
    if(game == 'pokemon'):
        return Agent(pokemon.obs_size, pokemon.actions, pokemon.loss_fn)
    elif(game == 'sonic'):
        return Agent(sonic.obs_size, sonic.actions, sonic.loss_fn)
    else:
        print("Unknown game " + game)
        return None


def make_environment(game, state):
    if(game == 'pokemon'):
        return Agent(pokemon.obs_size, pokemon.actions, pokemon.loss_fn)
    elif(game == 'sonic'):
        return Agent(sonic.obs_size, sonic.actions, sonic.loss_fn)
    else:
        print("Unknown game " + game)
        return None


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', type=str, default='', required=False,
                        help="Load a model to use for training/testing")

    parser.add_argument('-t', '--train_sequence', type=open, default=None,
                        help="A path to a text file containing the game to train, " +
                        "followed by each scene to train")

    parser.add_argument('-g', '--game', default='sonic', const='sonic',
                        nargs='?', choices=['sonic', 'pokemon'],
                        help="The game to train on")

    return parser.parse_args()


if __name__ == '__main__':
    main()