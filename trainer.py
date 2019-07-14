'''
Luc Kadletz, 7/14/2019

'''


# Standard Libraries

# Third Party Imports

# Local Imports
from agent import Agent
from enviroment import Environment

class Trainer:

    def __init__(self):
        pass

    def train(self, agent, enviroment):
        print("Training {agent} on {environment}")
        pass

    def test(self, agent, enviroment):
        print("Testing {agent} on {environment}")
        pass

    def replay(self, enviroment, filename):
        print("Replaying {filename} on {environment}")
        pass

def main():
    # By default, make a new agent, and train it on the first game, state listed
    trainer = Trainer()
    agent = Agent()
    enviroment = Environment(Environment.GAMES[0], Environment.STATES[0])
    trainer.test(agent, enviroment)

if __name__ == 'main':
    main()