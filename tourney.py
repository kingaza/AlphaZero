from game import Game
import loggers
import config
from model import ResCNN
from agent import Agent
from funcs import playMatches

run_version = 1
player1version = 10
player2version = 50
EPISODES = 7
logger = loggers.logger_tourney
turns_until_tau0 = 0

env = Game()
network = ResCNN(config.REG_CONST, config.LEARNING_RATE, env.input_shape,
                 env.action_size, config.HIDDEN_CNN_LAYERS)

network.load(env.name, run_version, player1version)
player1 = Agent('player1', env.state_size, env.action_size, config.MCTS_SIMS,
                config.CPUCT, network)

network.load(env.name, run_version, player2version)
player2 = Agent('player2', env.state_size, env.action_size, config.MCTS_SIMS,
                config.CPUCT, network)

print('Players are ready, Tourney begins!')

goes_first = 0
scores, memory, points, sp_scores = playMatches(
    player1, player2, EPISODES, logger, turns_until_tau0, None, goes_first)

print(scores)
print(points)
print(sp_scores)
