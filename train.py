# -*- coding: utf-8 -*-
# %matplotlib inline

import numpy as np
from shutil import copyfile
import random
import pickle
import loggers

from game import Game
from agent import Agent
from memory import Memory
from model import ResCNN
from funcs import playMatches
from settings import run_folder, run_archive_folder
import utils
import initialise
import config

np.set_printoptions(suppress=True)

loggers.logger_train.info('=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*')
loggers.logger_train.info('=*=*=*=*=*=.      NEW LOG      =*=*=*=*=*')
loggers.logger_train.info('=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*')

env = Game()

# If loading an existing neural network, copy the config file to root
# TODO: json instead
if initialise.INITIAL_RUN_NUMBER is not None:
    copyfile(run_archive_folder + env.name + '/run' +
             str(initialise.INITIAL_RUN_NUMBER).zfill(4) + '/config.py',
             './config.py')

# LOAD MEMORIES IF NECESSARY
if initialise.INITIAL_MEMORY_VERSION is None:
    memory = Memory(config.MEMORY_SIZE)
else:
    print('LOADING MEMORY VERSION ' + str(initialise.INITIAL_MEMORY_VERSION) +
          '...')
    memory_path = utils.get_memory_path(env.name,
                                        initialise.INITIAL_RUN_NUMBER,
                                        initialise.INITIAL_MEMORY_VERSION)
    memory = pickle.load(open(memory_path, "rb"))

# create an untrained neural network objects from the config file
current_NN = ResCNN(config.REG_CONST, config.LEARNING_RATE,
                    (2, ) + env.grid_shape, env.action_size,
                    config.HIDDEN_CNN_LAYERS)
best_NN = ResCNN(config.REG_CONST, config.LEARNING_RATE,
                 (2, ) + env.grid_shape, env.action_size,
                 config.HIDDEN_CNN_LAYERS)

# If loading an existing neural netwrok, set the weights from that model
best_player_version = 0
if initialise.INITIAL_MODEL_VERSION is not None:
    best_player_version = initialise.INITIAL_MODEL_VERSION
    print('LOADING MODEL VERSION ' + str(initialise.INITIAL_MODEL_VERSION) +
          '...')
    current_NN.load(env.name, initialise.INITIAL_RUN_NUMBER,
                    best_player_version)
    best_NN.load(env.name, initialise.INITIAL_RUN_NUMBER, best_player_version)
# otherwise just ensure the weights on the two players are the same
# not necessary?
#else:
#    best_player_version = 0
#    best_NN.model.set_weights(current_NN.model.get_weights())

# copy the config file to the run folder
# not interupted due to some reasons, i.e. graphviz is not installed
copyfile('./config.py', run_folder + 'config.py')
try:
    current_NN.plot_model(run_folder + 'models/model.png')
except Exception as ex:
    print('MODEL PLOTTING FAILED!')
    current_NN.show_summary()
finally:
    print('\n')

# create players
current_player = Agent('current_player', env.state_size, env.action_size,
                       config.MCTS_SIMS, config.CPUCT, current_NN)
best_player = Agent('best_player', env.state_size, env.action_size,
                    config.MCTS_SIMS, config.CPUCT, best_NN)
# user_player = User('player1', env.state_size, env.action_size)
iteration = 0

while 1:

    iteration += 1

    print('ITERATION NUMBER ' + str(iteration))

    loggers.logger_train.info('BEST PLAYER VERSION: %d', best_player_version)
    print('BEST PLAYER VERSION ' + str(best_player_version))

    # Self play
    print('SELF PLAYING ' + str(config.EPISODES) + ' EPISODES...')
    _, memory, _, _ = playMatches(
        best_player,
        best_player,
        config.EPISODES,
        loggers.logger_train,
        turns_until_tau0=config.TURNS_UNTIL_TAU0,
        memory=memory)

    memory.clear_stmemory()

    if len(memory.ltmemory) < config.MEMORY_SIZE:
        print('MEMORY SIZE: ' + str(len(memory.ltmemory)))

    else:
        # retrain
        print('\nRETRAINING...')
        current_player.replay(memory.ltmemory)
        print('')

        if iteration % 5 == 0:
            pickle.dump(memory,
                        open(run_folder + "memories/memory" + str(iteration) + ".p",
                             "wb"))

        loggers.logger_memory.info('====================')
        loggers.logger_memory.info('NEW MEMORIES')
        loggers.logger_memory.info('====================')

        memory_sample = random.sample(memory.ltmemory,
                                      min(1000, len(memory.ltmemory)))

        for s in memory_sample:
            current_value, current_probs, _ = current_player.get_preds(
                s['state'])
            best_value, best_probs, _ = best_player.get_preds(s['state'])

            loggers.logger_memory.info('MCTS VALUE FOR %s: %f',
                                       s['playerTurn'], s['value'])
            loggers.logger_memory.info('CUR PRED VALUE FOR %s: %f',
                                       s['playerTurn'], current_value)
            loggers.logger_memory.info('BES PRED VALUE FOR %s: %f',
                                       s['playerTurn'], best_value)
            loggers.logger_memory.info('THE MCTS ACTION VALUES: %s',
                                       ['%.2f' % elem for elem in s['AV']])
            loggers.logger_memory.info('CUR PRED ACTION VALUES: %s', [
                '%.2f' % elem for elem in current_probs
            ])
            loggers.logger_memory.info('BES PRED ACTION VALUES: %s',
                                       ['%.2f' % elem for elem in best_probs])
            loggers.logger_memory.info('ID: %s', s['state'].id)
            loggers.logger_memory.info(
                'INPUT TO MODEL: %s',
                current_player.model.convertToModelInput(s['state']))

            s['state'].render(loggers.logger_memory)

        # TOURNAMENT
        print('TOURNAMENT...')
        scores, _, points, sp_scores = playMatches(
            best_player,
            current_player,
            config.EVAL_EPISODES,
            loggers.logger_tourney,
            turns_until_tau0=0,
            memory=None)
        print('\nSCORES')
        print(scores)
        print('\nSTARTING PLAYER / NON-STARTING PLAYER SCORES')
        print(sp_scores)
        print(points)

        print('\n\n')

        if scores['current_player'] > scores['best_player'] * config.SCORING_THRESHOLD:
            best_player_version = best_player_version + 1
            best_NN.model.set_weights(current_NN.model.get_weights())
            best_NN.save(env.name, best_player_version)
