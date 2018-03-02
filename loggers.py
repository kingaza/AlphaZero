import logging

from settings import run_folder

# SET all LOGGER_DISABLED to True to disable logging
# WARNING: the mcts log file gets big quite quickly
# therefore it is recommended to use rotation file handler

LOGGER_DISABLED = {
    'main': False,
    'memory': False,
    'train': False,
    'tourney': False,
    'mcts': False,
    'model': False
}


def setup_logger(name, log_file, rotation=False, level=logging.INFO):

    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

    handler = logging.FileHandler(log_file)
    if rotation:
        handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=1024 * 1024 * 1024, backupCount=5)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(handler)

    return logger


logger_mcts = setup_logger(
    'logger_mcts', run_folder + 'logs/logger_mcts.log', rotation=True)
logger_mcts.disabled = LOGGER_DISABLED['mcts']

logger_main = setup_logger('logger_main', run_folder + 'logs/logger_main.log')
logger_main.disabled = LOGGER_DISABLED['main']

logger_train = setup_logger(
    'logger_train', run_folder + 'logs/logger_train.log', rotation=True)
logger_train.disabled = LOGGER_DISABLED['train']

logger_tourney = setup_logger('logger_tourney',
                              run_folder + 'logs/logger_tourney.log')
logger_tourney.disabled = LOGGER_DISABLED['tourney']

logger_memory = setup_logger('logger_memory',
                             run_folder + 'logs/logger_memory.log')
logger_memory.disabled = LOGGER_DISABLED['memory']

logger_model = setup_logger('logger_model',
                            run_folder + 'logs/logger_model.log')
logger_model.disabled = LOGGER_DISABLED['model']
