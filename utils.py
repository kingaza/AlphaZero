from settings import run_archive_folder


# path for retrieving the archived data
def get_memory_path(game_name, run_number, memory_version):
    return run_archive_folder + game_name + '/run' + str(run_number).zfill(
        4) + '/memories/memory' + str(memory_version).zfill(4) + '.p'


def get_model_path(game_name, run_number, model_version):
    return run_archive_folder + game_name + '/run' + str(run_number).zfill(
        4) + '/models/version' + str(model_version).zfill(4) + '.h5'
