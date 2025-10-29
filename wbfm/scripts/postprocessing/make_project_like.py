
# Experiment tracking
import sacred
from sacred import Experiment

# main function
from wbfm.utils.projects.project_config_classes import make_project_like

# Initialize sacred experiment
ex = Experiment(save_git_info=False)
ex.add_config(project_path=None, target_directory=None, target_suffix=None, steps_to_keep=None, config_files_to_update=None, DEBUG=False)


@ex.automain
def main(_config, _run):
    sacred.commands.print_config(_run)

    target_directory = _config['target_directory']
    project_path = _config['project_path']
    steps_to_keep = _config['steps_to_keep']
    target_suffix = _config['target_suffix']
    config_files_to_update = _config['config_files_to_update']

    make_project_like(project_path, target_directory=target_directory,
                      target_suffix=target_suffix, steps_to_keep=steps_to_keep, config_files_to_update=config_files_to_update, verbose=1)
