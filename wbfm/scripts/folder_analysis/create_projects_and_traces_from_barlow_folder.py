import argparse
import os
import subprocess
from pathlib import Path
import re
from pydantic.utils import deep_update

from wbfm.utils.general.utils_filenames import get_location_of_installed_project
from wbfm.utils.projects.finished_project_data import ProjectData
from wbfm.utils.projects.project_config_classes import make_project_like


def parse_args():
    parser = argparse.ArgumentParser(description="Run pipeline: copy → track → extract (all via SBATCH)")
    parser.add_argument("--finished-path", required=True, help="Path to finished project, usually an analyzed ground truth project")
    parser.add_argument("--new-location", required=True, help="Base path for new projects")
    parser.add_argument("--models-dir", required=True, help="Folder containing trial subfolders with models OR a single trial directory when --single-trial is used")
    parser.add_argument("--model-fname", default="resnet50.pth", help="Model filename inside each trial folder")
    parser.add_argument("--use_projection_space", action="store_true", help="Using projection space or final embedding space")
    parser.add_argument("--single-trial", action="store_true", help="Treat --models-dir as a single trial directory instead of a folder of trials")
    parser.add_argument("--use_tracklets", action="store_true", help="Use tracklets instead of clustering to build final tracks")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (runs only one trial with verbose output)")
    return parser.parse_args()


def main():
    args = parse_args()

    wbfm_home = get_location_of_installed_project()
    models_dir = Path(args.models_dir)
    use_projection_space = args.use_projection_space

    if args.single_trial:
        # Just one trial, directly from models_dir
        trial_dirs = [models_dir]
    else:
        # Collect all trial_* subdirectories
        trial_dirs = sorted([d for d in models_dir.iterdir() if d.is_dir() and re.match(r"trial_\d+", d.name)])

    if not trial_dirs:
        print(f"No trial folders found in {args.models_dir} matching pattern 'trial_#'")
        return

    if args.debug and not args.single_trial:
        print(f"[DEBUG] Found {len(trial_dirs)} trial folders")
        print(f"[DEBUG] Will only process the first trial: {trial_dirs[0].name}")
        trial_dirs = trial_dirs[:1]

    #########################################################################################
    # Create projects and update the config file to target the proper barlow model
    #########################################################################################
    for trial_dir in trial_dirs:
        trial_name = trial_dir.name
        # try:
        barlow_model_path = Path(trial_dir) / Path(args.model_fname)
        print(f"Starting pipeline for {trial_name}")
        if args.debug:
            print(f"[DEBUG] Model path: {barlow_model_path}")

        new_project_name = make_project_like(
            project_path=args.finished_path, 
            target_directory=args.new_location, 
            target_suffix=trial_name,
            steps_to_keep=['preprocessing', 'segmentation'],
            verbose=3 if args.debug else 0
        )
        if not barlow_model_path.is_file():
            print(f"Warning: Model file not found: {barlow_model_path} - skipping {trial_name}")
            continue

        # Two options: use tracklets or direct segmentation
        project_data = ProjectData.load_final_project_data(new_project_name, verbose=0)
        project_config = project_data.project_config
        if args.use_tracklets:
            tracklet_config = project_config.get_training_config()
            config_updates = dict(
                tracker_params=dict(
                    use_barlow_network=True,
                    encoder_opt=dict(
                        network_path=str(barlow_model_path),
                        use_projection_space=use_projection_space
                    )
                ),
                pairwise_matching_params=dict(add_affine_to_candidates=False)
            )
            deep_update(tracklet_config.config, config_updates)
            tracklet_config.update_self_on_disk()
        else:
            snakemake_config = project_config.get_snakemake_config()
            config_updates = dict(use_barlow_tracker=True, barlow_model_path=str(barlow_model_path))
            snakemake_config.config.update(config_updates)
            snakemake_config.update_self_on_disk()

    #########################################################################################
    # Actually submit jobs for full pipeline
    #########################################################################################
    # Note that the script is already recursive

    CMD = ["bash", os.path.join(wbfm_home, 'wbfm', 'scripts', 'cluster', 'run_all_projects_in_parent_folder.sh')]
    CMD.extend(["-t", args.new_location,  "-s" , "traces"])
    if args.debug:
        # Dryrun
        CMD.append("-n")
    subprocess.call(CMD)

    print(f"All jobs for {len(trial_dirs)} trials in folder {args.new_location} submitted successfully.")


if __name__ == "__main__":
    main()
