import os
import argparse
import itertools
import yaml
import subprocess


def update_yaml(yaml_path, updates):
    """Update a YAML file with nested dictionary values."""
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    for section, params in updates.items():
        if section not in config:
            config[section] = {}
        for key, value in params.items():
            config[section][key] = value

    with open(yaml_path, "w") as f:
        yaml.dump(config, f)


def run_cmd(cmd, cwd=None):
    """Run a shell command with logging."""
    print(f"[RUNNING] {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def main(args):
    # Define parameter sweep
    param_grid = {
        "opt_db": {
            "min_cluster_size": [0.5, 0.7],
            "min_samples": [0.01, 0.05],
        },
        "opt_umap": {
            "n_components": [10, 20],
            "n_neighbors": [5, 10, 30],
            "min_dist": [0.0],
        }
    }

    # Flatten the parameter grid into keys and values
    keys, values = zip(*[
        ((section, key), vals)
        for section, params in param_grid.items()
        for key, vals in params.items()
    ])

    # Iterate over all parameter combinations
    for combo in itertools.product(*values):
        updates = {}
        trial_suffix = []
        for (section, key), value in zip(keys, combo):
            updates.setdefault(section, {})[key] = value
            trial_suffix.append(f"{section.split('_')[-1]}-{key}{value}")

        new_trial_name = os.path.basename(args.source_trial) + "_" + "_".join(trial_suffix)
        new_trial_dir = os.path.join(args.new_location, new_trial_name)

        # 1. Make project like
        run_cmd([
            "python", "scripts/make_project_like.py",
            "--finished-path", args.finished_path,
            "--new-location", new_trial_dir
        ], cwd=args.wbfm_home)

        # 2. Modify YAML after project structure exists
        yaml_path = os.path.join(new_trial_dir, "3-tracking", "tracking_config.yaml")
        if os.path.exists(yaml_path):
            update_yaml(yaml_path, updates)
            print(f"[UPDATED] {yaml_path}")
        else:
            print(f"[ERROR] tracking_config.yaml not found in {yaml_path}")
            continue  # Skip this trial

        # 3. Track using Barlow
        run_cmd([
            "python", "scripts/track_using_barlow.py",
            "--project-path", new_trial_dir,
            "--model-fname", args.model_fname,
            "--use-projection-space", str(args.use_projection_space).lower(),
            "--source-trial", args.source_trial,
        ], cwd=args.wbfm_home)

        # 4. Extract traces
        run_cmd([
            "python", "scripts/create_traces.py",
            "--project-path", new_trial_dir,
        ], cwd=args.wbfm_home)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wbfm-home", required=True, help="Path to wbfm repo")
    parser.add_argument("--finished-path", required=True, help="Reference finished project path")
    parser.add_argument("--new-location", required=True, help="Base directory for new trials")
    parser.add_argument("--source-trial", required=True, help="Trial directory to copy")
    parser.add_argument("--model-fname", required=True, help="Model filename for tracking")
    parser.add_argument("--use-projection-space", type=bool, default=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    main(args)
