import argparse
import subprocess
from pathlib import Path
import re

SBATCH_TEMPLATES = {
    "copy_project": """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=logs/{job_name}_%j.out
#SBATCH --error=logs/{job_name}_%j.err
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

{command}
""",
    "track": """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=logs/{job_name}_%j.out
#SBATCH --error=logs/{job_name}_%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1

{command}
""",
    "extract_traces": """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=logs/{job_name}_%j.out
#SBATCH --error=logs/{job_name}_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G

{command}
"""
}


def submit_job(script_path, dependency=None, debug=False):
    cmd = ["sbatch"]
    if dependency:
        cmd.append(f"--dependency=afterok:{dependency}")
    cmd.append(str(script_path))

    if debug:
        print(f"[DEBUG] Submitting job: {' '.join(cmd)}")

    result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE)
    stdout = result.stdout.decode().strip()
    if debug:
        print(f"[DEBUG] Job submission output: {stdout}")
    job_id = stdout.strip().split()[-1]
    return job_id


def write_and_submit_job(trial_name, step_name, command, dependency=None, debug=False):
    job_name = f"{step_name}_{trial_name}"
    script_path = Path(f"sbatch_scripts/{job_name}.sh")

    script_template = SBATCH_TEMPLATES[step_name]
    script_content = script_template.format(job_name=job_name, command=command)
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script_content)

    if debug:
        print(f"[DEBUG] Wrote SBATCH script: {script_path}")
        print(f"[DEBUG] Script content:\n{script_content}")

    return submit_job(script_path, dependency=dependency, debug=debug)


def submit_copy_job(trial_name, finished_path, new_location, make_project_script, debug=False):
    cmd = (
        f"python {make_project_script} "
        f"with project_path={finished_path} "
        f"target_directory={new_location} "
        f"target_suffix={trial_name} "
        f'steps_to_keep="[\'preprocessing\', \'segmentation\']"'
    )
    return write_and_submit_job(trial_name, "copy_project", cmd, debug=debug)


def submit_tracking_job(trial_name, new_location, barlow_model, track_script, dependency_jobid, debug=False, use_projection_space=True):
    project_path = f"{new_location}{trial_name}"
    cmd = (
        f"python {track_script} "
        f"with project_path={project_path} "
        f"model_fname={barlow_model} "
        f"use_projection_space={use_projection_space}"
    )
    return write_and_submit_job(trial_name, "track", cmd, dependency=dependency_jobid, debug=debug)


def submit_trace_job(trial_name, new_location, dispatcher_script, dependency_jobid, debug=False):
    project_path = f"{new_location}{trial_name}"
    cmd = (
        f"sbatch {dispatcher_script} "
        f"-t {project_path} -s 4"
    )
    return write_and_submit_job(trial_name, "extract_traces", cmd, dependency=dependency_jobid, debug=debug)


def parse_args():
    parser = argparse.ArgumentParser(description="Run pipeline: copy → track → extract (all via SBATCH)")
    parser.add_argument("--wbfm-home", required=True, help="Path to the wbfm codebase root directory")
    parser.add_argument("--finished-path", required=True, help="Path to finished project")
    parser.add_argument("--new-location", required=True, help="Base path for new projects")
    parser.add_argument("--models-dir", required=True, help="Folder containing trial subfolders with models OR a single trial directory when --single-trial is used")
    parser.add_argument("--model-fname", default="resnet50.pth", help="Model filename inside each trial folder")
    parser.add_argument("--use_projection_space", required=True, help="Using projection space or final embedding space")
    parser.add_argument("--single-trial", action="store_true", help="Treat --models-dir as a single trial directory instead of a folder of trials")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (runs only one trial with verbose output)")
    return parser.parse_args()


def main():
    args = parse_args()
    models_dir = Path(args.models_dir)
    make_project_script = Path(args.wbfm_home) / "scripts/postprocessing/make_project_like.py"
    track_script = Path(args.wbfm_home) / "scripts/pipeline_alternate/3-track_using_barlow.py"
    dispatcher_script = Path(args.wbfm_home) / "scripts/cluster/single_step_dispatcher.sbatch"
    use_projection_space = args.use_projection_space

    # NEW: Extract parent folder name (e.g. "2025_07_01")
    finished_path_parent = Path(args.finished_path).parent.name

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

    for trial_dir in trial_dirs:
        trial_name = trial_dir.name
        barlow_model_path = trial_dir / args.model_fname
        if not barlow_model_path.is_file():
            print(f"Warning: Model file not found: {barlow_model_path} - skipping {trial_name}")
            continue

        try:
            if args.debug:
                print(f"[DEBUG] Starting pipeline for {trial_name}")
                print(f"[DEBUG] Model path: {barlow_model_path}")

            # Copy project
            copy_jobid = submit_copy_job(
                trial_name, args.finished_path, args.new_location, make_project_script, debug=args.debug
            )

            # Build project base path dynamically
            project_base_path = str(Path(args.new_location) / finished_path_parent)

            # Track
            track_jobid = submit_tracking_job(
                trial_name, project_base_path, str(barlow_model_path), track_script,
                dependency_jobid=copy_jobid, debug=args.debug, use_projection_space=use_projection_space
            )

            # Extract traces
            submit_trace_job(
                trial_name, project_base_path, dispatcher_script,
                dependency_jobid=track_jobid, debug=args.debug
            )

            if args.debug:
                print(f"[DEBUG] Submitted all jobs for {trial_name}")

        except subprocess.CalledProcessError as e:
            print(f"Error in {trial_name}: {e}")
            continue


if __name__ == "__main__":
    main()
