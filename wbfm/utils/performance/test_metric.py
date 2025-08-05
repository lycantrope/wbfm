import argparse
import os
import pandas as pd
import numpy as np
from wbfm.utils.projects.finished_project_data import ProjectData
from wbfm.utils.neuron_matching.utils_candidate_matches import rename_columns_using_matching


def pad_with_nan_rows(df: pd.DataFrame, target_length: int) -> pd.DataFrame:
    """
    Pads the given DataFrame with NaN rows until it reaches the specified target length.
    
    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to pad.
    target_length : int
        The desired number of rows.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with NaN rows appended if it was shorter than target_length.
    """
    if len(df) < target_length:
        missing_rows = target_length - len(df)
        new_index = range(df.index.max() + 1, df.index.max() + 1 + missing_rows)
        nan_rows = pd.DataFrame(np.nan, index=new_index, columns=df.columns)
        return pd.concat([df, nan_rows])
    return df


def calculate_accuracy(df_gt: pd.DataFrame, df_pred: pd.DataFrame) -> dict:
    """
    Calculate overall, per-neuron (column), and per-timepoint (row) accuracy
    between ground truth and predicted DataFrames.

    Parameters
    ----------
    df_gt : pd.DataFrame
        Ground truth DataFrame with shape (timepoints, neurons).
    df_pred : pd.DataFrame
        Predicted DataFrame with same shape and column names as df_gt.

    Returns
    -------
    dict
        Contains overall accuracy, per-neuron accuracy (Series),
        per-timepoint accuracy (Series), and counts of misses and mismatches.
    """
    # Align both DataFrames on the same rows and columns
    common_index = df_gt.index.intersection(df_pred.index)
    common_columns = df_gt.columns.intersection(df_pred.columns)

    df_gt = df_gt.loc[common_index, common_columns]
    df_pred = df_pred.loc[common_index, common_columns]

    # Validity masks
    gt_valid = ~df_gt.isna()
    pred_valid = ~df_pred.isna()

    # Conditions
    misses = gt_valid & ~pred_valid
    mismatches = gt_valid & pred_valid & (df_gt != df_pred)

    total_misses = misses.sum().sum()
    total_mismatches = mismatches.sum().sum()
    total_gt_detections = gt_valid.sum().sum()

    overall_accuracy = 1 - (total_misses + total_mismatches) / total_gt_detections

    # Per-column (neuron) accuracy
    total_per_neuron = gt_valid.sum(axis=0)
    errors_per_neuron = (misses + mismatches).sum(axis=0)
    per_neuron_accuracy = 1 - (errors_per_neuron / total_per_neuron)

    # Per-row (timepoint) accuracy
    total_per_timepoint = gt_valid.sum(axis=1)
    errors_per_timepoint = (misses + mismatches).sum(axis=1)
    per_timepoint_accuracy = 1 - (errors_per_timepoint / total_per_timepoint)

    return {
        "accuracy": overall_accuracy,
        "misses": int(total_misses),
        "mismatches": int(total_mismatches),
        "total_ground_truth": int(total_gt_detections),
        "per_neuron_accuracy": per_neuron_accuracy,
        "per_timepoint_accuracy": per_timepoint_accuracy
    }



def process_trial(trial: int, df_gt: pd.DataFrame, res_file: str) -> dict:
    """
    Process a single trial: load results, match columns, pad rows, and compute accuracy.

    Parameters
    ----------
    trial : int
        The trial number being processed.
    df_gt : pd.DataFrame
        Ground truth DataFrame.
    res_file : str
        Path to the trial's result configuration file.

    Returns
    -------
    dict
        Dictionary containing trial number and accuracy stats.
    """
    try:
        # Load result data
        project_data_res = ProjectData.load_final_project_data(res_file)
        df_res = project_data_res.final_tracks

        # Match lengths
        max_len = max(len(df_res), len(df_gt))
        df_res = pad_with_nan_rows(df_res, max_len)
        df_gt_padded = pad_with_nan_rows(df_gt, max_len)

        # Match columns using neuron matching
        df_res_renamed, _, _, _ = rename_columns_using_matching(
            df_gt_padded, df_res, column='raw_segmentation_id', try_to_fix_inf=True
        )
        # Reduce both DataFrames to raw_segmentation_id level
        df_res_renamed = df_res_renamed.xs('raw_segmentation_id', axis=1, level=1)
        df_gt_padded = df_gt_padded.xs('raw_segmentation_id', axis=1, level=1)

        # Calculate accuracy
        stats = calculate_accuracy(df_gt_padded, df_res_renamed)
        stats["trial"] = trial
        return stats

    except Exception as e:
        print(f"Trial {trial}: ERROR during processing -> {e}")
        return {"trial": trial, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Evaluate tracking accuracy across multiple trials.")
    parser.add_argument("--ground_truth_path", required=True, help="Path to the ground truth NWB file")
    parser.add_argument("--res_path", required=True, help="Path to the directory containing trial runs")
    parser.add_argument("--trial_dir_prefix", default="", help="Optional prefix for trial directories (e.g. 2025_07_01)")
    parser.add_argument("--trials", required=True, help="Either a list of trial numbers [1,2,3] or a single integer for range 0..N-1")

    args = parser.parse_args()

    # Parse trials
    if args.trials.startswith("["):
        trials = eval(args.trials)
    else:
        trials = list(range(int(args.trials)))

    print(f"Loading ground truth from {args.ground_truth_path} ...")
    project_data_gt = ProjectData.load_final_project_data(args.ground_truth_path)
    df_gt = project_data_gt.final_tracks

    results = []
    for trial in trials:
        trial_dir = f"{args.trial_dir_prefix}trial_{trial}" if args.trial_dir_prefix else f"trial_{trial}"
        res_file = os.path.join(args.res_path, trial_dir, "project_config.yaml")

        if not os.path.exists(res_file):
            print(f"Trial {trial}: Skipping, result file not found at {res_file}")
            continue

        print(f"\nProcessing trial {trial} ...")
        stats = process_trial(trial, df_gt, res_file)
        results.append(stats)
        if "error" not in stats:
            print(f"Trial {trial}: {stats}")

    # Optionally print summary
    print("\nSummary:")
    for res in results:
        if "error" in res:
            print(f"Trial {res['trial']}: ERROR -> {res['error']}")
        else:
            print(f"Trial {res['trial']}: Accuracy {res['accuracy']:.4f} (Misses: {res['misses']}, Mismatches: {res['mismatches']})")


if __name__ == "__main__":
    main()
