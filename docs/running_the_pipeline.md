# Running the full pipeline

## Preparation

See expected folder structure [here](docs/data_folder_organization.md).

1. Red channel (tracking) as an ndtiff folder
2. Green channel (signal) as an ndtiff folder
3. 1 conda environment (see above for installation instructions, or use pre-installed versions on the cluster)

There are two alternatives to ndtiff for raw data:
1. bigtiff for the raw data, but is deprecated.
2. .nwb file

## Full pipeline

### Creating a single project (single recording)

Most people will create multiple projects at once (next section), but for a single recording, see: [detailed pipeline steps](docs/detailed_pipeline_steps.md).

### Creating multiple projects

Each recording will generate one project, and you can use the create_projects_from_folder.sh script.

Important: the data folders must be organized in a specific way, see [here](docs/data_folder_organization.md).

```commandline
bash /path/to/this/code/wbfm/scripts/cluster/create_multiple_projects_from_data_parent_folder.sh -t /path/to/parent/data/folder -p /path/to/projects/folder
```

Where "/path/to/parent/data/folder" contains subfolders with red, green, and (optionally) behavioral data, and "/path/to/projects/folder" is a folder where the new projects will be generated. 
In principle "/path/to/projects/folder" should be empty, but this is not necessary.

As of September 2023, this is the proper path to this code on the cluster:
```commandline
ls /lisc/scratch/neurobiology/zimmer/wbfm/code/wbfm/wbfm/scripts/cluster
```

For running projects, you will most likely want to run them all simultaneously instead of one-by-one.
See this [section](#running-the-rest-of-the-workflow-for-multiple-projects).

#### *IMPORTANT*

Creating and running the project on different operating systems will cause problems.
See [Summary of common problems](#summary-of-common-problems) for more details.

### Checklist of most important parameters to change or validate

You should check that the correct files were found, and that the z-metadata is correct.

1. project_config.yaml
   1. red_fname
   2. green_fname
   3. behavior_fname (optional)
2. dat/preprocessing_config.yaml
   1. raw_number_of_planes (before any removal)
   2. starting_plane (set 1 to remove a single plane)
3. segment_config.yaml
   1. max_number_of_objects (increase for immobilized projects)

For all other settings, the defaults should work well.


### Running the rest of the workflow for single project

Most people will run multiple projects at once (next section), but for a single recording, see: [detailed pipeline steps](docs/detailed_pipeline_steps.md).

### Running the rest of the workflow for multiple projects

If you have many projects to run, you can use the run_all_projects_in_parent_folder.sh script.
This is especially useful if you created the projects using the create_multiple_projects_from_data_parent_folder.sh script.

```commandline
bash /path/to/this/code/wbfm/scripts/cluster/run_all_projects_in_parent_folder.sh -t /path/to/projects/folder
```

Note that you should run this script from a directory where you have permission to create files.
Otherwise the log files will be created in the directory where you ran the script, which will crash silently due to permission errors.

For more details on each step, see: [detailed pipeline steps](docs/detailed_pipeline_steps.md).

#### Check ongoing progress

There are three ways to check progress:
1. Check the currently running jobs
2. Check the log files in the snakemake/ subfolder
3. Check the produced analysis files using a gui

Method 1:
Run this command on the cluster:
```commandline
squeue -u <your_username>
```

Method 2:
Use the tail command to check the log files:
```commandline
tail -f /path/to/your/project/snakemake/log/[MOST_RECENT_LOG].log
```

Note that the -f flag will keep the terminal open and update the log file as it changes.

Method 3:
Use the progress_gui.py gui on a local machine with mounted data to check the actual images, segmentation, and tracking produced.
```commandline
cd /path/to/this/code/wbfm/gui
python progress_gui.py -p /path/to/your/project
```

### Manual annotation and rerunning

Tracking can be checked and corrected using the main trace_explorer [GUI](#Summary of GUIs).
However, this does not automatically regenerate the final trace dataframes.
For this, some steps must be rerun.
The steps are the same as running steps within an incomplete project, but in short:

1. Run step 3b
   - Note that setting only_use_previous_matches=True in tracking_config.yaml is suggested, and will speed the process dramatically
2. Possible: run step 1-alt (rebuild the segmentation metadata)
   - This is only necessary if segmentation was changed in the manual annotation step 
2. Run step 4

All steps can be run using multi_step_dispatcher.sh
Example usage is given within that file.

Note that it is possible to use snakemake to "know" which files need to be updated, if any files were copied to/from a local machine, snakemake will become confused and attempt to rerun the entire pipeline.

### Advanced: running steps within an incomplete project (including if it crashed)

See [detailed pipeline steps](docs/detailed_pipeline_steps.md)
