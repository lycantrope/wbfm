# Running the full pipeline

This file is organized as follows:
1. First, all information for the recommended workflow is described (using .nwb files)
2. After that are instructions for legacy workflows (especially for the Zimmer lab)

The following steps use the command line, and assume you have the helper scripts located in this repository (i.e., you have cloned it).

## Data preparation

The main starting point is a file in the Neurodata Without Borders (.nwb) file format.
This should contain at least raw imaging data, but may also contain segmentation, object detection, and/or object id information.

## Creating a project

While running, the raw data file will be "unpacked" into a [folder structure](docs/data_folder_organization.md) for easier processing, which can be re-exported as .nwb at any time. 

```commandline
cd /PATH_TO_THIS_CODE/wbfm/scripts
python 0-create_new_project_from_nwb.py with project_dir=PATH_TO_NEW_PROJECT_LOCATION nwb_file=PATH_TO_NWB_FILE
```

This will create a new project from the nwb file, and unpack any analysis steps that it can find.


## Main pipeline workflow via snakemake

The analysis steps are organized via snakemake, although they can be run step-by-step, see: [detailed pipeline steps](docs/detailed_pipeline_steps.md). 
Before anything, the parameters should be double-checked.
Most do not matter for most people, so I will just highlight the main ones


### Important parameters to check

The most important parameters are related to the tracking pipeline.
In particular, if you are using the newer [BarlowTrack](https://github.com/Zimmer-lab/barlow_track/) method, you must set the path to the neural network here:
```
project_folder/
├── snakemake/snakemake_config.yaml
```

And set:
```
use_barlow_tracker: true
barlow_model_path: YOUR_TRAINED_NETWORK_FOLDER/resnet50.pth
```

### Actually running the pipeline

Snakemake organizes all of the analysis steps; for more information I suggest the main [docs](https://snakemake.readthedocs.io/en/stable/).
No detailed knowledge should be necessary, and our helper script should run all steps.
It is highly recommended to do look at the help first, and then do a dryrun:

```commandline
cd /PATH_TO_THE_PROJECT/snakemake
conda activate MY_ENV
bash RUNME.sh -h
```

```commandline
cd /PATH_TO_THE_PROJECT/snakemake
conda activate MY_ENV
bash RUNME.sh -n
```

This will print which steps need to be run, which are dynamically determined based on the already existing files, if any.
To run **LOCALLY**, use `-c`:

```commandline
cd /PATH_TO_THE_PROJECT/snakemake
conda activate MY_ENV
bash RUNME.sh -c
```

For a more computationally efficient workflow, we run on a cluster via slurm.
However, the exact cluster configuration parameters can change dramatically across clusters, particularly for using gpu partitions.
Please check with your local admin and modify the `cluster_config.yaml` file in this snakemake folder according to their advice.
To run on a cluster (using slurm), no flags are needed:

```commandline
cd /PATH_TO_THE_PROJECT/snakemake
conda activate MY_ENV
bash RUNME.sh
```



# Running the full pipeline: legacy

Most of this information is only relevant to the Zimmer lab, and uses internal formatting and paths.

## Legacy data preparation

See expected folder structure [here](docs/data_folder_organization.md).

1. Red channel (tracking) as an ndtiff folder
2. Green channel (signal) as an ndtiff folder
3. 1 conda environment (see above for installation instructions, or use pre-installed versions on the cluster)

An alternatives to ndtiff for raw data:
1. bigtiff for the raw data, but is deprecated.


### Legacy creating a single project (single recording)

Most people will create multiple projects at once (next section), but for a single recording, see: [detailed pipeline steps](docs/detailed_pipeline_steps.md).

### Creating multiple projects

Each recording will generate one project, and you can use the create_projects_from_folder.sh script.

Important: the data folders must be organized in a specific way, see [here](docs/data_folder_organization.md).

```commandline
bash /PATH_TO_THIS_CODE/wbfm/scripts/cluster/create_multiple_projects_from_data_parent_folder.sh -t /path/to/parent/data/folder -p /path/to/projects/folder
```

Where "/path/to/parent/data/folder" contains subfolders with red, green, and (optionally) behavioral data, and "/path/to/projects/folder" is a folder where the new projects will be generated. 
In principle "/path/to/projects/folder" should be empty, but this is not necessary.

As of September 2023, this is the proper path to this code on the cluster:
```commandline
ls /lisc/data/scratch/neurobiology/zimmer/wbfm/code/wbfm/wbfm/scripts/cluster
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


## Running the rest of the workflow via snakemake

### Running the rest of the workflow for single project

Most people will run multiple projects at once (next section), but for a single recording, see: [detailed pipeline steps](docs/detailed_pipeline_steps.md).

### Running the rest of the workflow for multiple projects

If you have many projects to run, you can use the run_all_projects_in_parent_folder.sh script.
This is especially useful if you created the projects using the create_multiple_projects_from_data_parent_folder.sh script.

```commandline
bash /PATH_TO_THIS_CODE/wbfm/scripts/cluster/run_all_projects_in_parent_folder.sh -t /path/to/projects/folder
```

Note that you should run this script from a directory where you have permission to create files.
Otherwise the log files will be created in the directory where you ran the script, which will crash silently due to permission errors.

For more details on each step, see: [detailed pipeline steps](docs/detailed_pipeline_steps.md).

## Check ongoing progress

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
Use the `progress_gui.py` gui on a local machine with mounted data to check the actual images, segmentation, and tracking produced.
```commandline
cd /PATH_TO_THIS_CODE/wbfm/gui
python progress_gui.py -p /path/to/your/project
```

Method 4:
Use `wbfm_dashboard.py` on a local machine with mounted data to keep track of multiple projects at once:
```commandline
cd /PATH_TO_THIS_CODE/wbfm/gui
python wbfm_dashboard.py -p /path/to/your/parent_folder
```

## Manual annotation and rerunning

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

## Advanced: running steps within an incomplete project (including if it crashed)

See [detailed pipeline steps](docs/detailed_pipeline_steps.md)
