from wbfm.utils.projects.finished_project_data import ProjectData
from wbfm.utils.general.high_performance_pandas import get_names_from_df
from wbfm.utils.neuron_matching.utils_candidate_matches import rename_columns_using_matching


from wbfm.utils.performance.comparing_ground_truth import calculate_accuracy_from_dataframes, calc_accuracy_of_pipeline_steps



fname_gt = "/lisc/scratch/neurobiology/zimmer/fieseler/wbfm_projects_future/flavell_data/images_for_charlie/flavell_data.nwb"
project_data_gt = ProjectData.load_final_project_data(fname_gt)

fname_res = "/lisc/scratch/neurobiology/zimmer/schwartz/traces_mit_debug_embed/2025_07_01trial_28/project_config.yaml"
project_data_res = ProjectData.load_final_project_data(fname_res)




df_res = project_data_res.final_tracks
df_gt = project_data_gt.final_tracks
"""df_res = df_res.xs('raw_segmentation_id', axis=1, level=1)
df_gt = df_gt.xs('raw_segmentation_id', axis=1, level=1)"""

#df_gt = df_gt.iloc[1:1599]


df_res_renamed = rename_columns_using_matching(df_gt, df_res, column='raw_segmentation_id')

print(df_res_renamed)
print(df_gt)

column_names=['raw_segmentation_id']
df_acc_pipeline = calculate_accuracy_from_dataframes(df_gt, df_res, column_names)
print(df_acc_pipeline)

