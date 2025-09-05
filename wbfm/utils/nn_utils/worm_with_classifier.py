import os.path
from dataclasses import dataclass
from abc import ABC, abstractmethod
from re import Match
from typing import Dict

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from tqdm.auto import tqdm

from wbfm.utils.external.custom_errors import NoMatchesError
from wbfm.utils.neuron_matching.class_frame_pair import num_possible_matches_between_two_frames
from wbfm.utils.neuron_matching.class_reference_frame import ReferenceFrame
from wbfm.utils.neuron_matching.matches_class import MatchesWithConfidence
from wbfm.utils.nn_utils.model_image_classifier import NeuronEmbeddingModel
from wbfm.utils.nn_utils.superglue import SuperGlueModel, SuperGlueUnpackerWithTemplate
from wbfm.utils.projects.finished_project_data import ProjectData, template_matches_to_dataframe
from wbfm.utils.general.hardcoded_paths import load_hardcoded_neural_network_paths


# TODO: also save hyperparameters (doesn't work in jupyter notebooks)
HPARAMS = dict(num_classes=127)

@dataclass
class FeatureSpaceTemplateMatcher(ABC):
    """Abstract class for tracking neurons via matching in some feature space based on a template"""

    template_frame: ReferenceFrame

    # To be optimized
    confidence_gamma: float = 100.0
    cdist_p: int = 2

    @abstractmethod
    def match_target_frame(self, target_frame: ReferenceFrame) -> MatchesWithConfidence:
        pass

    def check_target_frame_can_be_matched(self, target_frame: ReferenceFrame) -> bool:
        # See FramePair.check_both_frames_valid
        num_possible_matches = num_possible_matches_between_two_frames(self.template_frame, target_frame)
        is_valid = True
        if num_possible_matches <= 1 or np.isnan(num_possible_matches):
            is_valid = False
        return is_valid

    def _match_using_linear_sum_assignment(self, query_embedding: torch.Tensor) -> MatchesWithConfidence:

        distances = torch.cdist(self.embedding_template, query_embedding, p=self.cdist_p)
        conf_matrix = torch.nan_to_num(torch.softmax(self.confidence_gamma / distances, dim=0), nan=1.0)

        matches = linear_sum_assignment(conf_matrix, maximize=True)
        matches = [[m0, m1] for (m0, m1) in zip(matches[0], matches[1])]
        matches = np.array(matches)
        conf = np.array([np.tanh(conf_matrix[i0, i1]) for i0, i1 in matches])
        matches_with_conf = [[m[0], m[1], c] for m, c in zip(matches, conf)]

        return MatchesWithConfidence.matches_from_array(matches_with_conf)

class DirectFeatureSpaceTemplateMatcher(FeatureSpaceTemplateMatcher):
    """Direct matching in the feature space without re-embedding or other postprocessing"""
    embedding_template: torch.tensor = None

    def __post_init__(self):
        self.embedding_template = torch.from_numpy(self.template_frame.all_features)

    def match_target_frame(self, target_frame: ReferenceFrame) -> MatchesWithConfidence:
        """
        Matches target frame (custom class) to the initialized template frame of this tracker

        Parameters
        ----------
        target_frame

        Returns
        -------
        Matches with confidence (n_matches x 3)

        """
        if not self.check_target_frame_can_be_matched(target_frame):
            raise NoMatchesError("Target frame cannot be matched")

        with torch.no_grad():
            query_embedding = torch.from_numpy(target_frame.all_features)
            matches_with_conf = self._match_using_linear_sum_assignment(query_embedding)
        return matches_with_conf


class ReembeddedFeatureSpaceTemplateMatcher(FeatureSpaceTemplateMatcher):
    """
    Tracks neurons using a feature-space embedding and pre-calculated Frame objects

    The feature space in the Frame objects is re-embedded using a pretrained superglue network
    """

    model_type: callable = NeuronEmbeddingModel
    model: NeuronEmbeddingModel = None
    path_to_model: str = None
    hparams: dict = None

    embedding_template: torch.tensor = None
    labels_template: list = None

    def __post_init__(self):
        if self.path_to_model is None:
            # Load hardcoded path to model
            path_dict = load_hardcoded_neural_network_paths()
            superglue_parent_folder = path_dict['tracking_paths']['model_parent_folder']
            superglue_model_name = path_dict['tracking_paths']['global_tracking_model_name']
            superglue_path = os.path.join(superglue_parent_folder, superglue_model_name)

            if os.path.exists(superglue_path):
                self.path_to_model = superglue_path
            else:
                raise FileNotFoundError(superglue_path)

        if self.hparams is None:
            self.hparams = HPARAMS
        if self.model is None:
            # TODO: just load Siamese directly, and ignore the number of classes?
            self.model = self.model_type.load_from_checkpoint(checkpoint_path=self.path_to_model, **self.hparams)

        self.initialize_template()

    def initialize_template(self, template_frame: ReferenceFrame = None):
        if template_frame is None:
            template_frame = self.template_frame
        else:
            self.template_frame = template_frame
        if template_frame is None:
            raise NotImplementedError("Must pass template_frame or initialize self.t_template")

        features = torch.from_numpy(template_frame.all_features)
        self.embedding_template = self.model.embed(features.to(self.model.device)).type(torch.float)
        # TODO: better naming?
        self.labels_template = list(range(features.shape[0]))

    def match_target_frame(self, target_frame: ReferenceFrame) -> MatchesWithConfidence:
        """
        Matches target frame (custom class) to the initialized template frame of this tracker

        Parameters
        ----------
        target_frame

        Returns
        -------
        Matches with confidence (n_matches x 3)

        """
        if not self.check_target_frame_can_be_matched(target_frame):
            raise NoMatchesError("Target frame cannot be matched")

        with torch.no_grad():
            query_embedding = self.embed_target_frame(target_frame)
            matches_with_conf = self._match_using_linear_sum_assignment(query_embedding)
        return matches_with_conf

    def embed_target_frame(self, target_frame):
        query_features = torch.tensor(target_frame.all_features).to(self.model.device)
        query_embedding = self.model.embed(query_features).type(torch.float)
        return query_embedding

    def __repr__(self):
        return f"Worm Tracker based on network: {self.path_to_model}"


@dataclass
class FullVideoTrackerWithTemplate:
    """
    Simpler reimplementation of FullVideoNeuronTrackerSuperglue that uses the FeatureSpaceTemplateMatcher class instead of a direct neural network
    """
    t_template: int
    time_dict_of_matcher_classes: Dict[int, FeatureSpaceTemplateMatcher]
    time_dict_of_matcher_classes: Dict[int, FeatureSpaceTemplateMatcher]

    def match_target_frame(self, t_target) -> MatchesWithConfidence:
        template_matcher = self.time_dict_of_matcher_classes[self.t_template]
        target_frame = self.time_dict_of_matcher_classes[t_target].template_frame

        matches_with_conf = template_matcher.match_target_frame(target_frame)
        return matches_with_conf


@dataclass
class SuperGlueFullVideoTrackerWithTemplate:
    """
    Tracks neurons using a superglue network and pre-calculated Frame objects

    Contains information (Frame objects) for the entire video

    Designed to be used for non-adjacent frame matching
    """
    model: SuperGlueModel = None
    superglue_unpacker: SuperGlueUnpackerWithTemplate = None  # Note: contains the reference frame

    path_to_model: str = None

    def __post_init__(self):
        if self.path_to_model is None:
            # Load hardcoded path to model
            path_dict = load_hardcoded_neural_network_paths()
            superglue_parent_folder = path_dict['tracking_paths']['model_parent_folder']
            superglue_model_name = path_dict['tracking_paths']['global_tracking_model_name']
            superglue_path = os.path.join(superglue_parent_folder, superglue_model_name)

            if os.path.exists(superglue_path):
                self.path_to_model = superglue_path
            else:
                raise FileNotFoundError(superglue_path)
        if self.model is None:
            self.model = SuperGlueModel.load_from_checkpoint(checkpoint_path=self.path_to_model)
        self.model.eval()

    def match_target_frame(self, target_frame: ReferenceFrame):

        with torch.no_grad():
            data, is_valid_frame = self.superglue_unpacker.convert_single_frame_to_superglue_format(target_frame,
                                                                                                    use_gt_matches=False)
            if is_valid_frame:
                data = self.superglue_unpacker.expand_all_data(data, device=self.model.device)
                matches_with_conf = self.model.superglue.match_and_output_list(data)
            else:
                matches_with_conf = []
        return matches_with_conf

    def embed_target_frame(self, target_frame: ReferenceFrame):
        """For debugging: no matching, just returns the features"""
        with torch.no_grad():
            data, is_valid_frame = self.superglue_unpacker.convert_single_frame_to_superglue_format(target_frame,
                                                                                                    use_gt_matches=False)
            if is_valid_frame:
                data = self.superglue_unpacker.expand_all_data(data, device=self.model.device)
                _, mdesc = self.model.superglue.embed_descriptors_and_keypoints(data)
            else:
                mdesc = None
        return mdesc

    def embed_target_frame_debug(self, target_frame: ReferenceFrame):
        """For debugging: no matching, just returns the features"""
        with torch.no_grad():
            data, is_valid_frame = self.superglue_unpacker.convert_single_frame_to_superglue_format(target_frame,
                                                                                                    use_gt_matches=False)
            if is_valid_frame:
                data = self.superglue_unpacker.expand_all_data(data, device=self.model.device)
                out = self.model.superglue.embed_descriptors_and_keypoints_debug(data)
            else:
                out = None
        return out

    def match_two_time_points(self, t0: int, t1: int) -> MatchesWithConfidence:
        with torch.no_grad():
            data, is_valid_pair = self.superglue_unpacker.convert_frames_to_superglue_format(t0, t1,
                                                                                             use_gt_matches=False)
            if not is_valid_pair:
                matches_with_conf = []
            else:
                data = self.superglue_unpacker.expand_all_data(data, device=self.model.device)
                matches_with_conf = self.model.superglue.match_and_output_list(data)
        return MatchesWithConfidence.matches_from_array(matches_with_conf)

    def match_two_time_points_return_full_loss(self, t0: int, t1: int):
        """
        Like match_two_time_points() but this does a full forward pass

        Note that this needs ground truth matches
        """
        with torch.no_grad():
            data, is_valid_pair = self.superglue_unpacker.convert_frames_to_superglue_format(t0, t1,
                                                                                             use_gt_matches=True)
            if not is_valid_pair:
                result = {}
            else:
                data = self.superglue_unpacker.expand_all_data(data, device=self.model.device)
                result = self.model(data)
        return result, data

    def __repr__(self):
        return f"Worm Tracker based on superglue network"


def track_using_template(all_frames, num_frames, project_data, tracker: SuperGlueFullVideoTrackerWithTemplate):
    """
    Tracks all the frames in all_frames using the tracker class.

    Project data should match all_frames, and is used to add information about centroids to the final dataframe
    However, the tracker object (which is initialized with a project_data object), does not need to be the same dataset

    Parameters
    ----------
    all_frames
    num_frames
    project_data
    tracker

    Returns
    -------

    """
    all_matches = []
    for t in tqdm(range(num_frames), leave=False):
        # Note: if there are no neurons, this list should be empty
        matches_with_conf = tracker.match_target_frame(all_frames[t])

        all_matches.append(matches_with_conf)
    df = template_matches_to_dataframe(project_data, all_matches)
    return df


def _unpack_project_for_global_tracking(DEBUG, project_cfg):
    project_data = ProjectData.load_final_project_data_from_config(project_cfg, to_load_frames=True)
    tracking_cfg = project_data.project_config.get_tracking_config()
    t_template = tracking_cfg.config['final_3d_tracks']['template_time_point']
    use_multiple_templates = tracking_cfg.config['leifer_params']['use_multiple_templates']
    num_random_templates = tracking_cfg.config['leifer_params']['num_random_templates']
    num_frames = project_data.project_config.get_num_frames_robust()
    if DEBUG:
        num_frames = 3
    all_frames = project_data.raw_frames
    return all_frames, num_frames, num_random_templates, project_data, t_template, tracking_cfg, use_multiple_templates
