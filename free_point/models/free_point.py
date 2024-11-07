from typing import Optional

import torch
from mmcv.ops import PointsSampler
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.utils import ConfigType
from mmengine.model import BaseModel
from torch_geometric.nn import knn


@MODELS.register_module(name="FreePointPointNet2", force=True)
class FreePointPointNet2(BaseModel):
    def __init__(
        self,
        feature_extractor: ConfigType,
        feature_extractor_callable: Optional[str] = None,
        input_dict=Optional[dict],
        graph_contructor: ConfigType = dict(
            type="FreePointGraphConstructor",
            weight_feature=0,
            weight_xyz=1,
            max_edges_per_node=8,
        ),
        graph_segmenter: ConfigType = dict(
            type="FreePointGraphSegmenter", rama_py_solver_mode="PD", num_times=5
        ),
        num_points: int = 20000,
        k2: int = 8,
        data_preprocessor: dict | None = None,
        init_cfg: dict | None = None,
    ):
        super().__init__(data_preprocessor, init_cfg)

        self.encoder = MODELS.build(feature_extractor)
        self.feature_extractor_callable = feature_extractor_callable
        self.input_dict = input_dict

        self.num_points = num_points
        self.k2 = k2

        self.fps_mode_list = ["D-FPS"]
        self.fps_sampler = PointsSampler([int(self.num_points / 2)], self.fps_mode_list)

        self.graph_contructor = MODELS.build(graph_contructor)
        self.graph_segmenter = MODELS.build(graph_segmenter)

    def extract_feat(self, batch_inputs_dict: dict):
        if self.feature_extractor_callable:
            batch_inputs_dict.update(self.input_dict)
            xyz, features = getattr(self.encoder, self.feature_extractor_callable)(
                batch_inputs_dict
            )
        else:
            xyz, features = self.encoder(batch_inputs_dict)
        return xyz, features

    def forward(
        self, batch_inputs_dict: dict, batch_data_samples: SampleList, **kwargs
    ):
        batch_size = len(batch_inputs_dict["points"])
        original_points = batch_inputs_dict["points"]

        # [B, N, 3], [B, N, c] or [B, c, N]
        encoder_xyz, encoder_features = self.extract_feat(batch_inputs_dict)

        if encoder_features.shape[1] != encoder_xyz.shape[1]:
            encoder_features = encoder_features.transpose(1, 2)

        assert encoder_features.shape[1] == encoder_xyz.shape[1]

        sampled_indices = (
            self.fps_sampler(
                torch.stack(batch_inputs_dict["points"])[:, :, :3], features=None
            )
            .long()
            .squeeze(-1)
        )

        xyz = []
        features = []
        for batch_index in range(batch_size):
            xyz.append(original_points[batch_index][sampled_indices[batch_index], :3])
            features.append(
                encoder_features[batch_index][sampled_indices[batch_index], :]
            )

        xyz = torch.stack(xyz)
        features = torch.stack(features)

        affinity_matrix_batch, adjacency_matrix_batch = self.graph_contructor(
            xyz, features
        )
        xyz_label_ids = self.graph_segmenter(
            affinity_matrix_batch, adjacency_matrix_batch
        )

        original_points_label_ids = []
        for batch_index in range(batch_size):
            # xyz[batch_index] -> [M, 3]
            # original_points[batch_index] -> [N, 3]; N >= M
            # target_indices -> [N*self.k2, ]
            source_target_indices = knn(
                xyz[batch_index], original_points[batch_index][:, :3], k=self.k2
            )
            target_indices = source_target_indices[1]

            # label_ids -> [N*self.k2, ]
            label_ids = xyz_label_ids[batch_index][target_indices]
            # target_label_ids -> [N, self.k2]
            label_ids = label_ids.reshape(original_points[batch_index].size(0), self.k2)
            # Vote pooling
            label_ids = torch.mode(label_ids, dim=1).values
            original_points_label_ids.append(label_ids)

        original_points_label_ids = torch.stack(original_points_label_ids)
        return original_points_label_ids
