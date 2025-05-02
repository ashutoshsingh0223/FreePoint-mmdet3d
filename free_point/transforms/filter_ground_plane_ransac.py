from typing import Dict, List, Tuple

import open3d as o3d
import torch
from mmcv.transforms import BaseTransform
from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures.points import LiDARPoints


@TRANSFORMS.register_module()
class FilterGroundPlaneRANSAC(BaseTransform):
    def __init__(
        self,
        num_iterations: int = 1000,
        ransac_n: int = 3,
        distance_threshold: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_iterations = num_iterations
        self.ransac_n = ransac_n
        self.distance_threshold = distance_threshold

    def transform(self, results: Dict) -> Dict | Tuple[List, List] | None:
        points: LiDARPoints = results["points"]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.tensor.cpu().numpy()[:, :3])
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=self.distance_threshold,
            ransac_n=self.ransac_n,
            num_iterations=self.num_iterations,
        )
        ground_point_mask = torch.zeros(points.tensor.shape[0], dtype=torch.bool)
        ground_point_mask[inliers] = True

        results["plane"] = torch.from_numpy(plane_model)
        results["ground_point_mask"] = ground_point_mask
        return results
