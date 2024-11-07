from typing import List

import torch
import torch.nn.functional as torch_functional
from mmdet3d.registry import MODELS
from mmengine.model import BaseModule
from torch import Tensor


@MODELS.register_module(name="FreePointGraphConstructor", force=True)
class FreePointGraphConstructor(BaseModule):
    def __init__(
        self,
        weight_xyz: float,
        weight_feature: float,
        max_edges_per_node: int = 4,
        init_cfg: dict | List[dict] | None = None,
    ):
        """
        Affinity matrix and graph construction based on
        FreePoint: https://arxiv.org/pdf/2305.06973


        NOTE: From FreePoint we want to use only k1 neighbours to construct the
        graph for each node. But to keep the indices of nodes in the graph
        edges consistent, first contruct the whole graph
        and then set edges other than the k1 neighbours to zero when creating
        `a_feature` and `a_xyz`. As the paper does not tell us how it measures distances
        to get these `k1` neighbours, so we use euclidean distances.

        Args:
            weight_xyz (float): Weight to be used for xyz distance affinity matrix.
            weight_feature (float): Weight to be used for feature affinity matrix.
            max_edges_per_node (int, optional): `k1` from FreePoint, the number of
                                                neighbours to use while constructing
                                                the graph. Defaults to 4.
            init_cfg (dict | List[dict] | None): init config. Defaults to None.
        """
        super().__init__(init_cfg)
        self.weight_feature = weight_feature
        self.weight_xyz = weight_xyz
        self.max_edges_per_node = max_edges_per_node

    def forward(self, xyz: Tensor, features: Tensor) -> Tensor:
        """
        Generates the graph: "A" from FreePoint paper using xyz and features.

        Args:
            xyz (Tensor): xyz coordinates of points. Shape: [B, N, 3]
            features (Tensor): Features fo points. Shape: [B, N, C]
        Returns:
            Tensor: Containing the final affinity matrix. Shape: [B, N, N]
        """

        batch_size, num_points = xyz.shape[0:2]

        # Normalize the features to get the cosine similarity
        normalized_features = torch_functional.normalize(features, p=2, dim=2)

        # Compute the cosine similarity matrix for A_feature
        a_feature = torch.matmul(
            normalized_features, normalized_features.transpose(1, 2)
        )

        # Compute norm of cordinate distances in xyz
        xyz_diff = xyz.unsqueeze(2) - xyz
        a_xyz = -1 * torch.linalg.norm(xyz_diff, ord=2, dim=3)

        i = torch.arange(num_points)
        # Set diagonals of A_xyz to -torch.inf to avoid self reference in top_k
        a_xyz[:, i, i] = -torch.inf

        # Find nearest k1 neighbours for all nodes
        _, indices = torch.topk(a_xyz, self.max_edges_per_node, dim=2)

        # Prepare adjacency matrix
        fill = torch.ones(batch_size, num_points, num_points, dtype=torch.int32)
        # Initialize adjanceny matrix as zeros
        adj_matrix = torch.zeros(batch_size, num_points, num_points, dtype=torch.int32)
        # Fill adjancency matrix with ones for all k1 neighbours using `indices`.
        adj_matrix.scatter_(2, indices.cpu(), fill)
        # Convert adj_matrix to upper triangular since its an undirected graph.
        adj_matrix = torch.triu(adj_matrix)

        # Normalize both A_feature and A_xyz to have zero mean and unit variance.
        # Only active k1 neighbours being used for mean and std calculation.

        # Normalize A_feature
        # Select values only from k1 neighbours to measure.
        # Create temp_matrix and fill with nan
        temp = torch.zeros(batch_size, num_points, num_points)
        temp[:, :, :] = torch.nan
        # Fill temp with 1 wherever an edge exists i.e. for k1 neighbours.
        temp = temp.masked_fill(adj_matrix.bool(), 1.0)
        # Multiply to get values  in A_feature for all k1 neighbours.
        temp = temp.to(a_feature.device) * a_feature
        # Calculate mean and std while ignoring nan values.
        # Std and mean are calculated for all valid edges
        feat_mean = torch.nanmean(temp, dim=(1, 2), keepdim=True)
        feat_std = torch.sqrt(
            torch.nansum(
                torch.pow(temp - feat_mean, exponent=2), dim=(1, 2), keepdim=True
            )
            / (self.max_edges_per_node - 1)
        )

        a_feature = (a_feature - feat_mean) / (
            feat_std + torch.finfo(torch.float32).eps
        )

        # Bring adj_matrix to device
        adj_matrix = adj_matrix.to(a_xyz.device)

        # Normalize A_xyz. Std and mean are calculated for all valid edges
        temp[:, :, :] = torch.nan
        temp = temp.masked_fill(adj_matrix.bool(), 1.0)
        temp = temp * a_xyz
        dist_mean = torch.nanmean(temp, dim=(1, 2), keepdim=True)
        dist_std = torch.sqrt(
            torch.nansum(
                torch.pow(temp - dist_mean, exponent=2), dim=(1, 2), keepdim=True
            )
            / (self.max_edges_per_node - 1)
        )

        a_xyz = (a_xyz - dist_mean) / (dist_std + torch.finfo(torch.float32).eps)

        return (self.weight_feature * a_feature) + (self.weight_xyz * a_xyz), adj_matrix
