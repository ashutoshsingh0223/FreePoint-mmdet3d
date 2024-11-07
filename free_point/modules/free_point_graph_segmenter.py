from typing import List

try:
    import rama_py
except ImportError:
    from torch_cluster import graclus_cluster

import torch
from mmdet3d.registry import MODELS
from mmengine.model import BaseModule
from torch import Tensor


@MODELS.register_module(name="FreePointGraphSegmenter", force=True)
class FreePointGraphSegmenter(BaseModule):
    def __init__(
        self,
        rama_py_solver_mode: str = "PD",
        num_times: int = 10,
        init_cfg: dict | List[dict] | None = None,
    ):
        super().__init__(init_cfg)

        self.use_rama = True
        try:
            self.rama_py_solver_mode = rama_py_solver_mode
            self.num_times = num_times
            self.opts = rama_py.multicut_solver_options(self.rama_py_solver_mode)
            self.opts.dump_timeline = False  # Set to true to get intermediate results.
            self.opts.verbose = False
        except Exception as e:
            print(e)
            self.use_rama = False

    def perform_graph_cut(
        self, affinity_matrix: Tensor, adjacency_matrix: Tensor
    ) -> Tensor:
        """
        Perform graph cut using RAMA. The `affinity_matrix` here as no self connections
        i.e. the diagonal elements are zero.

        Args:
            affinity_matrix (Tensor): Affinity matrix. Shape: [N, N]
            adjacency_matrix (Tensor): Adjaceny matrix. Shape: [N, N]

        Returns:
            Tensor: Node label ids. Shape: [N,]
        """
        num_nodes = affinity_matrix.size(0)
        edge_list = adjacency_matrix.nonzero().t().to(torch.int32)
        edge_weights = affinity_matrix[edge_list[0], edge_list[1]].to(torch.float32)

        if self.use_rama:
            node_labels = torch.ones(num_nodes, device=edge_list.device).to(torch.int32)
            # rama_py cuda usage. (edge_list_from, edge_list_to, edge_weights,
            # node_labels_result_tensor, num_nodes, num_edges, device options)
            _ = rama_py.rama_cuda_gpu_pointers(
                edge_list[0].data_ptr(),
                edge_list[1].data_ptr(),
                edge_weights.data_ptr(),
                node_labels.data_ptr(),
                num_nodes,
                edge_list.shape[1],
                edge_list.device.index,
                self.opts,
            )
        else:
            node_labels = graclus_cluster(
                edge_list[0], edge_list[1], num_nodes=num_nodes
            ).to(edge_list.device, torch.int32)
        return node_labels

    def forward(
        self, affinity_matrix_batch: Tensor, adjacency_matrix_batch: Tensor
    ) -> Tensor:
        """
        Segment the affinity matrix using rama.py: https://arxiv.org/pdf/2109.01838.

        1. Perform segmentation on each affinity matrix in the batch num_times.
        2. Store the segmentation label ids.
        3. Construct A_id according to ID-as-a-Feature from
           FreePoint: https://arxiv.org/pdf/2305.06973.
        4. Do final segmentation on A_id and return the node labels.

        Args:
            affinity_matrix_batch (Tensor): Affinity matrix for graph. Shape: [B, N, N].
            adjacency_matrix_batch (Tensor): Adjaceny matrix for grpah. Shape: [B, N, N]
        Returns:
            Tensor: Node label ids for each batch element. Shape: [B, N]
        """
        batch_size, num_nodes = affinity_matrix_batch.shape[0:2]

        # Set diagonals to zero to remove all self connections in the graph.
        i = torch.arange(num_nodes)
        adjacency_matrix_batch[:, i, i] = 0
        # Convert to upper triangular matrix since its an undirected graph.
        adjacency_matrix_batch = torch.triu(adjacency_matrix_batch)

        node_labels_batch = []
        for batch_index in range(batch_size):
            affinity_matrix = affinity_matrix_batch[batch_index]
            adjacency_matrix = adjacency_matrix_batch[batch_index]
            node_label_ids = torch.zeros(
                self.num_times, num_nodes, device=affinity_matrix.device
            ).to(torch.int32)

            # Perform RAMA num_times and store label ids
            for rama_iter in range(self.num_times):
                # Store predicted node labels here.
                node_labels = self.perform_graph_cut(affinity_matrix, adjacency_matrix)
                node_label_ids[rama_iter] = node_labels

            # Construct ID-as-a-Feature Graph
            node_label_ids = node_label_ids.transpose(1, 0)
            a_id = (
                torch.sum(
                    (node_label_ids.unsqueeze(1) == node_label_ids).to(torch.float32),
                    axis=2,
                )
                / self.num_times
            )

            # Set diagonals of A_id to zero
            a_id[i, i] = 0
            # Convert to upper triangular to get edges.
            a_id = torch.triu(a_id)
            id_adjacency = torch.zeros(
                num_nodes, num_nodes, device=a_id.device, dtype=torch.int32
            )
            id_adjacency[a_id > 0] = 1
            # Finally perform cut on A_id
            final_node_labels = self.perform_graph_cut(a_id, id_adjacency)

            node_labels_batch.append(final_node_labels)

        node_labels_batch = torch.stack(node_labels_batch)
        return node_labels_batch
