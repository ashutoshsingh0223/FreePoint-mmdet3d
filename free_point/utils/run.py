from pathlib import Path
from typing import Optional

import numpy as np
import torch
from mmdet3d.registry import DATASETS, MODELS
from mmdet3d.utils import register_all_modules

from configs.datasets.kitti_3d_3class import train_dataloader  # noqa
from configs.models.free_point_pointnet2 import model as model_config  # noqa


def run(
    out_dir: str,
    checkpoint_path: Optional[str] = None,
    weight_feature: float = 0.5,
    weight_xyz: float = 0.5,
    k1: int = 4,
    k2: int = 4,
    num_points: int = 20000,
    num_iters_for_graph_cut: int = 5,
    ransac_distance: float = 0.175,
):
    global model_config, train_dataloader

    model_config["graph_contructor"] = dict(
        type="FreePointGraphConstructor",
        weight_feature=weight_feature,
        weight_xyz=weight_xyz,
        max_edges_per_node=k1,
    )
    model_config["graph_segmenter"] = dict(
        type="FreePointGraphSegmenter",
        rama_py_solver_mode="PD",
        num_times=num_iters_for_graph_cut,
    )

    model_config["k2"] = k2
    model_config["num_points"] = num_points

    register_all_modules()
    train_dataloader["batch_size"] = 1

    dataset = train_dataloader["dataset"]
    if dataset["pipeline"][-2]["type"] == "FilterGroundPlaneRANSAC":
        dataset["pipeline"][-2]["distance_threshold"] = ransac_distance

    dataset = DATASETS.build(dataset)

    # out_dir = Path(dataset.data_prefix["pts"]) / out_dir
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    model = MODELS.build(model_config)
    model = model.eval()
    model = model.to("cuda")

    if checkpoint_path:
        state_dict = torch.load(checkpoint_path)
        print(model.encoder.load_state_dict(state_dict["state_dict"], strict=False))

    # pcd = o3d.geometry.PointCloud()
    r = dataset[4]
    batch_inputs_dict = dict()
    for k in r["inputs"]:
        if k == "points":
            batch_inputs_dict[k] = [r["inputs"][k][:, :4].to("cuda")]
        else:
            batch_inputs_dict[k] = [r["inputs"][k].to("cuda")]

    with torch.no_grad():
        labels = model(batch_inputs_dict, None)
        labels = labels[0].cpu().numpy()

    name = Path(r["data_samples"].lidar_path).stem
    np.save(str(out_dir / f"{name}.npy"), labels)
