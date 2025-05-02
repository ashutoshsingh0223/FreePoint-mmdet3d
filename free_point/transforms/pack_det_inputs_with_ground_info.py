from mmdet3d.datasets.transforms.formating import Pack3DDetInputs
from mmdet3d.registry import TRANSFORMS


@TRANSFORMS.register_module(force=True)
class Pack3DDetInputsWithGroundInfo(Pack3DDetInputs):
    INPUTS_KEYS = [
        "points",
        "plane",
        "ground_point_mask",
    ]

    # Add two more meta keys -> "tranformation_meta", "tranformation_meta_aug"
    def __init__(
        self,
        keys: tuple,
        meta_keys: tuple = (
            "img_path",
            "ori_shape",
            "img_shape",
            "lidar2img",
            "depth2img",
            "cam2img",
            "pad_shape",
            "scale_factor",
            "flip",
            "pcd_horizontal_flip",
            "pcd_vertical_flip",
            "box_mode_3d",
            "box_type_3d",
            "img_norm_cfg",
            "num_pts_feats",
            "pcd_trans",
            "sample_idx",
            "pcd_scale_factor",
            "pcd_rotation",
            "pcd_rotation_angle",
            "lidar_path",
            "transformation_3d_flow",
            "trans_mat",
            "affine_aug",
            "sweep_img_metas",
            "ori_cam2img",
            "cam2global",
            "crop_offset",
            "img_crop_offset",
            "resize_img_shape",
            "lidar2cam",
            "ori_lidar2img",
            "num_ref_frames",
            "num_views",
            "ego2global",
            "axis_align_matrix",
            "tranformation_meta",
        ),
    ) -> None:
        super().__init__(keys, meta_keys)
