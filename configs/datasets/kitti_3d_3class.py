# dataset settings
dataset_type = "KittiDataset"
data_root = "data/kitti_object3d/"
class_names = ["Pedestrian", "Cyclist", "Car"]
point_cloud_range = [-70.4, -40, -3, 70.4, 40, 1]
input_modality = dict(use_lidar=True, use_camera=False)
metainfo = dict(classes=class_names)
backend_args = None


train_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=4,  # x, y, z, intensity
        use_dim=4,
        backend_args=backend_args,
    ),
    dict(type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True),
    dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
    dict(
        type="FilterGroundPlaneRANSAC",
        num_iterations=1000,
        ransac_n=3,
        distance_threshold=0.2,
    ),
    dict(
        type="Pack3DDetInputsWithGroundInfo",
        keys=["points", "gt_bboxes_3d", "gt_labels_3d", "plane", "ground_point_mask"],
    ),
]

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="kitti_infos_train.pkl",
        data_prefix=dict(pts="training/velodyne"),
        pipeline=train_pipeline,
        modality=input_modality,
        test_mode=False,
        metainfo=metainfo,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d="LiDAR",
        backend_args=backend_args,
    ),
)
