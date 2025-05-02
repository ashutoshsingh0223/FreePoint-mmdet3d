default_scope = "mmdet3d"

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=20),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(type="CheckpointHook", interval=1),
    sampler_seed=dict(type="DistSamplerSeedHook"),
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)


custom_imports = dict(
    imports=[
        "free_point.models.free_point",
        "free_point.modules.free_point_graph_constructor",
        "free_point.modules.free_point_graph_segmenter",
        "free_point.modules.feature_extractor",
        "free_point.transforms.filter_ground_plane_ransac",
        "free_point.transforms.pack_det_inputs_with_ground_info",
    ]
)

log_processor = dict(type="LogProcessor", window_size=50, by_epoch=True)
log_level = "INFO"
load_from = None
resume = False
