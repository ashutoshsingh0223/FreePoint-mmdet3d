model = dict(
    type="FreePointPointNet2",
    feature_extractor_callable="features_after_encoding",
    input_dict=dict(num_neighbours=8, pooling="max"),
    feature_extractor=dict(
        type="ProposalContrastModel",
        backbone=dict(
            type="PointNet2SAMSG",
            in_channels=4,
            num_points=(4096, 1024, 256, 64),
            radii=((0.1, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0)),
            num_samples=((16, 32), (16, 32), (16, 32), (16, 32)),
            sa_channels=(
                ((16, 16, 32), (32, 32, 64)),
                ((64, 64, 128), (64, 96, 128)),
                ((128, 196, 256), (128, 196, 256)),
                ((256, 256, 512), (256, 384, 512)),
            ),
            fps_mods=(("D-FPS"), ("D-FPS"), ("D-FPS"), ("D-FPS")),
            fps_sample_range_lists=((-1), (-1), (-1), (-1)),
            aggregation_channels=(None, None, None, None),
            dilated_group=(False, False, False, False),
            out_indices=(0, 1, 2, 3),
            norm_cfg=dict(type="BN2d", eps=1e-3, momentum=0.1),
            sa_cfg=dict(
                type="PointSAModuleMSG",
                pool_mod="max",
                use_xyz=True,
                normalize_xyz=False,
            ),
        ),
        neck=dict(
            type="PointNetFPNeck",
            fp_channels=(
                (1536, 512, 512),
                (768, 512, 512),
                (608, 256, 256),
                (257, 128, 128),
            ),
        ),
        self_sup_head=dict(
            type="SelfSupHead",
            embed_layer_dims=[128, 128, 128],
            radius=1.0,
            sample_nums=16,
        ),
        cluster_prediction_head=dict(
            type="ClusterPrediction",
            embed_layer_dims=[128, 128],
        ),
        decode_head=dict(
            type="ProposalContrastDecodeHead",
            losses=[
                dict(
                    type="SSLInstanceLoss",
                    criterion="cross_entropy",
                    temperature=0.1,
                    loss_weight=1.0,
                ),
                dict(
                    type="SSLClusterLoss",
                    sinkhorn_iterations=3,
                    epsilon=0.03,
                    temperature=0.1,
                    loss_weight=1.0,
                ),
            ],
        ),
        n_proposals=1024,
    ),
)
