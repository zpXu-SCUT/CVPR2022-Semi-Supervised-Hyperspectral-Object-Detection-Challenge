model = dict(
    type='CascadeRCNN',
    backbone=dict(
        type='MobileNetV2HSI',
        out_indices=(1, 2, 4, 6),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True),
    neck=dict(
        type='FPN',
        in_channels=[24, 32, 96, 320],
        out_channels=256,
        num_outs=5,
        norm_cfg=dict(type='BN', requires_grad=True)),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0)),
    roi_head=dict(
        type='CascadeRoIHead',
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='Shared4Conv1FCBBoxHead',
                in_channels=256,
                fc_out_channels=512,
                roi_feat_size=7,
                num_classes=3,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared4Conv1FCBBoxHead',
                in_channels=256,
                fc_out_channels=512,
                roi_feat_size=7,
                num_classes=3,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared4Conv1FCBBoxHead',
                in_channels=256,
                fc_out_channels=512,
                roi_feat_size=7,
                num_classes=3,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ]),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)
        ]),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='soft_nms', iou_threshold=0.45, min_score=0.01),
            max_per_img=100)),
    pretrained=None)


dataset_type = 'RhsiDataset'
data_root = 'data/RIT-HS20/'

# albu_train_transforms 阶段二新加入的数据增强方法
albu_train_transforms = [
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.0625,
        scale_limit=0.0,
        rotate_limit=0,
        interpolation=1,
        p=0.5),
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.2),
    dict(type='RandomResizedCrop', height=188, width=1600, p=0.2),
    dict(type='Sharpen', p=0.5),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1),
            dict(type='MedianBlur', blur_limit=3, p=1),
            dict(type='GaussianBlur', blur_limit=3, p=1)
        ],
        p=0.1)
]
train_pipeline = [
    dict(type='LoadMaskedHSIImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=[(1600, 188), (1600, 189)], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='BrightnessTransform', level=3),
    dict(
        type='Albu',
        transforms=[
            dict(
                type='ShiftScaleRotate',
                shift_limit=0.0625,
                scale_limit=0.0,
                rotate_limit=0,
                interpolation=1,
                p=0.5),
            dict(
                type='RandomBrightnessContrast',
                brightness_limit=[0.1, 0.3],
                contrast_limit=[0.1, 0.3],
                p=0.2),
            dict(type='RandomResizedCrop', height=188, width=1600, p=0.2),
            dict(type='Sharpen', p=0.5),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='Blur', blur_limit=3, p=1),
                    dict(type='MedianBlur', blur_limit=3, p=1),
                    dict(type='GaussianBlur', blur_limit=3, p=1)
                ],
                p=0.1)
        ],
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap=dict(img='image', gt_masks='masks', gt_bboxes='bboxes'),
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(
        type='CutOut',
        n_holes=1,
        cutout_ratio=[[0.15, 0.15], [0.1, 0.1]],
        fill_in=[
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0
        ]),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadMaskedHSIImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(1600, 188), (1600, 189)],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type='RhsiDataset',
            ann_file=
            'data/RIT-HS20/annotationsjson/all_pseudo.json',
            img_prefix=
            'data/RIT-HS20/spectral/',
            mask_prefix=
            'data/RIT-HS20/masks/',
            pipeline=[
                dict(type='LoadMaskedHSIImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(
                    type='Resize',
                    img_scale=[(1600, 188), (1600, 189)],
                    keep_ratio=True),
                dict(type='RandomFlip', flip_ratio=0.5),
                dict(type='BrightnessTransform', level=3),
                dict(
                    type='Albu',
                    transforms=[
                        dict(
                            type='ShiftScaleRotate',
                            shift_limit=0.0625,
                            scale_limit=0.0,
                            rotate_limit=0,
                            interpolation=1,
                            p=0.5),
                        dict(
                            type='RandomBrightnessContrast',
                            brightness_limit=[0.1, 0.3],
                            contrast_limit=[0.1, 0.3],
                            p=0.2),
                        dict(
                            type='RandomResizedCrop',
                            height=188,
                            width=1600,
                            p=0.2),
                        dict(type='Sharpen', p=0.5),
                        dict(
                            type='OneOf',
                            transforms=[
                                dict(type='Blur', blur_limit=3, p=1),
                                dict(type='MedianBlur', blur_limit=3, p=1),
                                dict(type='GaussianBlur', blur_limit=3, p=1)
                            ],
                            p=0.1)
                    ],
                    bbox_params=dict(
                        type='BboxParams',
                        format='pascal_voc',
                        label_fields=['gt_labels'],
                        min_visibility=0.0,
                        filter_lost_elements=True),
                    keymap=dict(
                        img='image', gt_masks='masks', gt_bboxes='bboxes'),
                    update_pad_shape=False,
                    skip_img_without_anno=True),
                dict(
                    type='CutOut',
                    n_holes=1,
                    cutout_ratio=[ [0.1, 0.1],[0.15,0.15]],
                    fill_in=[
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                    ]),
                dict(type='Pad', size_divisor=32),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
            ])),
    val=dict(
        type='RhsiDataset',
        ann_file=
        'data/RIT-HS20/annotationsjson/test.json',
        img_prefix=
        'data/RIT-HS20/spectral/',
        mask_prefix=
        'data/RIT-HS20/masks/',
        pipeline=[
            dict(type='LoadMaskedHSIImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=[(1600, 188), (1600, 189)],
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='RhsiDataset',
        ann_file=
        'data/RIT-HS20/annotationsjson/instances_test_id.json',
        img_prefix=
        'data/RIT-HS20/spectral/',
        mask_prefix=
        'data/RIT-HS20/masks/',
        pipeline=[
            dict(type='LoadMaskedHSIImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=[(1600, 188), (1600, 189)],
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
evaluation = dict(interval=1, metric='bbox', classwise=True)
optimizer = dict(
    type='SGD',
    lr=0.02,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_cfg=dict(norm_decay_mult=0))
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.1,
    step=[10, 20, 25])
runner = dict(type='EpochBasedRunner', max_epochs=30)
checkpoint_config = dict(interval=1)
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'data/first.pth'
resume_from = None
workflow = [('train', 1)]
norm_cfg = dict(type='BN', requires_grad=True)
work_dir = './work_dirs/albu-001'
auto_resume = False
gpu_ids = [1]
