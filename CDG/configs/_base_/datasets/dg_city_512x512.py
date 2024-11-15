_base_ = [
    "./cityscapes_512x512.py",
    "./gta_512x512.py",
    "./syn_512x512.py",
    "./bdd100k_512x512.py",
    "./mapillary_512x512.py",
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=False,
    pin_memory=False,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset={{_base_.train_cityscapes}},
)
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="ConcatDataset",
        datasets=[
            # {{_base_.val_cityscapes}},
            # {{_base_.val_bdd}},
            {{_base_.val_mapillary}},
            # {{_base_.val_gta}},
            # {{_base_.val_syn}},
        ],
    ),
)
test_dataloader = val_dataloader
val_evaluator = dict(
    # type="DGIoUMetric", iou_metrics=["mIoU"], dataset_keys=["citys", "map", "bdd"]
    type="DGIoUMetric", iou_metrics=["mIoU"], dataset_keys=["map"]
)
test_evaluator=val_evaluator
