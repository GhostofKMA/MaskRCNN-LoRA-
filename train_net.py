import torch
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*autocast.*")
from detectron2.engine import DefaultTrainer, default_argument_parser, launch, default_setup
import detectron2.data.transforms as T
from detectron2.config import get_cfg
from detectron2.config import CfgNode as CN
from detectron2.data import DatasetCatalog
from detectron2 import model_zoo
from detectron2.data import DatasetMapper, build_detection_train_loader, build_detection_test_loader
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data.datasets import register_coco_instances
from core import backbone   

def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.INPUT.FORMAT = "RGB"
    cfg.MODEL.PIXEL_MEAN = [123.675, 116.280, 103.530]
    cfg.MODEL.PIXEL_STD = [58.395, 57.120, 57.375]
    cfg.INPUT.MIN_SIZE_TRAIN = (1024,)
    cfg.INPUT.MAX_SIZE_TRAIN = 1024
    cfg.INPUT.MIN_SIZE_TEST = 1024
    cfg.INPUT.MAX_SIZE_TEST = 1024
    cfg.INPUT.CROP.ENABLED = False
    cfg.MODEL.DEVICE = "cuda"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10
    cfg.MODEL.BACKBONE.CHECKPOINT = "weights/sam_vit_h_4b8939.pth"
    cfg.MODEL.BACKBONE.NAME = "ViTHugeBackbone"
    cfg.MODEL.RPN.IN_FEATURES = ["p2", "p3", "p4", "p5", "p6"]
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [
    [16, 32],       # P2 (Stride 4)
    [64, 128],      # P3 (Stride 8)
    [256, 512],     # P4 (Stride 16) 
    [512, 1024],    # P5 (Stride 32) 
    [1024, 2048]    # P6 (Stride 64) 
]
    cfg.MODEL.ROI_HEADS.IN_FEATURES = ["p2", "p3", "p4", "p5"]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.33, 0.5, 1.0, 2.0, 3.0]]
    cfg.MODEL.MASK_ON = True
    cfg.MODEL.KEYPOINT_ON = False
    DATA_ROOT = "/data/bailab_data/hoangnv/UIIS10K/"
    register_coco_instances("uiis10k_train", 
                            {}, os.path.join(DATA_ROOT, "annotations/multiclass_train.json"), 
                            os.path.join(DATA_ROOT, "img"))
    register_coco_instances("uiis10k_test", 
                            {}, os.path.join(DATA_ROOT, "annotations/multiclass_test.json"), 
                            os.path.join(DATA_ROOT, "img"))
    cfg.DATASETS.TRAIN = ("uiis10k_train",)
    cfg.DATASETS.TEST = ("uiis10k_test",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH  = 4
    cfg.SOLVER.BASE_LR = 0.0001 
    cfg.SOLVER.WEIGHT_DECAY = 0.05
    cfg.SOLVER.WARMUP_ITERS = 5000
    cfg.SOLVER.WARMUP_METHOD = "linear"
    cfg.SOLVER.BIAS_LR_FACTOR = 1.0
    cfg.SOLVER.WEIGHT_DECAY_BIAS = 0.0
    cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"
    dicts = DatasetCatalog.get("uiis10k_train")
    num_images = len(dicts)
    batch_size = cfg.SOLVER.IMS_PER_BATCH  
    one_epoch_iters = int(num_images / batch_size)
    cfg.SOLVER.MAX_ITER = one_epoch_iters * 24
    cfg.SOLVER.STEPS = (one_epoch_iters * 14, one_epoch_iters * 18)
    cfg.SOLVER.CHECKPOINT_PERIOD = one_epoch_iters 
    cfg.TEST.EVAL_PERIOD = one_epoch_iters
    cfg.SOLVER.AMP.ENABLED = True
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "value"
    cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
    cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0
    cfg.OUTPUT_DIR = "/data1/bailab_checkpoint/output/maskrcnn_vit_huge_lora"
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        from detectron2.evaluation import COCOEvaluator
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)
    @classmethod
    def build_train_loader(cls, cfg):
        # Data augmentation for underwater UIIS10K dataset
        augs = [
            T.RandomFlip(),  # Random flipping (horizontal)
            T.ResizeScale(
                min_scale=0.1, max_scale=2.0, target_height=1024, target_width=1024
            ),  # Random scaling (adjusted for underwater objects)
            T.FixedSizeCrop(crop_size=(1024, 1024), pad=True, pad_value=128.0), 
            T.RandomBrightness(0.8, 1.2),  # Brightness adjustment for lighting variations
            T.RandomContrast(0.8, 1.2),  # Contrast adjustment for underwater image clarity
        ]
        mapper = DatasetMapper(cfg, is_train=True, augmentations=augs, use_instance_mask=True, recompute_boxes=True)
        return build_detection_train_loader(cfg, mapper=mapper)
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        mapper = DatasetMapper(cfg, is_train=False, augmentations=[
            T.Resize((1024, 1024)) 
        ])
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)
    @classmethod
    def build_optimizer(cls, cfg, model):
        params = []
        for key, value in model.named_parameters():
            if not value.requires_grad:
                continue
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "bias" in key:
                lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
                weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
            if weight_decay is None:
                weight_decay = cfg.SOLVER.WEIGHT_DECAY
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
        return optimizer

def main(args):
    cfg = setup(args)
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        return res
    trainer = Trainer(cfg)
    model = trainer.model
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{'='*40}")
    print(f"[MODEL INFO] Tổng số tham số: {total_params / 1e6:.2f} M")
    print(f"[MODEL INFO] Tham số được học: {trainable_params / 1e6:.2f} M")
    print(f"[MODEL INFO] Tỷ lệ Unfreeze:   {(trainable_params/total_params)*100:.1f} %")
    print(f"{'='*40}\n")
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()

if __name__ == "__main__":

    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        dist_url=args.dist_url,
        args=(args,),
    )