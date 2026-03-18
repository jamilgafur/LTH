
CHECKPOINT_BASES = {
    "VGG16": {
        "Cifar10": safe_glob("../structured_study/pruning_checkpoints/*Vgg16*cifar10_*"),
        "Cifar100": safe_glob("../structured_study/pruning_checkpoints/*Vgg16*cifar100_*"),
        "imagenet": safe_glob("../structured_study/pruning_checkpoints/*Vgg16*datasetimagenet_*"),
        "tinyimagenet": safe_glob("../structured_study/pruning_checkpoints/*Vgg16*datasettinyimagenet_*"),
    },
    "RegNetX_400MF": {
        "Cifar10": safe_glob("../structured_study/pruning_checkpoints/*RegNetX*cifar10_*"),
        "Cifar100": safe_glob("../structured_study/pruning_checkpoints/*RegNetX*cifar100_*"),
        "imagenet": safe_glob("../structured_study/pruning_checkpoints/*RegNetX*datasetimagenet_*"),
        "tinyimagenet": safe_glob("../structured_study/pruning_checkpoints/*RegNetX*datasettinyimagenet_*"),
    },
    "InceptionNet": {
        "Cifar10": safe_glob("../structured_study/pruning_checkpoints/*InceptionNet*cifar10_*"),
        "Cifar100": safe_glob("../structured_study/pruning_checkpoints/*InceptionNet*cifar100_*"),
        "imagenet": safe_glob("../structured_study/pruning_checkpoints/*InceptionNet*datasetimagenet_*"),
        "tinyimagenet": safe_glob("../structured_study/pruning_checkpoints/*InceptionNet*datasettinyimagenet_*"),
    },
    "MobileNet": {
        "Cifar10": safe_glob("../structured_study/pruning_checkpoints/*MobileNet*cifar10_*"),
        "Cifar100": safe_glob("../structured_study/pruning_checkpoints/*MobileNet*cifar100_*"),
        "imagenet": safe_glob("../structured_study/pruning_checkpoints/*MobileNet*datasetimagenet_*"),
        "tinyimagenet": safe_glob("../structured_study/pruning_checkpoints/*MobileNet*datasettinyimagenet_*"),
    },
    "XceptionNet": {
        "Cifar10": safe_glob("../structured_study/pruning_checkpoints/*XceptionNet*cifar10_*"),
        "Cifar100": safe_glob("../structured_study/pruning_checkpoints/*XceptionNet*cifar100_*"),
        "imagenet": safe_glob("../structured_study/pruning_checkpoints/*XceptionNet*datasetimagenet_*"),
        "tinyimagenet": safe_glob("../structured_study/pruning_checkpoints/*XceptionNet*datasettinyimagenet_*"),
    },
    "ConvNeXt": {
        "Cifar10": safe_glob("../structured_study/pruning_checkpoints/*ConvNeXt*cifar10_*"),
        "Cifar100": safe_glob("../structured_study/pruning_checkpoints/*ConvNeXt*cifar100_*"),
        "imagenet": safe_glob("../structured_study/pruning_checkpoints/*ConvNeXt*datasetimagenet_*"),
        "tinyimagenet": safe_glob("../structured_study/pruning_checkpoints/*ConvNeXt*datasettinyimagenet_*"),
    },
    
}

CHECKPOINT_FILES = {
    "VGG16": {
        "Cifar10": (
            "checkpoint_Finetuned_0.914101.pth",
            "checkpoint_Original_0.000000.pth",
        ),
        "Cifar100": (
            "checkpoint_Finetuned_0.981986.pth",
            "checkpoint_Original_0.000000.pth",
        ),
        "imagenet": (
            "checkpoint_Finetuned_0.790285.pth",
            "checkpoint_Original_0.000000.pth",
        ),
        "tinyimagenet": (
            "checkpoint_Finetuned_0.000000.pth",
            "checkpoint_Original_0.000000.pth",
        ),
    },
    "RegNetX_400MF": {
        "Cifar10": (
            "checkpoint_Finetuned_0.945024.pth",
            "checkpoint_Original_0.000000.pth",
        ),
        "Cifar100": (
            "checkpoint_Finetuned_0.488000.pth",
            "checkpoint_Original_0.000000.pth",
        ),
        "imagenet": (
            "checkpoint_Finetuned_0.914101.pth",
            "checkpoint_Original_0.000000.pth",
        ),
        "tinyimagenet": (
            "checkpoint_Finetuned_0.000000.pth",
            "checkpoint_Original_0.000000.pth",
        ),
    },
    "InceptionNet": {
        "Cifar10": (
            "None",
            "None",
        ),
        "Cifar100": (
            "None",
            "None",
        ),
        "imagenet": (
            "None",
            "None",
        ),
        "tinyimagenet": (
            "None",
            "None",
        ),
    },
    "MobileNet": {
        "Cifar10": (
            "None",
            "None",
        ),
        "Cifar100": (
            "None",
            "None",
        ),
        "imagenet": (
            "None",
            "None",
        ),
        "tinyimagenet": (
            "None",
            "None",
        ),
    },
    "XceptionNet": {
        "Cifar10": (
            "None",
            "None",
        ),
        "Cifar100": (
            "None",
            "None",
        ),
        "imagenet": (
            "None",
            "None",
        ),
        "tinyimagenet": (
            "None",
            "None",
        ),
    },
    "ConvNeXt": {
        "Cifar10": (
            "None",
            "None",
        ),
        "Cifar100": (
            "None",
            "None",
        ),
        "imagenet": (
            "None",
            "None",
        ),
        "tinyimagenet": (
            "None",
            "None",
        ),
    },
}

Vgg_common = {
            "Original Model": None,
            "Last 2": ("features.conv_12", "features.conv_13"),
            "Stage 5": ("features.conv_11", "features.conv_13"),
            "Stage 4": ("features.conv_8", "features.conv_10"),
            "Stage 3": ("features.conv_5", "features.conv_7"),
            "Stage 2": ("features.conv_3", "features.conv_4"),
            "Stage 4-5": ("features.conv_8", "features.conv_13"),
            "Stage 3-5": ("features.conv_5", "features.conv_13"),
            "Stage 2-5": ("features.conv_3", "features.conv_13"),
        }
RegNetX_common =  {
            "Original Model": None,
            # Single-stage collapses (single tuples)
            "Last 2": (
                "stage4.stage4_block5.block.conv1",
                "stage4.stage4_block6.block.conv3",
            ),
            "Stage 4": (
                "stage4.stage4_block0.block.conv1",
                "stage4.stage4_block6.block.conv3",
            ),
            "Stage 3": (
                "stage3.stage3_block0.block.conv1",
                "stage3.stage3_block3.block.conv3",
            ),
            "Stage 2": (
                "stage2.stage2_block0.block.conv1",
                "stage2.stage2_block0.block.conv3",
            ),
            "Stage 1": (
                "stage1.stage1_block0.block.conv1",
                "stage1.stage1_block0.block.conv3",
            ),
            # Multi-stage collapses (lists of tuples)
            "Stage 3-4": [
                (
                    "stage3.stage3_block0.block.conv1",
                    "stage3.stage3_block3.block.conv3",
                ),  # Stage 3
                (
                    "stage4.stage4_block0.block.conv1",
                    "stage4.stage4_block6.block.conv3",
                ),  # Stage 4
            ],
            "Stage 2-4": [
                (
                    "stage2.stage2_block0.block.conv1",
                    "stage2.stage2_block0.block.conv3",
                ),  # Stage 2
                (
                    "stage3.stage3_block0.block.conv1",
                    "stage3.stage3_block3.block.conv3",
                ),  # Stage 3
                (
                    "stage4.stage4_block0.block.conv1",
                    "stage4.stage4_block6.block.conv3",
                ),  # Stage 4
            ],
            "Stage 1-4": [
                (
                    "stage1.stage1_block0.block.conv1",
                    "stage1.stage1_block0.block.conv3",
                ),  # Stage 1
                (
                    "stage2.stage2_block0.block.conv1",
                    "stage2.stage2_block0.block.conv3",
                ),  # Stage 2
                (
                    "stage3.stage3_block0.block.conv1",
                    "stage3.stage3_block3.block.conv3",
                ),  # Stage 3
                (
                    "stage4.stage4_block0.block.conv1",
                    "stage4.stage4_block6.block.conv3",
                ),  # Stage 4
            ],
            # Stage-specific first/last conv pairs
            "Stage 1 first 2 conv": (
                "stage1.stage1_block0.block.conv1",
                "stage1.stage1_block0.block.conv2",
            ),
            "Stage 2 first 2 conv": (
                "stage2.stage2_block0.block.conv1",
                "stage2.stage2_block0.block.conv2",
            ),
            "Stage 3 first 2 conv": (
                "stage3.stage3_block0.block.conv1",
                "stage3.stage3_block1.block.conv1",
            ),
            "Stage 4 first 2 conv": (
                "stage4.stage4_block0.block.conv1",
                "stage4.stage4_block1.block.conv1",
            ),
            "Stage 1 last 2 conv": (
                "stage1.stage1_block0.block.conv2",
                "stage1.stage1_block0.block.conv3",
            ),
            "Stage 2 last 2 conv": (
                "stage2.stage2_block0.block.conv2",
                "stage2.stage2_block0.block.conv3",
            ),
            "Stage 3 last 2 conv": (
                "stage3.stage3_block2.block.conv3",
                "stage3.stage3_block3.block.conv3",
            ),
            "Stage 4 last 2 conv": (
                "stage4.stage4_block4.block.conv3",
                "stage4.stage4_block5.block.conv3",
            ),
        }
XceptionNet_common =  {
            "Original Model": None,
            "Stage 5": ("block5.depthwise", "block5.bn2"),
            "Stage 4": ("block4.depthwise", "block5.depthwise"),
            "Stage 3": ("block3.depthwise", "block4.depthwise"),
            "Stage 2": ("block2.depthwise", "block3.depthwise"),
            "Stage 1": ("block1.depthwise", "block2.depthwise"),
            "Stage 3-5": ("block3.depthwise", "block5.depthwise"),
            "Stage 2-5": ("block2.depthwise", "block5.depthwise"),
            "Stage 1-5": ("block1.depthwise", "block5"),
        }
mobileNet_common =  {
            "Original Model": None,
            "Stage 7": ("block7.depthwise", "block7.bn2"),
            "Stage 6": ("block6.depthwise", "block7.depthwise"),
            "Stage 5": ("block5.depthwise", "block6.depthwise"),
            "Stage 4": ("block4.depthwise", "block5.depthwise"),
            "Stage 3": ("block3.depthwise", "block4.depthwise"),
            "Stage 2": ("block2.depthwise", "block3.depthwise"),
            "Stage 1": ("block1.depthwise", "block2.depthwise"),
            "Stage 5-7": ("block5.depthwise", "block7.depthwise"),
            "Stage 4-7": ("block4.depthwise", "block7.depthwise"),
            "Stage 6-7": ("block6.depthwise", "block7.depthwise"),
            "Stage 3-7": ("block3.depthwise", "block7.depthwise"),
            "Stage 2-7": ("block2.depthwise", "block7.depthwise"),
            "Stage 1-7": ("block1.depthwise", "block7.depthwise"),
            "Last 2": ("block6.depthwise", "block7.depthwise"),
        }
InceptionNet_common = {
            "Original Model": None,
            # Single-stage collapses
            "Stage 5": (
                "stage5.inception_5a",
                "stage5.inception_5b",
            ),
            "Stage 4": (
                "stage4.inception_4a",
                "stage4.inception_4b",
            ),
            "Stage 3": (
                "stage3.inception_3a",
                "stage3.inception_3b",
            ),
            "Stage 2": (
                "stage2.inception_2a",
                "stage2.inception_2b",
            ),
            "Stage 2-5": [
                (
                    "stage2.inception_2a",
                    "stage2.inception_2b",
                ),  # Stage 2
                (
                    "stage3.inception_3a",
                    "stage3.inception_3b",
                ),  # Stage 3
                (
                    "stage4.inception_4a",
                    "stage4.inception_4b",
                ),  # Stage 4
                (
                    "stage5.inception_5a",
                    "stage5.inception_5b",
                ),  # Stage 5
            ],
            "Stage 3-5": [
                (
                    "stage3.inception_3a",
                    "stage3.inception_3b",
                ),  # Stage 3
                (
                    "stage4.inception_4a",
                    "stage4.inception_4b",
                ),  # Stage 4
                (
                    "stage5.inception_5a",
                    "stage5.inception_5b",
                ),  # Stage 5
            ],
            "Stage 4-5": [
                (
                    "stage4.inception_4a",
                    "stage4.inception_4b",
                ),  # Stage 4
                (
                    "stage5.inception_5a",
                    "stage5.inception_5b",
                ),  # Stage 5
            ],  
            "Last 2": (
                "stage5.inception_5a",
                "stage5.inception_5b",
            ),
        }
ConvNeXt_common = {
    "Original Model": None,
    # Stage 1
    "Stage 1": ("stage1.block1_1", "stage1.block1_2"),

    # Stage 2
    "Stage 2": ("stage2.block2_1", "stage2.block2_2"),

    # Stage 3 (strong redundancy)
    "Stage 3": ("stage3.block3_1", "stage3.block3_3"),

    # Stage 4
    "Stage 4": ("stage4.block4_1", "stage4.block4_2"),
}

EXPERIMENTS = {
    "VGG16": {
        "Cifar10": Vgg_common,
        "Cifar100": Vgg_common,
        "tinyimagenet": Vgg_common,
        "imagenet": Vgg_common,
    },
    "RegNetX_400MF": {
        "Cifar10": RegNetX_common,
        "Cifar100": RegNetX_common,
        "tinyimagenet": RegNetX_common,
        "imagenet": RegNetX_common,
    },
    "XceptionNet": {
        "Cifar10": XceptionNet_common,
        "Cifar100": XceptionNet_common,
        "tinyimagenet": XceptionNet_common,
        "imagenet": XceptionNet_common,
    },
    "MobileNet": {
        "Cifar10": mobileNet_common,
        "Cifar100": mobileNet_common,
        "tinyimagenet": mobileNet_common,
        "imagenet": mobileNet_common,
    },
    "InceptionNet": {
        "Cifar10": InceptionNet_common,
        "Cifar100": InceptionNet_common,
        "tinyimagenet": InceptionNet_common,
        "imagenet":InceptionNet_common,
    },
    "ConvNeXt": {
        "Cifar10": ConvNeXt_common,
        "Cifar100": ConvNeXt_common,
        "tinyimagenet": ConvNeXt_common,
        "imagenet": ConvNeXt_common,
    }
}

