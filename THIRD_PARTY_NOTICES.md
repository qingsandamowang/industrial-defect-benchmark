# 来源与归属

- 真实图片与像素标注：[Magnetic Tile Defect 数据作者仓库](https://github.com/abin24/Magnetic-tile-defect-datasets.)。使用数据请遵守作者要求并引用其论文《Saliency of magnetic tile surface defects》。
- U-Net 基本方法：Ronneberger, Fischer, Brox, [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597), 2015。本仓库为轻量实现与实验流程，不宣称算法原创。
- 依赖 OpenCV、PyTorch、NumPy；其权利和许可分别归原作者。
- 升级编码器使用 [torchvision ResNet18 IMAGENET1K_V1](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html) 公开预训练权重；其训练数据为 ImageNet。模型设计、预训练成果及依赖许可归原作者，使用者应遵守相应条件。
- 代码由 Codex Agent 协助实现，记录的实验实际运行。公开包不含数据集图片、派生图片或模型权重；数值报告用于复查本次实验。
