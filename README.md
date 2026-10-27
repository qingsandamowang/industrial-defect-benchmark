# 工业表面缺陷检测与分割

**Industrial Defect Benchmark · OpenCV baseline + lightweight U-Net**

在公开的真实磁瓦图像上完成数据划分、传统分割基线、轻量 U-Net 训练、参数选择、独立测试与错误分析。代码由 Codex Agent 协助开发，实验在本机 CPU 实际运行。项目使用公开方法构建可复查的实验流程。

## 流程

照片与像素标注 → 按类别划分数据 → Otsu / U-Net → 恢复原图尺寸 → 像素与图像指标 → 失败样例

## 升级后的功能

- 保留原轻量 U-Net 实验，新增 ImageNet 预训练 ResNet18 编码器与分割解码器。
- 训练与测试分成两个命令；验证集选择权重与阈值，测试命令核对固定划分和数据内容指纹。
- 配对翻转、亮度/对比度与噪声增强；支持 CUDA AMP，损失的像素求和始终使用 FP32。
- 单图或图片文件夹预测，输出原尺寸概率数组、掩码、叠加图、缺陷区域面积/边框、JSON 与离线 HTML 索引。
- 评估报告增加混淆矩阵、正常样本误报率、AP/PR 曲线、逐图结果和按图像 bootstrap 的 Dice 区间。

测试文件的压缩字节参与哈希登记，以识别同名文件被替换；训练命令不会解码测试像素、评分测试集或用其选参数。该边界与“完全不读取测试文件”不同。

## 已完成实验

下面保留的是历史 `mtd_v2` 数值，便于与新实验比较。

- 数据：1,344 张真实磁瓦照片，952 张正常图，392 张来自五个缺陷目录；其中 4 张掩码为空。
- 划分：固定 seed=42，按目录类别分层；训练 940 / 验证 202 / 测试 202。
- U-Net：保持比例填充至 192×192，加权 BCE + soft Dice，训练 30 轮；由验证集选择第 27 轮权重与 0.60 分割阈值。
- 指标：恢复至原图尺寸后汇总像素 TP/FP/FN；图像检测阈值只在验证集选择。

| 方法 | 测试像素 Dice ↑ | IoU ↑ | 图像 F1 ↑ | 图像 AUROC ↑ |
|---|---:|---:|---:|---:|
| Otsu + 形态学 | 0.080 | 0.042 | 0.463 | 0.513 |
| 轻量 U-Net | **0.575** | **0.403** | **0.594** | **0.789** |

原始数值：[U-Net 报告](results/mtd_v2/unet_report.json)、[基线报告](results/mtd_v2/baseline_report.json)、[训练历史](results/mtd_v2/unet_history.json)、[数据名单与哈希](results/mtd_v2/manifest.json)。128×128、12 轮的初步实验也保存在 `results/mtd_v1`。

## 安装与复现

使用 Python 3.11 或更新版本。在仓库根目录打开 PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
```

下载[数据作者仓库](https://github.com/abin24/Magnetic-tile-defect-datasets.)的 master ZIP，保留压缩包并解压：

```powershell
New-Item -ItemType Directory -Force data/raw | Out-Null
Invoke-WebRequest 'https://github.com/abin24/Magnetic-tile-defect-datasets./archive/refs/heads/master.zip' -OutFile data/raw/mtd-source.zip
Expand-Archive data/raw/mtd-source.zip -DestinationPath data/raw/mtd
$env:PYTHONPATH='src'
python -m defectbench.run --epochs 30 --size 192 --batch-size 16 --out results/local_run
```

程序递归寻找 `MT_*/Imgs/*.jpg` 及同名 PNG 掩码。复现实验使用的原始 ZIP SHA-256：`39bd97027c027116dc97f858db6e5028f98fdcf94b70e695c063f74e40976cf5`。上游数据变化或软件环境不同可能影响结果。

运行后会生成报告、模型权重及 `case_images` 原图/真值/预测对比。仓库只保存代码和数值报告；原始图片、派生结果图及模型权重不随仓库分发。

## 加强模型与独立预测

普通依赖按上文安装。使用兼容 NVIDIA GPU 时，可先按 [PyTorch 官方说明](https://pytorch.org/get-started/locally/)安装匹配的 CUDA 构建。本次采用 PyTorch 2.11.0 / torchvision 0.26.0 的 CUDA 12.8 构建。

```powershell
$env:PYTHONPATH='src'
python -m defectbench.experiment train --data data/raw/mtd --out results/transfer_seed42 --epochs 40 --size 384 --batch-size 8 --device cuda --seed 42
# CPU 可用 --device cpu；--no-pretrained 用于不下载初始权重的实验。
```

`train` 仅保存训练历史、验证结果和最佳权重，不输出测试成绩。确定候选后显式评估：

```powershell
python -m defectbench.experiment evaluate --data data/raw/mtd --checkpoint results/transfer_seed42/model_best.pt --out results/mtd_v3 --device cuda
python -m defectbench.predict --checkpoint results/transfer_seed42/model_best.pt --input your_images --out results/predictions --device cuda
```

打开 `results/mtd_v3/report.html` 看数值评估，或 `results/predictions/index.html` 看预测叠加图。概率数组不是经过校准的缺陷概率；没有人工答案的自定义图片不输出精度。预测输出目录必须位于输入目录外，避免把上次生成的图片再次读为输入。

[升级原理和逐步复现](docs/upgrade-guide.md) · [预先记录的实验协议](docs/experiment-protocol-v3.json)

## 局限与学习入口

测试是同一公开数据集内部的留出划分，不能等同于产线性能。Blowhole 类 Dice 仅 0.262，细小缺陷和光照纹理仍会引起漏检、误报。Dice 不是图像分类准确率。

- [中文学习说明](docs/learning-guide.md)：掩码、数据划分、U-Net、阈值与指标。
- [实验入口](src/defectbench/run.py)：从数据准备到报告输出。
- [来源与归属](THIRD_PARTY_NOTICES.md)。
