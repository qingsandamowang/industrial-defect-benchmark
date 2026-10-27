# 升级版：从模型实验到可使用的预测工具

本项目的价值需要同时看方法、评估可信度和实际使用入口。原版 Dice 0.575 只是起点；旧图像检测精确率0.448，约44.8%的正常图被误报，不应称为检测效果优秀。

## 模型多学到了什么

`transfer.py` 使用 torchvision 的 ResNet18 ImageNet 权重作为编码器。输入仍是一张灰度图，在模型内部复制为三通道并执行 ImageNet 均值/标准差归一化；编码器提取多尺度特征，解码器结合跳跃连接找回位置细节。预训练方法与数据来源归原作者，本仓库没有提出新网络结构。

与旧版相比，输入从192提高至384，更容易保留小缺陷。训练时图像与掩码同时做水平/垂直翻转；亮度、对比度与噪声只作用于图像。验证和预测都不做随机增强。正常/缺陷图进行平衡采样，损失为未加类别权重的BCE加soft Dice。

AMP使用较少显存，但不能对大图的重叠像素直接做FP16求和：384×384张量中，即使每个值只有0.5，总和也可能超过65504。实现会将BCE和Dice归约转为FP32，回归测试比较half与float32结果，避免“loss有限但Dice信号已经丢失”。

## 数据和评估怎么核对

1. 固定seed=42的原类别分层划分：940训练、202验证、202历史留出测试。训练随机种子与划分种子分开记录。
2. 数据登记读取所有压缩文件的字节计算SHA256，不从测试集解码图像、计算特征、获取分数或选参数。每张图/掩码的哈希及总指纹保存在`data_fingerprint.json`。重复样本key会直接拒绝，避免重复解压数据导致同图跨组。
3. 每轮训练后只用验证集原始尺寸Dice选择权重，并用验证集选择分割阈值、整图判断阈值。配置、软件版本、硬件和每轮耗时写入实验目录。
4. `evaluate`核对名单与内容指纹，拒绝同名数据被修改，再对固定历史测试集报告。逐图JSON带输入图和掩码哈希。
5. Dice衡量区域重叠；图像级任务额外看TP/FP/FN/TN、精确率、召回率、F1、AUROC、AP与正常样本误报率。bootstrap按图片抽样，不能把每个相关像素当独立样本。

历史测试结果此前已经查看；这次不以测试分数挑权重或阈值，但它仍属于对同一数据集的持续迭代，不能代替新的外部盲测。原数据集的采集批次与物理工件独立性信息不足，也限制了随机划分所能证明的泛化能力。

## 三个实际动作

在仓库根目录安装依赖后：

```powershell
$env:PYTHONPATH='src'
python -m defectbench.experiment train --data data/raw/mtd --out results/my_training --device cuda --epochs 40 --size 384
python -m defectbench.experiment evaluate --data data/raw/mtd --checkpoint results/my_training/model_best.pt --out results/my_evaluation --device cuda
python -m defectbench.predict --checkpoint results/my_training/model_best.pt --input your_images --out results/my_predictions --device cuda
```

第一条默认下载torchvision官方ResNet18权重；`--no-pretrained`禁用下载。CPU运行使用`--device cpu`，但训练耗时会更长。实验目录已有配置时拒绝覆盖，换新输出路径可保留每次尝试。

预测保存原尺寸`*_mask.png`、`*_overlay.png`、`*_probability.npy`，每个区域的面积和边框写进`predictions.json`。`index.html`是离线浏览入口；输入图片没有真值时不声称精度。仅使用可信来源的本地checkpoint，加载采用`weights_only=True`。

## 自己应能解释的内容

- 输入尺寸变大为什么有助于小缺陷，也为什么会消耗更多显存？
- 预训练是借用已有特征，和自己提出新算法有什么区别？
- 为什么训练增强必须同时改变图像和掩码的位置？
- “正常图误报率”为什么比单看AUROC更能反映报警体验？
- 同一个文件名对应不同图片时，实验指纹怎样阻止错误比较？

权重、原数据及数据派生图片保存在本机，公开仓库发布代码和数值报告。HTML数值报告可公开；预测HTML依赖用户图片，仅供本地查看。生产使用前仍需新的采集环境、速度/稳定性和误报成本验证。
