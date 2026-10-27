# Portfolio Upgrade Implementation Plan

> **For agentic workers:** 按任务使用 executing-plans 或 subagent-driven-development 执行；每个独立项目都有单独计划，必须先完成安全测试再集成。

**Goal:** 完成加强训练、可复现评估、独立预测与HTML报告，并真实比较同一测试集。
**Architecture:** 原版run.py保持可复现。新建transfer.py、experiment.py、predict.py、html_report.py，分别负责模型、实验、推理和纯数值报告。
**Tech Stack:** PyTorch CUDA 12.8、torchvision、OpenCV、NumPy、pytest。

### Task 1: GPU与模型
Files: src/defectbench/transfer.py; tests/test_transfer.py; requirements.txt。
- [ ] 创建独立GPU虚拟环境，确认CUDA算子与RTX5060真实可用。
- [ ] 先加入模型形状/梯度测试，验证新增模块缺失时失败。
```python
model = ResNetUNet(pretrained=False)
assert model(torch.rand(2, 1, 64, 96)).shape == (2, 1, 64, 96)
```
- [ ] 实现预训练编码器与跳跃连接解码器；归一化与权重来源写入元数据。

### Task 2: 实验训练
Files: src/defectbench/experiment.py; tests/test_experiment.py。
- [ ] 先测试成对翻转保持image/mask几何一致、验证不增强、训练不读test。
- [ ] 实现train与evaluate命令；保存配置、历史、验证选择、checkpoint与固定名单。
```text
python -m defectbench.experiment train --data DATA --out results/transfer_seed42 --device cuda
python -m defectbench.experiment evaluate --data DATA --checkpoint CHECKPOINT --out results/mtd_v3
```
- [ ] 在验证集运行候选配置并选择；只对最终选中配置运行留出评估。

### Task 3: 预测与报告
Files: src/defectbench/predict.py; src/defectbench/html_report.py; tests/test_predict.py。
- [ ] 先测Unicode文件输入、原尺寸输出、同路径拒绝、可加载权重元数据。
- [ ] 实现文件/文件夹预测，生成掩码/叠加图/JSON；HTML报告展示真实数值与限制。
- [ ] 实际预测一张本地图片；审阅报告；跑完整测试。
- [ ] 更新README、来源、学习说明、结果比较，独立审查后提交并推送。
