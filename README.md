# 航空发动机零部件缺陷检测

基于 YOLOv8 的缺陷检测项目，用于检测航空发动机零部件的 scratch 和 damage 两类缺陷，并进行良品/残次品二分类判定。

## 环境搭建

推荐使用 uv 管理 Python 环境：

```bash
pip install uv
uv venv --python 3.9
source .venv/bin/activate
uv sync
```

## 数据准备

原始数据集位于 `/root/autodl-tmp/Aero-engine_defect-detect_new`，包含 4 类缺陷：scratch, dirty, stain, damage。

运行脚本筛选出目标 2 类数据（scratch, damage）：

```bash
source .venv/bin/activate
python src/filter_dataset.py
```

筛选后数据输出至 `datasets/aero-engine-2class`：
- 训练集：222 张图片
- 验证集：53 张图片

## 模型训练

```bash
source .venv/bin/activate
python src/train.py
```

训练配置：
- 预训练模型：yolov8s.pt（首次训练时自动下载）
- 训练轮数：100（早停耐心 50）
- 输入尺寸：640x640
- 批大小：8（GPU 显存不足时可调低）
- 设备：GPU 或 CPU
- 输出路径：`runs/seg/train/weights/best.pt`

## 缺陷检测与分类

使用训练好的模型进行推理：

```bash
source .venv/bin/activate
python src/predict.py
```

**分类逻辑：**
- 检测到 scratch 或 damage 缺陷 → 残次品（defective）
- 未检测到缺陷 → 良品（qualified）

**输出示例：**
```
============================================================
图片名称                           检测结果        判定
============================================================
xxx.jpg                           scratch(0.95)   残次品 [FAIL]
xxx.jpg                                               良品 [PASS]
...

统计结果:
  良品 (qualified): XX
  残次品 (defective): XX
  合计: 53
```

## 项目结构

```
yyc-seg/
├── src/
│   ├── filter_dataset.py    # 筛选2类数据
│   ├── train.py             # 模型训练
│   └── predict.py          # 缺陷检测 + 二分类
├── configs/
│   └── data.yaml           # 数据集配置
├── datasets/
│   ├── Aero-engine_defect-detect_new/  # 原始数据集
│   └── aero-engine-2class/            # 筛选后数据集
├── docs/                    # 报告文档
│   ├── 训练报告-2026-05-05.md
│   └── 验证报告-2026-05-05.md
└── runs/seg/train/         # 训练输出
```

## 训练结果

| 指标 | 最佳值 |
|------|--------|
| mAP@0.5 | 0.9556 |
| mAP@0.5:0.95 | 0.6896 |
| Precision | 0.9550 |
| Recall | 0.8900 |

- 实际训练 69 epoch（早停触发）
- 最佳模型权重: `runs/seg/train/weights/best.pt`

## 数据集类别

| 类别 | 标签 |
|------|------|
| scratch | 0 |
| damage | 1 |

检测到任一缺陷即判定为残次品。

## 验证结果

- 验证集 52 张图片全部检测到缺陷，全部判定为残次品
- 原始数据集为缺陷数据集，无良品图片用于验证"未检测到缺陷 = 良品"的分类逻辑
- 建议补充良品图片以完整验证分类系统
