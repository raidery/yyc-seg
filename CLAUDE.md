# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

航空发动机零部件良品/残次品检测，基于 YOLOv8 目标检测实现。

- **核心任务**：缺陷检测 + 二分类判定（scratch/damage → 残次品）
- **方案**：方案 A — 基于缺陷检测结果判定
- **预训练模型**：`yolov8s.pt`

## 环境搭建

推荐使用 uv：
```bash
pip install uv
uv venv --python 3.9
source .venv/bin/activate
uv sync
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
│   └── Aero-engine_defect-detect_new/  # 原始数据集
└── runs/seg/train/        # 训练输出
```

## 数据集

- 原始数据集：`datasets/Aero-engine_defect-detect_new`
- 4 类原始缺陷：scratch, dirty, stain, damage
- 目标缺陷：scratch, damage（2类）

## 关键配置

- **数据集类别映射**：`CLASS_MAP = {"scratch": 0, "damage": 1}`
- **输入尺寸**：`imgsz=640`
- **训练轮数**：默认 100，早停耐心 50

## 相关文档

- 设计规格：`docs/superpowers/specs/2026-05-02-aero-engine-defect-detection-design.md`
- 实施计划：`docs/superpowers/plans/2026-05-02-aero-engine-defect-detection-plan.md`
- 对话记录：`docs/superpowers/specs/2026-05-defect-detection/问卷调查对话记录.md`