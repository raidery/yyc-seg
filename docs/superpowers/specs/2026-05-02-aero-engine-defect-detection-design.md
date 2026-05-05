# 航空发动机零部件良品/残次品检测设计规格

**日期：** 2026-05-02
**项目：** YOLOv8 缺陷检测毕业设计

---

## 1. 项目概述

**目标：** 基于 YOLOv8 目标检测模型，实现航空发动机零部件的良品/残次品二分类判定。

**方案：** 方案 A — 基于缺陷检测结果判定
- 训练一个 YOLO 检测模型识别 `scratch`（划痕）和 `damage`（损坏）两类缺陷
- 推理时根据检测结果输出分类判定：
  - 检测到缺陷 → 残次品（defective）
  - 无缺陷 → 良品（qualified）

---

## 2. 数据集

**来源：** `datasets/Aero-engine_defect-detect_new`

**原始类别（4类）：** scratch, dirty, stain, damage
**目标类别（2类）：** scratch, damage

**数据量：**
- 训练集：257 张
- 验证集：73 张

**数据筛选规则：**
- 仅保留包含 `scratch` 或 `damage` 标注的图片和标签
- 剔除仅有 `dirty`、`stain` 标注的样本

---

## 3. 技术方案

**框架：** Ultralytics YOLOv8
**模型：** yolov8s.pt（预训练模型，可根据硬件情况切换 yolov8n.pt）
**任务类型：** Object Detection（目标检测）

**训练配置：**
| 参数 | 值 |
|------|-----|
| epochs | 100（早停patience=50） |
| imgsz | 640 |
| batch | 8（显存不足时可调低） |
| device | 0（GPU）或 cpu |
| patience | 50 |

**输出形式：**
- 检测结果：缺陷类别 + 边界框 + 置信度
- 分类判定：qualified / defective

---

## 4. 项目结构

```
yyc-seg/
├── datasets/
│   └── Aero-engine_defect-detect_new/
│       ├── images/train/, val/
│       └── labels/train/, val/
├── src/
│   ├── filter_dataset.py    # 筛选scratch/damage数据
│   ├── train.py             # 训练检测模型
│   └── predict.py          # 缺陷检测 + 二分类判定
├── configs/
│   └── data.yaml            # 数据集配置
├── yolov8s-seg.pt           # 预训练模型
└── runs/                    # 训练输出目录
```

---

## 5. 核心代码逻辑

### 5.1 数据筛选（filter_dataset.py）
```python
# 筛选条件：保留含有 scratch(0) 或 damage(3) 的标注
# 输出：筛选后的 train/val 数据集
```

### 5.2 训练（train.py）
```python
from ultralytics import YOLO
model = YOLO('yolov8s.pt')
results = model.train(data='configs/data.yaml', epochs=100, imgsz=640)
```

### 5.3 推理与判定（predict.py）
```python
def classify(defect_count):
    return "defective" if defect_count > 0 else "qualified"
```

---

## 6. 实施步骤

1. **环境检查** — 确认 Python 环境和依赖
2. **数据筛选** — 运行 filter_dataset.py 筛选 2 类数据
3. **模型训练** — 运行 train.py 训练检测模型
4. **推理验证** — 运行 predict.py 测试检测效果
5. **结果评估** — 统计检测精度和分类准确率

---

## 7. 验收标准

- [ ] 模型能检测出 scratch 和 damage 两类缺陷
- [ ] 检测到缺陷时输出 "defective"
- [ ] 未检测到缺陷时输出 "qualified"
- [ ] 验证集准确率满足毕业设计要求

---

## 8. 已知限制

- 数据量较小（330张），可能影响模型泛化能力
- dirty/stain 类别被剔除，相关样本被过滤
- 依赖 GPU 显存，显存不足时需调低 batch 或使用 yolov8n.pt