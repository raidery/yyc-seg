# 航空发动机零部件良品/残次品检测 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 基于 YOLOv8 目标检测实现航空发动机零部件良品/残次品二分类判定

**Architecture:** 训练 YOLOv8 检测模型识别 scratch/damage 两类缺陷，推理时根据检测结果输出分类判定（检测到缺陷→残次品，无缺陷→良品）

**Tech Stack:** Python 3.9, Ultralytics YOLOv8, PyTorch

---

## 文件结构

```
yyc-seg/
├── configs/
│   └── data.yaml              # 数据集配置（相对于项目根目录）
├── src/
│   ├── filter_dataset.py      # 数据筛选脚本
│   ├── train.py               # 模型训练脚本
│   └── predict.py             # 缺陷检测 + 二分类判定
├── datasets/
│   └── Aero-engine_defect-detect_new/
│       ├── images/train/, val/
│       └── labels/train/, val/
└── runs/seg/train/            # 训练输出（自动生成）
```

---

## 数据集映射

原始类别（classes.txt）：
- 0: scratch
- 1: dirty
- 2: stain
- 3: damage

目标类别：
- 0: scratch
- 1: damage

筛选规则：保留含有类别 0 或 3 的标注文件

---

## 任务分解

### Task 1: 创建数据集配置文件

**Files:**
- Create: `configs/data.yaml`

- [ ] **Step 1: 创建 data.yaml**

```yaml
path: datasets/Aero-engine_defect-detect_new
train: images/train
val: images/val

nc: 2
names:
  0: scratch
  1: damage
```

- [ ] **Step 2: 提交**

```bash
git add configs/data.yaml
git commit -m "feat: add dataset config for aero-engine defect detection"
```

---

### Task 2: 创建数据筛选脚本

**Files:**
- Create: `src/filter_dataset.py`

- [ ] **Step 1: 创建 filter_dataset.py**

```python
"""
数据筛选脚本：从4类缺陷数据中筛选出2类（scratch, damage）
原始类别：0-scratch, 1-dirty, 2-stain, 3-damage
目标类别：0-scratch, 1-damage
"""
import os
import shutil
from pathlib import Path

# 原始数据集路径
SRC_ROOT = Path("datasets/Aero-engine_defect-detect_new")
SRC_LABELS = SRC_ROOT / "labels"
SRC_IMAGES = SRC_ROOT / "images"

# 目标数据集路径
DST_ROOT = Path("datasets/aero-engine-2class")
DST_LABELS = DST_ROOT / "labels"
DST_IMAGES = DST_ROOT / "images"

# 要保留的原始类别（scratch=0, damage=3）
KEEP_CLASSES = {0, 3}
# 类别映射（原始类别 -> 新类别）
CLASS_MAP = {0: 0, 3: 1}

def filter_split(split: str):
    """筛选单个数据集分割（train/val）"""
    src_label_dir = SRC_LABELS / split
    dst_label_dir = DST_LABELS / split
    dst_image_dir = DST_IMAGES / split

    dst_label_dir.mkdir(parents=True, exist_ok=True)
    dst_image_dir.mkdir(parents=True, exist_ok=True)

    count_kept = 0
    count_total = 0

    for label_file in src_label_dir.glob("*.txt"):
        count_total += 1
        if label_file.name == "classes.txt":
            continue

        # 读取标签内容
        with open(label_file, "r") as f:
            lines = f.readlines()

        # 检查是否包含要保留的类别
        new_lines = []
        has_target_class = False
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            if cls in KEEP_CLASSES:
                has_target_class = True
                # 映射类别
                new_cls = CLASS_MAP[cls]
                new_lines.append(f"{new_cls} {' '.join(parts[1:])}\n")

        # 如果包含目标类别，复制文件和标签
        if has_target_class:
            # 复制标签（带新类别）
            with open(dst_label_dir / label_file.name, "w") as f:
                f.writelines(new_lines)

            # 复制图片
            img_name = label_file.stem + ".jpg"
            src_img = SRC_IMAGES / split / img_name
            if src_img.exists():
                shutil.copy(src_img, dst_image_dir / img_name)
            count_kept += 1

    print(f"{split}: 筛选 {count_kept}/{count_total} 张图片")
    return count_kept, count_total

def main():
    """主函数：筛选训练集和验证集"""
    print("开始筛选数据集...")

    # 创建目标目录
    DST_LABELS.mkdir(parents=True, exist_ok=True)
    DST_IMAGES.mkdir(parents=True, exist_ok=True)

    # 筛选训练集和验证集
    train_kept, train_total = filter_split("train")
    val_kept, val_total = filter_split("val")

    print(f"\n筛选完成！")
    print(f"训练集: {train_kept}/{train_total}")
    print(f"验证集: {val_kept}/{val_total}")
    print(f"输出目录: {DST_ROOT}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 提交**

```bash
git add src/filter_dataset.py
git commit -m "feat: add dataset filter script for 2-class defect detection"
```

---

### Task 3: 运行数据筛选

- [ ] **Step 1: 运行筛选脚本**

```bash
cd /Users/raidery/bench/yolo8/yyc-seg
python src/filter_dataset.py
```

预期输出：
```
开始筛选数据集...
train: 筛选 XX/257 张图片
val: 筛选 XX/73 张图片

筛选完成！
训练集: XX/257
验证集: XX/73
输出目录: datasets/aero-engine-2class
```

- [ ] **Step 2: 验证筛选结果**

```bash
ls datasets/aero-engine-2class/labels/train/ | head -10
ls datasets/aero-engine-2class/images/train/ | head -10
```

- [ ] **Step 3: 提交**

```bash
git add datasets/aero-engine-2class/
git commit -m "feat: add filtered 2-class dataset"
```

---

### Task 4: 检查/下载预训练模型

- [ ] **Step 1: 检查模型是否已存在**

```bash
ls yolov8s.pt 2>/dev/null && echo "模型已存在" || echo "模型不存在"
```

- [ ] **Step 2: 如果模型不存在，下载预训练模型**

```bash
# 使用 ultralytics 自动下载（首次训练时会自动下载）
# 或手动下载：
# wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt
```

**注意：** YOLOv8 会在首次训练时自动下载预训练模型，如果不存在的话。

- [ ] **Step 3: 提交（无需提交模型文件）**

---

### Task 5: 创建模型训练脚本

**Files:**
- Create: `src/train.py`

- [ ] **Step 1: 创建 train.py**

```python
"""
YOLOv8 缺陷检测模型训练脚本
目标：检测 scratch 和 damage 两类缺陷
"""
from ultralytics import YOLO
import torch

def main():
    # 检查GPU
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # 加载预训练模型
    model = YOLO("yolov8s.pt")
    print("已加载 yolov8s.pt 预训练模型")

    # 训练模型
    results = model.train(
        data="configs/data.yaml",
        epochs=100,
        imgsz=640,
        batch=8,
        device=device,
        patience=50,
        project="runs/seg",
        name="train",
        exist_ok=True,
        verbose=True,
    )

    print("\n训练完成！")
    print(f"最佳模型路径: runs/seg/train/weights/best.pt")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 提交**

```bash
git add src/train.py
git commit -m "feat: add YOLOv8 training script for defect detection"
```

---

### Task 6: 运行模型训练

- [ ] **Step 1: 运行训练**

```bash
cd /Users/raidery/bench/yolo8/yyc-seg
python src/train.py
```

预期输出：
```
使用设备: 0
已加载 yolov8s.pt 预训练模型
开始训练...
Epoch 100/100
...
训练完成！
最佳模型路径: runs/seg/train/weights/best.pt
```

**注意：** 训练时间取决于 GPU，首次训练可能需要 30 分钟至数小时。

- [ ] **Step 2: 提交训练结果**

```bash
git add runs/seg/train/
git commit -m "feat: add trained model weights"
```

---

### Task 7: 创建推理与分类脚本

**Files:**
- Create: `src/predict.py`

- [ ] **Step 1: 创建 predict.py**

```python
"""
YOLOv8 缺陷检测 + 良品/残次品分类脚本
逻辑：
- 检测到 scratch 或 damage 缺陷 -> defective（残次品）
- 未检测到缺陷 -> qualified（良品）
"""
from ultralytics import YOLO
from pathlib import Path

# 模型路径
MODEL_PATH = "runs/seg/train/weights/best.pt"
# 测试图片路径（默认为验证集）
TEST_PATH = "datasets/aero-engine-2class/images/val"

# 缺陷类别名称
DEFECT_CLASSES = ["scratch", "damage"]

def classify_image(result) -> str:
    """
    根据检测结果判定图片类别

    Args:
        result: YOLO 检测结果

    Returns:
        "defective" (残次品) - 检测到缺陷
        "qualified" (良品) - 未检测到缺陷
    """
    # 检查是否检测到任何缺陷
    if result.boxes is None or len(result.boxes) == 0:
        return "qualified"

    # 获取检测到的类别
    detected_classes = result.boxes.cls.cpu().numpy()
    detected_names = [result.names[int(cls)] for cls in detected_classes]

    # 检查是否包含缺陷类别
    for name in detected_names:
        if name in DEFECT_CLASSES:
            return "defective"

    return "qualified"

def main():
    # 加载模型
    model = YOLO(MODEL_PATH)
    print(f"已加载模型: {MODEL_PATH}\n")

    # 获取测试图片
    test_dir = Path(TEST_PATH)
    image_files = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))

    if not image_files:
        print(f"未找到测试图片: {TEST_PATH}")
        return

    print(f"找到 {len(image_files)} 张测试图片\n")
    print("=" * 60)
    print(f"{'图片名称':<30} {'检测结果':<15} {'判定'}")
    print("=" * 60)

    qualified_count = 0
    defective_count = 0

    # 遍历每张图片进行推理
    for img_path in sorted(image_files):
        # 执行检测
        results = model(img_path, verbose=False)

        # 获取第一张图片的结果
        result = results[0]

        # 分类判定
        classification = classify_image(result)

        # 统计
        if classification == "qualified":
            qualified_count += 1
        else:
            defective_count += 1

        # 获取检测到的缺陷信息
        defect_info = ""
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes
            defects = [result.names[int(cls)] for cls in boxes.cls.cpu().numpy()]
            confidences = boxes.conf.cpu().numpy()
            defect_info = ", ".join([f"{d}({c:.2f})" for d, c in zip(defects, confidences)])

        # 打印结果
        label = "良品" if classification == "qualified" else "残次品"
        status = "PASS" if classification == "qualified" else "FAIL"
        print(f"{img_path.name:<30} {defect_info:<15} {label} [{status}]")

    print("=" * 60)
    print(f"\n统计结果:")
    print(f"  良品 (qualified): {qualified_count}")
    print(f"  残次品 (defective): {defective_count}")
    print(f"  合计: {qualified_count + defective_count}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 提交**

```bash
git add src/predict.py
git commit -m "feat: add defect detection and classification script"
```

---

### Task 8: 运行推理验证

- [ ] **Step 1: 运行推理**

```bash
cd /Users/raidery/bench/yolo8/yyc-seg
python src/predict.py
```

预期输出：
```
已加载模型: runs/seg/train/weights/best.pt

找到 73 张测试图片

============================================================
图片名称                           检测结果        判定
============================================================
xxx.jpg                           scratch(0.95)   残次品 [FAIL]
xxx.jpg                                               良品 [PASS]
...
============================================================

统计结果:
  良品 (qualified): XX
  残次品 (defective): XX
  合计: 73
```

- [ ] **Step 2: 提交**

```bash
git add runs/seg/predict/
git commit -m "feat: add prediction results"
```

---

### Task 9: 结果评估

- [ ] **Step 1: 查看训练指标**

```bash
# 查看混淆矩阵
ls runs/seg/train/

# 查看验证集precision/recall
cat runs/seg/train/results.csv | head -20
```

- [ ] **Step 2: 手动验证（可选）**

选择几张验证集图片，肉眼检查检测结果是否正确。

- [ ] **Step 3: 提交评估结果**

```bash
git add runs/seg/train/results.csv
git commit -m "docs: add training results summary"
```

---

## 验收检查清单

- [ ] Task 1: configs/data.yaml 已创建
- [ ] Task 2: src/filter_dataset.py 已创建
- [ ] Task 3: 数据筛选完成，筛选出 2 类数据
- [ ] Task 4: 预训练模型已就绪
- [ ] Task 5: src/train.py 已创建
- [ ] Task 6: 模型训练完成，best.pt 已生成
- [ ] Task 7: src/predict.py 已创建
- [ ] Task 8: 推理验证完成，能正确输出分类判定
- [ ] Task 9: 结果评估完成

---

## 注意事项

1. **GPU 显存不足时**：将 train.py 中的 `batch=8` 改为 `batch=4` 或 `batch=2`
2. **无 GPU 时**：将 train.py 中的 `device="0"` 改为 `device="cpu"`（训练会非常慢）
3. **断点续训**：YOLOv8 支持断点续训，中断后重新运行 train.py 即可
4. **模型切换**：如需更轻量模型，将 `yolov8s.pt` 改为 `yolov8n.pt`