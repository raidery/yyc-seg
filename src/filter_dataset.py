"""
数据筛选脚本：从4类缺陷数据中筛选出2类（scratch, damage）
原始类别：0-scratch, 1-dirty, 2-stain, 3-damage
目标类别：0-scratch, 1-damage
"""
import shutil
from pathlib import Path

# 原始数据集路径
SRC_ROOT = Path("/root/autodl-tmp/Aero-engine_defect-detect_new")
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