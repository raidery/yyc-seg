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