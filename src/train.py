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