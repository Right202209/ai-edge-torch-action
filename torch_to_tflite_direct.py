# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torchvision import models
import ai_edge_torch  # 核心库
import os

def export_direct_tflite(model_path, output_path='models/direct_model.tflite'):
    print(f"🚀 开始加载模型: {model_path}")
    
    # 1. 重新构建模型结构 (必须与训练时一致)
    # 这里的 num_classes 需要根据你训练时的类别数修改，或者让脚本自动检测
    # 为了演示，假设我们要先加载权重来确定
    try:
        # 加载权重
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        
        # 尝试从权重中推断类别数 (根据 classifier.3.weight 的形状)
        # MobileNetV3 Small 的最后一层通常是 'classifier.3.weight'
        if 'classifier.3.weight' in checkpoint:
            num_classes = checkpoint['classifier.3.weight'].shape[0]
        else:
            # 如果找不到，请手动指定，例如 num_classes = 5
            print("⚠️ 无法自动检测类别数，默认设为 10，请检查代码！")
            num_classes = 10 
            
        print(f"ℹ️ 检测到类别数量: {num_classes}")

        # 实例化模型
        model = models.mobilenet_v3_small(weights=None)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
        
        # 加载权重
        model.load_state_dict(checkpoint)
        model.eval() # 切换到评估模式
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 2. 准备 Dummy Input (输入样本)
    # MobileNetV3 输入通常是 (Batch, Channel, Height, Width)
    sample_input = (torch.randn(1, 3, 224, 224),)

    print("🔄 正在直接转换为 TFLite (跳过 ONNX)...")

    try:
        # 3. 核心转换步骤
        edge_model = ai_edge_torch.convert(model, sample_input)
        
        # 4. 保存模型
        edge_model.export(output_path)
        
        print("\n" + "="*30)
        print("✅ 转换成功！")
        print(f"💾 输出文件: {os.path.abspath(output_path)}")
        
        # 打印大小
        size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"📦 模型体积: {size:.2f} MB")
        
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        print("提示: 确保你安装了 pip install ai-edge-torch")

if __name__ == "__main__":
    # 配置路径
    MODEL_PTH = 'crop_disease_v3.pth' # 你的 .pth 文件路径
    OUTPUT_TFLITE = 'models/crop_disease_direct.tflite'
    
    if os.path.exists(MODEL_PTH):
        export_direct_tflite(MODEL_PTH, OUTPUT_TFLITE)
    else:
        print(f"错误: 找不到文件 {MODEL_PTH}")
