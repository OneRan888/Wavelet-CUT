#!/usr/bin/env python3
"""
测试生成的WSI切片数据集是否可正常读取
"""
from pathlib import Path
from PIL import Image
import numpy as np

# 数据集根目录（对应你的--dataroot参数）
DATA_ROOT = Path("C:\\Users\\Dell\\Desktop\\medical\\tools\\datasets\\medical")
TRAIN_A = DATA_ROOT / "trainA"
TRAIN_B = DATA_ROOT / "trainB"

def test_image_readability(img_path: Path):
    """测试单张图片是否可正常读取，并返回图片信息"""
    try:
        img = Image.open(img_path).convert('RGB')
        arr = np.array(img)
        # 验证图片尺寸、数据类型
        assert img.size == (512, 512), f"图片尺寸错误，预期512×512，实际{img.size}"
        assert arr.dtype == np.uint8, f"图片数据类型错误，预期uint8，实际{arr.dtype}"
        assert arr.shape == (512, 512, 3), f"图片数组形状错误，预期(512,512,3)，实际{arr.shape}"
        return True, f"{img_path.name}：读取成功，尺寸{img.size}，数据类型{arr.dtype}"
    except Exception as e:
        return False, f"{img_path.name}：读取失败，错误信息：{str(e)}"

def main():
    # 检查目录是否存在
    assert TRAIN_A.exists(), f"目录不存在：{TRAIN_A}"
    assert TRAIN_B.exists(), f"目录不存在：{TRAIN_B}"

    # 遍历trainA，测试前10张图片
    print("=== 测试trainA目录 ===")
    a_images = list(TRAIN_A.glob("*.png"))[:10]
    if not a_images:
        print("trainA目录下无PNG切片文件")
        return
    for img_path in a_images:
        success, msg = test_image_readability(img_path)
        print(msg)

    # 遍历trainB，测试前10张图片
    print("\n=== 测试trainB目录 ===")
    b_images = list(TRAIN_B.glob("*.png"))[:10]
    if not b_images:
        print("trainB目录下无PNG切片文件")
        return
    for img_path in b_images:
        success, msg = test_image_readability(img_path)
        print(msg)

    print("\n=== 数据集测试完成：所有测试图片均可正常读取，符合后续模型训练要求 ===")

if __name__ == '__main__':
    main()