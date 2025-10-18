
import numpy as np
from PIL import Image
import os
import glob
import cv2
import albumentations as A
from pathlib import Path

# 禁用PIL读取大图像时的安全限制
Image.MAX_IMAGE_PIXELS = None


# --- 1. 定义我们的数据增强“管道” ---
def get_augmentation_pipeline(p=0.5):
    """
    定义一个Albumentations增强管道。
    p: 每种增强应用的概率。
    """
    return A.Compose([
        # --- 几何变换 (对图像和掩码同时应用) ---
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.0625,
            scale_limit=0.1,
            rotate_limit=15,  # 随机旋转-15到+15度
            p=0.7
        ),

        # --- 像素级变换 (只对图像应用) ---
        A.RandomBrightnessContrast(p=0.5),
        A.GaussNoise(p=0.3),
        A.GaussianBlur(p=0.3),
    ], p=1.0)  # p=1.0 确保每次调用都至少应用一种增强


# --- 2. 图像读写和处理的辅助函数 ---
def read_and_convert_frames(raw_path, label_path):
    """读取原始数据和标签，并转换为8-bit numpy列表"""
    # 读取32-bit原始数据并转为8-bit
    with Image.open(raw_path) as img:
        raw_frames = []
        for i in range(img.n_frames):
            img.seek(i)
            frame_32bit = np.array(img)
            min_val, max_val = frame_32bit.min(), frame_32bit.max()
            if max_val > min_val:
                frame_normalized = (frame_32bit - min_val) / (max_val - min_val)
            else:
                frame_normalized = np.zeros_like(frame_32bit)
            raw_frames.append((frame_normalized * 255).astype(np.uint8))

    # 读取RGB掩码并二值化
    with Image.open(label_path) as img:
        label_frames = []
        for i in range(img.n_frames):
            img.seek(i)
            img_np = np.array(img.convert('RGB'))
            binary_mask = (img_np.sum(axis=2) > 0).astype(np.uint8) * 255
            label_frames.append(binary_mask)

    return raw_frames, label_frames


def save_frames_as_multiframe_tif(frames, output_path):
    """将一个numpy数组列表保存为多帧TIF"""
    pil_images = [Image.fromarray(frame) for frame in frames]
    pil_images[0].save(
        output_path,
        save_all=True,
        append_images=pil_images[1:],
        compression="tiff_deflate"
    )


# --- 3. 主程序 ---
if __name__ == "__main__":
    # --- 配置 ---
    input_raw_dir = "../10.15data_1-10/train/"
    input_label_dir = "../10.15data_1-10/label/"
    output_dir = "../10.15data_1-10/data_enhance/"
    label_suffix = "_bw"

    # 您想为每个原始文件生成多少个增强版本
    num_augmentations_per_file = 5

    os.makedirs(output_dir, exist_ok=True)

    # 扫描输入文件
    raw_files = glob.glob(os.path.join(input_raw_dir, "*.tif")) + \
                glob.glob(os.path.join(input_raw_dir, "*.tiff"))

    if not raw_files:
        print(f"ERROR: No raw files found in '{input_raw_dir}'")
        exit()

    print(
        f"Found {len(raw_files)} raw file(s). Will generate {num_augmentations_per_file} augmented versions for each.")

    # 获取增强管道
    augmenter = get_augmentation_pipeline()

    # 循环处理每个原始文件
    for raw_file_path in raw_files:
        base_name, ext = os.path.splitext(os.path.basename(raw_file_path))
        label_file_name = f"{base_name}{label_suffix}{ext}"
        label_file_path = os.path.join(input_label_dir, label_file_name)

        print(f"\n--- Processing: {base_name}{ext} ---")

        if not os.path.exists(label_file_path):
            print(f"  -> WARNING: Label file not found. Skipping.")
            continue

        # 读取原始数据和标签
        try:
            raw_frames, label_frames = read_and_convert_frames(raw_file_path, label_file_path)
            print(f"  -> Successfully loaded {len(raw_frames)} frames.")
        except Exception as e:
            print(f"  -> ERROR loading file pair: {e}. Skipping.")
            continue

        # 循环生成增强数据
        for i in range(num_augmentations_per_file):

            augmented_raw_frames = []
            augmented_label_frames = []

            # 对序列中的每一帧应用相同的增强
            # (Albumentations每次调用都会产生不同的随机变换)
            for raw_frame, label_frame in zip(raw_frames, label_frames):
                # 'image'对应原始帧, 'mask'对应标签帧
                augmented = augmenter(image=raw_frame, mask=label_frame)
                augmented_raw_frames.append(augmented['image'])
                augmented_label_frames.append(augmented['mask'])

            # 定义新文件名
            new_raw_filename = f"{base_name}_aug_{i + 1:03d}{ext}"
            new_label_filename = f"{base_name}_aug_{i + 1:03d}{label_suffix}{ext}"

            # 保存增强后的多帧TIF文件
            # 将增强后的原始数据和标签分别保存到对应的train和label子文件夹中
            output_raw_dir = os.path.join(output_dir, "train")
            output_label_dir = os.path.join(output_dir, "label")
            os.makedirs(output_raw_dir, exist_ok=True)
            os.makedirs(output_label_dir, exist_ok=True)

            save_frames_as_multiframe_tif(augmented_raw_frames, os.path.join(output_raw_dir, new_raw_filename))
            save_frames_as_multiframe_tif(augmented_label_frames, os.path.join(output_label_dir, new_label_filename))

            print(f"  -> Saved augmented pair {i + 1}/{num_augmentations_per_file}")

    print("\n\nData augmentation complete.")
    print(
        f"Augmented files are saved in '{os.path.join(output_dir, 'train')}' and '{os.path.join(output_dir, 'label')}'")