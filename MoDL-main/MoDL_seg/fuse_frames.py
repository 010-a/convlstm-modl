import os
import glob
import numpy as np
import cv2
from PIL import Image
from pathlib import Path


def fuse_multiframe_tiff_by_majority_vote(input_tiff_path, output_dir, threshold_ratio=0.5):
    """
    读取一个多帧TIF文件，通过多数投票原则将其融合成一个单帧TIF。

    Args:
        input_tiff_path (str): 输入的多帧TIF文件路径。
        output_dir (str): 输出融合结果的目录。
        threshold_ratio (float): 投票的阈值比例 (例如0.5代表超过半数)。
    """
    try:
        print(f"\nProcessing file: {Path(input_tiff_path).name}")

        with Image.open(input_tiff_path) as img:
            num_frames = img.n_frames
            if num_frames == 0:
                print("  -> Warning: File contains no frames. Skipping.")
                return

            # 获取图像尺寸
            width, height = img.size

            # 初始化一个累加器，用于“投票”
            vote_accumulator = np.zeros((height, width), dtype=np.float32)

            print(f"  -> Found {num_frames} frames. Accumulating votes...")

            # 遍历每一帧
            for i in range(num_frames):
                img.seek(i)

                # ==========================================================
                # ##### 关键修复：在这里，我们强制将图像帧转换为灰度模式 ('L') #####
                frame_grayscale_img = img.convert("L")
                # ==========================================================

                # 将已经转为灰度的图像转为numpy数组，并归一化
                frame_np = np.array(frame_grayscale_img) / 255.0

                # 现在形状是兼容的: (2048, 2048) += (2048, 2048)
                vote_accumulator += frame_np

            # 计算投票阈值
            threshold_count = num_frames * threshold_ratio
            print(f"  -> Vote threshold set to > {threshold_count:.2f} frames.")

            # 根据阈值生成最终的二值掩码
            fused_mask = (vote_accumulator > threshold_count).astype(np.uint8) * 255

            # 构建输出路径并保存
            base_name = Path(input_tiff_path).stem
            # 修正路径拼接，避免Windows系统下的路径问题
            output_path = os.path.join(output_dir, f"{base_name}_fused.tif")

            cv2.imwrite(output_path, fused_mask)
            print(f"  -> Successfully saved fused mask to: {output_path}")

    except Exception as e:
        print(f"  -> An error occurred while processing {input_tiff_path}: {e}")


if __name__ == "__main__":
    # 定义输入和输出目录
    final_results_storage_dir = "../final_results_all-time/"
    fused_output_dir = "../fused_results/"

    # 确保输出目录存在
    os.makedirs(fused_output_dir, exist_ok=True)
    print(f"Output directory created at: {fused_output_dir}")

    # 找到所有需要处理的 *_bw.tif 文件
    files_to_process = glob.glob(os.path.join(final_results_storage_dir, "*_bw.tif"))

    if not files_to_process:
        print("\nNo '*_bw.tif' files found in the input directory. Make sure you have run 'segment_predict.py' first.")
    else:
        print(f"\nFound {len(files_to_process)} files to fuse.")
        for file_path in files_to_process:
            fuse_multiframe_tiff_by_majority_vote(file_path, fused_output_dir)

    print("\n\nFusion process complete.")