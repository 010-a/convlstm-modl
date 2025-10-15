import os
import glob
import numpy as np
import cv2
from PIL import Image
from pathlib import Path


def refine_segmentation_with_context(mask_file_path, output_dir):
    """
    读取一个多帧分割结果的TIF文件，利用前后帧信息优化中间帧的分割。

    Args:
        mask_file_path (str): 输入的多帧mask TIF文件路径。
        output_dir (str): 输出优化后单帧结果的目录。
    """
    try:
        print(f"\nProcessing file for refinement: {Path(mask_file_path).name}")

        with Image.open(mask_file_path) as img:
            num_frames = img.n_frames
            if num_frames < 3:
                print(
                    f"  -> Error: File must contain at least 3 frames for context refinement. Found {num_frames}. Skipping.")
                return

            # 读取第1, 2, 3帧并确保是灰度图
            img.seek(0)
            mask_1 = np.array(img.convert("L"))
            img.seek(1)
            mask_2 = np.array(img.convert("L"))
            img.seek(2)
            mask_3 = np.array(img.convert("L"))

            # --- 分水岭算法准备 ---

            # 1. 找到确定无疑的前景（种子）
            kernel = np.ones((3, 3), np.uint8)
            sure_fg_1 = cv2.erode(mask_1, kernel, iterations=2)
            sure_fg_2_only = cv2.erode(mask_2, kernel, iterations=2)
            sure_fg_3 = cv2.erode(mask_3, kernel, iterations=2)

            sure_fg_2_only[sure_fg_1 > 0] = 0
            sure_fg_2_only[sure_fg_3 > 0] = 0

            # 2. 创建分水岭算法的 markers (标记)
            _, markers_1 = cv2.connectedComponents(sure_fg_1)
            _, markers_2 = cv2.connectedComponents(sure_fg_2_only)
            _, markers_3 = cv2.connectedComponents(sure_fg_3)

            markers_2[markers_2 > 0] += np.max(markers_1)
            markers_3[markers_3 > 0] += np.max(markers_2)

            markers = markers_1 + markers_2 + markers_3
            markers += 1

            # 找到“未知区域”并标记为0
            # 这里我们定义未知区域为第2帧的所有前景
            unknown_area = mask_2
            markers[unknown_area == 255] = 0

            # 3. 应用分水岭
            mask_2_color = cv2.cvtColor(mask_2, cv2.COLOR_GRAY2BGR)
            cv2.watershed(mask_2_color, markers)

            # ==========================================================
            # #####           关键修改在这里           #####
            # ==========================================================

            # 4. 生成最终优化后的第2帧掩码
            # 创建一个全黑的画布
            final_refined_mask_2 = np.zeros_like(mask_2)

            # 在`markers`矩阵中，所有大于1的区域都代表被成功划分给了某个种子。
            # 标记为-1的是分割线，标记为1的是背景。
            # 我们只需要把所有>1的区域涂成白色，就能得到干净、分离的掩码。
            final_refined_mask_2[markers > 1] = 255

            # ==========================================================
            # #####             修改结束             #####
            # ==========================================================

            # 保存结果
            base_name = Path(mask_file_path).stem
            output_path = os.path.join(output_dir, f"{base_name}_refined_frame_2.tif")
            cv2.imwrite(output_path, final_refined_mask_2)
            print(f"  -> Successfully saved refined frame 2 to: {output_path}")

    except Exception as e:
        print(f"  -> An error occurred: {e}")


if __name__ == "__main__":
    # 定义输入和输出目录
    final_results_storage_dir = "../final_results_all-time/"
    refined_output_dir = "../refined_results/"

    os.makedirs(refined_output_dir, exist_ok=True)
    print(f"Refined output directory created at: {refined_output_dir}")

    # 找到所有需要处理的 *_bw.tif 文件
    files_to_process = glob.glob(os.path.join(final_results_storage_dir, "*_bw.tif"))

    if not files_to_process:
        print("\nNo '*_bw.tif' files found. Run 'segment_predict.py' first.")
    else:
        for file_path in files_to_process:
            refine_segmentation_with_context(file_path, refined_output_dir)

    print("\n\nRefinement process complete.")