import os
import glob
import numpy as np
import cv2
from PIL import Image
from pathlib import Path


def final_separation_canonical_watershed(multiframe_mask_path, output_dir):
    """
    使用教科书标准的“分水岭”算法范式，最终解决过度分割和全黑问题。
    """
    try:
        base_name = Path(multiframe_mask_path).stem
        debug_dir = os.path.join(output_dir, f"{base_name}_DEBUG")
        os.makedirs(debug_dir, exist_ok=True)

        print(f"\n--- Processing file: {base_name}.tif (Using Canonical Watershed) ---")

        with Image.open(multiframe_mask_path) as img:
            num_frames = img.n_frames
            if num_frames < 3:
                print(f"  -> Skipping: File needs at least 3 frames.")
                return

            original_center_frame_pil = img.seek(1) or img.copy()

            def robust_binarize(pil_image):
                img_np = np.array(pil_image.convert('RGB'))
                binary_mask = np.zeros((img_np.shape[0], img_np.shape[1]), dtype=np.uint8)
                non_black_pixels_mask = img_np.sum(axis=2) > 0
                binary_mask[non_black_pixels_mask] = 255
                return binary_mask

            img.seek(0);
            mask_prev = robust_binarize(img)
            img.seek(1);
            mask_center = robust_binarize(img)
            img.seek(2);
            mask_next = robust_binarize(img)

            cv2.imwrite(os.path.join(debug_dir, '1_center_mask.tif'), mask_center)

    except Exception as e:
        print(f"  -> ERROR during file loading: {e}")
        return

    # --- 1. 智能种子生成 (与之前一样，这个逻辑是正确的) ---
    intersection_prev = cv2.bitwise_and(mask_center, mask_prev)
    seed_past = cv2.subtract(intersection_prev, mask_next)
    intersection_next = cv2.bitwise_and(mask_center, mask_next)
    seed_future = cv2.subtract(intersection_next, mask_prev)
    kernel_erode = np.ones((3, 3), np.uint8)
    seed_past = cv2.erode(seed_past, kernel_erode, iterations=1)
    seed_future = cv2.erode(seed_future, kernel_erode, iterations=1)

    sure_foreground = cv2.bitwise_or(seed_past, seed_future)
    cv2.imwrite(os.path.join(debug_dir, '2_smart_seeds.tif'), sure_foreground)

    if np.sum(sure_foreground) == 0:
        print("  -> WARNING: No distinct seeds generated. Cannot perform separation. Saving original.")
        final_output_cv = cv2.cvtColor(np.array(original_center_frame_pil), cv2.COLOR_RGB2BGR)
        output_path = os.path.join(output_dir, f"{base_name}_separated_FAILED.tif")
        cv2.imwrite(output_path, final_output_cv)
        return

    # ====================================================================
    # #####           教科书标准的 Watershed 准备步骤           #####
    # ====================================================================

    # --- 2. 找到“确定无疑”的背景 ---
    # 对中心帧进行膨胀，得到的黑色区域就是确定无疑的背景
    kernel_dilate = np.ones((3, 3), np.uint8)
    sure_background = cv2.dilate(mask_center, kernel_dilate, iterations=3)
    cv2.imwrite(os.path.join(debug_dir, '3_sure_background.tif'), sure_background)

    # --- 3. 找到需要划分的“未知”区域 ---
    # 未知区域 = 确定背景 - 确定前景(种子)
    unknown = cv2.subtract(sure_background, sure_foreground)
    cv2.imwrite(os.path.join(debug_dir, '4_unknown_region.tif'), unknown)

    # --- 4. 创建 Markers ---
    # 用 connectedComponents 标记我们的种子
    _, markers = cv2.connectedComponents(sure_foreground)
    # 背景标记为1
    markers = markers + 1
    # 未知区域标记为0，这是算法要去“淹没”的地方
    markers[unknown == 255] = 0

    # 【诊断】可视化markers，不同的颜色代表不同的种子
    markers_visual = cv2.applyColorMap(markers.astype(np.uint8) * 20, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(debug_dir, '5_markers_for_watershed.tif'), markers_visual)

    # ====================================================================

    # --- 5. 在原始中心帧上执行分水岭 ---
    # 注意：这次的 markers 是基于标准方法生成的，包含了确定的前景和背景
    original_center_color_cv = cv2.cvtColor(np.array(original_center_frame_pil), cv2.COLOR_RGB2BGR)
    cv2.watershed(original_center_color_cv, markers)

    # 分割线会被标记为 -1
    # 【诊断】在原图上画出分割线（红色）
    original_center_color_cv[markers == -1] = [0, 0, 255]  # BGR for red
    cv2.imwrite(os.path.join(debug_dir, '6_watershed_result_on_original.tif'), original_center_color_cv)

    # --- 6. 生成最终输出 ---
    # 创建一个最终的模板，所有不是分割线和背景的区域都是前景
    final_separation_mask = np.zeros_like(mask_center)
    final_separation_mask[markers > 1] = 255

    # 应用这个模板到原始中心帧
    final_output = cv2.bitwise_and(original_center_color_cv, original_center_color_cv, mask=final_separation_mask)

    output_path = os.path.join(output_dir, f"{base_name}_separated.tif")
    cv2.imwrite(output_path, final_output)
    print(f"  -> Successfully saved final separated mask to: {output_path}")


# --- 主程序 (不变) ---
if __name__ == "__main__":
    input_data_dir = "../final_results_all-time/"
    output_dir = "../final_results_all-time-processv5/"
    os.makedirs(output_dir, exist_ok=True)
    files_to_process = glob.glob(os.path.join(input_data_dir, "*.tif")) + glob.glob(
        os.path.join(input_data_dir, "*.tiff"))

    if not files_to_process:
        print("No image files found.")
    else:
        for file_path in files_to_process:
            final_separation_canonical_watershed(file_path, output_dir)

    print("\n\nAll files have been processed.")