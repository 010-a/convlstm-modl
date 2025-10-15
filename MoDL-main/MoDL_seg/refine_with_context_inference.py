import os
import glob
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array


# --- 核心函数：分块预测并拼接 ---
def predict_full_frame_by_patching(model, full_frame_np):
    """
    对2048x2048的单帧图像进行分块、预测，然后拼接成完整掩码。
    """
    # 图像和分块尺寸
    img_size = 2048
    patch_size = 512
    num_patches_per_dim = img_size // patch_size  # 结果是 4

    # 1. 裁剪大图为小块 (patches)
    patches = []
    for y in range(num_patches_per_dim):
        for x in range(num_patches_per_dim):
            y_start, y_end = y * patch_size, (y + 1) * patch_size
            x_start, x_end = x * patch_size, (x + 1) * patch_size
            patch = full_frame_np[y_start:y_end, x_start:x_end]
            patches.append(patch)

    # 2. 批量准备模型输入
    patches_np = np.array(patches).astype('float32')
    patches_np /= 255.0
    mean = np.mean(patches_np)  # 使用所有小块的均值
    patches_np -= mean
    patches_np = np.expand_dims(patches_np, axis=-1)  # 增加channel维度 (16, 512, 512, 1)

    # 3. 批量预测所有小块
    predicted_patches = model.predict(patches_np)

    # 4. 拼接结果
    stitched_mask = np.zeros((img_size, img_size), dtype=np.uint8)
    patch_idx = 0
    for y in range(num_patches_per_dim):
        for x in range(num_patches_per_dim):
            # 获取预测结果并后处理
            predicted_patch = predicted_patches[patch_idx, :, :, 0]
            predicted_patch[predicted_patch > 0.7] = 255
            predicted_patch[predicted_patch <= 0.7] = 0

            # 将处理后的小块放回大图的正确位置
            y_start, y_end = y * patch_size, (y + 1) * patch_size
            x_start, x_end = x * patch_size, (x + 1) * patch_size
            stitched_mask[y_start:y_end, x_start:x_end] = predicted_patch.astype(np.uint8)
            patch_idx += 1

    return stitched_mask


# --- 核心函数：智能分水岭 (不变) ---
def refine_frame_by_watershed(mask_prev, mask_center, mask_next):
    # (这个函数无需改动)
    seed_from_prev = cv2.bitwise_and(mask_center, mask_prev)
    seed_from_next = cv2.bitwise_and(mask_center, mask_next)
    kernel = np.ones((3, 3), np.uint8)
    seed_from_prev = cv2.erode(seed_from_prev, kernel, iterations=2)
    seed_from_next = cv2.erode(seed_from_next, kernel, iterations=2)
    seed_from_next[seed_from_prev > 0] = 0
    _, markers_prev = cv2.connectedComponents(seed_from_prev)
    _, markers_next = cv2.connectedComponents(seed_from_next)
    if np.max(markers_prev) > 0 and np.max(markers_next) > 0:
        markers_next[markers_next > 0] += np.max(markers_prev)
    markers = markers_prev + markers_next
    markers += 1
    markers[mask_center == 255] = 0
    mask_center_color = cv2.cvtColor(mask_center, cv2.COLOR_GRAY2BGR)
    cv2.watershed(mask_center_color, markers)
    refined_mask = np.zeros_like(mask_center)
    refined_mask[markers > 1] = 255
    return refined_mask


# --- 主程序 (修改了调用方式) ---
if __name__ == "__main__":
    input_dir = "../testraw_multiframe/"
    output_dir = "../refined_inference_results/"
    model_path = "../model/U-RNet+.hdf5"

    os.makedirs(output_dir, exist_ok=True)

    print("Loading pre-trained model...")
    try:
        model = load_model(model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()

    print(f"Scanning for .tif and .tiff files in '{input_dir}'...")
    image_files = glob.glob(os.path.join(input_dir, "*.tif")) + glob.glob(os.path.join(input_dir, "*.tiff"))

    if not image_files:
        print("No image files found.")
        exit()

    print(f"Found {len(image_files)} file(s) to process.")

    for file_path in image_files:
        print(f"\n=============================================")
        print(f"Processing file: {os.path.basename(file_path)}")
        print(f"=============================================")

        try:
            with Image.open(file_path) as img:
                frames = []
                for i in range(img.n_frames):
                    img.seek(i)
                    frames.append(np.array(img.convert("L")))

            if len(frames) < 3:
                print(f"  -> Warning: File has fewer than 3 frames ({len(frames)}). Skipping.")
                continue

            print(f"  -> Loaded {len(frames)} frames.")

            for i in range(1, len(frames) - 1):
                print(f"\n--- Processing center frame {i} of {os.path.basename(file_path)} ---")

                frame_prev = frames[i - 1]
                frame_center = frames[i]
                frame_next = frames[i + 1]

                print(f"  -> Predicting mask for frame {i - 1} (by patching)...")
                mask_prev = predict_full_frame_by_patching(model, frame_prev)
                print(f"  -> Predicting mask for frame {i} (by patching)...")
                mask_center = predict_full_frame_by_patching(model, frame_center)
                print(f"  -> Predicting mask for frame {i + 1} (by patching)...")
                mask_next = predict_full_frame_by_patching(model, frame_next)

                print(f"  -> Refining mask for frame {i} using context...")
                refined_mask = refine_frame_by_watershed(mask_prev, mask_center, mask_next)

                file_base_name = Path(file_path).stem
                output_path = os.path.join(output_dir, f"{file_base_name}_refined_frame_{i}.tif")
                cv2.imwrite(output_path, refined_mask)
                print(f"  -> Saved refined mask to {output_path}")

        except Exception as e:
            print(f"  -> An error occurred while processing {os.path.basename(file_path)}: {e}")
            continue

    print("\n\nAll files processed.")