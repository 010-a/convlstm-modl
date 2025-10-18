# 文件名: MoDL-main/MoDL_seg/predict_center_frame.py (新文件)
import numpy as np
from PIL import Image
import os
import glob
import cv2
from pathlib import Path
from tensorflow.keras.models import load_model


# 禁用PIL和TensorFlow的一些冗余日志
Image.MAX_IMAGE_PIXELS = None
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def predict_center_frame_with_2_5d_model(model, raw_frames_3_seq, target_size=(512, 512)):
    """
    使用2.5D模型，根据一个3帧的序列，预测中心帧的分割掩码。

    Args:
        model: 加载好的2.5D Keras模型。
        raw_frames_3_seq (list of np.array): 包含3个原始帧的列表 (8-bit, 0-255)。
        target_size (tuple): 模型期望的输入尺寸 (宽度, 高度)。

    Returns:
        np.array: 预测出的单帧分割掩码。
    """
    print("  -> Preparing 3-frame sequence for model...")

    # 1. 尺寸调整和数据类型转换
    frames_to_stack = []
    for frame in raw_frames_3_seq:
        if frame.shape[1] != target_size[0] or frame.shape[0] != target_size[1]:
            frame_resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
        else:
            frame_resized = frame
        frames_to_stack.append(frame_resized.astype(np.float32) / 255.0)

    # 2. 堆叠并塑造成模型期望的输入形状
    # (1, 3, 512, 512, 1) = (batch, timesteps, height, width, channels)
    input_sequence = np.stack(frames_to_stack, axis=0)
    input_sequence = np.expand_dims(input_sequence, axis=0)
    input_sequence = np.expand_dims(input_sequence, axis=-1)

    # 3. 模型预测
    print("  -> Predicting center frame...")
    predicted_mask_prob = model.predict(input_sequence)

    # 4. 后处理预测结果
    predicted_mask = predicted_mask_prob[0, :, :, 0]
    predicted_mask[predicted_mask > 0.5] = 255
    predicted_mask[predicted_mask <= 0.5] = 0
    predicted_mask = predicted_mask.astype(np.uint8)

    # 5. 将结果放大回原始尺寸
    original_center_frame_shape = raw_frames_3_seq[1].shape  # (高度, 宽度)
    original_size_wh = (original_center_frame_shape[1], original_center_frame_shape[0])

    if original_size_wh != target_size:
        predicted_mask = cv2.resize(predicted_mask, original_size_wh, interpolation=cv2.INTER_NEAREST)

    return predicted_mask


# --- 主程序 ---
if __name__ == "__main__":
    # --- 配置 ---
    input_dir = "../10.15data_1-10/test/"
    output_dir = "../10.15data_1-10/test_predictions_center_frame_stable/"
    model_path = "../model/convlstm_unet_stable_final.hdf5"

    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载模型
    print("Loading fine-tuned 2.5D model...")
    try:
        model = load_model(model_path, compile=False)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        exit()

    # 2. 扫描输入目录
    print(f"Scanning for .tif and .tiff files in '{input_dir}'...")
    raw_files = glob.glob(os.path.join(input_dir, "*.tif")) + glob.glob(os.path.join(input_dir, "*.tiff"))

    if not raw_files:
        print("No image files found to predict.")
        exit()

    print(f"Found {len(raw_files)} file(s) to process.")

    # 3. 循环处理每一个文件
    for file_path in raw_files:
        print(f"\n=============================================")
        print(f"Processing file: {os.path.basename(file_path)}")

        try:
            with Image.open(file_path) as img:
                if img.n_frames < 3:
                    print("  -> WARNING: File has fewer than 3 frames. Skipping.")
                    continue

                # 只读取前3帧
                raw_frames_32bit = [(img.seek(i) or np.array(img)) for i in range(3)]

            # 转换数据类型
            raw_frames_8bit = []
            for frame_32bit in raw_frames_32bit:
                min_val, max_val = frame_32bit.min(), frame_32bit.max()
                if max_val > min_val:
                    frame_normalized = (frame_32bit - min_val) / (max_val - min_val)
                else:
                    frame_normalized = np.zeros_like(frame_32bit)
                raw_frames_8bit.append((frame_normalized * 255).astype(np.uint8))
            print(f"  -> Loaded and converted the first 3 frames to 8-bit.")

            # 使用新模型进行预测
            predicted_center_mask = predict_center_frame_with_2_5d_model(model, raw_frames_8bit)

            # 保存单帧的预测结果
            output_filename = f"{Path(file_path).stem}_center_frame_predicted.tif"
            output_path = os.path.join(output_dir, output_filename)

            # 使用cv2.imwrite直接保存单帧numpy数组
            cv2.imwrite(output_path, predicted_center_mask)
            print(f"  -> Successfully saved SINGLE FRAME prediction to: {output_path}")

        except Exception as e:
            print(f"  -> An ERROR occurred while processing {os.path.basename(file_path)}: {e}")
            continue

    print("\n\nAll predictions complete.")