# 文件名: MoDL-main/MoDL_seg/data_load.py (最终完整版 - 路径已更新)
import numpy as np
from PIL import Image
import os
import glob
import cv2  # 引入OpenCV库

# 禁用PIL读取大图像时的安全限制
Image.MAX_IMAGE_PIXELS = None


class DataProcess25D_Auto_Scan_Advanced(object):

    def __init__(self, out_rows, out_cols,
                 data_dir,  # <-- 改为参数传入
                 label_dir,  # <-- 改为参数传入
                 label_suffix="_bw",
                 npy_path="../npydata"):
        self.out_rows = out_rows
        self.out_cols = out_cols
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.label_suffix = label_suffix
        self.npy_path = npy_path

    def read_32bit_tif_and_convert_to_8bit(self, file_path):
        with Image.open(file_path) as img:
            frames = []
            for i in range(img.n_frames):
                img.seek(i)
                frame_32bit = np.array(img)
                min_val, max_val = frame_32bit.min(), frame_32bit.max()
                if max_val > min_val:
                    frame_normalized = (frame_32bit - min_val) / (max_val - min_val)
                else:
                    frame_normalized = np.zeros_like(frame_32bit)
                frame_8bit = (frame_normalized * 255).astype(np.uint8)

                if frame_8bit.shape[0] != self.out_rows or frame_8bit.shape[1] != self.out_cols:
                    frame_8bit = cv2.resize(frame_8bit, (self.out_cols, self.out_rows), interpolation=cv2.INTER_AREA)
                frames.append(frame_8bit)
        return frames

    def binarize_rgb_mask_from_path(self, file_path):
        with Image.open(file_path) as img:
            frames = []
            for i in range(img.n_frames):
                img.seek(i)
                img_np = np.array(img.convert('RGB'))
                binary_mask = np.zeros((img_np.shape[0], img_np.shape[1]), dtype=np.uint8)
                non_black_pixels_mask = img_np.sum(axis=2) > 0
                binary_mask[non_black_pixels_mask] = 255

                if binary_mask.shape[0] != self.out_rows or binary_mask.shape[1] != self.out_cols:
                    binary_mask = cv2.resize(binary_mask, (self.out_cols, self.out_rows),
                                             interpolation=cv2.INTER_NEAREST)
                frames.append(binary_mask)
        return frames

    def create_train_data(self):
        print('-' * 30)
        print('Creating 2.5D training data (handling 32-bit raw & RGB mask with RESIZING)...')
        print('-' * 30)

        raw_files = glob.glob(os.path.join(self.data_dir, "*.tif")) + glob.glob(os.path.join(self.data_dir, "*.tiff"))

        if not raw_files:
            print(f"ERROR: No .tif or .tiff files found in '{self.data_dir}'.")
            return

        print(f"Found {len(raw_files)} raw data file(s) to process.")

        all_img_stacks, all_label_stacks = [], []

        for raw_file_path in raw_files:
            base_name, ext = os.path.splitext(os.path.basename(raw_file_path))
            label_file_name = f"{base_name}{self.label_suffix}{ext}"
            label_file_path = os.path.join(self.label_dir, label_file_name)

            print(f"\n--- Processing raw file: {os.path.basename(raw_file_path)} ---")
            print(f"    Expecting label file: {label_file_name}")

            if not os.path.exists(label_file_path):
                print(f"  -> WARNING: Label file not found. Skipping.")
                continue

            try:
                raw_frames = self.read_32bit_tif_and_convert_to_8bit(raw_file_path)
                print(f"  -> Loaded, converted, and RESIZED {len(raw_frames)} raw frames.")
                label_frames = self.binarize_rgb_mask_from_path(label_file_path)
                print(f"  -> Loaded, binarized, and RESIZED {len(label_frames)} mask frames.")

            except Exception as e:
                print(f"  -> ERROR loading this file pair: {e}. Skipping.")
                continue

            if len(raw_frames) != len(label_frames) or len(raw_frames) < 3:
                print(f"  -> WARNING: Frame count mismatch or insufficient frames. Skipping.")
                continue

            for i in range(1, len(raw_frames) - 1):
                img_prev, img_center, img_next = raw_frames[i - 1], raw_frames[i], raw_frames[i + 1]
                label_center = label_frames[i]

                stacked_images = np.stack([img_prev, img_center, img_next], axis=0)
                stacked_images = np.expand_dims(stacked_images, axis=-1)
                label_center = np.expand_dims(label_center, axis=-1)

                all_img_stacks.append(stacked_images)
                all_label_stacks.append(label_center)

        if not all_img_stacks:
            print("\nERROR: No valid samples were generated.")
            return

        imgdatas = np.array(all_img_stacks)
        imglabels = np.array(all_label_stacks)

        print(f'\nTotal samples generated: {len(imgdatas)}.')
        print(f'Final input data shape: {imgdatas.shape}')
        print(f'Final label data shape: {imglabels.shape}')

        os.makedirs(self.npy_path, exist_ok=True)
        np.save(os.path.join(self.npy_path, 'imgs_train_2.5D_seq.npy'), imgdatas)
        np.save(os.path.join(self.npy_path, 'imgs_mask_train_2.5D_seq.npy'), imglabels)
        print("Successfully saved final 2.5D sequence dataset.")


# ==========================================================
# #####         这是之前被截断的、现在已完整的代码         #####
# ==========================================================
if __name__ == "__main__":
    # 已经按照您的要求，更新了数据文件夹路径
    # 如果未来需要修改，请在这里修改

    data_directory = "../10.15data_1-10/train/"
    label_directory = "../10.15data_1-10/label/"

    # 实例化数据处理类
    mydata = DataProcess25D_Auto_Scan_Advanced(
        out_rows=512,
        out_cols=512,
        data_dir=data_directory,
        label_dir=label_directory
    )

    # 运行数据创建流程
    mydata.create_train_data()