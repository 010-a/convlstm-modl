import cv2
import numpy as np
import os
from glob import glob
from pathlib import Path
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
import shutil

# (您的 import 语句...)
from PIL import Image
import shutil # 确保导入了 shutil


def refine_masks_with_watershed(mask_path_t, mask_path_t_plus_1):
    """
    使用分水岭算法处理两个相邻帧的分割掩码，解决重叠问题。
    此函数会直接修改 t 帧的掩码文件。
    """
    # 1. 读取两个连续的掩码
    img_t = cv2.imread(mask_path_t, cv2.IMREAD_GRAYSCALE)
    img_t_plus_1 = cv2.imread(mask_path_t_plus_1, cv2.IMREAD_GRAYSCALE)

    if img_t is None or img_t_plus_1 is None:
        return

    # 2. 找到确定无疑的前景 (Sure Foreground) 作为种子
    kernel = np.ones((3, 3), np.uint8)
    # 对 t 帧进行腐蚀
    sure_fg_t = cv2.erode(img_t, kernel, iterations=2)
    # 对 t+1 帧进行腐蚀，并确保它与 t 帧的种子不重叠
    sure_fg_t_plus_1 = cv2.erode(img_t_plus_1, kernel, iterations=2)
    sure_fg_t_plus_1[sure_fg_t > 0] = 0  # 关键：避免种子重叠

    # 3. 创建分水岭算法的 markers
    # 使用 connectedComponents 标记种子，每个独立区域一个唯一标签
    _, markers_t = cv2.connectedComponents(sure_fg_t)
    _, markers_t_plus_1 = cv2.connectedComponents(sure_fg_t_plus_1)

    # 将 t+1 帧的标签值调整，使其不与 t 帧的标签冲突
    markers_t_plus_1[markers_t_plus_1 > 0] = markers_t_plus_1[markers_t_plus_1 > 0] + np.max(markers_t)

    # 合并两帧的 markers
    markers = markers_t + markers_t_plus_1

    # 背景标记为1（分水岭算法要求背景是1，未知区域是0）
    # 我们将确定前景外的所有区域暂时标记为未知(0)，背景信息稍后处理
    markers = markers + 1  # 所有标签+1，这样背景就是0

    # 找到"未知区域"
    unknown = cv2.subtract(cv2.bitwise_or(img_t, img_t_plus_1), cv2.bitwise_or(sure_fg_t, sure_fg_t_plus_1))
    markers[unknown == 255] = 0

    # 4. 应用分水岭
    # 需要将原始图像转为3通道给分水岭函数
    img_for_watershed = cv2.bitwise_or(img_t, img_t_plus_1)
    img_for_watershed_color = cv2.cvtColor(img_for_watershed, cv2.COLOR_GRAY2BGR)

    cv2.watershed(img_for_watershed_color, markers)

    # 5. 生成新的、无重叠的 t 帧掩码
    # 分割线在 markers 中被标记为 -1
    # 新的 t 帧掩码只包含原属于 t 帧的标签 (标签值在 2 到 np.max(markers_t)+1 之间)
    new_mask_t = np.zeros_like(img_t, dtype=np.uint8)
    max_label_t = np.max(markers_t) + 1
    new_mask_t[(markers > 1) & (markers <= max_label_t)] = 255

    # 6. 用优化后的结果覆盖原文件
    cv2.imwrite(mask_path_t, new_mask_t)

    print(f"Refined mask saved to: {mask_path_t}")
def split_multiframe_tiff(source_path, dest_folder):
    """将一个多帧TIF文件拆分成多个单帧文件。"""
    with Image.open(source_path) as img:
        for i in range(img.n_frames):
            img.seek(i)
            # 文件名补零，确保后续处理顺序正确
            frame_path = os.path.join(dest_folder, f"{str(i).zfill(4)}.tif")
            img.save(frame_path)
    return img.n_frames

def clear_files(folders):
    for root, dirs, files in os.walk(folders):
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)


# We initially cropped 1 original stack (2048 × 2048 pixel2 resolution) into 16 patches (512 × 512 pixel2 resolution)
# in three ways, later reassembling these segmentation patches to reconstruct a complete image.
def crop_images(input_val):
    if input_val == "44":
        crop_boxes = [[0, 0, 1 / 4, 1 / 4], [1 / 4, 0, 1 / 2, 1 / 4], [1 / 2, 0, 3 / 4, 1 / 4], [3 / 4, 0, 1, 1 / 4],
                      [0, 1 / 4, 1 / 4, 1 / 2], [1 / 4, 1 / 4, 1 / 2, 1 / 2], [1 / 2, 1 / 4, 3 / 4, 1 / 2],
                      [3 / 4, 1 / 4, 1, 1 / 2],
                      [0, 1 / 2, 1 / 4, 3 / 4], [1 / 4, 1 / 2, 1 / 2, 3 / 4], [1 / 2, 1 / 2, 3 / 4, 3 / 4],
                      [3 / 4, 1 / 2, 1, 3 / 4],
                      [0, 3 / 4, 1 / 4, 1], [1 / 4, 3 / 4, 1 / 2, 1], [1 / 2, 3 / 4, 3 / 4, 1], [3 / 4, 3 / 4, 1, 1]]

    elif input_val == "43":
        crop_boxes = [[0, 1 / 8, 1 / 4, 3 / 8], [1 / 4, 1 / 8, 1 / 2, 3 / 8], [1 / 2, 1 / 8, 3 / 4, 3 / 8],
                      [3 / 4, 1 / 8, 1, 3 / 8], [0, 3 / 8, 1 / 4, 5 / 8], [1 / 4, 3 / 8, 1 / 2, 5 / 8],
                      [1 / 2, 3 / 8, 3 / 4, 5 / 8], [3 / 4, 3 / 8, 1, 5 / 8], [0, 5 / 8, 1 / 4, 7 / 8],
                      [1 / 4, 5 / 8, 1 / 2, 7 / 8], [1 / 2, 5 / 8, 3 / 4, 7 / 8], [3 / 4, 5 / 8, 1, 7 / 8]]

    elif input_val == "34":
        crop_boxes = [[1 / 8, 0, 3 / 8, 1 / 4], [3 / 8, 0, 5 / 8, 1 / 4], [5 / 8, 0, 7 / 8, 1 / 4],
                      [1 / 8, 1 / 4, 3 / 8, 1 / 2], [3 / 8, 1 / 4, 5 / 8, 1 / 2], [5 / 8, 1 / 4, 7 / 8, 1 / 2],
                      [1 / 8, 1 / 2, 3 / 8, 3 / 4], [3 / 8, 1 / 2, 5 / 8, 3 / 4], [5 / 8, 1 / 2, 7 / 8, 3 / 4],
                      [1 / 8, 3 / 4, 3 / 8, 1], [3 / 8, 3 / 4, 5 / 8, 1], [5 / 8, 3 / 4, 7 / 8, 1]]

    raw = glob(os.path.join(img_paths, "*"))
    for i, file in enumerate(raw):
        img = Image.open(file)

        for j, crop_box in enumerate(crop_boxes):
            img_cropped = img.crop([crop_box[0] * input_pixel, crop_box[1] * input_pixel, crop_box[2] * input_pixel,
                                    crop_box[3] * input_pixel]).convert("L")
            img_cropped.save(os.path.join(path_save, f"{i}({j+1})" + ".tif"))


def rename_images(path):
    files = glob(os.path.join(path, "*"))
    for i, file in enumerate(files):
        count = str(i).zfill(4)
        new_filename = os.path.join(path, f"{count}.tif")
        os.rename(file, new_filename)


def test(test_path):
    # Create test data
    i = 0
    imgs = glob(test_path + "*")
    imgdatas = np.ndarray((len(imgs), output_pixel, output_pixel, 1), dtype=np.uint8)
    for imgname in imgs:
        midname = imgname[imgname.rindex("/") + 1:]
        img = load_img('../test/' + midname, color_mode='grayscale')
        img = img_to_array(img)
        imgdatas[i] = img
        i += 1
    np.save(npy_path + 'imgs_test.npy', imgdatas)

    # Convert images to float32 and normalize it
    imgs_test = imgdatas.astype('float32')
    imgs_test /= 255
    mean = imgs_test.mean(axis=0)
    imgs_test -= mean

    # Load the trained model and predict
    model = load_model('../model/U-RNet+.hdf5')
    imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
    np.save(npy_path + 'imgs_mask_test.npy', imgs_mask_test)
    imgs_mask_test[imgs_mask_test > 0.7] = 1
    imgs_mask_test[imgs_mask_test <= 0.7] = 0
    os.makedirs(test_save + "bw/", exist_ok=True)
    for i in range(imgs_mask_test.shape[0]):
        img = imgs_mask_test[i]
        img = array_to_img(img)
        img.save(test_save + "bw/%d.tif" % i)
    return model


def color_enhance(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    img_color = np.zeros_like(img)

    # --- 亮度区间 1: < 64 ---
    mask1 = gray_img < 64
    img_color[mask1, 0] = 255
    img_color[mask1, 1] = gray_img[mask1] * 4
    img_color[mask1, 2] = 0

    # --- 亮度区间 2: 64-127 ---
    mask2 = (gray_img >= 64) & (gray_img < 128)
    green_val = (128 - gray_img[mask2]) * 4 - 1
    img_color[mask2, 0] = 0
    img_color[mask2, 1] = np.clip(green_val, 0, 255)
    img_color[mask2, 2] = 255

    # --- 亮度区间 3: 128-191 ---
    mask3 = (gray_img >= 128) & (gray_img < 192)
    green_val = (gray_img[mask3] - 128) * 4
    img_color[mask3, 0] = 0
    img_color[mask3, 1] = np.clip(green_val, 0, 255)
    img_color[mask3, 2] = 255

    # --- 亮度区间 4: >= 192 ---
    mask4 = gray_img >= 192
    green_val = (192 - gray_img[mask4]) * 4 - 1
    img_color[mask4, 0] = 0
    img_color[mask4, 1] = np.clip(green_val, 0, 255)
    img_color[mask4, 2] = 255

    img_color = img_color.astype(np.uint8)
    b = np.count_nonzero(img_color[:, :, 0])
    g = np.count_nonzero(img_color[:, :, 1])
    r = np.count_nonzero(img_color[:, :, 2])
    return img_color, b, g, r

def process_pseudo(path, pseudo):
    file_list = glob(os.path.join(path, '*'))
    for infile in file_list:
        i = Path(infile).stem
        test_img = cv2.imread(infile, 1)
        pseudo_img, b, g, r = color_enhance(test_img)
        bw_img = cv2.imread(test_save + f'bw/{int(i)}.tif')

        # Merge the color-enhanced image with the binary mask
        merge = cv2.bitwise_and(pseudo_img, bw_img)
        cv2.imwrite(pseudo + f'pseudo{int(i)}.tif', merge)


def stitch(image_dir, output_dir, positions, order, num_positions):
    image_files = sorted(glob(os.path.join(image_dir, '*')), key=os.path.getctime)
    if not image_files:
        print(f"Warning: No images found in '{image_dir}', skipping stitch.")
        return
    os.makedirs(output_dir, exist_ok=True)
    images = [Image.open(image_file) for image_file in image_files]
    n = len(image_files) // num_positions
    for i in range(n):
        new_image = Image.new('RGB', (2048, 2048), 'black')

        # Paste patches onto the new image according to the specified positions and order
        for j, pos_index in enumerate(order):
            new_image.paste(images[pos_index + num_positions * i], positions[j])
        new_image.save(os.path.join(output_dir, str(i) + '.tif'))


def merge_images(input_path, output_path):
    num_raw = len(glob('../testraw/*'))
    for i in range(num_raw):
        stitch44 = cv2.imread('../results/results_44/' + input_path + '/' + str(i) + '.tif')
        stitch34 = cv2.imread('../results/results_34/' + input_path + '/' + str(i) + '.tif')
        stitch43 = cv2.imread('../results/results_43/' + input_path + '/' + str(i) + '.tif')

        stitch44 = cv2.bitwise_not(stitch44)
        stitch34 = cv2.bitwise_not(stitch34)
        stitch43 = cv2.bitwise_not(stitch43)

        merge1 = cv2.bitwise_and(stitch44, stitch34)
        merge2 = cv2.bitwise_and(merge1, stitch43)
        merge2 = cv2.bitwise_not(merge2)
        cv2.imwrite(output_path + '/' + str(i) + '.tif', merge2)


if __name__ == "__main__":


    multiframe_source_dir = "../testraw_multiframe/"
    final_results_storage_dir = "../final_results_all/"

    os.makedirs(final_results_storage_dir, exist_ok=True)

    multiframe_files = sorted(glob(os.path.join(multiframe_source_dir, "*.tif")))
    print(f"Found {len(multiframe_files)} multi-frame files to process.")

    for multiframe_filepath in multiframe_files:
        base_name = Path(multiframe_filepath).stem
        print(f"\n{'=' * 20} Processing: {base_name} {'=' * 20}")

        img_paths = "../testraw/"
        all_work_dirs = [
            img_paths,
            "../test/test_44/", "../test/test_43/", "../test/test_34/",
            "../npydata/npydata_44/", "../npydata/npydata_43/", "../npydata/npydata_34/",
            "../results/results_44/", "../results/results_43/", "../results/results_34/",
            "../final_results/"
        ]
        for work_dir in all_work_dirs:
            if os.path.exists(work_dir):
                shutil.rmtree(work_dir)
            os.makedirs(work_dir)

        num_frames = split_multiframe_tiff(multiframe_filepath, img_paths)
        if num_frames == 0:
            continue

        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            except RuntimeError as e:
                print(e)
        else:
            tf.config.set_visible_devices([], 'GPU')

        input_pixel = 2048
        output_pixel = 512

        test_folders = ["44", "43", "34"]
        for folder in test_folders:
            path_save = f"../test/test_{folder}/"
            os.makedirs(path_save, exist_ok=True)
            npy_path = f"../npydata/npydata_{folder}/"
            os.makedirs(npy_path, exist_ok=True)
            test_save = f"../results/results_{folder}/"
            os.makedirs(test_save, exist_ok=True)
            pseudo_save = f"../results/results_{folder}/pseudo/"
            os.makedirs(pseudo_save, exist_ok=True)

            crop_images(folder)
            rename_images(path_save)
            test(path_save)
            print(f"\n{'=' * 10} Starting Temporal Refinement for '{folder}' {'=' * 10}")
            bw_mask_dir = os.path.join(test_save, "bw/")
            mask_files = sorted(glob(os.path.join(bw_mask_dir, "*.tif")))

            # 遍历所有相邻的掩码对
            for i in range(len(mask_files) - 1):
                mask_path_1 = mask_files[i]
                mask_path_2 = mask_files[i + 1]
                print(f"Processing pair: {os.path.basename(mask_path_1)} and {os.path.basename(mask_path_2)}")
                refine_masks_with_watershed(mask_path_1, mask_path_2)

            print(f"{'=' * 10} Temporal Refinement Finished {'=' * 10}\n")
            process_pseudo(path_save, pseudo_save)

        final_results = '../final_results/'
        source_results = {'bw': '../final_results/bw', 'pseudo': '../final_results/pseudo'}
        results_folder = '../results/'
        for category, folder in source_results.items():
            if not os.path.exists(folder): os.makedirs(folder)
            if not os.path.isdir(folder): continue
            for filename in os.listdir(folder):
                if filename.endswith('.tif'):
                    src_file_path = os.path.join(folder, filename)
                    results_filename = f"{filename.split('.')[0]}_{category}.tif"
                    dest_file_path = os.path.join(results_folder, results_filename)
                    shutil.copy(src_file_path, dest_file_path)
        print("Images have been saved to 'final_results' and 'results'")

        clear_files(final_results)
        p_values = ["bw", "pseudo"]
        positions_44 = [(0, 0), (512, 0), (1024, 0), (1536, 0),
                        (0, 512), (512, 512), (1024, 512), (1536, 512),
                        (0, 1024), (512, 1024), (1024, 1024), (1536, 1024),
                        (0, 1536), (512, 1536), (1024, 1536), (1536, 1536)]
        order_44 = [0, 8, 9, 10, 11, 12, 13, 14, 15, 1, 2, 3, 4, 5, 6, 7]

        positions_43 = [(0, 256), (512, 256), (1024, 256), (1536, 256),
                        (0, 768), (512, 768), (1024, 768), (1536, 768),
                        (0, 1280), (512, 1280), (1024, 1280), (1536, 1280)]
        order_43 = [0, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3]

        positions_34 = [(256, 0), (768, 0), (1280, 0),
                        (256, 512), (768, 512), (1280, 512),
                        (256, 1024), (768, 1024), (1280, 1024),
                        (256, 1536), (768, 1536), (1280, 1536)]
        order_34 = [0, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2, 3]

        for p in p_values:
            os.makedirs(f'../final_results/{p}/', exist_ok=True)
            stitch(f"../results/results_44/{p}/", f"../results/results_44/{p}_stitch/", positions_44, order_44, 16)
            stitch(f"../results/results_43/{p}/", f"../results/results_43/{p}_stitch/", positions_43, order_43, 12)
            stitch(f"../results/results_34/{p}/", f"../results/results_34/{p}_stitch/", positions_34, order_34, 12)
            merge_images(f'{p}_stitch', f'../final_results/{p}/')



        print("Saving final results...")
        for p in p_values:
            source_folder = os.path.join(final_results, p)
            if not os.path.isdir(source_folder): continue

            final_frames = []
            result_files = sorted(glob(os.path.join(source_folder, '*.tif')))
            for result_file in result_files:
                final_frames.append(Image.open(result_file))

            if final_frames:
                output_path = os.path.join(final_results_storage_dir, f"{base_name}_{p}.tif")
                final_frames[0].save(
                    output_path,
                    save_all=True,
                    append_images=final_frames[1:],
                    compression="tiff_deflate"
                )
                print(f"Saved {len(final_frames)} frames to: {output_path}")

    print("\nAll files processed.")
    print("\nCleaning up intermediate files...")


    dirs_to_delete = [
        "../testraw",
        "../test",
        "../npydata",
        # "../npydata/npydata_34",
        # "../npydata/npydata_43",
        # "../npydata/npydata_44",
        "../results",
        # "../results/results_34",
        # "../results/results_43",
        # "../results/results_44/",
        "../final_results"
    ]

    for dir_path in dirs_to_delete:
        try:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
                print(f"Successfully deleted: {dir_path}")
            else:
                print(f"Directory not found, skipping: {dir_path}")
        except OSError as e:
            print(f"Error deleting directory {dir_path}: {e}")

    print("Cleanup complete.")

