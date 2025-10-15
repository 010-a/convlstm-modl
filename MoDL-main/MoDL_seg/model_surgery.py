import os
import glob
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Concatenate, Conv2DTranspose
from tensorflow.keras.models import load_model


# ==============================================================================
# 模块1: 手动构建与原始模型100%匹配的3通道U-Net
# 这是最可靠的方式
# ==============================================================================
def build_3_channel_unet(img_rows=512, img_cols=512):
    inputs = Input((img_rows, img_cols, 3))

    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv2d")(
        inputs)  # 注意命名
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv2d_1")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv2d_2")(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv2d_3")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv2d_4")(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv2d_5")(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv2d_6")(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv2d_7")(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv2d_8")(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv2d_9")(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2DTranspose(512, 2, activation='relu', padding='same', kernel_initializer='he_normal',
                          name="conv2d_transpose")(UpSampling2D(size=(2, 2))(drop5))
    merge6 = Concatenate(axis=3)([drop4, up6])
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv2d_10")(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv2d_11")(conv6)

    up7 = Conv2DTranspose(256, 2, activation='relu', padding='same', kernel_initializer='he_normal',
                          name="conv2d_transpose_1")(UpSampling2D(size=(2, 2))(conv6))
    merge7 = Concatenate(axis=3)([conv3, up7])
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv2d_12")(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv2d_13")(conv7)

    up8 = Conv2DTranspose(128, 2, activation='relu', padding='same', kernel_initializer='he_normal',
                          name="conv2d_transpose_2")(UpSampling2D(size=(2, 2))(conv7))
    merge8 = Concatenate(axis=3)([conv2, up8])
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv2d_14")(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv2d_15")(conv8)

    up9 = Conv2DTranspose(64, 2, activation='relu', padding='same', kernel_initializer='he_normal',
                          name="conv2d_transpose_3")(UpSampling2D(size=(2, 2))(conv8))
    merge9 = Concatenate(axis=3)([conv1, up9])
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv2d_16")(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv2d_17")(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal', name="conv2d_18")(conv9)

    conv10 = Conv2D(1, 1, activation='sigmoid', name="conv2d_19")(conv9)
    model = Model(inputs=inputs, outputs=conv10)
    return model


# ==============================================================================
# 模块2: 权重嫁接
# ==============================================================================
def create_grafted_model(pretrained_model_1ch):
    print("--- Creating Grafted Model ---")

    # 1. 构建3通道U-Net“空壳”
    new_model_3ch = build_3_channel_unet()
    print("New 3-channel U-Net architecture created.")

    # 2. 转移权重，手动处理命名偏移
    print("Transferring weights with manual name mapping...")
    for layer_new in new_model_3ch.layers[1:]:  # 跳过Input层
        # 手动构建新旧层名的对应关系
        # 这是解决所有问题的关键！
        old_name = layer_new.name

        # 尝试直接用新层名在旧模型中查找
        try:
            layer_old = pretrained_model_1ch.get_layer(name=old_name)
            if layer_old.get_weights():
                # 第一个卷积层特殊处理，其他直接复制
                if layer_new.name == "conv2d":  # 明确指定第一个卷积层
                    continue
                else:
                    layer_new.set_weights(layer_old.get_weights())
        except ValueError:
            # 如果找不到，说明命名规则可能不一致，但对于这个模型结构，应该都能找到
            print(f"  - Warning: Could not find layer '{old_name}' in the old model. Skipping.")
            continue

    # 3. 对第一个卷积层进行“权重复用”嫁接
    first_conv_old = pretrained_model_1ch.get_layer(name="conv2d")
    first_conv_new = new_model_3ch.get_layer(name="conv2d")

    print(f"Performing weight grafting for layer: '{first_conv_old.name}'")
    old_weights, old_biases = first_conv_old.get_weights()
    W_new = np.concatenate([old_weights] * 3, axis=2) / 3.0

    first_conv_new.set_weights([W_new, old_biases])

    print("✅ Weight grafting complete.")
    print("--- Grafted Model Ready ---")
    return new_model_3ch


# ==============================================================================
# 模块3 & 4: 推理函数和主程序 (保持不变)
# ==============================================================================
def predict_with_grafted_model(model, frame_prev, frame_center, frame_next):
    stacked_input = np.stack([frame_prev, frame_center, frame_next], axis=-1)
    img_for_model = stacked_input.astype('float32') / 255.0
    img_for_model = np.expand_dims(img_for_model, axis=0)
    result = model.predict(img_for_model)
    result = result[0, :, :, 0]
    mask = (result > 0.5).astype(np.uint8) * 255
    return mask


if __name__ == "__main__":

    input_dir = "../testraw_multiframe/"
    output_dir = "../pseudo_temporal_results/"
    model_path = "../model/U-RNet+.hdf5"

    os.makedirs(output_dir, exist_ok=True)

    print("Loading original 1-channel pre-trained model...")
    try:
        original_model = load_model(model_path)
        # 我们可以打印旧模型的层名来确认
        # print("Original model summary:")
        # original_model.summary()
    except Exception as e:
        print(f"Fatal: Error loading model at '{model_path}'. Check filename. Error: {e}");
        exit()

    grafted_model = create_grafted_model(original_model)

    image_files = glob.glob(os.path.join(input_dir, "*.tif")) + glob.glob(os.path.join(input_dir, "*.tiff"))
    if not image_files:
        print(f"Fatal: No image files found in '{input_dir}'");
        exit()

    for file_path in image_files:
        print(f"\n--- Processing file: {os.path.basename(file_path)} ---")
        try:
            with Image.open(file_path) as img:
                frames = [np.array(img.convert("L")) for i in range(img.n_frames)]
        except Exception as e:
            print(f"  -> Error reading file: {e}. Skipping.");
            continue

        if len(frames) < 3:
            print(f"  -> Warning: File has fewer than 3 frames. Skipping.");
            continue

        center_frame_index = 1
        frame_prev, frame_center, frame_next = frames[0], frames[1], frames[2]

        if frame_center.shape != (512, 512):
            print(f"  -> Resizing frames from {frame_center.shape} to (512, 512) for demonstration.")
            frame_prev = cv2.resize(frame_prev, (512, 512))
            frame_center = cv2.resize(frame_center, (512, 512))
            frame_next = cv2.resize(frame_next, (512, 512))

        print(f"Predicting frame {center_frame_index} using pseudo-temporal inference...")
        refined_mask = predict_with_grafted_model(grafted_model, frame_prev, frame_center, frame_next)

        base_name = Path(file_path).stem
        output_path = os.path.join(output_dir, f"{base_name}_pseudo_temporal_frame_{center_frame_index}.tif")
        cv2.imwrite(output_path, refined_mask)
        print(f"  -> Saved refined mask to {output_path}")

    print("\n\nAll files processed.")