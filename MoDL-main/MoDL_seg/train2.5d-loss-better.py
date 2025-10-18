# 文件名: MoDL-main/MoDL_seg/train_2.5d_boundary_loss.py (简洁最终版)
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, ConvLSTM2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras import backend as K

# 禁用冗余日志
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# ==========================================================
# #####           边界增强Dice损失函数 (核心)           #####
# ==========================================================
def boundary_enhanced_dice_loss(y_true, y_pred, alpha=0.5, smooth=1e-6):
    """
    一个专门为分离交叠物体设计的损失函数。
    它结合了Dice损失(关注区域重叠)和边界损失(关注轮廓匹配)。
    """
    # --- 1. 计算标准Dice损失 ---
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice_loss_val = 1 - (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    # --- 2. 计算边界损失 ---
    # 定义Sobel滤波器来检测图像边界
    sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=tf.float32)
    sobel_x = tf.reshape(sobel_x, [3, 3, 1, 1])
    sobel_y = tf.transpose(sobel_x, [1, 0, 2, 3])

    # 提取真实标签的边界
    y_true_edges_x = tf.nn.conv2d(y_true, sobel_x, strides=[1, 1, 1, 1], padding='SAME')
    y_true_edges_y = tf.nn.conv2d(y_true, sobel_y, strides=[1, 1, 1, 1], padding='SAME')
    y_true_edges = tf.sqrt(tf.square(y_true_edges_x) + tf.square(y_true_edges_y))

    # 提取预测结果的边界
    y_pred_edges_x = tf.nn.conv2d(y_pred, sobel_x, strides=[1, 1, 1, 1], padding='SAME')
    y_pred_edges_y = tf.nn.conv2d(y_pred, sobel_y, strides=[1, 1, 1, 1], padding='SAME')
    y_pred_edges = tf.sqrt(tf.square(y_pred_edges_x) + tf.square(y_pred_edges_y))

    # 计算边界之间的均方误差损失
    boundary_loss_val = K.mean(K.square(y_true_edges - y_pred_edges))

    # --- 3. 组合两种损失 ---
    # alpha是边界损失的权重，(1-alpha)是Dice损失的权重
    total_loss = (1 - alpha) * dice_loss_val + alpha * boundary_loss_val

    return total_loss


# ==========================================================
# #####                训练主流程                #####
# ==========================================================
def train():
    # --- 数据加载 ---
    print("Loading 2.5D sequence data...")
    try:
        imgs_train = np.load("../npydata/imgs_train_2.5D_seq.npy")
        imgs_mask_train = np.load("../npydata/imgs_mask_train_2.5D_seq.npy")
    except FileNotFoundError:
        print("\nERROR: .npy data files not found! Please run 'data_load.py' first.")
        return

    # --- 数据预处理 ---
    imgs_train = imgs_train.astype('float32') / 255.0
    imgs_mask_train = imgs_mask_train.astype('float32')
    # 确保掩码是0或1的浮点数，因为损失函数需要
    imgs_mask_train[imgs_mask_train > 128] = 1.0  # 假设原掩码是0/255
    imgs_mask_train[imgs_mask_train <= 128] = 0.0
    print(f"Data loading complete. Training with {len(imgs_train)} samples.")

    # --- 模型构建与权重加载 ---
    print("Loading pre-trained U-Net and building 2.5D model...")
    try:
        original_unet = load_model('../model/U-RNet+.hdf5', compile=False)
        original_unet.trainable = False  # 冻结U-Net
    except Exception as e:
        print(f"\nERROR: Could not load pre-trained model: {e}")
        return

    sequence_input = Input(shape=(3, 512, 512, 1))
    lstm_encoder = ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=False)(sequence_input)
    feature_adapter = Conv2D(1, (1, 1), activation='relu', padding='same')(lstm_encoder)
    final_output = original_unet(feature_adapter)
    model = Model(inputs=sequence_input, outputs=final_output)

    # --- 模型编译 ---
    print("\nCompiling model with Boundary Enhanced Dice Loss...")
    model.compile(optimizer=Adam(learning_rate=1e-4), loss=boundary_enhanced_dice_loss, metrics=['accuracy'])
    model.summary()

    # --- 模型保存点 ---
    output_model_path = '../model/convlstm_unet_boundary_loss.hdf5'
    model_checkpoint = ModelCheckpoint(output_model_path, monitor='loss', verbose=1, save_best_only=True)

    # --- 阶段1: 微调头部 ---
    print("\n--- PHASE 1: Fine-tuning new layers (ConvLSTM Head) ---")
    history1 = model.fit(imgs_train, imgs_mask_train, batch_size=1, epochs=15, shuffle=True,
                         callbacks=[model_checkpoint])

    # --- 阶段2: 整体微调 ---
    print("\n--- PHASE 2: Unfreezing and fine-tuning the entire model ---")
    original_unet.trainable = True
    model.compile(optimizer=Adam(learning_rate=1e-5), loss=boundary_enhanced_dice_loss,
                  metrics=['accuracy'])  # 使用更低的学习率
    history2 = model.fit(imgs_train, imgs_mask_train, batch_size=1, epochs=10, shuffle=True,
                         callbacks=[model_checkpoint])

    print(f"\nTraining complete! Final model saved to {output_model_path}")

    # --- 可视化训练过程 ---
    # (这部分可以保持原样或根据需要简化，这里提供一个简洁版本)
    plt.figure(figsize=(10, 5))
    plt.plot(history1.history['loss'] + history2.history['loss'], label='Training Loss')
    plt.title('Model Loss during Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('../model/loss_boundary_enhanced.png')
    print("Training loss plot saved to ../model/loss_boundary_enhanced.png")


if __name__ == '__main__':
    train()