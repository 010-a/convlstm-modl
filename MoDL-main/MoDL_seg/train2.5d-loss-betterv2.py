# 文件名: train_2.5d_stable_final.py (已修正文件保存)
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, ConvLSTM2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras import backend as K

# ... (顶部的损失函数定义，完全不变) ...
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def boundary_enhanced_dice_loss(y_true, y_pred, alpha=0.5, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice_loss_val = 1 - (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=tf.float32)
    sobel_x = tf.reshape(sobel_x, [3, 3, 1, 1])
    sobel_y = tf.transpose(sobel_x, [1, 0, 2, 3])
    y_true_edges_x = tf.nn.conv2d(y_true, sobel_x, strides=[1, 1, 1, 1], padding='SAME')
    y_true_edges_y = tf.nn.conv2d(y_true, sobel_y, strides=[1, 1, 1, 1], padding='SAME')
    y_true_edges = tf.sqrt(tf.square(y_true_edges_x) + tf.square(y_true_edges_y) + K.epsilon())
    y_pred_edges_x = tf.nn.conv2d(y_pred, sobel_x, strides=[1, 1, 1, 1], padding='SAME')
    y_pred_edges_y = tf.nn.conv2d(y_pred, sobel_y, strides=[1, 1, 1, 1], padding='SAME')
    y_pred_edges = tf.sqrt(tf.square(y_pred_edges_x) + tf.square(y_pred_edges_y) + K.epsilon())
    boundary_loss_val = K.mean(K.square(y_true_edges - y_pred_edges))
    total_loss = (1 - alpha) * dice_loss_val + alpha * boundary_loss_val
    return total_loss


def train():
    # ... (数据加载和模型构建部分，完全不变) ...
    print("Loading 2.5D sequence data...")
    imgs_train = np.load("../npydata/imgs_train_2.5D_seq.npy")
    imgs_mask_train = np.load("../npydata/imgs_mask_train_2.5D_seq.npy")
    imgs_train = imgs_train.astype('float32') / 255.0
    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train[imgs_mask_train > 0.5] = 1.0
    imgs_mask_train[imgs_mask_train <= 0.5] = 0.0
    print(f"Data loading complete.")

    print("Loading pre-trained U-Net and building 2.5D model...")
    original_unet = load_model('../model/U-RNet+.hdf5', compile=False)
    original_unet.trainable = False
    sequence_input = Input(shape=(3, 512, 512, 1))
    lstm_encoder = ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=False)(sequence_input)
    feature_adapter = Conv2D(1, (1, 1), activation='relu', padding='same')(lstm_encoder)
    final_output = original_unet(feature_adapter)
    model = Model(inputs=sequence_input, outputs=final_output)

    # ==========================================================
    # #####           核心修正：分离模型保存路径           #####
    # ==========================================================

    # --- 阶段1: 微调头部 ---
    print("\n--- PHASE 1: Fine-tuning new layers with Gradient Clipping ---")

    # 为第一阶段创建一个独立的保存点
    phase1_model_path = '../model/convlstm_unet_phase1_best.hdf5'
    checkpoint_phase1 = ModelCheckpoint(phase1_model_path, monitor='loss', verbose=1, save_best_only=True)

    optimizer_phase1 = Adam(learning_rate=1e-4, clipnorm=1.0)
    model.compile(optimizer=optimizer_phase1, loss=boundary_enhanced_dice_loss, metrics=['accuracy'])
    model.summary()
    history1 = model.fit(imgs_train, imgs_mask_train, batch_size=1, epochs=15, shuffle=True,
                         callbacks=[checkpoint_phase1])

    # --- 阶段2: 整体微调 ---
    print("\n--- PHASE 2: Unfreezing and fine-tuning the entire model with Gradient Clipping ---")

    # 加载第一阶段训练出的最佳模型，确保状态干净
    print(f"Reloading best model from Phase 1: {phase1_model_path}")
    model = load_model(phase1_model_path, custom_objects={'boundary_enhanced_dice_loss': boundary_enhanced_dice_loss})

    # 获取U-Net部分并解冻
    # 我们需要通过名字找到U-Net模型。原始的U-Net模型名默认为 "model"
    unet_body = model.layers[-1]  # 在我们的结构中，unet是最后一层
    unet_body.trainable = True

    # 为第二阶段创建独立的保存点
    phase2_model_path = '../model/convlstm_unet_phase2_best.hdf5'
    checkpoint_phase2 = ModelCheckpoint(phase2_model_path, monitor='loss', verbose=1, save_best_only=True)

    optimizer_phase2 = Adam(learning_rate=1e-5, clipnorm=1.0)
    model.compile(optimizer=optimizer_phase2, loss=boundary_enhanced_dice_loss, metrics=['accuracy'])

    history2 = model.fit(imgs_train, imgs_mask_train, batch_size=1, epochs=10, shuffle=True,
                         callbacks=[checkpoint_phase2])

    # ==========================================================

    final_model_path = '../model/convlstm_unet_stable_final.hdf5'
    print(f"\nTraining complete! Loading best model from Phase 2 and saving to final path: {final_model_path}")
    # 加载第二阶段训练出的最佳模型
    final_model = load_model(phase2_model_path,
                             custom_objects={'boundary_enhanced_dice_loss': boundary_enhanced_dice_loss})
    # 保存为最终的文件名
    final_model.save(final_model_path)

    # （可选）清理临时模型文件
    # os.remove(phase1_model_path)
    # os.remove(phase2_model_path)

    # --- 可视化训练过程 (不变) ---
    print("\nGenerating training history plot...")
    # ... (画图代码和之前一样) ...
    plt.figure(figsize=(12, 6))
    loss1 = history1.history['loss']
    loss2 = history2.history['loss']
    full_loss = loss1 + loss2
    epochs_range = range(len(full_loss))
    plt.plot(epochs_range, full_loss, label='Training Loss', color='blue')
    plt.axvline(x=len(loss1) - 1, color='gray', linestyle='--', label='Phase 1/2 Boundary')
    plt.title('Model Loss during Training (Stable with Save Fix)', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plot_path = '../model/loss_stable_final.png'
    plt.savefig(plot_path, dpi=300)
    print(f"Training loss plot saved to {plot_path}")


if __name__ == '__main__':
    train()