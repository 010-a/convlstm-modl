# 文件名: MoDL-main/MoDL_seg/train_2.5d.py (最终版 - 修正权重加载)
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, ConvLSTM2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import os

# 禁用TensorFlow的一些冗余日志
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# --- 1. 定义新的、完整的2.5D模型结构 ---
def build_convlstm_unet_model(img_rows=512, img_cols=512):
    """
    直接构建一个完整的、集成了ConvLSTM和U-Net结构的新模型。
    """
    # 输入是一个序列
    inputs = Input(shape=(3, img_rows, img_cols, 1), name='sequence_input')

    # ConvLSTM编码器
    fused_features = ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=False,
                                name='conv_lstm_encoder')(inputs)

    # 将融合后的特征降维，作为U-Net部分的输入
    unet_input_like = Conv2D(1, (1, 1), activation='relu', padding='same', name='feature_adapter')(fused_features)

    # --- 在这里直接构建U-Net的编码器部分 ---
    # 我们给每一层都加上独特的名字，以便权重加载
    conv1 = Conv2D(64, 3, activation='relu', padding='same', name='conv1a')(unet_input_like)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', name='conv1b')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), name='pool1')(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', name='conv2a')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', name='conv2b')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), name='pool2')(conv2)

    # ... (省略了U-Net的完整结构，因为解码器部分也需要命名) ...
    # 鉴于此，我们采用一个更聪明的加载方法，见下文的 train 函数

    # 为了简化，我们还是用模块化的方法，但加载权重时更智能
    return None  # 我们将在train函数中直接操作


# --- 2. 训练流程 (包含正确的权重加载逻辑) ---
def train():
    """
    完整的2.5D模型微调流程，包含正确的权重加载逻辑。
    """
    # --- 数据加载 ---
    print("Loading 2.5D sequence data...")
    try:
        imgs_train = np.load("../npydata/imgs_train_2.5D_seq.npy")
        imgs_mask_train = np.load("../npydata/imgs_mask_train_2.5D_seq.npy")
    except FileNotFoundError:
        print("\nERROR: .npy data files not found!")
        print("Please run 'data_load.py' first.")
        return

    # --- 数据预处理 ---
    imgs_train = imgs_train.astype('float32') / 255.0
    imgs_mask_train = imgs_mask_train.astype('float32') / 255.0
    imgs_mask_train[imgs_mask_train > 0.5] = 1
    imgs_mask_train[imgs_mask_train <= 0.5] = 0
    print("Data loading complete.")

    # ====================================================================
    # #####           全新的、正确的模型构建与权重加载逻辑           #####
    # ====================================================================

    # --- 步骤 A: 加载完整的预训练U-Net模型 ---
    original_unet_path = '../model/U-RNet+.hdf5'
    print(f"Loading full pre-trained U-Net model from {original_unet_path}...")
    try:
        # 使用 load_model 加载完整的模型
        original_unet = load_model(original_unet_path, compile=False)
        # 将U-Net部分设置为不可训练，先“冻结”起来
        original_unet.trainable = False
        print("Pre-trained U-Net loaded and frozen.")
    except Exception as e:
        print(f"\nERROR: Could not load pre-trained model: {e}")
        return

    # --- 步骤 B: 构建新的2.5D模型的“头部”（ConvLSTM部分） ---
    sequence_input = Input(shape=(3, 512, 512, 1), name='sequence_input')
    lstm_encoder = ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=False,
                              name='conv_lstm_encoder')(sequence_input)
    # 将LSTM的输出通道数调整为与U-Net输入通道数(1)匹配
    feature_adapter = Conv2D(1, (1, 1), activation='relu', padding='same', name='feature_adapter')(lstm_encoder)

    # --- 步骤 C: 将新的“头部”和“冻结”的U-Net身体连接起来 ---
    # original_unet(feature_adapter) 就像一个函数调用
    final_output = original_unet(feature_adapter)

    # --- 步骤 D: 组装成最终的2.5D模型 ---
    convlstm_unet_model = Model(inputs=sequence_input, outputs=final_output, name='convlstm_unet_finetuned')

    print("\nNew 2.5D model constructed:")
    convlstm_unet_model.summary()

    # --- 步骤 E: 编译模型，准备微调 ---
    # 我们只训练新添加的层 (ConvLSTM和适配器层)
    convlstm_unet_model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    # ====================================================================

    # --- 设置模型保存回调 ---
    output_model_path = '../model/convlstm_unet_finetuned.hdf5'
    model_checkpoint = ModelCheckpoint(output_model_path, monitor='loss', verbose=1, save_best_only=True)

    # --- 模型微调 ---
    print('\nStarting to fine-tune the new layers of the 2.5D model...')
    val_split = 0.1 if len(imgs_train) >= 10 else 0.0

    history = convlstm_unet_model.fit(
        imgs_train,
        imgs_mask_train,
        batch_size=1,
        epochs=15,  # 先只训练头部15个epoch
        verbose=1,
        validation_split=val_split,
        shuffle=True,
        callbacks=[model_checkpoint]
    )

    # --- （可选）第二阶段微调：解冻部分U-Net层 ---
    print("\n--- Phase 2: Unfreezing U-Net and fine-tuning end-to-end ---")
    original_unet.trainable = True  # 解冻U-Net

    # 重新编译整个模型，使用一个更低的学习率
    convlstm_unet_model.compile(optimizer=Adam(learning_rate=1e-6), loss='binary_crossentropy', metrics=['accuracy'])

    history_phase2 = convlstm_unet_model.fit(
        imgs_train,
        imgs_mask_train,
        batch_size=1,
        epochs=10,  # 再用极低学习率，整体微调10个epoch
        verbose=1,
        validation_split=val_split,
        shuffle=True,
        callbacks=[model_checkpoint]
    )

    print(f"Fine-tuning complete. Final model saved to {output_model_path}")

    # --- 训练过程可视化 ---
    print("\nGenerating training history plots...")
    output_plot_dir = '../model/'

    # 准确率图
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'], 'b', label='Training accuracy')
    if val_split > 0:
        plt.plot(history.history['val_accuracy'], 'r', label='Validation accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(os.path.join(output_plot_dir, 'accuracy_2.5d_finetune.png'))
    print(f"Accuracy plot saved to {os.path.join(output_plot_dir, 'accuracy_2.5d_finetune.png')}")

    # 损失图
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], 'b', label='Training loss')
    if val_split > 0:
        plt.plot(history.history['val_loss'], 'r', label='Validation loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(os.path.join(output_plot_dir, 'loss_2.5d_finetune.png'))
    print(f"Loss plot saved to {os.path.join(output_plot_dir, 'loss_2.5d_finetune.png')}")


if __name__ == '__main__':
    train()
