import numpy as np
import tensorflow as tf
from model3D import *
from model2D import *


# -------------------------- 工具函数 --------------------------
def sigmoid(x):
    """Sigmoid函数"""
    return 1 / (1 + np.exp(-x))


def split_block_3d(data, block_size, overlap):
    """
    3D数据分块（带重叠）
    """
    blocks = []
    step = [int(s * (1 - overlap)) for s in block_size]
    for i in range(0, data.shape[0], step[0]):
        for j in range(0, data.shape[1], step[1]):
            for k in range(0, data.shape[2], step[2]):
                block = data[i:min(i + block_size[0], data.shape[0]),
                        j:min(j + block_size[1], data.shape[1]),
                        k:min(k + block_size[2], data.shape[2])]
                blocks.append((block, i, j, k))
    return blocks


def create_weight_map(block_shape, overlap):
    """
    创建3D权重图，用于重叠区域融合
    """
    weights = np.ones(block_shape)
    overlap_size = [int(s * overlap) for s in block_shape]

    # X方向权重
    for i in range(overlap_size[0]):
        weights[i, :, :] *= (i + 1) / (overlap_size[0] + 1)
        weights[-i - 1, :, :] *= (i + 1) / (overlap_size[0] + 1)
    # Y方向权重
    for j in range(overlap_size[1]):
        weights[:, j, :] *= (j + 1) / (overlap_size[1] + 1)
        weights[:, -j - 1, :] *= (j + 1) / (overlap_size[1] + 1)
    # Z方向权重
    for k in range(overlap_size[2]):
        weights[:, :, k] *= (k + 1) / (overlap_size[2] + 1)
        weights[:, :, -k - 1] *= (k + 1) / (overlap_size[2] + 1)
    return weights


def combine_blocks_3d(blocks, data_shape, block_size, overlap):
    """
    融合带重叠的3D分块结果
    """
    result = np.zeros(data_shape)
    count = np.zeros(data_shape)
    step = [int(s * (1 - overlap)) for s in block_size]

    for block, i, j, k in blocks:
        block_shape = block.shape
        weights = create_weight_map(block_shape, overlap)

        i_end = min(i + block_shape[0], data_shape[0])
        j_end = min(j + block_shape[1], data_shape[1])
        k_end = min(k + block_shape[2], data_shape[2])

        block_i_slice = slice(0, i_end - i)
        block_j_slice = slice(0, j_end - j)
        block_k_slice = slice(0, k_end - k)

        result[i:i_end, j:j_end, k:k_end] += block[block_i_slice, block_j_slice, block_k_slice] * weights[
            block_i_slice, block_j_slice, block_k_slice]
        count[i:i_end, j:j_end, k:k_end] += weights[block_i_slice, block_j_slice, block_k_slice]

    count[count == 0] = 1  # 避免除零
    result /= count
    return result


# -------------------------- 主函数 --------------------------
def main():
    # GPU配置（按需开启，取消注释即可使用）
    # config = tf.compat.v1.ConfigProto()
    # config.gpu_options.allow_growth = True
    # sess = tf.compat.v1.Session(config=config)

    # 数据维度参数
    len1, len2, len3 = 201, 251, 801

    # 模型路径与分块参数
    model_path1 = "unet2D.hdf5"
    model_path2 = "unet3D.hdf5"
    block_size = (128, 128, 256)
    overlap = 0.25  # 50%重叠

    # 输入输出文件路径
    data_path = r"G:\大庆高分辨项目\3月全家桶\201_251_801.dat"

    output_path = r"G:\大庆高分辨项目\3月全家桶\201_251_801断层识别.dat"

    # 1. 加载模型
    print("加载模型中...")
    model1 = load_model(model_path1, custom_objects={"loss1": loss1})
    model2 = load_model(model_path2, custom_objects={"loss1": loss1})

    print("模型加载完成！")

    # 2. 读取三维数据
    print("读取原始数据...")
    data = np.fromfile(data_path, dtype=np.single).reshape(len1, len2, len3)
    dataup=np.zeros((len1, len2, len3))
    datadown=np.zeros((len1, len2, len3))
    data2 =np.zeros((len1, len2, len3))
    data3 =np.zeros((len1, len2, len3))
    for i in range(len1):
        dataper = data[i, :, :]
        dataper = np.reshape(dataper, (1, len2, len3, 1))
        dataper = model1.predict(dataper)
        dataup[i, :, :] = dataper[0, :, :, 0]
        datadown[i, :, :] = dataper[0, :, :, 1]
        data2[i, :, :] = dataup[i, :, :] + datadown[i, :, :]

    for j in range(len2):
        dataper = data[:, j, :]
        dataper = np.reshape(dataper, (1, len1, len3, 1))
        dataper = model1.predict(dataper)
        dataup[:, j, :] = dataper[0, :, :, 0]
        datadown[:, j, :]= dataper[0, :, :, 1]
        data3[i, :, :] = dataup[:, j, :] + datadown[:, j, :]

    # 3. 计算注意力权重（取data2和data3的最大值）
    print("计算注意力权重...")
    dataattention = np.maximum(data2, data3)

    # 4. 数据加权 + 归一化
    data = data * tf.sigmoid(dataattention).numpy()
    data = (data - np.min(data)) / (np.max(data) - np.min(data))

    # 5. 3D数据分块
    print("数据分块中...")
    blocks = split_block_3d(data, block_size, overlap)

    # 6. 分块预测
    print("开始模型预测...")
    dataups = []
    datadowns = []

    for idx, (block, i, j, k) in enumerate(blocks):
        print(f"正在预测第 {idx + 1}/{len(blocks)} 个块...")
        # 维度调整：(H,W,D) -> (1, H,W,D)
        block_input = np.reshape(block, (1, *block.shape))
        # 模型推理
        predicted_block = model2.predict(block_input, verbose=0)
        # 提取输出结果
        dataup = predicted_block[0, :, :, :, 0]
        datadown = predicted_block[0, :, :, :, 1]
        # 保存结果与坐标
        dataups.append((dataup, i, j, k))
        datadowns.append((datadown, i, j, k))

    # 7. 融合分块结果
    print("融合预测结果...")
    result1 = combine_blocks_3d(dataups, data.shape, block_size, overlap)
    result2 = combine_blocks_3d(datadowns, data.shape, block_size, overlap)
    result = result1 + result2

    # 8. 保存最终结果
    print("保存输出文件...")
    result.astype(np.float32).tofile(output_path)
    print(f"处理完成！文件已保存至：\n{output_path}")


# -------------------------- 程序入口 --------------------------
if __name__ == "__main__":
    main()