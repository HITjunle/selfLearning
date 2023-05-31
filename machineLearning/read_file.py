import numpy as np


def read_idx3_file(images_file, labels_file):
    # 读取图像数据文件
    with open(images_file, 'rb') as f:
        # 解析文件头
        magic_number = int.from_bytes(f.read(4), 'big')
        num_images = int.from_bytes(f.read(4), 'big')
        num_rows = int.from_bytes(f.read(4), 'big')
        num_cols = int.from_bytes(f.read(4), 'big')

        # 读取图像数据
        images_data = np.frombuffer(f.read(), dtype=np.uint8)
        images_data = images_data.reshape(num_images, num_rows * num_cols)

    # 读取标签数据文件
    with open(labels_file, 'rb') as f:
        # 解析文件头
        magic_number = int.from_bytes(f.read(4), 'big')
        num_labels = int.from_bytes(f.read(4), 'big')

        # 读取标签数据
        labels_data = np.frombuffer(f.read(), dtype=np.uint8)

    return images_data, labels_data
