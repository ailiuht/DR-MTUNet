import os

import torch
import numpy as np

from load_data import logger


def multi_tensor_collate(batch):
    """高效安全地处理包含多个张量的批量数据，添加类型转换确保兼容性

    参数:
        batch (list): 包含多个张量数据的批次列表

    返回:
        tuple: 包含6个堆叠张量的元组
    """
    # 确保所有元素都是Tensor
    processed_batch = []
    for sample in batch:
        # 转换样本中的每个元素为Tensor
        tensor_sample = []
        for item in sample:
            if isinstance(item, np.ndarray):
                # 转换numpy数组为Tensor
                tensor_sample.append(torch.from_numpy(item))
            elif isinstance(item, torch.Tensor):
                # 已经是Tensor，直接使用
                tensor_sample.append(item)
            else:
                # 其他类型（如标量）转换为Tensor
                tensor_sample.append(torch.tensor(item))
        processed_batch.append(tuple(tensor_sample))

    # 将批次数据按元素类型分组
    item0_list = []
    item1_list = []
    item2_list = []
    item3_list = []
    item4_list = []
    item5_list = []

    # 遍历处理后的批次
    for item in processed_batch:
        item0_list.append(item[0])
        item1_list.append(item[1])
        item2_list.append(item[2])
        item3_list.append(item[3])
        item4_list.append(item[4])
        item5_list.append(item[5])

    # 堆叠每个元素的张量
    return (
        torch.stack(item0_list, 0),
        torch.stack(item1_list, 0),
        torch.stack(item2_list, 0),
        torch.stack(item3_list, 0),
        torch.stack(item4_list, 0),
        torch.stack(item5_list, 0)
    )


def clean_cache_files(cache_names):
    """清除缓存文件"""
    for name in cache_names:
        try:
            # 删除缓存文件
            cache_file = f"{name}.pkl"
            if os.path.exists(cache_file):
                os.remove(cache_file)
                logger.info(f"已删除缓存文件: {cache_file}")

            # 删除哈希文件
            hash_file = f"{name}_hash.txt"
            if os.path.exists(hash_file):
                os.remove(hash_file)
                logger.info(f"已删除哈希文件: {hash_file}")
        except Exception as e:
            logger.warning(f"删除缓存文件失败: {str(e)}")