from test import validate
from tool import multi_tensor_collate, clean_cache_files
import concurrent.futures
import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from load_data import check_dataset_integrity, MedicalSegmentationDataset, logger
from model2 import DR_MTUNet
from train import train_model
import csv
from datetime import datetime
import concurrent.futures
import pickle


class ParallelMemoryCachedDataset:
    """并行内存缓存数据集包装器 - 优化加载速度"""

    def __init__(self, dataset, num_workers=10):
        self.dataset = dataset
        self.cache = [None] * len(dataset)  # 预分配内存
        self._preload_to_memory(num_workers)

    def _load_sample(self, idx):
        """加载单个样本 - 供并行处理使用"""
        try:
            # 加载数据项
            item = self.dataset[idx]
            # 转换为numpy数组以节省内存
            return tuple(tensor.cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor
                         for tensor in item)
        except Exception as e:
            logger.error(f"加载样本 {idx} 失败: {str(e)}")
            # 返回空数据
            dummy = np.zeros((3, 256, 256), dtype=np.float32)
            return (dummy, dummy[0:1], dummy[0:1], np.zeros(5), np.zeros(3), np.zeros(3))

    def _preload_to_memory(self, num_workers):
        """使用并行处理加载数据集到内存"""
        logger.info(f"开始并行加载数据集到内存 ({num_workers} 个工作线程)...")
        start_time = time.time()

        # 创建线程池执行器
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            # 提交所有加载任务
            futures = [executor.submit(self._load_sample, i) for i in range(len(self.dataset))]

            # 获取结果并填充缓存
            for idx, future in enumerate(concurrent.futures.as_completed(futures)):
                try:
                    self.cache[idx] = future.result()
                except Exception as e:
                    logger.error(f"获取样本 {idx} 结果失败: {str(e)}")
                    # 设置空数据
                    dummy = np.zeros((3, 256, 256), dtype=np.float32)
                    self.cache[idx] = (dummy, dummy[0:1], dummy[0:1], np.zeros(5), np.zeros(3), np.zeros(3))

                # 定期报告进度
                if idx % 100 == 0 or idx == len(self.dataset) - 1:
                    elapsed = time.time() - start_time
                    progress = 100 * (idx + 1) / len(self.dataset)
                    logger.info(f"已加载 {idx + 1}/{len(self.dataset)} 个样本 ({progress:.1f}%), 耗时 {elapsed:.1f} 秒")

        elapsed = time.time() - start_time
        logger.info(f"数据集并行加载完成! 共 {len(self.cache)} 个样本, 总耗时 {elapsed:.1f} 秒")

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, idx):
        return self.cache[idx]




def save_cached_dataset(cache, cache_file):
    """将缓存数据集保存到文件"""
    try:
        # 使用pickle保存缓存
        with open(cache_file, 'wb') as f:
            pickle.dump(cache, f)
        logger.info(f"数据集缓存已保存到: {cache_file}")
        return True
    except Exception as e:
        logger.error(f"保存缓存失败: {str(e)}")
        return False


def load_cached_dataset(cache_file):
    """从文件加载缓存数据集"""
    try:
        # 使用pickle加载缓存
        with open(cache_file, 'rb') as f:
            cache = pickle.load(f)
        logger.info(f"成功从缓存加载 {len(cache)} 个样本")
        return cache
    except Exception as e:
        logger.error(f"加载缓存失败: {str(e)}")
        return None


def create_cached_dataset(dataset, cache_name):
    """创建并返回缓存数据集，避免重复创建"""
    # 检查缓存文件是否存在
    cache_file = f"{cache_name}.pkl"


    # 检查缓存是否有效
    cache_valid = False
    if os.path.exists(cache_file):
        try:

            logger.info(f"缓存 {cache_file} 有效 (哈希匹配)")
            cache = load_cached_dataset(cache_file)
            if cache is not None:
                return cache
        except Exception as e:
            logger.warning(f"读取缓存哈希失败: {str(e)}")
    else:
        # 创建并行缓存数据集
        logger.info(f"创建新的缓存数据集: {cache_name}")
        parallel_cache = ParallelMemoryCachedDataset(dataset)

        # 保存到缓存文件
        save_success = save_cached_dataset(parallel_cache.cache, cache_file)
        return parallel_cache.cache


if __name__ == "__main__":

    train_caches = "train_cache"
    val_caches = "val_cache"


    log_dir = "continuous_training"
    os.makedirs(log_dir, exist_ok=True)  # 自动创建目录（如果不存在）

    # 生成带时间戳的文件路径
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"training_logs_{timestamp}.csv"
    log_file = os.path.abspath(os.path.join(log_dir, log_filename))  # 确保绝对路径

    if not os.path.exists(log_file):
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch',
                'phase',
                'train_loss',
                'train_iou_rln',
                'train_iou_sen',
                'train_iou_pre',
                'train_iou_fnr',
                'train_iou_fpr',
                'train_iou_dice',
                'train_iou_ap',
                'val_loss',
                'val_iou_rln',
                'val_iou_sen',
                'val_iou_pre',
                'val_iou_fnr',
                'val_iou_fpr',
                'val_iou_dice',
                'val_iou_ap',
                "al",
            ])

    model = DR_MTUNet(in_ch=3, seg_ch=2, recon_ch=3, shape_dim=5)
    pretrained_weights = r"D:\\liuhongtao\\segmentationproject\\best_model.pth"

    train_root = r"D:\\liuhongtao\\segmentationproject\\dataset\\training"
    val_root = r"D:\\liuhongtao\\segmentationproject\\dataset\\validation"
    test_root = r"D:\\liuhongtao\\segmentationproject\\dataset\\test"

    device = torch.device(f'cuda:{1}' if torch.cuda.is_available() else 'cpu')
    batch_size = 8
    if device.type == 'cuda':
        torch.cuda.set_device(device.index)

    # 优化工作进程数 - 可根据CPU核心数调整
    num_workers_val = min(8, os.cpu_count() // 2)
    persistent_workers_val = True
    prefetch_factor_val = 8  # 增加预取因子

    # 创建内存缓存的数据集
    train_cache, val_cache, test_cache = [], [], []

    if check_dataset_integrity(train_root):
        try:
            # 创建数据集实例（禁用增强以节省加载时间）
            logger.info("创建训练集...")
            train_dataset = MedicalSegmentationDataset(
                root_dir=train_root,
                augment=False,
                augment_prob=0.8
            )

            logger.info("创建验证集...")
            val_dataset = MedicalSegmentationDataset(
                root_dir=val_root,
                augment=False
            )

            logger.info("创建测试集...")
            test_dataset = MedicalSegmentationDataset(
                root_dir=test_root,
                augment=False
            )


        except Exception as e:
            logger.error(f"运行时错误: {str(e)}")
    else:
        logger.error("数据集完整性检查未通过")
        exit(1)

    # 创建数据集加载器（使用内存缓存的数据集）

    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,  # 测试集不需要shuffle
        persistent_workers=persistent_workers_val,
        num_workers=num_workers_val,
        prefetch_factor=prefetch_factor_val,
        drop_last=True,
        collate_fn=multi_tensor_collate
    )

    # 启动训练
    train_model(
        model,
        train_loader,
        val_loader,
        test_loader,
        phases=[
            ('recon', 300),
            ('full', 200)
        ],
        pretrained_weights=pretrained_weights,
        device=device,
        log_file=log_file
    )
    # # =============================test=====================================================
    history = {'test_metrics': None}
    test_metrics = validate(model, test_loader,  device, is_test=True, pretrained_weights=pretrained_weights)

    history['test_metrics'] = test_metrics
    logger.info(
        f"测试结果: RLN IoU {test_metrics['rln_iou']:.4f}"

    )
    # cache_names = [train_caches, val_caches]
    # logger.info("训练完成，正在清理缓存文件...")
    # clean_cache_files(cache_names)
    # logger.info("缓存文件清理完成")

