import os
import re
import json
from collections import defaultdict
import random
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import imageio.v3 as iio
import logging
import cv2
import numpy as np
import torch
import logging

from matplotlib import pyplot as plt
from skimage.draw import polygon
from skimage.transform import resize
from torch.utils.data import Dataset
from scipy.spatial import ConvexHull
from pathlib import Path
from typing import Tuple, List, Dict


# 配置日志系统
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MedicalSegmentationDataset(Dataset):
    def __init__(self, root_dir: str,
                 target_size: Tuple[int, int] = (256, 256),
                 simplify_tolerance: int = 5,
                 sobel_kernel: int = 3,  # Sobel算子的大小
                 edge_threshold: float = 0.2, # 新增控制边缘检测的灵敏度 值越小，检测到的边缘越多参数控制增强
                 augment: bool = False,
                 augment_prob: float = 0.5):  # 新增增强概率控制
        """
        医学图像分割数据集类
        返回四元组数据：(图像, 掩码, 边缘图, 位置参数)
        新增参数:
            augment: 是否启用数据增强
            augment_prob: 单样本增强概率

        参数:
            root_dir: 数据集根目录
            target_size: 目标图像尺寸 (H, W)
            simplify_tolerance: 多边形简化强度参数
            sobel_kernel: 边缘检测Sobel核大小
            edge_threshold: 边缘检测阈值
        """
        self.root_dir = Path(root_dir).resolve()
        if not self.root_dir.is_dir():
            raise NotADirectoryError(f"数据集目录不存在: {self.root_dir}")

        # 初始化参数
        self.augment = augment
        self.augment_prob = augment_prob
        # rotation_range: 控制图像的随机旋转范围。
        # scale_range: 缩放范围，影响图像的大小调整。
        # crop_range: 随机裁剪的比例范围，
        # contrast_range: 对比度调整范围，
        # elastic_alpha: 弹性形变的强度，类似用手指在图像上按压产生的变形效果
        # elastic_sigma: 弹性形变的平滑度，σ值越大，形变区域越宽大平滑
        self.augment_params = {
            'rotation_range': (-30, 30),
            'scale_range': (0.85, 1.15),
            'crop_range': (0.7, 0.9),
            'contrast_range': (0.8, 1.2),
            'elastic_alpha': 100,
            'elastic_sigma': 8
        }
        self.target_size = target_size
        self.simplify_tolerance = simplify_tolerance
        self.sobel_kernel = sobel_kernel
        self.edge_threshold = edge_threshold

        # 获取排序后的文件列表
        self.image_paths = self._get_sorted_files(r'^PR\d+\.png$')
        self.json_paths = self._get_sorted_files(r'^PR\d+\.json$')

        # 验证文件对
        self._validate_file_pairs()
        logger.info(f"成功加载数据集，共 {len(self.image_paths)} 个样本")

    def _get_sorted_files(self, pattern: str) -> List[Path]:
        """自然排序获取匹配文件"""
        files = []
        for f in self.root_dir.iterdir():
            if re.match(pattern, f.name, re.IGNORECASE):
                files.append(f)

        # 自然排序函数
        def natural_sort_key(path: Path):
            return [int(s) if s.isdigit() else s.lower()
                    for s in re.split(r'(\d+)', path.name)]

        return sorted(files, key=natural_sort_key)

    def _validate_file_pairs(self):
        """验证图片与标注文件对应关系"""
        if len(self.image_paths) != len(self.json_paths):
            raise ValueError(
                f"图片({len(self.image_paths)})与标注文件({len(self.json_paths)})数量不匹配"
            )

        for img_path, json_path in zip(self.image_paths, self.json_paths):
            img_num = re.search(r'PR(\d+)', img_path.name, re.I).group(1)
            json_num = re.search(r'PR(\d+)', json_path.name, re.I).group(1)

            if img_num != json_num:
                raise ValueError(
                    f"文件编号不匹配: {img_path.name} vs {json_path.name}"
                )

    def __len__(self) -> int:
        return len(self.image_paths)

    def _smart_resize(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """智能缩放图像并保持宽高比"""
        try:
            h, w = image.shape[:2]
            scale = min(self.target_size[0] / h, self.target_size[1] / w)
            new_h, new_w = int(h * scale), int(w * scale)

            # 双三次插值缩放
            resized = resize(image, (new_h, new_w),
                             order=3,  # 双三次插值
                             preserve_range=True,
                             anti_aliasing=True)

            # 对称填充
            pad_h = self.target_size[0] - new_h
            pad_w = self.target_size[1] - new_w
            padded = np.pad(resized,
                            ((0, pad_h), (0, pad_w), (0, 0)),
                            mode='constant',
                            constant_values=255)

            return padded.astype(np.float32), scale, (pad_h, pad_w)
        except Exception as e:
            logger.error(f"图像缩放失败: {str(e)}")
            raise

    def _simplify_polygon(self, points: np.ndarray) -> np.ndarray:
        """凸包简化多边形顶点"""
        try:
            if len(points) < 3:
                logger.warning("多边形点数不足，无法简化")
                return points

            hull = ConvexHull(points)
            hull_points = points[hull.vertices]
            # step = max(1, len(hull_points) // self.simplify_tolerance)
            # return hull_points[::step]
            return hull_points
        except Exception as e:
            logger.warning(f"多边形简化失败: {str(e)}")
            return points

    def _create_mask(self, shapes: List[Dict],
                     scale: float) -> np.ndarray:
        """生成多类别分割掩码"""
        mask = np.zeros(self.target_size, dtype=np.float32)

        for shape in shapes:
            try:
                # 验证标注结构
                if 'points' not in shape or 'label' not in shape:
                    logger.warning("无效的标注结构，跳过该形状")
                    continue

                # 坐标转换与裁剪
                scaled_points = [
                    (
                        max(0, min(x * scale, self.target_size[1] - 1)),
                        max(0, min(y * scale, self.target_size[0] - 1))
                    )
                    for (x, y) in shape['points']
                ]

                # 生成多边形区域
                rr, cc = polygon(
                    [p[1] for p in scaled_points],  # y坐标
                    [p[0] for p in scaled_points],  # x坐标
                    shape=self.target_size
                )

                # 有效性检查
                valid = (rr >= 0) & (rr < self.target_size[0]) & \
                        (cc >= 0) & (cc < self.target_size[1])

                if not valid.any():
                    logger.warning("生成的有效区域为空，跳过该形状")
                    continue

                # 根据类别设置掩码值
                if shape['label'] == 'pangxian':
                    mask_value = 1.0
                else:
                    continue

                mask[rr[valid], cc[valid]] = mask_value

            except Exception as e:
                logger.error(f"处理形状时出错[{shape.get('label', '未知')}]: {str(e)}")
                continue

        return mask

    def _generate_edges(self, mask: np.ndarray) -> np.ndarray:
        """生成双通道边缘图（优化版本）"""
        edges = np.zeros((1, *self.target_size), dtype=np.float32)

        for channel, class_id in enumerate([1]):
            class_mask = (mask == class_id).astype(np.uint8)
            if np.max(class_mask) == 0:
                continue

            # 梯度计算
            sobel_x = cv2.Sobel(class_mask, cv2.CV_64F, 1, 0, ksize=self.sobel_kernel)
            sobel_y = cv2.Sobel(class_mask, cv2.CV_64F, 0, 1, ksize=self.sobel_kernel)
            grad_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

            # 动态阈值计算
            # 它基于一个基础阈值self.edge_threshold加上梯度幅值平均值的0.5倍。
            # 这种动态调整阈值的方法是为了根据当前图像的实际边缘强度自适应地调整阈值，
            # 使得在边缘较强时提高阈值，减少噪声误检，而在边缘较弱时降低阈值，避免漏检。
            # 使用动态阈值可以自适应不同图像的特性，提高边缘检测的稳定性和准确性。
            # 动态阈值会相应提高，导致只有梯度幅值较高的区域被识别为边缘。
            mean_grad = np.mean(grad_magnitude) if np.max(grad_magnitude) > 0 else 0
            dynamic_thresh = self.edge_threshold + mean_grad * 0.5

            # 边缘检测与后处理
            edge_map = (grad_magnitude > dynamic_thresh).astype(np.float32)

            # 形态学后处理 ,优化边缘检测结果，通过形态学操作消除噪声和连接断裂的边缘。
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            edge_map = cv2.morphologyEx(edge_map, cv2.MORPH_CLOSE, kernel)

            edges[channel] = edge_map

        return edges

    def _get_shape_params(self, shapes: List[Dict],
                          scale: float) -> Dict[str, List[float]]:
        """获取形状参数：外接椭圆特征（支持多实例）"""
        params = {'pangxian': [0.0] * 5}

        # 按类别收集所有有效点
        category_points = defaultdict(list)
        for shape in shapes:
            try:
                label = shape['label']
                if label not in ['pangxian']:
                    continue

                # 坐标转换与凸包简化
                scaled_points = [
                    (max(0, min(x * scale, self.target_size[1] - 1)),
                     max(0, min(y * scale, self.target_size[0] - 1)))
                    for (x, y) in shape['points']
                ]
                simplified = self._simplify_polygon(np.array(scaled_points))

                # 合并所有同类实例的点
                if len(simplified) >= 3:  # 允许后续凸包合并
                    category_points[label].extend(simplified)

            except Exception as e:
                logger.warning(f"处理{label}形状时出错: {str(e)}")
                continue

        # 为每个类别计算椭圆参数
        for category in ['pangxian']:
            points = np.array(category_points[category], dtype=np.float32)

            # 有效性检查
            if len(points) < 5:
                logger.debug(f"{category}总点数不足({len(points)})")
                continue

            try:
                # 合并所有实例后的凸包计算
                hull = ConvexHull(points)
                hull_points = points[hull.vertices]

                if len(hull_points) < 5:
                    logger.debug(f"{category}凸包点数不足({len(hull_points)})")
                    continue

                # 椭圆拟合
                ellipse = cv2.fitEllipse(hull_points)
                params[category] = [
                    ellipse[0][0],  # 中心x
                    ellipse[0][1],  # 中心y
                    ellipse[1][0],  # 长轴
                    ellipse[1][1],  # 短轴
                    ellipse[2]  # 旋转角度
                ]
            except Exception as e:
                logger.warning(f"{category}椭圆拟合失败: {str(e)}")

        return params

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        try:
            # 加载原始数据path
            img_path = self.image_paths[idx]
            json_path = self.json_paths[idx]

            # 加载并处理原始数据（保持原有代码不变）
            image = self._load_image(img_path)
            processed_img, scale, (pad_h, pad_w) = self._smart_resize(image)
            label_data = self._load_annotation(json_path)
            target_shapes = [s for s in label_data.get('shapes', [])
                             if s.get('label') in ['pangxian']]
            mask = self._create_mask(target_shapes, scale)

            # 转换为numpy数组进行处理
            image_np = processed_img.astype(np.uint8)
            mask_np = mask.astype(np.uint8)

            # 应用数据增强
            if self.augment and random.random() < self.augment_prob:
                image_np, mask_np = self._apply_augmentations(image_np, mask_np)

            # 生成边缘图和位置参数（使用可能增强后的mask）
            edge_map = self._generate_edges(mask_np)
            position_params = self._get_shape_params_from_mask(mask_np)

            # 归一化处理
            image_np, max_vals, min_vals = self._norm_method(image_np)

            # 转换为张量
            image_tensor = torch.as_tensor(image_np, dtype=torch.float32).permute(2, 0, 1)
            max_vals = torch.as_tensor(max_vals, dtype=torch.float32).permute(2, 0, 1)
            min_vals = torch.as_tensor(min_vals, dtype=torch.float32).permute(2, 0, 1)
            mask_tensor = torch.as_tensor(mask_np, dtype=torch.float32).unsqueeze(0)
            edge_tensor = torch.as_tensor(edge_map, dtype=torch.float32)
            params_tensor = torch.tensor(position_params, dtype=torch.float32)

            return image_tensor, mask_tensor, edge_tensor, params_tensor, max_vals, min_vals

        except Exception as e:
            logger.error(f"处理样本{idx}失败: {str(e)}")
            raise

    def _norm_method(self, image_np):
        image_float = image_np.astype(np.float32)
        min_vals = np.min(image_float, axis=(0,1),keepdims=True)
        max_vals = np.max(image_float, axis=(0, 1), keepdims=True)
        safe_range = np.where(max_vals - min_vals == 0, 1e-8, max_vals - min_vals)
        normalized_image = (image_float - min_vals) / safe_range
        return normalized_image, max_vals, min_vals




    def _apply_augmentations(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        应用随机数据增强组合
        返回增强后的图像和掩膜
        """
        # 深拷贝原始数据
        aug_image = image.copy()
        aug_mask = mask.copy()

        # 随机应用增强组合（保持空间变换顺序）
        if random.random() > 0.5:
            aug_image, aug_mask = self._random_rotation(aug_image, aug_mask)

        if random.random() > 0.5:
            aug_image, aug_mask = self._random_scale(aug_image, aug_mask)

        if random.random() > 0.5:
            aug_image, aug_mask = self._random_crop(aug_image, aug_mask)

        if random.random() > 0.5:
            aug_image = self._random_contrast(aug_image)

        if random.random() > 1:
            aug_image, aug_mask = self._elastic_deform(aug_image, aug_mask)

        return aug_image, aug_mask

    def _random_rotation(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """随机旋转增强"""
        angle = np.random.uniform(*self.augment_params['rotation_range'])
        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        # 生成旋转矩阵
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # 应用旋转
        image_rot = cv2.warpAffine(image, M, (w, h),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=0)
        mask_rot = cv2.warpAffine(mask, M, (w, h),
                                  flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=0)
        return image_rot, mask_rot

    def _random_scale(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """随机缩放增强（修复维度错误版）"""
        scale = np.random.uniform(*self.augment_params['scale_range'])
        h, w = image.shape[:2]

        # 确保缩放后至少1像素
        new_h = max(1, int(h * scale))
        new_w = max(1, int(w * scale))

        # 缩放图像和掩码
        image_scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask_scaled = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        def process_dimension(dim_original: int, dim_scaled: int, data: np.ndarray, is_height: bool) -> np.ndarray:
            """处理单维度（自动适配图像为2D或3D）"""
            # 确定数据维度（假设通道在最后一维）
            is_3d = (data.ndim == 3)

            if dim_scaled > dim_original:
                # 随机裁剪
                start = np.random.randint(0, dim_scaled - dim_original)
                if is_height:
                    return data[start:start + dim_original, :] if not is_3d else data[start:start + dim_original, :, :]
                else:
                    return data[:, start:start + dim_original] if not is_3d else data[:, start:start + dim_original, :]
            else:
                # 计算填充参数（适配2D和3D图像）
                pad_total = dim_original - dim_scaled
                pad_start = pad_total // 2
                pad_end = pad_total - pad_start

                if is_height:
                    # 高度填充：((top, bottom), (left, right), (0,0)) 如果是3D
                    padding = ((pad_start, pad_end), (0, 0)) if not is_3d else ((pad_start, pad_end), (0, 0), (0, 0))
                else:
                    # 宽度填充：((0, 0), (left, right), (0,0)) 如果是3D
                    padding = ((0, 0), (pad_start, pad_end)) if not is_3d else ((0, 0), (pad_start, pad_end), (0, 0))

                return np.pad(data, padding, mode='constant', constant_values=0)

        # 处理高度
        image_out = process_dimension(h, new_h, image_scaled, is_height=True)
        mask_out = process_dimension(h, new_h, mask_scaled, is_height=True)

        # 处理宽度
        image_out = process_dimension(w, new_w, image_out, is_height=False)
        mask_out = process_dimension(w, new_w, mask_out, is_height=False)

        return image_out, mask_out

    def _random_crop(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """随机裁切增强"""
        h, w = image.shape[:2]
        crop_ratio = np.random.uniform(*self.augment_params['crop_range'])
        crop_h = int(h * crop_ratio)
        crop_w = int(w * crop_ratio)

        # 随机起始位置
        y_start = np.random.randint(0, h - crop_h)
        x_start = np.random.randint(0, w - crop_w)

        # 裁剪
        image_crop = image[y_start:y_start + crop_h, x_start:x_start + crop_w]
        mask_crop = mask[y_start:y_start + crop_h, x_start:x_start + crop_w]

        # 插值缩放到原尺寸
        image_out = cv2.resize(image_crop, (w, h), interpolation=cv2.INTER_LINEAR)
        mask_out = cv2.resize(mask_crop, (w, h), interpolation=cv2.INTER_NEAREST)
        return image_out, mask_out

    def _random_contrast(self, image: np.ndarray) -> np.ndarray:
        """随机对比度调整"""
        contrast = np.random.uniform(*self.augment_params['contrast_range'])
        mean = image.mean()
        return np.clip((image - mean) * contrast + mean, 0, 255).astype(np.uint8)

    def _elastic_deform(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """弹性形变增强（修复版本）"""
        alpha = self.augment_params['elastic_alpha']
        sigma = self.augment_params['elastic_sigma']
        h, w = image.shape[:2]

        # 生成归一化位移场（基于图像尺寸）
        dx = np.random.uniform(-1, 1, (h, w)) * alpha / w
        dy = np.random.uniform(-1, 1, (h, w)) * alpha / h

        # 应用高斯滤波
        dx = cv2.GaussianBlur(dx, (0, 0), sigma)
        dy = cv2.GaussianBlur(dy, (0, 0), sigma)

        # 生成网格坐标（标准化到[-1,1]范围）
        x, y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))

        # 应用位移场
        x = x + dx
        y = y + dy

        # 将坐标转换到像素空间 [0, w-1] 和 [0, h-1]
        map_x = ((x + 1) * (w - 1) / 2).astype(np.float32)
        map_y = ((y + 1) * (h - 1) / 2).astype(np.float32)

        # 确保坐标在有效范围内
        map_x = np.clip(map_x, 0, w - 1)
        map_y = np.clip(map_y, 0, h - 1)

        # 映射图像
        deformed_image = np.zeros_like(image)
        for c in range(image.shape[2]):
            deformed_image[:, :, c] = cv2.remap(
                image[:, :, c],
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )

        # 映射mask（使用最近邻插值）
        deformed_mask = cv2.remap(
            mask,
            map_x,
            map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        return deformed_image, deformed_mask

    def _get_shape_params_from_mask(self, mask: np.ndarray) -> List[float]:
        """从增强后的mask重新计算形状参数"""
        params = {'pangxian': [0.0] * 5}

        for class_id, label in enumerate(['pangxian'], start=1):
            class_mask = (mask == class_id).astype(np.uint8)

            if np.any(class_mask):
                # 寻找最大连通域
                contours, _ = cv2.findContours(class_mask,
                                               cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    max_contour = max(contours, key=cv2.contourArea)

                    if len(max_contour) >= 5:  # 椭圆拟合需要至少5个点
                        try:
                            ellipse = cv2.fitEllipse(max_contour)
                            params[label] = [
                                ellipse[0][0],  # 中心x
                                ellipse[0][1],  # 中心y
                                ellipse[1][0],  # 长轴
                                ellipse[1][1],  # 短轴
                                ellipse[2]  # 旋转角度
                            ]
                        except:
                            pass
        return params['pangxian']

    def _load_image(self, path: Path) -> np.ndarray:
        """加载图像文件"""
        try:
            with open(path, 'rb') as f:
                img_bytes = np.frombuffer(f.read(), dtype=np.uint8)
                image = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError("图像解码失败")
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转为RGB格式
        except Exception as e:
            logger.error(f"图像加载失败{path}: {str(e)}")
            raise

    def _load_annotation(self, path: Path) -> dict:
        """加载标注文件"""
        encodings = ['utf-8', 'gbk', 'latin1']
        for enc in encodings:
            try:
                with open(path, 'r', encoding=enc) as f:
                    data = json.load(f)

                if 'shapes' not in data:
                    raise ValueError("标注文件缺少'shapes'字段")
                return data
            except UnicodeDecodeError:
                continue
            except json.JSONDecodeError as e:
                logger.error(f"JSON解析失败: {path}")
                raise

        raise ValueError(f"无法解码标注文件: {path}")


def check_dataset_integrity(root_dir: str) -> bool:
    """数据集完整性检查"""
    try:
        dataset = MedicalSegmentationDataset(root_dir)
        logger.info("基础文件检查通过")

        # 随机检查三个样本
        for idx in np.random.choice(len(dataset), 3):
            img, mask, edge, params, max_vals, min_vals = dataset[idx]

            # 验证张量形状
            assert img.shape == (3, 256, 256), f"图像形状异常: {img.shape}"
            assert mask.shape == (1, 256, 256), f"掩码形状异常: {mask.shape}"
            assert edge.shape == (1, 256, 256), f"边缘图形状异常: {edge.shape}"
            assert params.shape == (5,), f"参数形状异常: {params.shape}"

            # 验证数值范围
            assert 0 <= img.min() <= img.max() <= 1, "图像数值范围异常"
            assert set(np.unique(mask.numpy()).tolist()).issubset({0.0, 1.0}), "掩码数值异常"
            assert 0 <= edge.min() <= edge.max() <= 1.0, "边缘图数值范围异常"

        logger.info("随机样本检查通过")
        return True
    except Exception as e:
        logger.error(f"数据集检查失败: {str(e)}")
        return False


