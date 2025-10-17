import math

import numpy as np
import torch


def compute_class_iou(pred, target, class_idx, epsilon=1e-6, device="cpu"):

    iou_value = 0.1
    batch_size = pred.size(0)
    TPs, FPs, TNs, FNs = 0, 0, 0, 0
    total_iou = 0.0
    valid_samples = 0
    iou_scores = []

    for i in range(batch_size):
        sample_pred = pred[i].unsqueeze(0)
        sample_target = target[i].unsqueeze(0)

        pred_mask = (sample_pred.argmax(1) == class_idx)
        target_mask = (sample_target == class_idx)
        target_exists = target_mask.any()

        # 处理四类情况
        if not target_exists:
            if pred_mask.any():
                FPs += 1  # 假阳性
                iou_scores.append(float('nan'))
            else:
                TNs += 1  # 真阴性
                iou_scores.append(float('nan'))
            continue
        else:
            valid_samples += 1  # 真实存在目标的样本

        # 计算IoU组件
        intersection = (pred_mask & target_mask).sum()
        union = (pred_mask | target_mask).sum()
        iou = intersection.float() / (union.float() + epsilon)
        iou_scores.append(iou.item())

        # 根据IoU阈值判断
        if iou >= iou_value:
            TPs += 1
            total_iou += 1.0
        else:
            FNs += 1
            total_iou += iou.item()

    # 指标计算（增加稳定性处理）
    sen = TPs / (TPs + FNs + epsilon) if (TPs + FNs) > 0 else float('nan')
    pre = TPs / (TPs + FPs + epsilon) if (TPs + FPs) > 0 else float('nan')
    fnr = 1 - sen if not math.isnan(sen) else float('nan')
    fpr = 1 - sen if not math.isnan(sen) else float('nan')
    # dice = (2 * TPs) / (2 * TPs + FPs + FNs + epsilon)

    mean_iou = total_iou / valid_samples if valid_samples > 0 else float('nan')
    dice = (2 * mean_iou) / (1 + mean_iou + epsilon) if mean_iou > 0 else float('nan')
    # ap = compute_class_ap(pred, target, class_idx, iou_scores)
    #
    ap = compute_class_ap(pred, target, class_idx, iou_scores, iou_threshold=iou_value)

    return (
        torch.tensor(mean_iou, device=pred.device),
        torch.tensor(sen, device=pred.device),
        torch.tensor(pre, device=pred.device),
        torch.tensor(fnr, device=pred.device),
        torch.tensor(fpr, device=pred.device),
        torch.tensor(dice, device=pred.device),
        ap
    )

def compute_class_ap(pred, target, class_idx, iou_scores, iou_threshold=0.5, num_thresholds=100):
    """修复要点：
    1. 置信度计算逻辑修正
    2. 排序逻辑优化
    3. 精确率-召回率计算改进
    """
    device = pred.device
    batch_size = pred.size(0)

    # 获取所有样本的置信度和目标存在标记
    confidences = []
    has_target = []
    for i in range(batch_size):
        prob_map = pred[i, class_idx]
        target_mask = (target[i] == class_idx)
        confidences.append(prob_map.max().item())  # 始终取最大置信度
        has_target.append(target_mask.any().item())

    confidences = np.array(confidences)
    has_target = np.array(has_target)
    iou_scores = np.array(iou_scores)

    # 按置信度降序排序
    sorted_indices = np.argsort(-confidences)
    sorted_confidences = confidences[sorted_indices]
    sorted_has_target = has_target[sorted_indices]
    sorted_ious = iou_scores[sorted_indices]

    # 生成评估阈值
    thresholds = np.linspace(0, 1, num_thresholds)
    precisions = []
    recalls = []

    total_positive = sum(sorted_has_target)
    if total_positive == 0:
        return torch.tensor(float("nan"), device=device)

    for th in thresholds:
        selected = sorted_confidences >= th
        tp, fp = 0, 0

        # 处理选中的预测
        for idx in np.where(selected)[0]:
            if sorted_has_target[idx]:
                if sorted_ious[idx] >= iou_threshold:
                    tp += 1
                else:
                    fp += 1  # IoU不达标
            else:
                fp += 1  # 假阳性

        # 处理未选中的真实目标
        fn = sum(sorted_has_target & ~selected)

        precision = tp / (tp + fp + 1e-6)
        recall = tp / (total_positive + 1e-6)
        precisions.append(precision)
        recalls.append(recall)

    # 11点插值法计算AP
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        mask = np.array(recalls) >= t
        if np.any(mask):
            p = max(np.array(precisions)[mask])
        else:
            p = 0.0
        ap += p / 11.0

    return torch.tensor(ap, device=device)