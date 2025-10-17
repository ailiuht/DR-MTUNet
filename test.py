import os

import numpy as np
import torch
from tqdm import tqdm
from load_data import logger
from loos import MultiTaskLoss
from indicator import compute_class_iou
from show_data import generate_comprehensive_report


def validate(model, loader, device, is_test=False, pretrained_weights=None):
    """测试函数"""

    model.eval()
    model = model.to(device)
    if pretrained_weights is not None:
        # 预处理路径：扩展用户目录和变量
        pretrained_weights = os.path.abspath(os.path.expanduser(os.path.expandvars(pretrained_weights)))

        if not os.path.isfile(pretrained_weights):
            logger.error(f"\n{'!' * 20} 预训练权重文件不存在 {'!' * 20}\n路径: {pretrained_weights}")
        else:
            logger.info(f"\n{'=' * 20} 加载预训练权重 {'=' * 20}")
            try:
                checkpoint = torch.load(pretrained_weights, map_location=device, weights_only=True)

                # 解析状态字典
                state_dict = checkpoint.get('model_state', checkpoint)

                # 非严格模式加载
                load_result = model.load_state_dict(state_dict, strict=False)
                logger.info(f"成功加载权重文件: {os.path.basename(pretrained_weights)}")

                # 记录加载详细信息
                if load_result.missing_keys:
                    logger.warning(f"缺失的层 ({len(load_result.missing_keys)}个):\n{load_result.missing_keys}")
                if load_result.unexpected_keys:
                    logger.warning(f"多余的层 ({len(load_result.unexpected_keys)}个):\n{load_result.unexpected_keys}")

                # 记录设备信息
                logger.info(f"权重加载到设备: {next(model.parameters()).device}")

            except Exception as e:
                logger.error(f"\n{'!' * 20} 权重加载失败 {'!' * 20}\n错误类型: {type(e).__name__}\n详细信息: {str(e)}")
                raise
    total_loss = 0.0
    iou_tracker = {'rln': [],"sen": [], "pre": [], "fnr": [], "fpr": [], "dice": [], "ap": []}

    val_bar = tqdm(loader, desc='Testing' if is_test else 'Validating',
                   bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}',
                   dynamic_ncols=True)
    criterion = MultiTaskLoss(num_classes=2)

    with torch.no_grad():
        for batch_idx, (img, seg_gt, edge_gt, shape_gt, max_vals, min_vals) in enumerate(val_bar):
            try:
                img = img.to(device, non_blocking=True)
                seg_gt = seg_gt.to(device, non_blocking=True).long().squeeze(1)
                edge_gt = edge_gt.to(device, non_blocking=True).float()
                shape_gt = shape_gt.to(device, non_blocking=True).float()
                seg_pred, recon_pred, shape_pred, al = model(img)
                # seg_pred, recon_pred, al = model(img)
                loss = criterion(seg_pred, seg_gt, edge_gt, shape_pred, shape_gt, recon_pred, img)
                # loss = criterion(seg_pred, seg_gt, edge_gt, shape_gt, recon_pred, img)
                rln_iou, sen, pre, fnr, fpr, dice, ap = compute_class_iou(seg_pred, seg_gt, class_idx=1)
                # =================================可视化部分=====================
                # 测试模式下的可视化
                if is_test:
                    debug_dir = "test_reports"
                    # shape_pred = torch.tensor([[0,0,0,0,0],[0,0,0,0,0]])
                    generate_comprehensive_report(
                        img=img,
                        seg_gt=seg_gt,
                        shape_gt=shape_gt,
                        seg_pred=seg_pred,
                        recon_pred=recon_pred,
                        shape_pred=shape_pred,
                        debug_dir=debug_dir,
                        max_vals=max_vals,
                        min_vals=min_vals
                    )
                    print(f"生成综合报告至 {debug_dir} 目录")
                # ===============可视化部分结束=====================================================
                val_bar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'RLN IoU': f"{rln_iou:.3f}",
                    'GPU Mem': f"{torch.cuda.memory_allocated() / 1e9:.2f}G"
                })
                total_loss += loss.item()
                iou_tracker['rln'].append(rln_iou.item())
                iou_tracker['sen'].append(sen.item())
                iou_tracker['pre'].append(pre.item())
                iou_tracker['fnr'].append(fnr.item())
                iou_tracker['fpr'].append(fpr.item())
                iou_tracker['dice'].append(dice.item())
                iou_tracker['ap'].append(ap.item())
            except Exception as e:
                logger.error(f"验证批次异常: {str(e)}")
                continue

    val_bar.close()
    metrics = {
        'loss': total_loss / len(loader),
        'rln_iou': np.nanmean(iou_tracker['rln']),
        'sen': np.nanmean(iou_tracker['sen']),
        'pre': np.nanmean(iou_tracker['pre']),
        'fnr': np.nanmean(iou_tracker['fnr']),
        'fpr': np.nanmean(iou_tracker['fpr']),
        'dice': np.nanmean(iou_tracker['dice']),
        'ap': np.nanmean(iou_tracker['ap'])
    }

    if is_test:
        logger.info("\n=== 最终测试结果 ===")
        logger.info(f"平均损失: {metrics['loss']:.4f}")
        logger.info(f"喉返神经IoU: {metrics['rln_iou']:.4f}")

    return metrics