import csv
import os


from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import torch
import torch.nn as nn
import numpy as np
import logging
from loos import MultiTaskLoss
import time
from tqdm import tqdm
from indicator import compute_class_iou
from show_data import generate_comprehensive_report

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AverageMeter:
    """用于统计平均值的工具类"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_model(model, train_loader, val_loader, test_loader, phases=[('recon', 5), ('full', 15)],
                pretrained_weights=None, device='cpu',log_file=None):

    """完整的训练流程，包含验证和测试阶段"""
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

    optim_groups = [
        {'name': 'enc1', 'params': list(model.enc1.parameters()), 'active': True},
        {'name': 'enc2', 'params': list(model.enc2.parameters()), 'active': True},
        {'name': 'enc3', 'params': list(model.enc3.parameters()), 'active': True},
        {'name': 'enc4', 'params': list(model.enc4.parameters()), 'active': True},
        {'name': 'gate1', 'params': list(model.gate1.parameters()), 'active': True},
        {'name': 'gate2', 'params': list(model.gate2.parameters()), 'active': True},
        {'name': 'gate3', 'params': list(model.gate3.parameters()), 'active': True},
        {'name': 'gate4', 'params': list(model.gate4.parameters()), 'active': True},

        {'name': 'pool1', 'params': list(model.pool1.parameters()), 'active': True},
        {'name': 'pool2', 'params': list(model.pool2.parameters()), 'active': True},
        {'name': 'pool3', 'params': list(model.pool3.parameters()), 'active': True},
        {'name': 'up1', 'params': list(model.up1.parameters()), 'active': True},
        {'name': 'up2', 'params': list(model.up2.parameters()), 'active': True},
        {'name': 'up3', 'params': list(model.up3.parameters()), 'active': True},
        {'name': 'ups1', 'params': list(model.ups1.parameters()), 'active': True},
        {'name': 'ups2', 'params': list(model.ups2.parameters()), 'active': True},
        {'name': 'ups3', 'params': list(model.ups3.parameters()), 'active': True},

        {'name': 'dec4', 'params': list(model.dec4.parameters()), 'active': True},
        {'name': 'dec3', 'params': list(model.dec3.parameters()), 'active': True},
        {'name': 'dec2', 'params': list(model.dec2.parameters()), 'active': True},
        {'name': 'dec1', 'params': list(model.dec1.parameters()), 'active': True},
        {'name': 'decs4', 'params': list(model.decs4.parameters()), 'active': True},
        {'name': 'decs3', 'params': list(model.decs3.parameters()), 'active': True},
        {'name': 'decs2', 'params': list(model.decs2.parameters()), 'active': True},
        {'name': 'decs1', 'params': list(model.decs1.parameters()), 'active': True},
        {'name': 'seg_head', 'params': list(model.seg_head.parameters()), 'active': False},
        {'name': 'segs_head', 'params': list(model.segs_head.parameters()), 'active': True},
        {'name': 'recon_head', 'params': list(model.recon_head.parameters()), 'active': True},
        {'name': 'ellipse_predictor', 'params': list(model.ellipse_predictor.parameters()), 'active': True},
        {'name': 'bottleneck', 'params': list(model.bottleneck.parameters()), 'active': True},
        {'name': 'alpha', 'params': [model.alpha], 'active': True}
    ]


    # 参数组验证
    for group in optim_groups:
        if not isinstance(group['params'], list):
            raise TypeError(f"参数组 {group['name']} 的params必须是列表")
        if not all(isinstance(p, nn.Parameter) for p in group['params']):
            raise ValueError(f"参数组 {group['name']} 包含非参数对象")

    # 优化器构建函数
    def create_optimizer(groups):
        params = []
        for g in groups:
            if g['active'] and len(g['params']) > 0:
                params.extend(g['params'])

        if not params:
            raise RuntimeError("没有可训练参数！请检查模型配置")

        if device.type == 'cuda':
            mem_info = torch.cuda.mem_get_info()
            if mem_info[0] < 1024 ** 3:
                logger.warning("可用显存不足，建议减小batch_size")

        return torch.optim.AdamW(params, lr=1e-2, weight_decay=0.01)

    criterion = MultiTaskLoss(num_classes=2, device=device)
    best_val_iou = 0.0
    history = {'train_loss': [], 'val_loss': [], 'test_metrics': None}


    def create_scheduler(optimizer, phase_name,total_epochss):
        """根据阶段创建学习率调度器"""
        if phase_name == 'recon':
            # 重建阶段使用余弦退火
            return CosineAnnealingLR(optimizer, T_max=total_epochss, eta_min=1e-12)
        else:
            # 完整训练阶段使用自适应衰减
            return ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=3)


    # 多阶段训练
    for phase_idx, (phase_name, total_epochs) in enumerate(phases):
        logger.info(f"\n{'=' * 40}\nStarting {phase_name.upper()} phase ({total_epochs} epochs)\n{'=' * 40}")

        # 动态激活参数组
        for g in optim_groups:
            if g['name'] == 'seg_head':
                g['active'] = (phase_name == 'full')
        optimizer = create_optimizer(optim_groups)
        scheduler = create_scheduler(optimizer, phase_name, total_epochs)


        # 学习率阶段调整
        if phase_name == 'full':
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-3

        # 混合精度训练
        # scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')

        # 阶段训练循环
        for epoch in range(1, total_epochs + 1):
            model.train()
            epoch_loss = 0.0
            iou_tracker = {'rln': [], "sen": [], "pre": [], "fnr": [], "fpr": [], "dice": [], "ap": [], "al": []}
            loss_value_train = []

            # 初始化性能统计
            batch_time = AverageMeter()
            data_time = AverageMeter()
            end_time = time.time()
            # 创建训练进度条
            train_bar = tqdm(enumerate(train_loader), total=len(train_loader),
                             desc=f'Epoch {epoch}/{total_epochs} ({phase_name})',
                             bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}',
                             dynamic_ncols=True)

            for batch_idx, (img, seg_gt, edge_gt, shape_gt, max_vals, min_vals) in train_bar:
                # 数据加载时间统计
                data_time.update(time.time() - end_time)
                try:
                    # 数据加载
                    img = img.to(device, non_blocking=True)
                    seg_gt = seg_gt.to(device, non_blocking=True).long().squeeze(1)
                    edge_gt = edge_gt.to(device, non_blocking=True).float()
                    shape_gt = shape_gt.to(device, non_blocking=True).float()


                    seg_pred, recon_pred, shape_pred, al = model(img)
                    # seg_pred, recon_pred, al = model(img)
                    loss = criterion(seg_pred, seg_gt, edge_gt, shape_pred, shape_gt, recon_pred, img)
                    # loss = criterion(seg_pred, seg_gt, edge_gt, shape_gt, recon_pred, img)
                    loss_value_train.append(loss.item())
                    # 梯度管理
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)


                    # # 对需要batch级更新的调度器立即执行
                    # if isinstance(scheduler, CosineAnnealingLR):
                    #     scheduler.step()  # 在optimizer.step()之后调用
                    #     scheduler_updated = True


                    # 指标计算
                    with torch.no_grad():
                        rln_iou, sen, pre, fnr, fpr, dice, ap = compute_class_iou(seg_pred.detach(), seg_gt, class_idx=1)
                        iou_tracker['rln'].append(rln_iou.item())
                        iou_tracker['sen'].append(sen.item())
                        iou_tracker['pre'].append(pre.item())
                        iou_tracker['fnr'].append(fnr.item())
                        iou_tracker['fpr'].append(fpr.item())
                        iou_tracker['dice'].append(dice.item())
                        iou_tracker['ap'].append(ap.item())
                    iou_tracker['al'].append(al.item())

                    epoch_loss += loss.item()

                    batch_time.update(time.time() - end_time)
                    end_time = time.time()
                    # 更新进度条显示
                    train_bar.set_postfix({
                        'Loss': f"{loss.item():.4f}",
                        'LR': f"{scheduler.optimizer.param_groups[0]['lr']:.2e}",
                        "rln_iou": f"{rln_iou.item():.4f}",
                        'DataTime': f"{data_time.avg:.3f}s",
                        'BatchTime': f"{batch_time.avg:.3f}s",
                        'GPU Mem': f"{torch.cuda.memory_allocated() / 1e9:.2f}G"
                    })
                    # 内存监控
                    if device.type == 'cuda' and batch_idx % 10 == 0:
                        mem_used = torch.cuda.memory_allocated() / 1024 ** 3
                        logger.debug(f"Batch {batch_idx}: GPU内存使用 {mem_used:.2f}GB")

                except RuntimeError as e:
                    if 'CUDA out of memory' in str(e):
                        logger.error(f"批次 {batch_idx} 显存溢出! 建议:")
                        logger.error("1. 减小batch_size  2. 降低分辨率  3. 简化模型")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise
            train_bar.close()

            # 验证阶段
            val_metrics = validate(model, val_loader, criterion, device)
            history['train_loss'].append(epoch_loss / len(train_loader))
            history['val_loss'].append(val_metrics['loss'])

            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch,
                    phase_name,
                    # 训练指标
                    loss_value_train,
                    iou_tracker['rln'],
                    iou_tracker['sen'],
                    iou_tracker['pre'],
                    iou_tracker['fnr'],
                    iou_tracker['fpr'],
                    iou_tracker['dice'],
                    iou_tracker['ap'],
                    # 验证指标
                    val_metrics['loss_value'],
                    val_metrics['rln'],
                    val_metrics['sen'],
                    val_metrics['pre'],
                    val_metrics['fnr'],
                    val_metrics['fpr'],
                    val_metrics['dice'],
                    val_metrics['ap'],
                    iou_tracker['al'],
                ])


            criterion.update_weights(val_metrics['rln_iou'], phase_name, epoch, total_epochs)
            # 模型保存逻辑
            if val_metrics['rln_iou'] > best_val_iou:
                best_val_iou = val_metrics['rln_iou']
                torch.save({
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                    'epoch': epoch
                }, f"best_model.pth")
                logger.info(f"模型已保存: 验证IoU {best_val_iou:.4f}")

            # 在epoch循环末尾添加调度器更新
            if isinstance(scheduler, CosineAnnealingLR):
                scheduler.step()  # 每个 epoch 更新一次
            elif isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_metrics['loss'])  # 使用验证损失更新
            # if not scheduler_updated:  # 避免重复更新
            #     if isinstance(scheduler, ReduceLROnPlateau):
            #         scheduler.step(val_metrics['loss'])  # 使用验证指标
            #     else:
            #         scheduler.step()

            # 训练日志
            logger.info(
                f"Epoch {epoch}/{total_epochs} | "
                f"Train Loss: {history['train_loss'][-1]:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Train_RLN IoU: {np.nanmean(iou_tracker['rln']):.4f} |"
                f"Val_RLN IoU: {val_metrics['rln_iou']:.4f} |"
                f"lr: {scheduler.optimizer.param_groups[0]['lr']:.2e}"
            )

        # 阶段结束保存优化器状态
        optimizer_state = optimizer.state_dict()
        logger.info(f"阶段状态保存完成: {len(optimizer_state)} 优化器参数已存储")

    # 最终测试
    logger.info("\n正在进行最终测试...")
    test_metrics = validate(model, test_loader, criterion, device, is_test=True)
    history['test_metrics'] = test_metrics
    logger.info(
        f"测试结果: RLN IoU {test_metrics['rln_iou']:.4f} | "
    )

    return history


def validate(model, loader, criterion, device, is_test=False):
    """验证/测试函数"""
    model.eval()
    total_loss = 0.0
    iou_tracker = {'rln': [],"sen": [], "pre": [], "fnr": [], "fpr": [], "dice": [], "ap": []}

    loss_value = []
    val_bar = tqdm(loader, desc='Testing' if is_test else 'Validating',
                   bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}',
                   dynamic_ncols=True)

    with torch.no_grad():
        for batch_idx, (img, seg_gt, edge_gt, shape_gt, max_vals, min_vals) in enumerate(val_bar):
            try:
                img = img.to(device, non_blocking=True)
                seg_gt = seg_gt.to(device, non_blocking=True).long().squeeze(1)
                edge_gt = edge_gt.to(device, non_blocking=True).float()

                shape_gt = shape_gt.to(device, non_blocking=True).float()

                # seg_pred, recon_pred, al = model(img)
                # loss = criterion(seg_pred, seg_gt, edge_gt, shape_gt, recon_pred, img)
                seg_pred, recon_pred, shape_pred, al = model(img)
                loss = criterion(seg_pred, seg_gt, edge_gt, shape_pred, shape_gt, recon_pred, img)


                loss_value.append(loss.item())
                rln_iou, sen, pre, fnr, fpr, dice, ap = compute_class_iou(seg_pred, seg_gt, class_idx=1)
                # =================================可视化部分=====================
                # 测试模式下的可视化
                if is_test and batch_idx == 0:
                    debug_dir = "test_reports"
                    # shape_pred = torch.tensor([[1,1,1,1,1],[1,1,1,1,1]])

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
        'loss_value': loss_value,

        'rln': iou_tracker['rln'],
        'sen': iou_tracker['sen'],
        'pre': iou_tracker['pre'],
        'fnr': iou_tracker['fnr'],
        'fpr': iou_tracker['fpr'],
        'dice': iou_tracker['dice'],
        'ap': iou_tracker['ap']
    }

    if is_test:
        logger.info("\n=== 最终测试结果 ===")
        logger.info(f"平均损失: {metrics['loss']:.4f}")
        logger.info(f"喉返神经IoU: {metrics['rln_iou']:.4f}")

    return metrics