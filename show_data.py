import time
from matplotlib.patches import Ellipse, Patch
import matplotlib

matplotlib.use('Agg')
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams


def generate_comprehensive_report(img, seg_gt, shape_gt, seg_pred, recon_pred, shape_pred, debug_dir, max_vals,
                                  min_vals):
    """Generate comprehensive report with only the composite view"""
    batch_size = img.shape[0]
    for batch_idx in range(batch_size):
        plt.close('all')
        # 设置更专业的matplotlib参数
        rcParams['font.family'] = 'DejaVu Sans'  # 使用更清晰的字体
        rcParams['font.size'] = 12
        rcParams['axes.linewidth'] = 1.5
        rcParams['lines.linewidth'] = 2.0
        # Create single figure for composite view
        fig = plt.figure(figsize=(12, 10), dpi=300)
        ax = fig.add_subplot(1, 1, 1)  # Composite View only


        # 定义高对比度颜色方案，避免红色
        colors = {
            'gt_mask': [0.0, 0.4, 0.8, 0.6],    # 蓝色半透明，增加不透明度
            'pred_mask': [0.9, 0.7, 0.0, 0.6],   # 黄色半透明，增加不透明度
            'gt_ellipse': [0.0, 0.2, 0.9],       # 深蓝色
            'pred_ellipse': [0.9, 0.6, 0.0],     # 深黄色
            'gt_edge': [0.0, 0.5, 1.0],         # 亮蓝色
            'pred_edge': [1.0, 0.8, 0.0]         # 亮黄色
        }

        # Composite View
        def plot_composite(ax):
            # Original image (denormalized)
            current_max = max_vals[batch_idx].detach().cpu().numpy()
            current_min = min_vals[batch_idx].detach().cpu().numpy()

            img_tensor = img[batch_idx].detach().cpu()
            denormalized_img = img_tensor * (current_max - current_min) + current_min
            img_np = denormalized_img.permute(1, 2, 0).numpy()
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)

            ax.imshow(img_np, alpha=0.9, interpolation='lanczos')

            # Ground truth overlay
            mask_gt = seg_gt[batch_idx].detach().cpu().numpy()
            overlay_gt = np.zeros((256, 256, 4))
            overlay_gt[mask_gt == 1] = colors['gt_mask']
            ax.imshow(overlay_gt, interpolation='nearest')

            # Predicted overlay
            mask_pred = seg_pred[batch_idx].detach().cpu().argmax(0).numpy()
            overlay_pred = np.zeros((256, 256, 4))
            overlay_pred[mask_pred == 1] = colors['pred_mask']
            ax.imshow(overlay_pred, interpolation='nearest')


            def draw_gt_ellipse(params, color):
                if np.all(params[:4] < 1e-3):
                    return
                cx, cy = params[0], params[1]
                major, minor = params[2] * 2, params[3] * 2
                angle = params[4]

                # 确保坐标在合理范围内
                cx = np.clip(cx, 10, 246)
                cy = np.clip(cy, 10, 246)

                ellipse = Ellipse((cx, cy), major, minor, angle=angle,
                                  fill=False, ec=color, linestyle='--', linewidth=2.5, alpha=0.9)
                ax.add_patch(ellipse)

            # def draw_pred_ellipse(params, color):
            #     if np.all(params[:4] < 1e-3):
            #         return
            #     cx, cy = params[0], params[1]
            #     major, minor = params[2] * 2, params[3] * 2
            #     angle = params[4]
            #
            #     # 确保坐标在合理范围内
            #     cx = np.clip(cx, 10, 246)
            #     cy = np.clip(cy, 10, 246)
            #
            #     ellipse = Ellipse((cx, cy), major, minor, angle=angle,
            #                       fill=False, ec=color, linestyle='--', linewidth=2.5, alpha=0.9)
            #     ax.add_patch(ellipse)

            # Draw ellipses
            draw_gt_ellipse(shape_pred[batch_idx].detach().cpu().numpy(), colors['pred_ellipse'])
            # draw_pred_ellipse(shape_gt[batch_idx].detach().cpu().numpy(), colors['gt_ellipse'])
         
            ax.axis('off')
            # Legend
            legend_elements = [
                Patch(facecolor=colors['gt_mask'], edgecolor=colors['gt_edge'], label='GT PTGs'),
                Patch(facecolor=colors['pred_mask'], edgecolor=colors['pred_edge'], label='Pred PTGs'),
                Patch(facecolor='none', edgecolor=colors['pred_ellipse'], linestyle='--', linewidth=2.5, label='Pred Ellipse'),

            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=12,
                      framealpha=0.9, fancybox=True, shadow=True,
                      handlelength=2.0,  # 增加标记长度
                      handleheight=2.0,  # 增加标记高度
                      markerscale=2.0)  # 增加标记比例

        plot_composite(ax)

        # Save report
        plt.tight_layout()
        os.makedirs(debug_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        save_path = os.path.join(debug_dir, f'Composite_Report_{timestamp}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=300, facecolor='white')
        plt.close()
        print(f"Composite report saved to: {save_path}")