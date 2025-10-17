import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import List, Dict


def process_training_logss(input_path: str, output_path: str) -> None:
    """处理训练日志并生成可视化图表及统计表格

    Args:
        input_path (str): 输入CSV文件路径
        output_path (str): 输出图片路径（推荐.png格式）
    """
    # 内部配置参数
    CONFIG = {
        "sep": ",",
        "engine": "python",
        "plot_style": {
            "figsize": (18, 18),
            "train_marker": "o",
            "val_marker": "x",
            "grid_alpha": 0.3
        }
    }

    def safe_literal_eval(s: str) -> List[float]:
        """安全解析列表字符串"""
        try:
            s = str(s).strip()
            s = s.replace("'", '"').replace("nan", "null")
            return json.loads(s)
        except (json.JSONDecodeError, TypeError) as e:
            print(f"解析失败: {str(e)} | 原始内容: {s}")
            return []

    def preprocess_data(raw_df: pd.DataFrame) -> pd.DataFrame:
        """数据预处理流水线"""
        df = raw_df.copy()
        # 扩展包含所有指标列
        list_columns = [
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
        ]

        for col in list_columns:
            df[col] = df[col].apply(safe_literal_eval)
            invalid_count = df[col].apply(lambda x: not isinstance(x, list)).sum()
            if invalid_count > 0:
                print(f"警告: 列 {col} 中存在 {invalid_count} 条无效数据")

        expanded_data = []
        for _, row in df.iterrows():
            max_len = max(len(lst) for lst in row[list_columns])
            for i in range(max_len):
                record = {
                    "epoch": row["epoch"],
                    "phase": row["phase"],
                    **{
                        col: row[col][i] if i < len(row[col]) else np.nan
                        for col in list_columns
                    }
                }
                expanded_data.append(record)
        return pd.DataFrame(expanded_data)

    def analyze_data(df: pd.DataFrame) -> pd.DataFrame:
        """数据分析处理"""
        grouped = df.groupby(['phase', 'epoch']).agg({
            'train_loss': np.nanmean,
            'train_iou_rln': np.nanmean,
            'train_iou_sen': np.nanmean,
            'train_iou_pre': np.nanmean,
            'train_iou_fnr': np.nanmean,
            'train_iou_fpr': np.nanmean,
            'train_iou_dice': np.nanmean,
            'train_iou_ap': np.nanmean,
            'val_loss': np.nanmean,
            'val_iou_rln': np.nanmean,
            'val_iou_sen': np.nanmean,
            'val_iou_pre': np.nanmean,
            'val_iou_fnr': np.nanmean,
            'val_iou_fpr': np.nanmean,
            'val_iou_dice': np.nanmean,
            'val_iou_ap': np.nanmean,
        }).reset_index()
        return grouped

    def calculate_statistics(analyzed_df: pd.DataFrame) -> pd.DataFrame:
        """按指标分组的统计计算"""
        metrics_columns = [col for col in analyzed_df.columns if col not in ['phase', 'epoch']]

        # 定义统计计算函数
        def calculate_stats(group):
            return pd.Series({
                'Mean': np.nanmean(group),
                'Median': np.nanmedian(group),
                'Variance': np.nanvar(group, ddof=0),
                'Max': np.nanmax(group),
                'Min': np.nanmin(group)
            })

        # 按阶段和指标展开
        stats_data = []
        for metric in metrics_columns:
            for phase, group in analyzed_df.groupby('phase'):
                metric_values = group[metric]
                stats = calculate_stats(metric_values)
                stats_data.append({
                    'Metric': metric.replace('_', ' ').title().replace('Iou', 'IOU'),
                    'Phase': phase.capitalize(),
                    **stats
                })

        stats_df = pd.DataFrame(stats_data)

        # 按指标分组并排序
        metric_order = [
            'Train Loss', 'Val Loss',
            'Train IOU Rln', 'Val IOU Rln',
            'Train IOU Sen', 'Val IOU Sen',
            'Train IOU Pre', 'Val IOU Pre',
            'Train IOU Fnr', 'Val IOU Fnr',
            'Train IOU Fpr', 'Val IOU Fpr',
            'Train IOU Dice', 'Val IOU Dice',
            'Train IOU AP', 'Val IOU AP'
        ]

        # 创建排序键
        stats_df['Sort Key'] = stats_df['Metric'] + ' ' + stats_df['Phase']
        stats_df['Sort Key'] = pd.Categorical(
            stats_df['Sort Key'],
            categories=metric_order,
            ordered=True
        )
        return stats_df.sort_values('Sort Key').drop('Sort Key', axis=1).reset_index(drop=True)

    def generate_visualization(df: pd.DataFrame, output_path: str) -> None:
        """生成可视化图表"""
        plt.style.use('seaborn-v0_8')
        metrics = [
            'loss',
            'iou_rln',
            'iou_sen',
            'iou_pre',
            'iou_fnr',
            'iou_fpr',
            'iou_dice',
            'iou_ap',
        ]
        metric_labels = {
            'loss': 'Loss',
            'iou_rln': 'IOU RLN',
            'iou_sen': 'Sensitivity',
            'iou_pre': 'Precision',
            'iou_fnr': 'False Negative Rate',
            'iou_fpr': 'False Positive Rate',
            'iou_dice': 'Dice Coefficient',
            'iou_ap': 'Average Precision'
        }
        phases = sorted(df['phase'].unique())
        n_metrics = len(metrics)
        n_phases = len(phases)

        color_map = {
            'train': '#D62728',
            'val': '#2CA02C'
        }

        fig, axes = plt.subplots(
            n_metrics,
            n_phases,
            figsize=(6 * n_phases, 4 * n_metrics),
            squeeze=False
        )
        plt.subplots_adjust(hspace=0.5, wspace=0.3)

        for metric_idx, metric in enumerate(metrics):
            for phase_idx, phase in enumerate(phases):
                ax = axes[metric_idx, phase_idx]
                phase_data = df[df['phase'] == phase]

                ax.plot(
                    phase_data['epoch'],
                    phase_data[f'train_{metric}'],
                    label='Train',
                    color=color_map['train'],
                    linewidth=1,
                    alpha=0.8
                )
                ax.plot(
                    phase_data['epoch'],
                    phase_data[f'val_{metric}'],
                    label='Val',
                    color=color_map['val'],
                    linewidth=1,
                    alpha=0.8
                )

                ax.set_title(
                    f"{phase.upper()} Phase - {metric_labels[metric]}",
                    fontsize=12
                )
                ax.set_xlabel("Epoch", fontsize=10)
                ax.set_ylabel(metric_labels[metric], fontsize=10)
                ax.legend(fontsize=8, frameon=False)
                ax.grid(alpha=CONFIG["plot_style"]["grid_alpha"])
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

        plt.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    def generate_statistics_table(stats_df: pd.DataFrame, output_path: str) -> None:
        """生成紧凑型统计表格"""
        plt.figure(figsize=(14, 4 + len(stats_df) * 0.3))
        ax = plt.gca()
        ax.axis('off')

        # 优化列显示格式
        formatted_df = stats_df.copy()
        for col in ['Mean', 'Median', 'Variance', 'Max', 'Min']:
            formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.4f}")

        # 创建颜色映射
        colors = [
            ['#F5F5F5' if i % 2 == 0 else 'white' for _ in range(len(formatted_df.columns))]
            for i in range(len(formatted_df))
        ]

        # 创建带颜色映射的表格
        table = ax.table(
            cellText=formatted_df.values,
            colLabels=formatted_df.columns.map(lambda x: x.replace('_', ' ').title()),
            cellColours=colors,
            cellLoc='center',
            loc='center'
        )

        # 优化表格样式
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.3)

        # 设置标题
        plt.title("Training Statistics by Metric and Phase", fontsize=11, pad=20, weight='bold')

        # 保存输出
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    # 主流程
    try:
        raw_df = pd.read_csv(input_path, sep=CONFIG["sep"], engine=CONFIG["engine"])
        processed_df = preprocess_data(raw_df)
        analyzed_df = analyze_data(processed_df)

        generate_visualization(analyzed_df, output_path)
        print(f"可视化图表已保存至: {output_path}")

        # 计算统计量并生成表格
        stats_df = calculate_statistics(analyzed_df)
        table_output_path = output_path.replace('.png', '_stats.png')
        generate_statistics_table(stats_df, table_output_path)
        print(f"统计表格已保存至: {table_output_path}")

    except Exception as e:
        print(f"处理失败: {str(e)}")
        raise


# 使用示例
if __name__ == "__main__":
    input_path = r"/media/a/Data/liuhongtao/2025/segmentationProject/continuous_training/training_logs_20250611_103628.csv"
    output_path = r"/media/a/Data/liuhongtao/2025/segmentationProject/continuous_training_plots/training_logs_20250611_103628.png"

    process_training_logss(
        input_path=input_path,
        output_path=output_path
    )
