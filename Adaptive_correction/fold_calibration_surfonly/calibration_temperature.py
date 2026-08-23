#!/usr/bin/env python3
"""
Temperature Calibration for Binary Classification

温度缩放（Temperature Scaling）是一种后处理校准方法，用于调整模型预测的置信度。

原理：
- 原始模型输出 logit，通过 sigmoid(logit) 得到预测概率
- 引入温度参数 T，使用 sigmoid(logit/T) 来调整预测的置信度
- T > 1: 使预测更平滑（降低过度自信）
- T < 1: 使预测更尖锐（提高置信度）
- T = 1: 不做调整

优化目标：
    BCE = -1/N * Σ[y_i * log(p_i) + (1-y_i) * log(1-p_i)]
    其中 p_i = sigmoid(logit_i / T) = 1 / (1 + exp(-logit_i/T))

优化方法：
- LBFGS: 二阶优化方法，适合小参数量问题，收敛快

评估指标：
- ECE (Expected Calibration Error): 期望校准误差
- MCE (Maximum Calibration Error): 最大校准误差
- Reliability Diagram: 可靠性图，展示预测置信度与实际准确率的关系
"""

import os
import glob
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score


# 设置日志
def setup_logging(output_dir):
    """设置日志系统"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"calibration_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file


def load_and_merge_predictions(data_dir):
    """
    读取所有fold的预测结果CSV文件并合并

    Args:
        data_dir: 包含CSV文件的目录

    Returns:
        merged_df: 合并后的DataFrame，包含 id, id_tcr, label, logit, prediction
    """
    logging.info(f"正在从 {data_dir} 读取CSV文件...")

    # 查找所有fold的CSV文件
    csv_files = sorted(glob.glob(os.path.join(data_dir, "fold_*_val_predictions.csv")))

    if not csv_files:
        raise FileNotFoundError(f"在 {data_dir} 中未找到任何 fold_*_val_predictions.csv 文件")

    logging.info(f"找到 {len(csv_files)} 个CSV文件:")
    for f in csv_files:
        logging.info(f"  - {os.path.basename(f)}")

    # 读取并合并所有CSV文件
    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        fold_num = os.path.basename(csv_file).split('_')[1]
        df['fold'] = int(fold_num)
        dfs.append(df)

    merged_df = pd.concat(dfs, ignore_index=True)
    logging.info(f"合并完成，总样本数: {len(merged_df)}")

    # 提取需要的列
    required_cols = ['id', 'id_tcr', 'label', 'logit', 'prediction', 'fold']
    merged_df = merged_df[required_cols]

    # 检查数据完整性
    logging.info(f"数据统计:")
    logging.info(f"  - 正样本数: {(merged_df['label'] == 1).sum()}")
    logging.info(f"  - 负样本数: {(merged_df['label'] == 0).sum()}")
    logging.info(f"  - Logit范围: [{merged_df['logit'].min():.4f}, {merged_df['logit'].max():.4f}]")
    logging.info(f"  - Prediction范围: [{merged_df['prediction'].min():.4f}, {merged_df['prediction'].max():.4f}]")

    return merged_df


class TemperatureScaling(nn.Module):
    """
    温度缩放模型

    只有一个可学习参数：温度 T
    """
    def __init__(self, init_temperature=1.0):
        super(TemperatureScaling, self).__init__()
        # 使用log(T)作为参数，确保T始终为正
        self.log_temperature = nn.Parameter(torch.tensor(np.log(init_temperature)))

    def forward(self, logits):
        """
        前向传播：计算 sigmoid(logit / T)

        Args:
            logits: 原始logit值

        Returns:
            calibrated_probs: 校准后的概率
        """
        temperature = torch.exp(self.log_temperature)
        return torch.sigmoid(logits / temperature)

    def get_temperature(self):
        """获取当前温度值"""
        return torch.exp(self.log_temperature).item()


def compute_ece(probs, labels, n_bins=15):
    """
    计算 Expected Calibration Error (ECE)

    ECE = Σ (n_k/N) * |acc_k - conf_k|

    其中：
    - n_k: 第k个bin中的样本数
    - N: 总样本数
    - acc_k: 第k个bin中的实际准确率
    - conf_k: 第k个bin中的平均预测置信度

    Args:
        probs: 预测概率 (N,)
        labels: 真实标签 (N,)
        n_bins: bin的数量

    Returns:
        ece: Expected Calibration Error
        bin_data: 每个bin的统计信息（用于绘制reliability diagram）
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    bin_data = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # 找到落在当前bin中的样本
        in_bin = (probs > bin_lower) & (probs <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = labels[in_bin].mean()
            avg_confidence_in_bin = probs[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

            bin_data.append({
                'bin_lower': bin_lower,
                'bin_upper': bin_upper,
                'confidence': avg_confidence_in_bin,
                'accuracy': accuracy_in_bin,
                'count': in_bin.sum()
            })
        else:
            bin_data.append({
                'bin_lower': bin_lower,
                'bin_upper': bin_upper,
                'confidence': (bin_lower + bin_upper) / 2,
                'accuracy': 0,
                'count': 0
            })

    return ece, bin_data


def compute_mce(probs, labels, n_bins=15):
    """
    计算 Maximum Calibration Error (MCE)

    MCE = max_k |acc_k - conf_k|

    Args:
        probs: 预测概率
        labels: 真实标签
        n_bins: bin的数量

    Returns:
        mce: Maximum Calibration Error
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    mce = 0.0

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (probs > bin_lower) & (probs <= bin_upper)

        if in_bin.sum() > 0:
            accuracy_in_bin = labels[in_bin].mean()
            avg_confidence_in_bin = probs[in_bin].mean()
            mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))

    return mce


def plot_reliability_diagram(bin_data, title, output_path):
    """
    绘制 Reliability Diagram（可靠性图）

    展示预测置信度与实际准确率的关系
    完美校准的模型应该在对角线上

    Args:
        bin_data: 每个bin的统计信息
        title: 图表标题
        output_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # 提取数据
    confidences = [d['confidence'] for d in bin_data]
    accuracies = [d['accuracy'] for d in bin_data]
    counts = [d['count'] for d in bin_data]

    # 绘制柱状图（样本数量）
    ax2 = ax.twinx()
    ax2.bar(range(len(bin_data)), counts, alpha=0.3, color='gray', label='Sample Count')
    ax2.set_ylabel('Sample Count', fontsize=12)
    ax2.legend(loc='upper left')

    # 绘制可靠性曲线
    ax.plot(confidences, accuracies, 'o-', linewidth=2, markersize=8, label='Model')

    # 绘制完美校准线（对角线）
    ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=2, label='Perfect Calibration')

    ax.set_xlabel('Confidence (Predicted Probability)', fontsize=12)
    ax.set_ylabel('Accuracy (Actual Probability)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"Reliability diagram 已保存至: {output_path}")


def calibrate_temperature(logits, labels, init_temp=1.0, max_iter=100, lr=0.01):
    """
    使用LBFGS优化温度参数

    Args:
        logits: 原始logit值 (N,)
        labels: 真实标签 (N,)
        init_temp: 初始温度
        max_iter: 最大迭代次数
        lr: 学习率

    Returns:
        optimal_temp: 最优温度
        calibrated_probs: 校准后的概率
        loss_history: 损失历史
    """
    # 转换为PyTorch张量
    logits_tensor = torch.FloatTensor(logits)
    labels_tensor = torch.FloatTensor(labels)

    # 初始化模型
    model = TemperatureScaling(init_temperature=init_temp)

    # 定义损失函数
    criterion = nn.BCELoss()

    # 使用LBFGS优化器
    optimizer = optim.LBFGS([model.log_temperature], lr=lr, max_iter=max_iter)

    loss_history = []

    def closure():
        optimizer.zero_grad()
        probs = model(logits_tensor)
        loss = criterion(probs, labels_tensor)
        loss.backward()
        loss_history.append(loss.item())
        return loss

    logging.info("开始优化温度参数...")
    logging.info(f"初始温度: T = {init_temp:.4f}")

    # 优化
    optimizer.step(closure)

    # 获取最优温度
    optimal_temp = model.get_temperature()

    # 计算校准后的概率
    with torch.no_grad():
        calibrated_probs = model(logits_tensor).numpy()

    logging.info(f"优化完成！")
    logging.info(f"最优温度: T = {optimal_temp:.4f}")
    logging.info(f"最终BCE Loss: {loss_history[-1]:.6f}")
    logging.info(f"总迭代次数: {len(loss_history)}")

    return optimal_temp, calibrated_probs, loss_history


def plot_loss_curve(loss_history, output_path):
    """绘制损失曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, linewidth=2)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('BCE Loss', fontsize=12)
    plt.title('Temperature Calibration - Loss Curve', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"损失曲线已保存至: {output_path}")


def evaluate_calibration(probs, labels, name="Model"):
    """
    评估校准效果

    Args:
        probs: 预测概率
        labels: 真实标签
        name: 模型名称

    Returns:
        metrics: 评估指标字典
    """
    # 计算分类指标
    preds = (probs > 0.5).astype(int)
    acc = accuracy_score(labels, preds)
    auc = roc_auc_score(labels, probs)
    auprc = average_precision_score(labels, probs)

    # 计算校准指标
    ece = compute_ece(probs, labels)[0]
    mce = compute_mce(probs, labels)

    # 计算BCE loss
    epsilon = 1e-7  # 避免log(0)
    probs_clipped = np.clip(probs, epsilon, 1 - epsilon)
    bce = -np.mean(labels * np.log(probs_clipped) + (1 - labels) * np.log(1 - probs_clipped))

    metrics = {
        'name': name,
        'accuracy': acc,
        'auc': auc,
        'auprc': auprc,
        'bce_loss': bce,
        'ece': ece,
        'mce': mce
    }

    logging.info(f"\n{name} 评估结果:")
    logging.info(f"  Accuracy: {acc:.4f}")
    logging.info(f"  AUC: {auc:.4f}")
    logging.info(f"  AUPRC: {auprc:.4f}")
    logging.info(f"  BCE Loss: {bce:.6f}")
    logging.info(f"  ECE: {ece:.6f}")
    logging.info(f"  MCE: {mce:.6f}")

    return metrics


def main():
    """主函数"""
    # 设置路径
    data_dir = "./analysis_code/fold_calibration_surfonly"
    output_dir = "./analysis_code/fold_calibration_surfonly/calibration/temperature"

    # 设置日志
    log_file = setup_logging(output_dir)
    logging.info("=" * 80)
    logging.info("Temperature Calibration for Binary Classification")
    logging.info("=" * 80)

    # 1. 读取并合并数据
    logging.info("\n步骤 1: 读取并合并预测数据")
    merged_df = load_and_merge_predictions(data_dir)

    # 保存合并后的数据
    summary_path = os.path.join(output_dir, "prediction_summary.csv")
    merged_df.to_csv(summary_path, index=False)
    logging.info(f"合并数据已保存至: {summary_path}")

    # 2. 准备数据
    logits = merged_df['logit'].values
    labels = merged_df['label'].values
    original_probs = merged_df['prediction'].values

    # 3. 评估原始模型
    logging.info("\n步骤 2: 评估原始模型（未校准）")
    original_metrics = evaluate_calibration(original_probs, labels, name="Original Model")

    # 绘制原始模型的reliability diagram
    ece_original, bin_data_original = compute_ece(original_probs, labels)
    plot_reliability_diagram(
        bin_data_original,
        f"Original Model (ECE={ece_original:.4f})",
        os.path.join(output_dir, "reliability_diagram_original.png")
    )

    # 4. 温度校准
    logging.info("\n步骤 3: 温度校准")
    optimal_temp, calibrated_probs, loss_history = calibrate_temperature(
        logits, labels, init_temp=1.0, max_iter=100, lr=0.01
    )

    # 绘制损失曲线
    plot_loss_curve(loss_history, os.path.join(output_dir, "loss_curve.png"))

    # 5. 评估校准后的模型
    logging.info("\n步骤 4: 评估校准后的模型")
    calibrated_metrics = evaluate_calibration(calibrated_probs, labels, name="Calibrated Model")

    # 绘制校准后的reliability diagram
    ece_calibrated, bin_data_calibrated = compute_ece(calibrated_probs, labels)
    plot_reliability_diagram(
        bin_data_calibrated,
        f"Calibrated Model (T={optimal_temp:.4f}, ECE={ece_calibrated:.4f})",
        os.path.join(output_dir, "reliability_diagram_calibrated.png")
    )

    # 6. 保存校准结果
    logging.info("\n步骤 5: 保存校准结果")

    # 保存校准后的预测
    merged_df['calibrated_prediction'] = calibrated_probs
    merged_df['temperature'] = optimal_temp
    calibrated_path = os.path.join(output_dir, "calibrated_predictions.csv")
    merged_df.to_csv(calibrated_path, index=False)
    logging.info(f"校准后的预测已保存至: {calibrated_path}")

    # 保存评估指标
    metrics_df = pd.DataFrame([original_metrics, calibrated_metrics])
    metrics_path = os.path.join(output_dir, "calibration_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    logging.info(f"评估指标已保存至: {metrics_path}")

    # 保存温度参数
    temp_info = {
        'optimal_temperature': optimal_temp,
        'initial_temperature': 1.0,
        'final_bce_loss': loss_history[-1],
        'n_iterations': len(loss_history)
    }
    temp_df = pd.DataFrame([temp_info])
    temp_path = os.path.join(output_dir, "temperature_info.csv")
    temp_df.to_csv(temp_path, index=False)
    logging.info(f"温度参数信息已保存至: {temp_path}")

    # 7. 总结
    logging.info("\n" + "=" * 80)
    logging.info("校准完成！")
    logging.info("=" * 80)
    logging.info(f"\n最优温度: T = {optimal_temp:.4f}")
    logging.info(f"\n校准效果对比:")
    logging.info(f"  BCE Loss: {original_metrics['bce_loss']:.6f} → {calibrated_metrics['bce_loss']:.6f} "
                 f"(改善 {(original_metrics['bce_loss'] - calibrated_metrics['bce_loss']):.6f})")
    logging.info(f"  ECE: {original_metrics['ece']:.6f} → {calibrated_metrics['ece']:.6f} "
                 f"(改善 {(original_metrics['ece'] - calibrated_metrics['ece']):.6f})")
    logging.info(f"  MCE: {original_metrics['mce']:.6f} → {calibrated_metrics['mce']:.6f} "
                 f"(改善 {(original_metrics['mce'] - calibrated_metrics['mce']):.6f})")
    logging.info(f"\n所有结果已保存至: {output_dir}")
    logging.info(f"日志文件: {log_file}")


if __name__ == "__main__":
    main()
