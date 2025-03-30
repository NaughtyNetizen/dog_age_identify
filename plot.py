import matplotlib.pyplot as plt
import re
import pandas as pd

def parse_log_file(log_file_path):
    """
    逐行解析训练日志文件并提取指标。

    Args:
        log_file_path (str): 日志文件路径。

    Returns:
        pandas.DataFrame: 包含训练指标的 DataFrame。
    """
    epochs = []
    train_losses = []
    val_losses = []
    maes = []
    rmses = []
    acc_plus5s = []
    acc_plus3s = []
    acc_plus1s = []
    kappas = []
    scores = []

    current_epoch = None  # 用于记录当前 Epoch 编号

    with open(log_file_path, 'r') as f:
        for line in f:
            # 1. 匹配 Epoch 行 (例如: Epoch 1/150)
            epoch_match = re.match(r'Epoch (\d+)/(\d+)', line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1)) # 记录当前 Epoch 编号
                continue # 继续读取下一行

            # 2. 匹配 Loss 行 (例如: Train Loss: 48.1279 | Val Loss: 37.4267)
            loss_match = re.match(r'Train Loss: ([\d.]+) \| Val Loss: ([\d.]+)', line)
            if loss_match:
                train_loss = float(loss_match.group(1))
                val_loss = float(loss_match.group(2))
                continue # 继续读取下一行

            # 3. 匹配 Metrics 行 (例如: MAE: ... | ... | Score: ...)
            metrics_match = re.match(r'MAE: ([\d.]+) \| RMSE: ([\d.]+) \| Accuracy \(±5\): ([\d.]+)% \| Accuracy \(±3\): ([\d.]+)% \| Accuracy \(±1\): ([\d.]+)% \| Kappa: ([\d.]+) \| Score: ([\d.]+)', line)
            if metrics_match:
                if current_epoch is not None: # 确保已经读取了 Epoch 行
                    epochs.append(current_epoch)
                    train_losses.append(train_loss) # 使用之前提取的 train_loss
                    val_losses.append(val_loss)     # 使用之前提取的 val_loss
                    maes.append(float(metrics_match.group(1)))
                    rmses.append(float(metrics_match.group(2)))
                    acc_plus5s.append(float(metrics_match.group(3)))
                    acc_plus3s.append(float(metrics_match.group(4)))
                    acc_plus1s.append(float(metrics_match.group(5)))
                    kappas.append(float(metrics_match.group(6)))
                    scores.append(float(metrics_match.group(7)))
                    current_epoch = None # 重置 current_epoch，等待下一个 Epoch 开始

    df = pd.DataFrame({
        'Epoch': epochs,
        'Train Loss': train_losses,
        'Val Loss': val_losses,
        'MAE': maes,
        'RMSE': rmses,
        'Accuracy (±5)': acc_plus5s,
        'Accuracy (±3)': acc_plus3s,
        'Accuracy (±1)': acc_plus1s,
        'Kappa': kappas,
        'Score': scores
    })
    return df

def plot_training_metrics(df, save_prefix="training_metrics"):
    """
    生成并保存训练指标的图表。

    Args:
        df (pandas.DataFrame): 包含训练指标的 DataFrame。
        save_prefix (str, optional): 保存图表文件的前缀。默认为 "training_metrics"。
    """
    fig_dpi = 300 # 设置 dpi 以提高分辨率

    # 1. 损失曲线
    plt.figure(figsize=(8, 6), dpi=fig_dpi)
    plt.plot(df['Epoch'], df['Train Loss'], label='Train Loss', linewidth=1)
    plt.plot(df['Epoch'], df['Val Loss'], label='Validation Loss', linewidth=1)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_loss.pdf')
    plt.close()

    # 2. MAE 和 RMSE
    plt.figure(figsize=(8, 6), dpi=fig_dpi)
    plt.plot(df['Epoch'], df['MAE'], label='MAE', linewidth=1)
    plt.plot(df['Epoch'], df['RMSE'], label='RMSE', linewidth=1)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Error Value', fontsize=12)
    plt.title('MAE and RMSE', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_mae_rmse.pdf')
    plt.close()

    # 3. 准确率指标
    plt.figure(figsize=(8, 6), dpi=fig_dpi)
    plt.plot(df['Epoch'], df['Accuracy (±5)'], label='Accuracy (±5)', linewidth=1)
    plt.plot(df['Epoch'], df['Accuracy (±3)'], label='Accuracy (±3)', linewidth=1)
    plt.plot(df['Epoch'], df['Accuracy (±1)'], label='Accuracy (±1)', linewidth=1)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Accuracy Metrics', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_accuracy.pdf')
    plt.close()

    # 4. Kappa 和 Score
    plt.figure(figsize=(8, 6), dpi=fig_dpi)
    plt.plot(df['Epoch'], df['Kappa'], label='Kappa', linewidth=1)
    plt.plot(df['Epoch'], df['Score'], label='Score', linewidth=1)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title('Kappa and Score', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_kappa_score.pdf')
    plt.close()

if __name__ == "__main__":
    log_file = 'training152.log'  # 替换为您的日志文件名（如果不同）
    training_data = parse_log_file(log_file)

    # --- 再次添加这些打印语句来检查 DataFrame ---
    print("DataFrame 信息:")
    training_data.info()
    print("\nDataFrame 前几行数据:")
    print(training_data.head())
    print("\nDataFrame 是否包含 NaN 值:")
    print(training_data.isnull().sum())
    # --- 结束打印语句 ---

    plot_training_metrics(training_data, save_prefix="training_plot")
    print(f"图表已保存为 training_plot_loss.pdf, training_plot_mae_rmse.pdf, training_plot_accuracy.pdf, training_plot_kappa_score.pdf")
    print("请确保您已安装 'science' 风格的 matplotlib。 如果未安装，请使用: pip install SciencePlots")
    print("或者您可以注释掉 `plt.style.use('science')` 行以使用默认的 matplotlib 风格。")