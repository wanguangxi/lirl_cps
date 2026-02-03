# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
import glob

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 噪声等级
noise_levels = ['0', '0.1', '0.3', '0.5']
base_dir = os.path.dirname(os.path.abspath(__file__))

# 颜色设置
colors = {'0': 'blue', '0.1': 'green', '0.3': 'orange', '0.5': 'red'}

plt.figure(figsize=(10, 6))

for noise in noise_levels:
    # 查找对应噪声等级的文件夹
    pattern = os.path.join(base_dir, f'ddpg_lirl_pi_multi_run_*_noise_{noise}')
    folders = glob.glob(pattern)
    
    if not folders:
        print(f"未找到噪声等级 {noise} 的文件夹，跳过。")
        continue
    
    folder_path = folders[0]  # 取第一个匹配的文件夹
    print(f"处理噪声等级 {noise}: {folder_path}")
    
    # 查找所有的 .npy 文件
    npy_files = glob.glob(os.path.join(folder_path, 'ddpg_lirl_pi_all_scores_*.npy'))
    
    if not npy_files:
        print(f"{folder_path} 中没有找到 .npy 文件，跳过。")
        continue
    
    # 读取所有 .npy 文件的数据
    all_runs = []
    for npy_file in npy_files:
        scores = np.load(npy_file)
        print(f"  读取文件: {os.path.basename(npy_file)}, 形状: {scores.shape}")
        # 确保数据是一维的
        if scores.ndim > 1:
            scores = scores.flatten()
        all_runs.append(scores)
    
    # 找到最短的运行长度（防止不同运行的长度不一致）
    min_length = min(len(run) for run in all_runs)
    
    # 截断所有运行到相同长度
    all_runs_truncated = [run[:min_length] for run in all_runs]
    all_runs = np.array(all_runs_truncated)
    
    print(f"  所有运行的形状: {all_runs.shape}")
    
    # 计算平均值和标准差
    mean_scores = np.mean(all_runs, axis=0)
    std_scores = np.std(all_runs, axis=0)
    
    # 绘制均值曲线
    episodes = np.arange(len(mean_scores))
    plt.plot(episodes, mean_scores, label=f'Noise Level {noise}', color=colors[noise], linewidth=2)
    
    # 绘制标准差阴影
    plt.fill_between(episodes, 
                     mean_scores - std_scores, 
                     mean_scores + std_scores, 
                     alpha=0.2, 
                     color=colors[noise])

plt.xlabel('Episode')
plt.ylabel('Score')
plt.title('不同噪声等级下的训练曲线对比')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# 保存图片
output_path = os.path.join(base_dir, 'training_curves_comparison.png')
plt.savefig(output_path, dpi=300)
plt.show()

print(f"训练曲线已保存至: {output_path}")
