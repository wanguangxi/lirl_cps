# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 噪声等级
noise_levels = ['0', '0.1', '0.3', '0.5']
base_dir = os.path.dirname(os.path.abspath(__file__))

# 存储所有数据
all_data = {
    'convergence_scores': {},
    'completion_times': {},
    'energy_consumption': {}
}

for noise in noise_levels:
    # 查找对应噪声等级的文件夹
    pattern = os.path.join(base_dir, f'ddpg_lirl_pi_multi_run_*_noise_{noise}')
    folders = glob.glob(pattern)
    
    if not folders:
        print(f"未找到噪声等级 {noise} 的文件夹，跳过。")
        continue
    
    folder_path = folders[0]
    print(f"处理噪声等级 {noise}: {folder_path}")
    
    # 查找所有的 CSV 文件
    csv_files = glob.glob(os.path.join(folder_path, 'ddpg_lirl_pi_all_episode_stats_*.csv'))
    
    if not csv_files:
        print(f"{folder_path} 中没有找到 CSV 文件，跳过。")
        continue
    
    # 查找对应的 npy 文件用于收敛性能
    npy_files = glob.glob(os.path.join(folder_path, 'ddpg_lirl_pi_all_scores_*.npy'))
    
    convergence_scores = []
    completion_times = []
    energy_consumption = []
    
    # 读取每个运行的收敛分数（最后100个episode的平均值）
    for npy_file in npy_files:
        scores = np.load(npy_file)
        if scores.ndim > 1:
            scores = scores.flatten()
        # 取最后100个episode的平均值作为收敛性能
        convergence_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
        convergence_scores.append(convergence_score)
    
    # 读取每个CSV文件的数据
    for i, csv_file in enumerate(csv_files):
        try:
            # 尝试不同的编码方式
            encodings = ['utf-8', 'gbk', 'gb2312', 'latin1', 'iso-8859-1']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(csv_file, encoding=encoding)
                    print(f"  成功读取 {os.path.basename(csv_file)} (编码: {encoding})")
                    break
                except:
                    continue
            
            if df is None:
                print(f"  无法读取 {csv_file}，尝试了所有编码方式")
                continue
                
            # 打印列名，帮助调试
            print(f"  CSV文件列名: {list(df.columns)}")
            
            # 尝试从对应的npy文件获取episode数据，确保数据对齐
            if i < len(npy_files):
                scores = np.load(npy_files[i])
                if scores.ndim > 1:
                    scores = scores.flatten()
                
                # 确保CSV和npy文件的数据长度一致
                min_len = min(len(df), len(scores))
                df = df.iloc[:min_len]
                
                # 从最后100个episode计算平均完工时间
                last_100_df = df.tail(100) if len(df) >= 100 else df
                
                # 查找完工时间相关的列
                time_columns = [ 'makespan', 'flow_time', 'Time']
                time_found = False
                for col in time_columns:
                    if col in df.columns:
                        avg_completion_time = last_100_df[col].mean()
                        if not np.isnan(avg_completion_time):
                            completion_times.append(avg_completion_time)
                            print(f"    找到时间列 '{col}': 平均值 = {avg_completion_time:.2f}")
                            time_found = True
                            break
                
                # 如果没有找到时间列，尝试从其他可能的列名中查找
                if not time_found:
                    for col in df.columns:
                        if 'time' in col.lower() or 'completion' in col.lower() or 'makespan' in col.lower():
                            avg_completion_time = last_100_df[col].mean()
                            if not np.isnan(avg_completion_time):
                                completion_times.append(avg_completion_time)
                                print(f"    找到时间列 '{col}': 平均值 = {avg_completion_time:.2f}")
                                time_found = True
                                break
                
                # 查找能耗相关的列
                energy_columns = ["total_energy"]
                energy_found = False
                for col in energy_columns:
                    if col in df.columns:
                        avg_energy = last_100_df[col].mean()
                        if not np.isnan(avg_energy):
                            energy_consumption.append(avg_energy)
                            print(f"    找到能耗列 '{col}': 平均值 = {avg_energy:.2f}")
                            energy_found = True
                            break
                
                # 如果没有找到能耗列，尝试从其他可能的列名中查找
                if not energy_found:
                    for col in df.columns:
                        if 'energy' in col.lower() or 'power' in col.lower():
                            avg_energy = last_100_df[col].mean()
                            if not np.isnan(avg_energy):
                                energy_consumption.append(avg_energy)
                                print(f"    找到能耗列 '{col}': 平均值 = {avg_energy:.2f}")
                                break
                
        except Exception as e:
            print(f"  读取 {csv_file} 时出错: {e}")
            import traceback
            traceback.print_exc()
    
    # 存储数据
    if convergence_scores:
        all_data['convergence_scores'][noise] = convergence_scores
    if completion_times:
        all_data['completion_times'][noise] = completion_times
    if energy_consumption:
        all_data['energy_consumption'][noise] = energy_consumption

# 检查是否有数据
has_data = False
for metric, data in all_data.items():
    if any(data.values()):
        has_data = True
        break

if not has_data:
    print("\n警告: 没有找到任何可用的数据！")
    print("请检查CSV文件是否存在，以及列名是否正确。")
    exit()

# 创建箱式图
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 1. 收敛性能箱式图
if all_data['convergence_scores'] and any(all_data['convergence_scores'].values()):
    ax1 = axes[0]
    data_to_plot = [all_data['convergence_scores'][noise] for noise in noise_levels if noise in all_data['convergence_scores'] and all_data['convergence_scores'][noise]]
    labels = [f'Noise {noise}' for noise in noise_levels if noise in all_data['convergence_scores'] and all_data['convergence_scores'][noise]]
    
    if data_to_plot:
        bp1 = ax1.boxplot(data_to_plot, labels=labels, patch_artist=True)
        for patch, noise in zip(bp1['boxes'], [n for n in noise_levels if n in all_data['convergence_scores'] and all_data['convergence_scores'][n]]):
            patch.set_facecolor({'0': 'lightblue', '0.1': 'lightgreen', '0.3': 'lightyellow', '0.5': 'lightcoral'}[noise])
    
    ax1.set_ylabel('收敛分数')
    ax1.set_title('不同噪声等级下的收敛性能对比')
    ax1.grid(True, alpha=0.3)
else:
    axes[0].text(0.5, 0.5, '无收敛分数数据', ha='center', va='center', transform=axes[0].transAxes)

# 2. 平均完工时间箱式图
if all_data['completion_times'] and any(all_data['completion_times'].values()):
    ax2 = axes[1]
    data_to_plot = [all_data['completion_times'][noise] for noise in noise_levels if noise in all_data['completion_times'] and all_data['completion_times'][noise]]
    labels = [f'Noise {noise}' for noise in noise_levels if noise in all_data['completion_times'] and all_data['completion_times'][noise]]
    
    if data_to_plot:
        bp2 = ax2.boxplot(data_to_plot, labels=labels, patch_artist=True)
        for patch, noise in zip(bp2['boxes'], [n for n in noise_levels if n in all_data['completion_times'] and all_data['completion_times'][n]]):
            patch.set_facecolor({'0': 'lightblue', '0.1': 'lightgreen', '0.3': 'lightyellow', '0.5': 'lightcoral'}[noise])
    
    ax2.set_ylabel('平均完工时间')
    ax2.set_title('不同噪声等级下的平均完工时间对比')
    ax2.grid(True, alpha=0.3)
else:
    axes[1].text(0.5, 0.5, '无完工时间数据', ha='center', va='center', transform=axes[1].transAxes)

# 3. 平均能耗箱式图
if all_data['energy_consumption'] and any(all_data['energy_consumption'].values()):
    ax3 = axes[2]
    data_to_plot = [all_data['energy_consumption'][noise] for noise in noise_levels if noise in all_data['energy_consumption'] and all_data['energy_consumption'][noise]]
    labels = [f'Noise {noise}' for noise in noise_levels if noise in all_data['energy_consumption'] and all_data['energy_consumption'][noise]]
    
    if data_to_plot:
        bp3 = ax3.boxplot(data_to_plot, labels=labels, patch_artist=True)
        for patch, noise in zip(bp3['boxes'], [n for n in noise_levels if n in all_data['energy_consumption'] and all_data['energy_consumption'][n]]):
            patch.set_facecolor({'0': 'lightblue', '0.1': 'lightgreen', '0.3': 'lightyellow', '0.5': 'lightcoral'}[noise])
    
    ax3.set_ylabel('平均能耗')
    ax3.set_title('不同噪声等级下的平均能耗对比')
    ax3.grid(True, alpha=0.3)
else:
    axes[2].text(0.5, 0.5, '无能耗数据', ha='center', va='center', transform=axes[2].transAxes)

plt.tight_layout()

# 保存图片
output_path = os.path.join(base_dir, 'boxplot_comparison.png')
plt.savefig(output_path, dpi=300)
plt.show()

# 输出统计数据
print("\n统计结果:")
print("-" * 50)

for metric, data in all_data.items():
    if data and any(data.values()):
        print(f"\n{metric}:")
        for noise, values in data.items():
            if values:
                print(f"  Noise {noise}: 均值={np.mean(values):.2f}, 标准差={np.std(values):.2f}, 样本数={len(values)}")

print(f"\n箱式图已保存至: {output_path}")

def read_noise_level_stats(noise_level_dir):
    """
    读取 Noise_level 文件夹下所有子文件夹中的 episode stats CSV 文件
    """
    all_data = {}
    
    # 遍历所有噪声等级的子文件夹
    for noise_folder in os.listdir(noise_level_dir):
        folder_path = os.path.join(noise_level_dir, noise_folder)
        
        if os.path.isdir(folder_path) and 'noise_' in noise_folder:
            # 提取噪声等级
            noise_level = noise_folder.split('noise_')[-1]
            
            # 查找该文件夹下的所有 CSV 文件
            csv_pattern = os.path.join(folder_path, 'ddpg_lirl_pi_all_episode_stats_*.csv')
            csv_files = glob.glob(csv_pattern)
            
            if csv_files:
                all_data[noise_level] = []
                
                for csv_file in csv_files:
                    try:
                        df = pd.read_csv(csv_file)
                        all_data[noise_level].append(df)
                        print(f"读取 noise_{noise_level}: {os.path.basename(csv_file)}")
                    except Exception as e:
                        print(f"读取失败: {csv_file}, 错误: {e}")
    
    return all_data

# 使用示例
noise_level_dir = r"d:\1 工作\2 科研论文\2-LIRL\LIRL 最新实验数据\鲁棒性实验\Noise_level"
noise_data = read_noise_level_stats(noise_level_dir)