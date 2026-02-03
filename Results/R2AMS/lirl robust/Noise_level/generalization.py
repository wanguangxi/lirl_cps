import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置中文字体和图形样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def filter_last_n_data(df, noise_levels, n=100):
    """
    对每个噪声等级，只保留最后n个数据
    
    参数:
        df: 原始数据框
        noise_levels: 噪声等级列表
        n: 保留的数据数量
    
    返回:
        过滤后的数据框
    """
    filtered_data = []
    
    for noise_level in noise_levels:
        # 获取当前噪声等级的数据
        noise_data = df[df['parameter_value'] == noise_level]
        
        # 只取最后n个数据
        if len(noise_data) > n:
            noise_data = noise_data.tail(n)
        
        filtered_data.append(noise_data)
    
    # 合并所有过滤后的数据
    if filtered_data:
        return pd.concat(filtered_data, ignore_index=True)
    else:
        return pd.DataFrame()

def create_noise_level_boxplots():
    """
    创建基于噪声等级的箱式图对比
    """
    # 读取数据
    try:
        # 读取makespan数据
        makespan_df = pd.read_csv(r'D:\1 工作\2 科研论文\2-LIRL\LIRL 最新实验数据\鲁棒性实验\boxplot_makespan_plot_data.csv')
        # 读取energy数据
        energy_df = pd.read_csv(r'D:\1 工作\2 科研论文\2-LIRL\LIRL 最新实验数据\鲁棒性实验\boxplot_total_energy_plot_data.csv')
        print("数据读取成功！")
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return
    
    # 筛选指定的噪声等级
    noise_levels = [0.0, 0.1, 0.3, 0.5]
    
    # 对每个噪声等级只取最后100个数据
    makespan_filtered = filter_last_n_data(makespan_df, noise_levels, n=100)
    energy_filtered = filter_last_n_data(energy_df, noise_levels, n=100)
    
    # 添加noise_level列
    makespan_filtered['noise_level'] = makespan_filtered['parameter_value']
    energy_filtered['noise_level'] = energy_filtered['parameter_value']
    
    print(f"Makespan数据过滤后: {makespan_filtered.shape[0]}条")
    print(f"Energy数据过滤后: {energy_filtered.shape[0]}条")
    
    # 创建图形 - 两个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Makespan箱式图
    if not makespan_filtered.empty:
        # 确保噪声等级按顺序排列
        makespan_filtered['noise_level'] = pd.Categorical(
            makespan_filtered['noise_level'], 
            categories=noise_levels, 
            ordered=True
        )
        
        sns.boxplot(data=makespan_filtered, x='noise_level', y='makespan', 
                   ax=ax1, color='skyblue', width=0.6)
        
        # 添加均值点
        means = makespan_filtered.groupby('noise_level', observed=True)['makespan'].mean()
        for i, (noise, mean_val) in enumerate(means.items()):
            ax1.scatter(i, mean_val, color='red', s=100, marker='D', zorder=5, 
                       label='Mean' if i == 0 else '')
        
        ax1.set_title('Makespan Distribution by Noise Level (Last 100 samples)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Noise Level', fontsize=12)
        ax1.set_ylabel('Makespan', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # 添加统计信息
        for i, noise in enumerate(noise_levels):
            data = makespan_filtered[makespan_filtered['noise_level'] == noise]['makespan']
            if len(data) > 0:
                ax1.text(i, ax1.get_ylim()[1]*0.98, f'n={len(data)}', 
                        ha='center', va='top', fontsize=10)
    
    # 2. Total Energy箱式图
    if not energy_filtered.empty:
        # 确保噪声等级按顺序排列
        energy_filtered['noise_level'] = pd.Categorical(
            energy_filtered['noise_level'], 
            categories=noise_levels, 
            ordered=True
        )
        
        sns.boxplot(data=energy_filtered, x='noise_level', y='total_energy', 
                   ax=ax2, color='lightgreen', width=0.6)
        
        # 添加均值点
        means = energy_filtered.groupby('noise_level', observed=True)['total_energy'].mean()
        for i, (noise, mean_val) in enumerate(means.items()):
            ax2.scatter(i, mean_val, color='red', s=100, marker='D', zorder=5,
                       label='Mean' if i == 0 else '')
        
        ax2.set_title('Total Energy Distribution by Noise Level (Last 100 samples)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Noise Level', fontsize=12)
        ax2.set_ylabel('Total Energy', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # 添加统计信息
        for i, noise in enumerate(noise_levels):
            data = energy_filtered[energy_filtered['noise_level'] == noise]['total_energy']
            if len(data) > 0:
                ax2.text(i, ax2.get_ylim()[1]*0.98, f'n={len(data)}', 
                        ha='center', va='top', fontsize=10)
    
    # 添加图例
    if ax1.get_legend_handles_labels()[0]:
        ax1.legend(loc='upper right')
    if ax2.get_legend_handles_labels()[0]:
        ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('noise_level_comparison_boxplot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印统计信息
    print("\n" + "="*60)
    print("统计分析结果 (基于每个噪声等级最后100个样本)")
    print("="*60)
    
    print("\n--- Makespan Statistics by Noise Level ---")
    if not makespan_filtered.empty:
        makespan_stats = makespan_filtered.groupby('noise_level', observed=True)['makespan'].agg([
            'count', 'mean', 'std', 'min', 'max', 'median'
        ]).round(2)
        print(makespan_stats)
        
        # 计算噪声等级间的变化
        print("\n--- Makespan Changes ---")
        baseline = makespan_stats.loc[0.0, 'mean'] if 0.0 in makespan_stats.index else None
        if baseline:
            for noise in noise_levels[1:]:
                if noise in makespan_stats.index:
                    current = makespan_stats.loc[noise, 'mean']
                    change = ((current - baseline) / baseline) * 100
                    print(f"Noise {noise} vs 0.0: {change:+.2f}% "
                          f"({baseline:.2f} → {current:.2f})")
    
    print("\n--- Total Energy Statistics by Noise Level ---")
    if not energy_filtered.empty:
        energy_stats = energy_filtered.groupby('noise_level', observed=True)['total_energy'].agg([
            'count', 'mean', 'std', 'min', 'max', 'median'
        ]).round(2)
        print(energy_stats)
        
        # 计算噪声等级间的变化
        print("\n--- Energy Changes ---")
        baseline = energy_stats.loc[0.0, 'mean'] if 0.0 in energy_stats.index else None
        if baseline:
            for noise in noise_levels[1:]:
                if noise in energy_stats.index:
                    current = energy_stats.loc[noise, 'mean']
                    change = ((current - baseline) / baseline) * 100
                    print(f"Noise {noise} vs 0.0: {change:+.2f}% "
                          f"({baseline:.2f} → {current:.2f})")

def create_separate_boxplots():
    """
    为每个指标创建单独的箱式图
    """
    # 读取数据
    try:
        makespan_df = pd.read_csv(r'D:\1 工作\2 科研论文\2-LIRL\LIRL 最新实验数据\鲁棒性实验\boxplot_makespan_plot_data.csv')
        energy_df = pd.read_csv(r'D:\1 工作\2 科研论文\2-LIRL\LIRL 最新实验数据\鲁棒性实验\boxplot_total_energy_plot_data.csv')
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return
    
    noise_levels = [0.0, 0.1, 0.3, 0.5]
    
    # 1. Makespan单独箱式图
    plt.figure(figsize=(10, 8))
    
    # 对每个噪声等级只取最后100个数据
    makespan_filtered = filter_last_n_data(makespan_df, noise_levels, n=100)
    makespan_filtered['noise_level'] = pd.Categorical(
        makespan_filtered['parameter_value'], 
        categories=noise_levels, 
        ordered=True
    )
    
    sns.boxplot(data=makespan_filtered, x='noise_level', y='makespan', 
               color='skyblue', width=0.5)
    
    # 添加数据点
    sns.stripplot(data=makespan_filtered, x='noise_level', y='makespan', 
                 color='black', alpha=0.3, size=3)
    
    plt.title('Makespan vs Noise Level (Last 100 samples per level)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Noise Level', fontsize=14)
    plt.ylabel('Makespan', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    
    # 添加趋势线
    means = makespan_filtered.groupby('noise_level', observed=True)['makespan'].mean()
    x_positions = range(len(means))
    plt.plot(x_positions, means.values, 'r--', linewidth=2, label='Mean trend')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('makespan_by_noise_level.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Energy单独箱式图
    plt.figure(figsize=(10, 8))
    
    # 对每个噪声等级只取最后100个数据
    energy_filtered = filter_last_n_data(energy_df, noise_levels, n=100)
    energy_filtered['noise_level'] = pd.Categorical(
        energy_filtered['parameter_value'], 
        categories=noise_levels, 
        ordered=True
    )
    
    sns.boxplot(data=energy_filtered, x='noise_level', y='total_energy', 
               color='lightgreen', width=0.5)
    
    # 添加数据点
    sns.stripplot(data=energy_filtered, x='noise_level', y='total_energy', 
                 color='black', alpha=0.3, size=3)
    
    plt.title('Total Energy vs Noise Level (Last 100 samples per level)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Noise Level', fontsize=14)
    plt.ylabel('Total Energy', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    
    # 添加趋势线
    means = energy_filtered.groupby('noise_level', observed=True)['total_energy'].mean()
    x_positions = range(len(means))
    plt.plot(x_positions, means.values, 'r--', linewidth=2, label='Mean trend')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('energy_by_noise_level.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_violin_plots():
    """
    创建小提琴图以更好地显示数据分布
    """
    # 读取数据
    try:
        makespan_df = pd.read_csv(r'D:\1 工作\2 科研论文\2-LIRL\LIRL 最新实验数据\鲁棒性实验\boxplot_makespan_plot_data.csv')
        energy_df = pd.read_csv(r'D:\1 工作\2 科研论文\2-LIRL\LIRL 最新实验数据\鲁棒性实验\boxplot_total_energy_plot_data.csv')
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return
    
    noise_levels = [0.0, 0.1, 0.3, 0.5]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Makespan小提琴图
    makespan_filtered = filter_last_n_data(makespan_df, noise_levels, n=100)
    makespan_filtered['noise_level'] = pd.Categorical(
        makespan_filtered['parameter_value'], 
        categories=noise_levels, 
        ordered=True
    )
    
    sns.violinplot(data=makespan_filtered, x='noise_level', y='makespan', 
                  ax=ax1, color='skyblue', inner='box')
    ax1.set_title('Makespan Distribution (Violin Plot - Last 100 samples)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Noise Level', fontsize=12)
    ax1.set_ylabel('Makespan', fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Energy小提琴图
    energy_filtered = filter_last_n_data(energy_df, noise_levels, n=100)
    energy_filtered['noise_level'] = pd.Categorical(
        energy_filtered['parameter_value'], 
        categories=noise_levels, 
        ordered=True
    )
    
    sns.violinplot(data=energy_filtered, x='noise_level', y='total_energy', 
                  ax=ax2, color='lightgreen', inner='box')
    ax2.set_title('Total Energy Distribution (Violin Plot - Last 100 samples)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Noise Level', fontsize=12)
    ax2.set_ylabel('Total Energy', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('noise_level_violin_plots.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_data_summary():
    """
    打印数据摘要信息
    """
    try:
        makespan_df = pd.read_csv(r'D:\1 工作\2 科研论文\2-LIRL\LIRL 最新实验数据\鲁棒性实验\boxplot_makespan_plot_data.csv')
        energy_df = pd.read_csv(r'D:\1 工作\2 科研论文\2-LIRL\LIRL 最新实验数据\鲁棒性实验\boxplot_total_energy_plot_data.csv')
        
        print("\n" + "="*60)
        print("原始数据信息")
        print("="*60)
        
        noise_levels = [0.0, 0.1, 0.3, 0.5]
        
        print("\n--- Makespan数据分布 ---")
        for noise in noise_levels:
            count = len(makespan_df[makespan_df['parameter_value'] == noise])
            print(f"Noise level {noise}: {count} 条数据")
        
        print("\n--- Energy数据分布 ---")
        for noise in noise_levels:
            count = len(energy_df[energy_df['parameter_value'] == noise])
            print(f"Noise level {noise}: {count} 条数据")
            
    except Exception as e:
        print(f"无法读取数据文件: {e}")

# 主执行部分
if __name__ == "__main__":
    # 首先打印数据摘要
    print_data_summary()
    
    print("\n创建噪声等级对比箱式图...")
    
    # 创建组合箱式图
    create_noise_level_boxplots()
    
    # 创建单独的箱式图
    print("\n创建单独的箱式图...")
    create_separate_boxplots()
    
    # 创建小提琴图
    print("\n创建小提琴图...")
    create_violin_plots()
    
    print("\n所有图表创建完成！")
    print("注意: 所有图表都基于每个噪声等级的最后100个样本")