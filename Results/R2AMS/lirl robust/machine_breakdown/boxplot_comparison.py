import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置中文字体和图形样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def filter_last_n_data(df, failure_rates, n=100):
    """
    对每个故障率，只保留最后n个数据
    
    参数:
        df: 原始数据框
        failure_rates: 故障率列表
        n: 保留的数据数量
    
    返回:
        过滤后的数据框
    """
    filtered_data = []
    
    for failure_rate in failure_rates:
        # 获取当前故障率的数据
        rate_data = df[df['parameter_value'] == failure_rate]
        
        # 只取最后n个数据
        if len(rate_data) > n:
            rate_data = rate_data.tail(n)
        
        filtered_data.append(rate_data)
    
    # 合并所有过滤后的数据
    if filtered_data:
        return pd.concat(filtered_data, ignore_index=True)
    else:
        return pd.DataFrame()

def create_failure_rate_boxplots():
    """
    创建基于故障率的箱式图对比
    """
    # 读取数据
    try:
        # 读取makespan数据
        makespan_df = pd.read_csv(r'D:\1 工作\2 科研论文\2-LIRL\LIRL 最新实验数据\鲁棒性实验\machine_breakdown\boxplot_makespan_plot_data.csv')
        # 读取energy数据
        energy_df = pd.read_csv(r'D:\1 工作\2 科研论文\2-LIRL\LIRL 最新实验数据\鲁棒性实验\machine_breakdown\boxplot_total_energy_plot_data.csv')
        print("数据读取成功！")
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return
    
    # 筛选指定的故障率
    failure_rates = [0.0, 0.1, 0.3, 0.5]
    
    # 对每个故障率只取最后100个数据
    makespan_filtered = filter_last_n_data(makespan_df, failure_rates, n=100)
    energy_filtered = filter_last_n_data(energy_df, failure_rates, n=100)
    
    # 添加failure_rate列
    makespan_filtered['failure_rate'] = makespan_filtered['parameter_value']
    energy_filtered['failure_rate'] = energy_filtered['parameter_value']
    
    print(f"Makespan数据过滤后: {makespan_filtered.shape[0]}条")
    print(f"Energy数据过滤后: {energy_filtered.shape[0]}条")
    
    # 创建图形 - 两个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Makespan箱式图
    if not makespan_filtered.empty:
        # 确保故障率按顺序排列
        makespan_filtered['failure_rate'] = pd.Categorical(
            makespan_filtered['failure_rate'], 
            categories=failure_rates, 
            ordered=True
        )
        
        sns.boxplot(data=makespan_filtered, x='failure_rate', y='makespan', 
                   ax=ax1, color='skyblue', width=0.6)
        
        # 添加均值点
        means = makespan_filtered.groupby('failure_rate', observed=True)['makespan'].mean()
        for i, (rate, mean_val) in enumerate(means.items()):
            ax1.scatter(i, mean_val, color='red', s=100, marker='D', zorder=5, 
                       label='Mean' if i == 0 else '')
        
        ax1.set_title('Makespan Distribution by Machine Failure Rate (Last 100 samples)', 
                      fontsize=14, fontweight='bold')
        ax1.set_xlabel('Failure Rate', fontsize=12)
        ax1.set_ylabel('Makespan', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # 添加统计信息
        for i, rate in enumerate(failure_rates):
            data = makespan_filtered[makespan_filtered['failure_rate'] == rate]['makespan']
            if len(data) > 0:
                ax1.text(i, ax1.get_ylim()[1]*0.98, f'n={len(data)}', 
                        ha='center', va='top', fontsize=10)
    
    # 2. Total Energy箱式图
    if not energy_filtered.empty:
        # 确保故障率按顺序排列
        energy_filtered['failure_rate'] = pd.Categorical(
            energy_filtered['failure_rate'], 
            categories=failure_rates, 
            ordered=True
        )
        
        sns.boxplot(data=energy_filtered, x='failure_rate', y='total_energy', 
                   ax=ax2, color='lightgreen', width=0.6)
        
        # 添加均值点
        means = energy_filtered.groupby('failure_rate', observed=True)['total_energy'].mean()
        for i, (rate, mean_val) in enumerate(means.items()):
            ax2.scatter(i, mean_val, color='red', s=100, marker='D', zorder=5,
                       label='Mean' if i == 0 else '')
        
        ax2.set_title('Total Energy Distribution by Machine Failure Rate (Last 100 samples)', 
                      fontsize=14, fontweight='bold')
        ax2.set_xlabel('Failure Rate', fontsize=12)
        ax2.set_ylabel('Total Energy', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # 添加统计信息
        for i, rate in enumerate(failure_rates):
            data = energy_filtered[energy_filtered['failure_rate'] == rate]['total_energy']
            if len(data) > 0:
                ax2.text(i, ax2.get_ylim()[1]*0.98, f'n={len(data)}', 
                        ha='center', va='top', fontsize=10)
    
    # 添加图例
    if ax1.get_legend_handles_labels()[0]:
        ax1.legend(loc='upper right')
    if ax2.get_legend_handles_labels()[0]:
        ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('failure_rate_comparison_boxplot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印统计信息
    print("\n" + "="*60)
    print("统计分析结果 (基于每个故障率最后100个样本)")
    print("="*60)
    
    print("\n--- Makespan Statistics by Failure Rate ---")
    if not makespan_filtered.empty:
        makespan_stats = makespan_filtered.groupby('failure_rate', observed=True)['makespan'].agg([
            'count', 'mean', 'std', 'min', 'max', 'median'
        ]).round(2)
        print(makespan_stats)
        
        # 计算故障率间的变化
        print("\n--- Makespan Changes ---")
        baseline = makespan_stats.loc[0.0, 'mean'] if 0.0 in makespan_stats.index else None
        if baseline:
            for rate in failure_rates[1:]:
                if rate in makespan_stats.index:
                    current = makespan_stats.loc[rate, 'mean']
                    change = ((current - baseline) / baseline) * 100
                    print(f"Failure rate {rate} vs 0.0: {change:+.2f}% "
                          f"({baseline:.2f} → {current:.2f})")
    
    print("\n--- Total Energy Statistics by Failure Rate ---")
    if not energy_filtered.empty:
        energy_stats = energy_filtered.groupby('failure_rate', observed=True)['total_energy'].agg([
            'count', 'mean', 'std', 'min', 'max', 'median'
        ]).round(2)
        print(energy_stats)
        
        # 计算故障率间的变化
        print("\n--- Energy Changes ---")
        baseline = energy_stats.loc[0.0, 'mean'] if 0.0 in energy_stats.index else None
        if baseline:
            for rate in failure_rates[1:]:
                if rate in energy_stats.index:
                    current = energy_stats.loc[rate, 'mean']
                    change = ((current - baseline) / baseline) * 100
                    print(f"Failure rate {rate} vs 0.0: {change:+.2f}% "
                          f"({baseline:.2f} → {current:.2f})")

def create_separate_boxplots():
    """
    为每个指标创建单独的箱式图
    """
    # 读取数据
    try:
        makespan_df = pd.read_csv(r'D:\1 工作\2 科研论文\2-LIRL\LIRL 最新实验数据\鲁棒性实验\machine_breakdown\boxplot_makespan_plot_data.csv')
        energy_df = pd.read_csv(r'D:\1 工作\2 科研论文\2-LIRL\LIRL 最新实验数据\鲁棒性实验\machine_breakdown\boxplot_total_energy_plot_data.csv')
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return
    
    failure_rates = [0.0, 0.1, 0.3, 0.5]
    
    # 1. Makespan单独箱式图
    plt.figure(figsize=(10, 8))
    
    # 对每个故障率只取最后100个数据
    makespan_filtered = filter_last_n_data(makespan_df, failure_rates, n=100)
    makespan_filtered['failure_rate'] = pd.Categorical(
        makespan_filtered['parameter_value'], 
        categories=failure_rates, 
        ordered=True
    )
    
    sns.boxplot(data=makespan_filtered, x='failure_rate', y='makespan', 
               color='skyblue', width=0.5)
    
    # 添加数据点
    sns.stripplot(data=makespan_filtered, x='failure_rate', y='makespan', 
                 color='black', alpha=0.3, size=3)
    
    plt.title('Makespan vs Machine Failure Rate (Last 100 samples per rate)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Failure Rate', fontsize=14)
    plt.ylabel('Makespan', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    
    # 添加趋势线
    means = makespan_filtered.groupby('failure_rate', observed=True)['makespan'].mean()
    x_positions = range(len(means))
    plt.plot(x_positions, means.values, 'r--', linewidth=2, label='Mean trend')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('makespan_by_failure_rate.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Energy单独箱式图
    plt.figure(figsize=(10, 8))
    
    # 对每个故障率只取最后100个数据
    energy_filtered = filter_last_n_data(energy_df, failure_rates, n=100)
    energy_filtered['failure_rate'] = pd.Categorical(
        energy_filtered['parameter_value'], 
        categories=failure_rates, 
        ordered=True
    )
    
    sns.boxplot(data=energy_filtered, x='failure_rate', y='total_energy', 
               color='lightgreen', width=0.5)
    
    # 添加数据点
    sns.stripplot(data=energy_filtered, x='failure_rate', y='total_energy', 
                 color='black', alpha=0.3, size=3)
    
    plt.title('Total Energy vs Machine Failure Rate (Last 100 samples per rate)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Failure Rate', fontsize=14)
    plt.ylabel('Total Energy', fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    
    # 添加趋势线
    means = energy_filtered.groupby('failure_rate', observed=True)['total_energy'].mean()
    x_positions = range(len(means))
    plt.plot(x_positions, means.values, 'r--', linewidth=2, label='Mean trend')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('energy_by_failure_rate.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_violin_plots():
    """
    创建小提琴图以更好地显示数据分布
    """
    # 读取数据
    try:
        makespan_df = pd.read_csv(r'D:\1 工作\2 科研论文\2-LIRL\LIRL 最新实验数据\鲁棒性实验\machine_breakdown\boxplot_makespan_plot_data.csv')
        energy_df = pd.read_csv(r'D:\1 工作\2 科研论文\2-LIRL\LIRL 最新实验数据\鲁棒性实验\machine_breakdown\boxplot_total_energy_plot_data.csv')
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return
    
    failure_rates = [0.0, 0.1, 0.3, 0.5]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Makespan小提琴图
    makespan_filtered = filter_last_n_data(makespan_df, failure_rates, n=100)
    makespan_filtered['failure_rate'] = pd.Categorical(
        makespan_filtered['parameter_value'], 
        categories=failure_rates, 
        ordered=True
    )
    
    sns.violinplot(data=makespan_filtered, x='failure_rate', y='makespan', 
                  ax=ax1, color='skyblue', inner='box')
    ax1.set_title('Makespan Distribution (Violin Plot - Last 100 samples)', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Machine Failure Rate', fontsize=12)
    ax1.set_ylabel('Makespan', fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Energy小提琴图
    energy_filtered = filter_last_n_data(energy_df, failure_rates, n=100)
    energy_filtered['failure_rate'] = pd.Categorical(
        energy_filtered['parameter_value'], 
        categories=failure_rates, 
        ordered=True
    )
    
    sns.violinplot(data=energy_filtered, x='failure_rate', y='total_energy', 
                  ax=ax2, color='lightgreen', inner='box')
    ax2.set_title('Total Energy Distribution (Violin Plot - Last 100 samples)', 
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('Machine Failure Rate', fontsize=12)
    ax2.set_ylabel('Total Energy', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('failure_rate_violin_plots.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_data_summary():
    """
    打印数据摘要信息
    """
    try:
        makespan_df = pd.read_csv(r'D:\1 工作\2 科研论文\2-LIRL\LIRL 最新实验数据\鲁棒性实验\machine_breakdown\boxplot_makespan_plot_data.csv')
        energy_df = pd.read_csv(r'D:\1 工作\2 科研论文\2-LIRL\LIRL 最新实验数据\鲁棒性实验\machine_breakdown\boxplot_total_energy_plot_data.csv')
        
        print("\n" + "="*60)
        print("原始数据信息")
        print("="*60)
        
        failure_rates = [0.0, 0.1, 0.3, 0.5]
        
        print("\n--- Makespan数据分布 ---")
        for rate in failure_rates:
            count = len(makespan_df[makespan_df['parameter_value'] == rate])
            print(f"Failure rate {rate}: {count} 条数据")
        
        print("\n--- Energy数据分布 ---")
        for rate in failure_rates:
            count = len(energy_df[energy_df['parameter_value'] == rate])
            print(f"Failure rate {rate}: {count} 条数据")
            
    except Exception as e:
        print(f"无法读取数据文件: {e}")

def create_trend_analysis():
    """
    创建趋势分析图
    """
    # 读取数据
    try:
        makespan_df = pd.read_csv(r'D:\1 工作\2 科研论文\2-LIRL\LIRL 最新实验数据\鲁棒性实验\machine_breakdown\boxplot_makespan_plot_data.csv')
        energy_df = pd.read_csv(r'D:\1 工作\2 科研论文\2-LIRL\LIRL 最新实验数据\鲁棒性实验\machine_breakdown\boxplot_total_energy_plot_data.csv')
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return
    
    failure_rates = [0.0, 0.1, 0.3, 0.5]
    
    # 计算每个故障率的统计指标
    makespan_means = []
    makespan_stds = []
    energy_means = []
    energy_stds = []
    
    for rate in failure_rates:
        # Makespan统计
        makespan_data = makespan_df[makespan_df['parameter_value'] == rate]['makespan']
        if len(makespan_data) > 100:
            makespan_data = makespan_data.tail(100)
        makespan_means.append(makespan_data.mean())
        makespan_stds.append(makespan_data.std())
        
        # Energy统计
        energy_data = energy_df[energy_df['parameter_value'] == rate]['total_energy']
        if len(energy_data) > 100:
            energy_data = energy_data.tail(100)
        energy_means.append(energy_data.mean())
        energy_stds.append(energy_data.std())
    
    # 创建趋势图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Makespan趋势图
    ax1.errorbar(failure_rates, makespan_means, yerr=makespan_stds, 
                 marker='o', markersize=10, capsize=5, capthick=2,
                 linewidth=2, color='blue', ecolor='gray')
    ax1.fill_between(failure_rates, 
                     np.array(makespan_means) - np.array(makespan_stds),
                     np.array(makespan_means) + np.array(makespan_stds),
                     alpha=0.2, color='blue')
    
    ax1.set_title('Makespan Trend with Machine Failure Rate', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Failure Rate', fontsize=12)
    ax1.set_ylabel('Makespan (Mean ± Std)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(failure_rates)
    
    # 添加百分比变化标签
    for i in range(1, len(failure_rates)):
        if makespan_means[0] > 0:
            change = ((makespan_means[i] - makespan_means[0]) / makespan_means[0]) * 100
            ax1.text(failure_rates[i], makespan_means[i] + makespan_stds[i] + 5,
                    f'+{change:.1f}%', ha='center', va='bottom', fontsize=10, color='red')
    
    # Energy趋势图
    ax2.errorbar(failure_rates, energy_means, yerr=energy_stds, 
                 marker='s', markersize=10, capsize=5, capthick=2,
                 linewidth=2, color='green', ecolor='gray')
    ax2.fill_between(failure_rates, 
                     np.array(energy_means) - np.array(energy_stds),
                     np.array(energy_means) + np.array(energy_stds),
                     alpha=0.2, color='green')
    
    ax2.set_title('Total Energy Trend with Machine Failure Rate', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Failure Rate', fontsize=12)
    ax2.set_ylabel('Total Energy (Mean ± Std)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(failure_rates)
    
    # 添加百分比变化标签
    for i in range(1, len(failure_rates)):
        if energy_means[0] > 0:
            change = ((energy_means[i] - energy_means[0]) / energy_means[0]) * 100
            ax2.text(failure_rates[i], energy_means[i] + energy_stds[i] + 100,
                    f'+{change:.1f}%', ha='center', va='bottom', fontsize=10, color='red')
    
    plt.tight_layout()
    plt.savefig('failure_rate_trend_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# 主执行部分
if __name__ == "__main__":
    # 首先打印数据摘要
    print_data_summary()
    
    print("\n创建故障率对比箱式图...")
    
    # 创建组合箱式图
    create_failure_rate_boxplots()
    
    # 创建单独的箱式图
    print("\n创建单独的箱式图...")
    create_separate_boxplots()
    
    # 创建小提琴图
    print("\n创建小提琴图...")
    create_violin_plots()
    
    # 创建趋势分析图
    print("\n创建趋势分析图...")
    create_trend_analysis()
    
    print("\n所有图表创建完成！")
    print("注意: 所有图表都基于每个故障率的最后100个样本")