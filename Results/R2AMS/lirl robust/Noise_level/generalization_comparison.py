import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置中文字体和图形样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def remove_outliers(df, column, group_column=None, method='iqr', threshold=1.5):
    """
    移除异常值
    
    参数:
        df: 数据框
        column: 要检测异常值的列
        group_column: 分组列（如noise_level）
        method: 'iqr' 或 'zscore'
        threshold: IQR方法的倍数（默认1.5）或z-score的阈值（默认3）
    
    返回:
        过滤后的数据框
    """
    if group_column:
        # 按组移除异常值
        filtered_groups = []
        for group_value in df[group_column].unique():
            group_data = df[df[group_column] == group_value]
            
            if method == 'iqr':
                Q1 = group_data[column].quantile(0.25)
                Q3 = group_data[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                filtered = group_data[(group_data[column] >= lower_bound) & 
                                    (group_data[column] <= upper_bound)]
                
                removed_count = len(group_data) - len(filtered)
                if removed_count > 0:
                    print(f"  {group_column}={group_value}: 移除了 {removed_count} 个异常值 "
                          f"(范围: [{lower_bound:.2f}, {upper_bound:.2f}])")
                
            elif method == 'zscore':
                z_scores = np.abs((group_data[column] - group_data[column].mean()) / 
                                group_data[column].std())
                filtered = group_data[z_scores <= threshold]
                
                removed_count = len(group_data) - len(filtered)
                if removed_count > 0:
                    print(f"  {group_column}={group_value}: 移除了 {removed_count} 个异常值 "
                          f"(Z-score > {threshold})")
            
            filtered_groups.append(filtered)
        
        return pd.concat(filtered_groups, ignore_index=True)
    else:
        # 整体移除异常值
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            filtered = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
            
        elif method == 'zscore':
            z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
            filtered = df[z_scores <= threshold]
        
        removed_count = len(df) - len(filtered)
        if removed_count > 0:
            print(f"  移除了 {removed_count} 个异常值")
        
        return filtered

def read_generalization_data():
    """
    读取generalization_results.csv文件并筛选数据
    """
    try:
        # 读取generalization_results.csv
        gen_df = pd.read_csv(r'D:\1 工作\2 科研论文\2-LIRL\LIRL 最新实验数据\鲁棒性实验\Noise_level\generalization_results.csv')
        print("Generalization数据读取成功！")
        print(f"原始数据形状: {gen_df.shape}")
        print(f"列名: {gen_df.columns.tolist()}")
        
        # 筛选failure_rate=0的数据
        gen_filtered = gen_df[gen_df['failure_rate'] == 0.0].copy()
        print(f"\nfailure_rate=0的数据: {gen_filtered.shape[0]}条")
        
        # 进一步筛选指定的noise_level（不包括0.0）
        noise_levels = [0.1, 0.3, 0.5]  # 移除0.0
        gen_filtered = gen_filtered[gen_filtered['nose_level'].isin(noise_levels)].copy()
        print(f"筛选后的数据: {gen_filtered.shape[0]}条")
        
        # 显示数据分布
        print("\n数据分布:")
        for noise in noise_levels:
            count = len(gen_filtered[gen_filtered['nose_level'] == noise])
            print(f"  Noise level {noise}: {count}条")
        
        # 移除异常值
        print("\n移除Generalization数据中的异常值...")
        print("- Makespan异常值检测:")
        gen_filtered = remove_outliers(gen_filtered, 'avg_makespan', 'nose_level', method='iqr', threshold=1.5)
        print("- Energy异常值检测:")
        gen_filtered = remove_outliers(gen_filtered, 'avg_energy', 'nose_level', method='iqr', threshold=1.5)
        
        print(f"\n移除异常值后的数据: {gen_filtered.shape[0]}条")
        
        return gen_filtered
        
    except Exception as e:
        print(f"读取generalization数据时出错: {e}")
        return None

def filter_last_n_data(df, noise_levels, n=100):
    """
    对每个噪声等级，只保留最后n个数据，并移除异常值
    """
    filtered_data = []
    
    print(f"\n处理训练数据...")
    for noise_level in noise_levels:
        noise_data = df[df['parameter_value'] == noise_level]
        original_count = len(noise_data)
        
        if len(noise_data) > n:
            noise_data = noise_data.tail(n)
        
        # 移除异常值
        if 'makespan' in noise_data.columns:
            print(f"  Noise level {noise_level} - Makespan异常值检测:")
            noise_data = remove_outliers(noise_data, 'makespan', method='iqr', threshold=1.5)
        elif 'total_energy' in noise_data.columns:
            print(f"  Noise level {noise_level} - Energy异常值检测:")
            noise_data = remove_outliers(noise_data, 'total_energy', method='iqr', threshold=1.5)
        
        filtered_data.append(noise_data)
    
    if filtered_data:
        return pd.concat(filtered_data, ignore_index=True)
    else:
        return pd.DataFrame()

def create_comparison_violin_plots():
    """
    创建小提琴图对比：Generalization vs Training（不包括noise_level=0）
    """
    # 读取generalization数据
    gen_data = read_generalization_data()
    if gen_data is None or gen_data.empty:
        print("无法读取generalization数据")
        return
    
    # 读取训练数据
    try:
        makespan_df = pd.read_csv(r'D:\1 工作\2 科研论文\2-LIRL\LIRL 最新实验数据\鲁棒性实验\boxplot_makespan_plot_data.csv')
        energy_df = pd.read_csv(r'D:\1 工作\2 科研论文\2-LIRL\LIRL 最新实验数据\鲁棒性实验\boxplot_total_energy_plot_data.csv')
        print("\n训练数据读取成功！")
    except Exception as e:
        print(f"读取训练数据时出错: {e}")
        return
    
    noise_levels = [0.1, 0.3, 0.5]  # 不包括0.0
    
    # 对训练数据进行过滤（包括异常值处理）
    makespan_filtered = filter_last_n_data(makespan_df, noise_levels, n=100)
    energy_filtered = filter_last_n_data(energy_df, noise_levels, n=100)
    
    # 准备对比数据
    # 1. Makespan对比数据
    makespan_comparison = []
    
    # 添加训练数据
    for _, row in makespan_filtered.iterrows():
        makespan_comparison.append({
            'noise_level': row['parameter_value'],
            'makespan': row['makespan'],
            'data_type': 'Training'
        })
    
    # 添加generalization数据（需要扩展数据以显示分布）
    for _, row in gen_data.iterrows():
        # 为了显示分布，我们基于均值和标准差生成一些数据点
        for _ in range(10):  # 每个点重复10次以增加密度
            makespan_comparison.append({
                'noise_level': row['nose_level'],
                'makespan': row['avg_makespan'],
                'data_type': 'Generalization'
            })
    
    makespan_comp_df = pd.DataFrame(makespan_comparison)
    
    # 2. Energy对比数据
    energy_comparison = []
    
    # 添加训练数据
    for _, row in energy_filtered.iterrows():
        energy_comparison.append({
            'noise_level': row['parameter_value'],
            'energy': row['total_energy'],
            'data_type': 'Training'
        })
    
    # 添加generalization数据
    for _, row in gen_data.iterrows():
        for _ in range(10):  # 每个点重复10次以增加密度
            energy_comparison.append({
                'noise_level': row['nose_level'],
                'energy': row['avg_energy'],
                'data_type': 'Generalization'
            })
    
    energy_comp_df = pd.DataFrame(energy_comparison)
    
    # 创建小提琴图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Makespan小提琴图（分离式）
    makespan_comp_df['noise_level'] = pd.Categorical(
        makespan_comp_df['noise_level'], 
        categories=noise_levels, 
        ordered=True
    )
    
    sns.violinplot(data=makespan_comp_df, x='noise_level', y='makespan', 
                   hue='data_type', ax=ax1, palette=['skyblue', 'lightcoral'],
                   split=True, inner='quartile')
    
    ax1.set_title('Makespan Distribution: Training vs Generalization (Split Violin)', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Noise Level', fontsize=12)
    ax1.set_ylabel('Makespan', fontsize=12)
    ax1.legend(title='Data Type', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Energy小提琴图（分离式）
    energy_comp_df['noise_level'] = pd.Categorical(
        energy_comp_df['noise_level'], 
        categories=noise_levels, 
        ordered=True
    )
    
    sns.violinplot(data=energy_comp_df, x='noise_level', y='energy', 
                   hue='data_type', ax=ax2, palette=['lightgreen', 'orange'],
                   split=True, inner='quartile')
    
    ax2.set_title('Total Energy Distribution: Training vs Generalization (Split Violin)', 
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('Noise Level', fontsize=12)
    ax2.set_ylabel('Total Energy', fontsize=12)
    ax2.legend(title='Data Type', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Makespan小提琴图（叠加式）
    sns.violinplot(data=makespan_comp_df, x='noise_level', y='makespan', 
                   hue='data_type', ax=ax3, palette=['skyblue', 'lightcoral'],
                   split=False, inner='box')
    
    ax3.set_title('Makespan Distribution: Training vs Generalization (Overlay Violin)', 
                  fontsize=14, fontweight='bold')
    ax3.set_xlabel('Noise Level', fontsize=12)
    ax3.set_ylabel('Makespan', fontsize=12)
    ax3.legend(title='Data Type', fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Energy小提琴图（叠加式）
    sns.violinplot(data=energy_comp_df, x='noise_level', y='energy', 
                   hue='data_type', ax=ax4, palette=['lightgreen', 'orange'],
                   split=False, inner='box')
    
    ax4.set_title('Total Energy Distribution: Training vs Generalization (Overlay Violin)', 
                  fontsize=14, fontweight='bold')
    ax4.set_xlabel('Noise Level', fontsize=12)
    ax4.set_ylabel('Total Energy', fontsize=12)
    ax4.legend(title='Data Type', fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('training_vs_generalization_violin_plots_no_zero.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_combined_violin_boxplot():
    """
    创建组合的小提琴图和箱线图（不包括noise_level=0）
    """
    # 读取generalization数据
    gen_data = read_generalization_data()
    if gen_data is None or gen_data.empty:
        print("无法读取generalization数据")
        return
    
    # 读取训练数据
    try:
        makespan_df = pd.read_csv(r'D:\1 工作\2 科研论文\2-LIRL\LIRL 最新实验数据\鲁棒性实验\boxplot_makespan_plot_data.csv')
        energy_df = pd.read_csv(r'D:\1 工作\2 科研论文\2-LIRL\LIRL 最新实验数据\鲁棒性实验\boxplot_total_energy_plot_data.csv')
    except Exception as e:
        print(f"读取训练数据时出错: {e}")
        return
    
    noise_levels = [0.1, 0.3, 0.5]  # 不包括0.0
    
    # 对训练数据进行过滤
    makespan_filtered = filter_last_n_data(makespan_df, noise_levels, n=100)
    energy_filtered = filter_last_n_data(energy_df, noise_levels, n=100)
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # 1. Makespan - 训练数据小提琴图
    makespan_filtered['noise_level'] = pd.Categorical(
        makespan_filtered['parameter_value'], 
        categories=noise_levels, 
        ordered=True
    )
    
    # 绘制训练数据的小提琴图
    violin_parts = ax1.violinplot(
        [makespan_filtered[makespan_filtered['noise_level'] == noise]['makespan'].values 
         for noise in noise_levels],
        positions=range(len(noise_levels)),
        widths=0.7,
        showmeans=True,
        showmedians=True
    )
    
    # 设置小提琴图颜色
    for pc in violin_parts['bodies']:
        pc.set_facecolor('skyblue')
        pc.set_alpha(0.7)
    
    # 添加泛化数据的散点
    gen_x_positions = []
    gen_y_values = []
    for i, noise in enumerate(noise_levels):
        gen_noise_data = gen_data[gen_data['nose_level'] == noise]
        for _, row in gen_noise_data.iterrows():
            gen_x_positions.append(i)
            gen_y_values.append(row['avg_makespan'])
    
    ax1.scatter(gen_x_positions, gen_y_values, color='red', s=100, 
               marker='o', label='Generalization', zorder=5, alpha=0.8)
    
    ax1.set_xticks(range(len(noise_levels)))
    ax1.set_xticklabels(noise_levels)
    ax1.set_xlabel('Noise Level', fontsize=12)
    ax1.set_ylabel('Makespan', fontsize=12)
    ax1.set_title('Makespan: Training Distribution (violin) vs Generalization Points (red)', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend()
    
    # 2. Energy - 训练数据小提琴图
    energy_filtered['noise_level'] = pd.Categorical(
        energy_filtered['parameter_value'], 
        categories=noise_levels, 
        ordered=True
    )
    
    # 绘制训练数据的小提琴图
    violin_parts = ax2.violinplot(
        [energy_filtered[energy_filtered['noise_level'] == noise]['total_energy'].values 
         for noise in noise_levels],
        positions=range(len(noise_levels)),
        widths=0.7,
        showmeans=True,
        showmedians=True
    )
    
    # 设置小提琴图颜色
    for pc in violin_parts['bodies']:
        pc.set_facecolor('lightgreen')
        pc.set_alpha(0.7)
    
    # 添加泛化数据的散点
    gen_x_positions = []
    gen_y_values = []
    for i, noise in enumerate(noise_levels):
        gen_noise_data = gen_data[gen_data['nose_level'] == noise]
        for _, row in gen_noise_data.iterrows():
            gen_x_positions.append(i)
            gen_y_values.append(row['avg_energy'])
    
    ax2.scatter(gen_x_positions, gen_y_values, color='red', s=100, 
               marker='o', label='Generalization', zorder=5, alpha=0.8)
    
    ax2.set_xticks(range(len(noise_levels)))
    ax2.set_xticklabels(noise_levels)
    ax2.set_xlabel('Noise Level', fontsize=12)
    ax2.set_ylabel('Total Energy', fontsize=12)
    ax2.set_title('Energy: Training Distribution (violin) vs Generalization Points (red)', 
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('combined_violin_scatter_plot_no_zero.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_ridge_plots():
    """
    创建Ridge图（山脊图）以展示不同噪声等级的分布（不包括noise_level=0）
    """
    # 读取数据
    try:
        makespan_df = pd.read_csv(r'D:\1 工作\2 科研论文\2-LIRL\LIRL 最新实验数据\鲁棒性实验\boxplot_makespan_plot_data.csv')
        energy_df = pd.read_csv(r'D:\1 工作\2 科研论文\2-LIRL\LIRL 最新实验数据\鲁棒性实验\boxplot_total_energy_plot_data.csv')
    except Exception as e:
        print(f"读取训练数据时出错: {e}")
        return
    
    noise_levels = [0.1, 0.3, 0.5]  # 不包括0.0
    
    # 过滤数据
    makespan_filtered = filter_last_n_data(makespan_df, noise_levels, n=100)
    energy_filtered = filter_last_n_data(energy_df, noise_levels, n=100)
    
    # 创建Ridge图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # 设置颜色映射
    colors = ['#3498db', '#e74c3c', '#f39c12']  # 蓝色、红色、橙色
    
    # 1. Makespan Ridge图
    for i, (noise, color) in enumerate(zip(noise_levels, colors)):
        data = makespan_filtered[makespan_filtered['parameter_value'] == noise]['makespan']
        
        # 创建核密度估计
        density = sns.kdeplot(data, ax=ax1, fill=True, alpha=0.6, 
                            label=f'Noise {noise}', color=color)
        
    ax1.set_title('Makespan Distribution Density by Noise Level', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Makespan', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.legend(title='Noise Level')
    ax1.grid(True, alpha=0.3)
    
    # 2. Energy Ridge图
    for i, (noise, color) in enumerate(zip(noise_levels, colors)):
        data = energy_filtered[energy_filtered['parameter_value'] == noise]['total_energy']
        
        # 创建核密度估计
        density = sns.kdeplot(data, ax=ax2, fill=True, alpha=0.6, 
                            label=f'Noise {noise}', color=color)
        
    ax2.set_title('Total Energy Distribution Density by Noise Level', 
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('Total Energy', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.legend(title='Noise Level')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('distribution_density_plots_no_zero.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_comparison_boxplots():
    """
    创建对比箱线图：Generalization vs Training（不包括noise_level=0）
    """
    # 读取generalization数据
    gen_data = read_generalization_data()
    if gen_data is None or gen_data.empty:
        print("无法读取generalization数据")
        return
    
    # 读取训练数据
    try:
        makespan_df = pd.read_csv(r'D:\1 工作\2 科研论文\2-LIRL\LIRL 最新实验数据\鲁棒性实验\boxplot_makespan_plot_data.csv')
        energy_df = pd.read_csv(r'D:\1 工作\2 科研论文\2-LIRL\LIRL 最新实验数据\鲁棒性实验\boxplot_total_energy_plot_data.csv')
        print("\n训练数据读取成功！")
    except Exception as e:
        print(f"读取训练数据时出错: {e}")
        return
    
    noise_levels = [0.1, 0.3, 0.5]  # 不包括0.0
    
    # 对训练数据进行过滤（包括异常值处理）
    makespan_filtered = filter_last_n_data(makespan_df, noise_levels, n=100)
    energy_filtered = filter_last_n_data(energy_df, noise_levels, n=100)
    
    # 准备对比数据
    # 1. Makespan对比数据
    makespan_comparison = []
    
    # 添加训练数据
    for _, row in makespan_filtered.iterrows():
        makespan_comparison.append({
            'noise_level': row['parameter_value'],
            'makespan': row['makespan'],
            'data_type': 'Training'
        })
    
    # 添加generalization数据
    for _, row in gen_data.iterrows():
        makespan_comparison.append({
            'noise_level': row['nose_level'],
            'makespan': row['avg_makespan'],
            'data_type': 'Generalization'
        })
    
    makespan_comp_df = pd.DataFrame(makespan_comparison)
    
    # 2. Energy对比数据
    energy_comparison = []
    
    # 添加训练数据
    for _, row in energy_filtered.iterrows():
        energy_comparison.append({
            'noise_level': row['parameter_value'],
            'energy': row['total_energy'],
            'data_type': 'Training'
        })
    
    # 添加generalization数据
    for _, row in gen_data.iterrows():
        energy_comparison.append({
            'noise_level': row['nose_level'],
            'energy': row['avg_energy'],
            'data_type': 'Generalization'
        })
    
    energy_comp_df = pd.DataFrame(energy_comparison)
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # 1. Makespan对比箱线图
    makespan_comp_df['noise_level'] = pd.Categorical(
        makespan_comp_df['noise_level'], 
        categories=noise_levels, 
        ordered=True
    )
    
    sns.boxplot(data=makespan_comp_df, x='noise_level', y='makespan', 
               hue='data_type', ax=ax1, palette=['skyblue', 'lightcoral'])
    
    # 添加均值点
    for dtype in ['Training', 'Generalization']:
        for i, noise in enumerate(noise_levels):
            subset = makespan_comp_df[
                (makespan_comp_df['data_type'] == dtype) & 
                (makespan_comp_df['noise_level'] == noise)
            ]
            if not subset.empty:
                mean_val = subset['makespan'].mean()
                offset = -0.2 if dtype == 'Training' else 0.2
                ax1.scatter(i + offset, mean_val, color='red', s=50, marker='D', zorder=5)
    
    ax1.set_title('Makespan Comparison: Training vs Generalization', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Noise Level', fontsize=12)
    ax1.set_ylabel('Makespan', fontsize=12)
    ax1.legend(title='Data Type', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Energy对比箱线图
    energy_comp_df['noise_level'] = pd.Categorical(
        energy_comp_df['noise_level'], 
        categories=noise_levels, 
        ordered=True
    )
    
    sns.boxplot(data=energy_comp_df, x='noise_level', y='energy', 
               hue='data_type', ax=ax2, palette=['lightgreen', 'orange'])
    
    # 添加均值点
    for dtype in ['Training', 'Generalization']:
        for i, noise in enumerate(noise_levels):
            subset = energy_comp_df[
                (energy_comp_df['data_type'] == dtype) & 
                (energy_comp_df['noise_level'] == noise)
            ]
            if not subset.empty:
                mean_val = subset['energy'].mean()
                offset = -0.2 if dtype == 'Training' else 0.2
                ax2.scatter(i + offset, mean_val, color='red', s=50, marker='D', zorder=5)
    
    ax2.set_title('Total Energy Comparison: Training vs Generalization', 
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('Noise Level', fontsize=12)
    ax2.set_ylabel('Total Energy', fontsize=12)
    ax2.legend(title='Data Type', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_vs_generalization_boxplot_no_zero.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印统计信息
    print("\n" + "="*70)
    print("统计分析结果（不包括noise_level=0）")
    print("="*70)
    
    print("\n--- Makespan Statistics ---")
    makespan_stats = makespan_comp_df.groupby(['data_type', 'noise_level'])['makespan'].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).round(2)
    print(makespan_stats)
    
    print("\n--- Energy Statistics ---")
    energy_stats = energy_comp_df.groupby(['data_type', 'noise_level'])['energy'].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).round(2)
    print(energy_stats)
    
    # 计算性能差异
    print("\n--- Performance Comparison ---")
    for noise in noise_levels:
        print(f"\nNoise Level {noise}:")
        
        # Makespan比较
        train_makespan = makespan_comp_df[
            (makespan_comp_df['data_type'] == 'Training') & 
            (makespan_comp_df['noise_level'] == noise)
        ]['makespan'].mean()
        
        gen_makespan = makespan_comp_df[
            (makespan_comp_df['data_type'] == 'Generalization') & 
            (makespan_comp_df['noise_level'] == noise)
        ]['makespan'].mean()
        
        if not np.isnan(train_makespan) and not np.isnan(gen_makespan):
            diff_makespan = ((gen_makespan - train_makespan) / train_makespan) * 100
            print(f"  Makespan: Training={train_makespan:.2f}, Generalization={gen_makespan:.2f}, "
                  f"Diff={diff_makespan:+.2f}%")
        
        # Energy比较
        train_energy = energy_comp_df[
            (energy_comp_df['data_type'] == 'Training') & 
            (energy_comp_df['noise_level'] == noise)
        ]['energy'].mean()
        
        gen_energy = energy_comp_df[
            (energy_comp_df['data_type'] == 'Generalization') & 
            (energy_comp_df['noise_level'] == noise)
        ]['energy'].mean()
        
        if not np.isnan(train_energy) and not np.isnan(gen_energy):
            diff_energy = ((gen_energy - train_energy) / train_energy) * 100
            print(f"  Energy: Training={train_energy:.2f}, Generalization={gen_energy:.2f}, "
                  f"Diff={diff_energy:+.2f}%")

# 更新主执行部分
if __name__ == "__main__":
    print("开始对比分析（不包括noise_level=0）...")
    
    # 创建箱线图对比
    print("\n1. 创建Training vs Generalization箱线图...")
    create_comparison_boxplots()
    
    # 创建小提琴图对比
    print("\n2. 创建Training vs Generalization小提琴图...")
    create_comparison_violin_plots()
    
    # 创建组合的小提琴图和散点图
    print("\n3. 创建组合小提琴图（训练分布）和散点图（泛化数据）...")
    create_combined_violin_boxplot()
    
    # 创建密度分布图
    print("\n4. 创建分布密度图...")
    create_ridge_plots()
    
    print("\n所有分析完成！")