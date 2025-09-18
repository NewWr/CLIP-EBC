import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse
import sys

# 设置中文字体和图表样式
# 尝试多种中文字体，确保兼容性
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def format_number(value):
    """
    智能数值格式化函数
    如果数值是整数，则显示为整数；如果是小数，则保留两位小数
    """
    if pd.isna(value):
        return 'N/A'
    
    # 检查是否为数值类型
    if not isinstance(value, (int, float, np.integer, np.floating)):
        return str(value)
    
    # 检查是否为整数（包括浮点数但值为整数的情况）
    if isinstance(value, (int, np.integer)) or (isinstance(value, (float, np.floating)) and value.is_integer()):
        return f"{int(value)}"
    else:
        return f"{value:.1f}" # 保留一位小数


def load_excel_data(file_path):
    """
    加载Excel文件数据
    """
    try:
        df = pd.read_excel(file_path)
        print(f"成功加载数据，共 {len(df)} 行，{len(df.columns)} 列")
        print(f"可用列名: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"加载文件失败: {e}")
        return None

def save_distribution_to_csv(df, column_name, csv_path=None):
    """
    将分布统计信息保存到CSV文件
    """
    if column_name not in df.columns:
        print(f"错误：列 '{column_name}' 不存在于数据中")
        return
    
    # 移除缺失值
    data = df[column_name].dropna()
    
    if len(data) == 0:
        print(f"警告：列 '{column_name}' 没有有效数据")
        return
    
    # 判断数据类型
    is_numeric = pd.api.types.is_numeric_dtype(data)
    
    if is_numeric:
        # 数值型数据：直接统计每个数值的出现次数
        value_counts = data.value_counts().sort_index()
        distribution_stats = []
        
        cumulative_count = 0
        for value, count in value_counts.items():
            cumulative_count += count
            percentage = (count / len(data)) * 100
            cumulative_percentage = (cumulative_count / len(data)) * 100
            
            distribution_stats.append({
                '数值': format_number(value),
                '数量': count,
                '百分比': f'{percentage:.2f}%',
                '累计数量': cumulative_count,
                '累计百分比': f'{cumulative_percentage:.2f}%'
            })
        
        distribution_df = pd.DataFrame(distribution_stats)
        
        # 添加总体统计信息
        summary_stats = pd.DataFrame({
            '统计指标': ['总数', '缺失值', '唯一值数量', '均值', '标准差', '中位数', '最小值', '最大值', '25%分位数', '75%分位数'],
            '数值': [
                len(data),
                df[column_name].isnull().sum(),
                data.nunique(),
                format_number(data.mean()),
                format_number(data.std()),
                format_number(data.median()),
                format_number(data.min()),
                format_number(data.max()),
                format_number(data.quantile(0.25)),
                format_number(data.quantile(0.75))
            ]
        })
    
    else:
        # 分类型数据：值计数统计
        value_counts = data.value_counts()
        distribution_stats = []
        
        cumulative_count = 0
        for value, count in value_counts.items():
            cumulative_count += count
            percentage = (count / len(data)) * 100
            cumulative_percentage = (cumulative_count / len(data)) * 100
            
            distribution_stats.append({
                '类别值': str(value),
                '数量': count,
                '百分比': f'{percentage:.2f}%',
                '累计数量': cumulative_count,
                '累计百分比': f'{cumulative_percentage:.2f}%'
            })
        
        distribution_df = pd.DataFrame(distribution_stats)
        
        # 添加总体统计信息
        summary_stats = pd.DataFrame({
            '统计指标': ['总数', '缺失值', '唯一值数量', '最频繁的值', '最频繁值的数量'],
            '数值': [
                len(data),
                df[column_name].isnull().sum(),
                data.nunique(),
                data.mode().iloc[0] if not data.mode().empty else 'N/A',
                value_counts.iloc[0] if len(value_counts) > 0 else 0
            ]
        })
    
    # 保存CSV文件
    if csv_path:
        csv_filename = csv_path
    else:
        csv_filename = f"{column_name}_distribution_stats.csv"
    
    # 使用ExcelWriter保存多个sheet到一个文件，或者分别保存
    try:
        # 保存分布统计
        distribution_df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
        
        # 保存总体统计到另一个文件
        summary_filename = csv_filename.replace('.csv', '_summary.csv')
        summary_stats.to_csv(summary_filename, index=False, encoding='utf-8-sig')
        
        print(f"\n分布统计已保存为: {csv_filename}")
        print(f"总体统计已保存为: {summary_filename}")
        
        # 显示前几行预览
        print(f"\n=== {column_name} 分布统计预览 ===")
        print(distribution_df.head(10))
        print(f"\n=== {column_name} 总体统计 ===")
        print(summary_stats)
        
    except Exception as e:
        print(f"保存CSV文件失败: {e}")

def plot_column_distribution(df, column_name, save_path=None, csv_output=True):
    """
    绘制指定列的分布图
    """
    if column_name not in df.columns:
        print(f"错误：列 '{column_name}' 不存在于数据中")
        print(f"可用列名: {list(df.columns)}")
        return
    
    # 移除缺失值
    data = df[column_name].dropna()
    
    if len(data) == 0:
        print(f"警告：列 '{column_name}' 没有有效数据")
        return
    
    # 判断数据类型
    is_numeric = pd.api.types.is_numeric_dtype(data)
    
    plt.figure(figsize=(12, 8))
    
    if is_numeric:
        # 数值型数据：绘制直方图和密度曲线
        plt.subplot(2, 2, 1)
        plt.hist(data, bins=30, alpha=0.7, color='skyblue', edgecolor='black', density=True)
        plt.title(f'{column_name} - 直方图')
        plt.xlabel(column_name)
        plt.ylabel('密度')
        
        # 添加统计信息
        mean_val = data.mean()
        std_val = data.std()
        median_val = data.median()
        plt.axvline(mean_val, color='red', linestyle='--', label=f'均值: {format_number(mean_val)}')
        plt.axvline(median_val, color='green', linestyle='--', label=f'中位数: {format_number(median_val)}')
        plt.legend()
        
        # 密度图
        plt.subplot(2, 2, 2)
        sns.kdeplot(data, fill=True, color='lightcoral')
        plt.title(f'{column_name} - 密度图')
        plt.xlabel(column_name)
        
        # 箱线图
        plt.subplot(2, 2, 3)
        plt.boxplot(data, vert=True)
        plt.title(f'{column_name} - 箱线图')
        plt.ylabel(column_name)
        
        # Q-Q图
        plt.subplot(2, 2, 4)
        from scipy import stats
        stats.probplot(data, dist="norm", plot=plt)
        plt.title(f'{column_name} - Q-Q图')
        
        # 打印统计信息
        print(f"\n=== {column_name} 统计信息 ===")
        print(f"数据类型: 数值型")
        print(f"有效数据量: {len(data)}")
        print(f"缺失值: {df[column_name].isnull().sum()}")
        print(f"均值: {format_number(mean_val)}")
        print(f"标准差: {format_number(std_val)}")
        print(f"中位数: {format_number(median_val)}")
        print(f"最小值: {format_number(data.min())}")
        print(f"最大值: {format_number(data.max())}")
        print(f"25%分位数: {format_number(data.quantile(0.25))}")
        print(f"75%分位数: {format_number(data.quantile(0.75))}")
        
    else:
        # 分类型数据：绘制条形图
        value_counts = data.value_counts().head(20)  # 只显示前20个最频繁的值
        
        plt.subplot(2, 1, 1)
        bars = plt.bar(range(len(value_counts)), value_counts.values, 
                      color='lightcoral', alpha=0.8, edgecolor='black')
        plt.xticks(range(len(value_counts)), 
                  [str(x)[:20] + '...' if len(str(x)) > 20 else str(x) 
                   for x in value_counts.index], 
                  rotation=45, ha='right')
        plt.title(f'{column_name} - 频次分布')
        plt.ylabel('频次')
        
        # 在条形图上添加数值标签
        for bar, count in zip(bars, value_counts.values):
            plt.text(bar.get_x() + bar.get_width()/2, 
                    bar.get_height() + max(value_counts)*0.01,
                    str(count), ha='center', va='bottom')
        
        # 饼图（如果类别不太多）
        if len(value_counts) <= 10:
            plt.subplot(2, 1, 2)
            plt.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%')
            plt.title(f'{column_name} - 比例分布')
        
        # 打印统计信息
        print(f"\n=== {column_name} 统计信息 ===")
        print(f"数据类型: 分类型")
        print(f"有效数据量: {len(data)}")
        print(f"缺失值: {df[column_name].isnull().sum()}")
        print(f"唯一值数量: {data.nunique()}")
        print(f"最频繁的值: {data.mode().iloc[0] if not data.mode().empty else 'N/A'}")
        print(f"\n前10个最频繁的值:")
        for value, count in value_counts.head(10).items():
            percentage = (count / len(data)) * 100
            print(f"  {value}: {count} ({percentage:.1f}%)")
    
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n图表已保存为: {save_path}")
    else:
        save_name = f"{column_name}_distribution.png"
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        print(f"\n图表已保存为: {save_name}")
    
    plt.show()
    
    # 保存CSV统计文件
    if csv_output:
        csv_path = save_path.replace('.png', '.csv') if save_path else None
        save_distribution_to_csv(df, column_name, csv_path)

def main():
    """
    主函数 - 支持命令行参数和交互式使用
    """
    parser = argparse.ArgumentParser(description='Excel数据列分布可视化工具')
    parser.add_argument('--file', '-f', type=str, default='/opt/DM/OCT/CLIP_Code/CLIP-EBC/GPTprompts/Bins/ukb_train_val_set.xlsx', help='Excel文件路径')
    parser.add_argument('--column', '-c', type=str, default='hba1c', help='要分析的列名')
    parser.add_argument('--output', '-o', type=str, default='/opt/DM/OCT/CLIP_Code/CLIP-EBC/GPTprompts/Bins/hba1c.png', help='输出图片路径')
    parser.add_argument('--no-csv', action='store_true', help='不生成CSV统计文件')
    
    args = parser.parse_args()
    
    # 如果没有提供命令行参数，使用交互式模式
    if not args.file or not args.column:
        print("=== Excel数据列分布可视化工具 ===")
        
        # 获取Excel文件
        if args.file:
            excel_file = args.file
        else:
            # 显示当前目录下的Excel文件
            current_dir = Path('.')
            excel_files = list(current_dir.glob('*.xlsx')) + list(current_dir.glob('*.xls'))
            
            if excel_files:
                print("\n当前目录下的Excel文件:")
                for i, file in enumerate(excel_files):
                    print(f"{i+1}. {file.name}")
                
                try:
                    choice = int(input("\n请选择文件编号 (或输入0手动输入路径): "))
                    if choice == 0:
                        excel_file = input("请输入Excel文件路径: ")
                    elif 1 <= choice <= len(excel_files):
                        excel_file = excel_files[choice-1]
                    else:
                        print("无效选择")
                        return
                except ValueError:
                    print("无效输入")
                    return
            else:
                excel_file = input("请输入Excel文件路径: ")
        
        # 加载数据
        df = load_excel_data(excel_file)
        if df is None:
            return
        
        # 获取列名
        if args.column:
            column_name = args.column
        else:
            print("\n请选择要分析的列:")
            for i, col in enumerate(df.columns):
                print(f"{i+1}. {col}")
            
            try:
                choice = int(input("\n请输入列编号: "))
                if 1 <= choice <= len(df.columns):
                    column_name = df.columns[choice-1]
                else:
                    print("无效选择")
                    return
            except ValueError:
                print("无效输入")
                return
    else:
        # 使用命令行参数
        excel_file = args.file
        column_name = args.column
        
        # 加载数据
        df = load_excel_data(excel_file)
        if df is None:
            return
    
    # 生成可视化
    print(f"\n正在分析列: {column_name}")
    plot_column_distribution(df, column_name, args.output, csv_output=not args.no_csv)

if __name__ == "__main__":
    main()