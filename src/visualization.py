import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src import data_processing as dp

def plot_missing_summary(missing_summary, title="Tỷ lệ giá trị thiếu theo cột"):
    """
    Tạo biểu đồ cột thể hiện tỷ lệ phần trăm giá trị thiếu cho các cột.

    Args:
        missing_summary (dict): Từ điển từ dp.get_missing_summary.
        title (str): Tiêu đề của biểu đồ.
    """
    # Lọc các cột không có giá trị thiếu
    missing_info = {k: v for k, v in missing_summary.items() if v['missing_count'] > 0}
    if not missing_info:
        print("Không có giá trị thiếu để hiển thị.")
        return

    columns = list(missing_info.keys())
    missing_counts = [missing_info[col]['missing_count'] for col in columns]
    missing_percentages = [missing_info[col]['missing_percentage'] for col in columns]

    # Dữ liệu đã được sắp xếp từ get_missing_summary
    
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x=missing_percentages, y=columns, orient='h')
    plt.title(title, fontsize=16)
    plt.xlabel('Tỷ lệ phần trăm thiếu (%)', fontsize=12)
    plt.ylabel('Đặc trưng', fontsize=12)

    for i, (count, percentage) in enumerate(zip(missing_counts, missing_percentages)):
        ax.text(percentage + 0.5, i, f'{count} ({percentage:.2f}%)', color='black', va='center')

    plt.xlim(right=max(missing_percentages) * 1.15 if missing_percentages else 10)
    plt.tight_layout()
    plt.show()


def plot_target_distribution(target_counts, target_percentages):
    """
    Tạo biểu đồ cột thể hiện phân phối của biến mục tiêu.

    Args:
        target_counts (dict): Từ điển chứa số lượng của mỗi lớp mục tiêu.
        target_percentages (dict): Từ điển chứa tỷ lệ phần trăm của mỗi lớp mục tiêu.
    """
    labels = list(target_counts.keys())
    counts = list(target_counts.values())
    percentages = list(target_percentages.values())

    fig, ax = plt.subplots()
    bars = ax.bar(labels, counts)

    ax.set_title('Phân phối của Biến Mục tiêu', fontsize=16)
    ax.set_xlabel('Lớp Mục tiêu', fontsize=12)
    ax.set_ylabel('Số lượng', fontsize=12)

    for bar, percentage in zip(bars, percentages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height,
                f'{height}\n({percentage:.2f}%)',
                ha='center', va='bottom', fontsize=10)

    plt.show()

def plot_numerical_vs_target(data, numerical_column, target_column='target'):
    """
    Tạo biểu đồ boxplot hoặc violin plot cho đặc trưng số so với mục tiêu.

    Args:
        data (np.ndarray): Mảng cấu trúc NumPy chứa dữ liệu.
        numerical_column (str): Tên của cột số.
        target_column (str): Tên của cột mục tiêu.
    """
    # Convert to a dictionary that seaborn can understand
    plot_data = {
        numerical_column: data[numerical_column],
        target_column: data[target_column]
    }
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=target_column, y=numerical_column, data=plot_data)
    plt.title(f'{numerical_column} vs {target_column}', fontsize=16)
    plt.xlabel(target_column, fontsize=12)
    plt.ylabel(numerical_column, fontsize=12)
    plt.show()

def plot_target_rate_by_missing(missing_comparison_data, column_name):
    """
    Tạo biểu đồ cột so sánh tỷ lệ mục tiêu cho các bản ghi có và không có giá trị thiếu.

    Args:
        missing_comparison_data (dict): Dữ liệu so sánh giá trị thiếu (output từ dp.compare_target_rate_by_missing).
        column_name (str): Tên của cột được phân tích.
    """
    labels = ['Có giá trị thiếu', 'Không có giá trị thiếu']
    target_rates = [missing_comparison_data['missing']['target_rate'],
                    missing_comparison_data['not_missing']['target_rate']]
    counts = [missing_comparison_data['missing']['count'],
              missing_comparison_data['not_missing']['count']]

    # Loại bỏ các giá trị NaN nếu không có dữ liệu
    valid_indices = ~np.isnan(target_rates)
    labels = np.array(labels)[valid_indices]
    target_rates = np.array(target_rates)[valid_indices]
    counts = np.array(counts)[valid_indices]

    if len(labels) == 0:
        print(f"Không có đủ dữ liệu để so sánh tỷ lệ mục tiêu cho cột {column_name}.")
        return

    plt.figure(figsize=(8, 6))
    ax = sns.barplot(x=labels, y=target_rates)
    plt.title(f'Tỷ lệ Target=1 khi có/không có giá trị thiếu trong {column_name}', fontsize=14)
    plt.xlabel('Trạng thái giá trị thiếu', fontsize=12)
    plt.ylabel('Tỷ lệ Target=1', fontsize=12)
    plt.ylim(0, max(target_rates) * 1.2 if len(target_rates) > 0 else 0.5) # Đảm bảo trục y bắt đầu từ 0 và có khoảng trống

    for i, rate in enumerate(target_rates):
        ax.text(i, rate + 0.01, f'{rate:.2f} (n={counts[i]})', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.show()

def plot_numerical_distribution_and_boxplot(data, column_name):
    """
    Trực quan hóa phân phối của một biến số bằng histogram và box plot.
    """
    # Loại bỏ NaN
    col_data = data[column_name][~np.isnan(data[column_name])]
    
    plt.figure(figsize=(16, 6))

    # Histogram
    plt.subplot(1, 2, 1)
    sns.histplot(col_data, kde=True, bins=30)
    plt.title(f'Phân phối của {column_name} (Histogram)')
    plt.xlabel(column_name)
    plt.ylabel('Tần suất')

    # Box plot
    plt.subplot(1, 2, 2)
    sns.boxplot(x=col_data)
    plt.title(f'Phân phối của {column_name} (Box Plot)')
    plt.xlabel(column_name)

    plt.tight_layout()
    plt.show()

def plot_scaling_comparison(data, col1, col2):
    """
    Tạo scatter plot để so sánh trực quan thang đo của hai đặc trưng số.

    Args:
        data (np.ndarray): Mảng cấu trúc NumPy chứa dữ liệu.
        col1 (str): Tên cột thứ nhất.
        col2 (str): Tên cột thứ hai.
    """
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=data[col1], y=data[col2], alpha=0.5)
    plt.title(f'So sánh thang đo giữa {col1} và {col2}', fontsize=16)
    plt.xlabel(f'{col1} (Thang đo: {data[col1].min():.2f} - {data[col1].max():.2f})', fontsize=12)
    plt.ylabel(f'{col2} (Thang đo: {data[col2].min():.2f} - {data[col2].max():.2f})', fontsize=12)
    plt.grid(True)
    plt.show()

def plot_categorical_analysis(data, column_name, target_column='target', top_n=15):
    """
    Tạo hai biểu đồ cho một đặc trưng phân loại:
    1. Biểu đồ cột thể hiện phân phối của đặc trưng (top N).
    2. Biểu đồ cột thể hiện tỷ lệ mục tiêu theo từng loại của đặc trưng (top N).

    Args:
        data (np.ndarray): Mảng cấu trúc NumPy chứa dữ liệu.
        column_name (str): Tên của cột phân loại.
        target_column (str): Tên của cột mục tiêu.
        top_n (int): Số lượng các loại hàng đầu để hiển thị.
    """
    # Lấy phân phối của đặc trưng
    dist = dp.get_categorical_distribution(data, column_name)
    
    # Lấy tỷ lệ target theo đặc trưng
    target_rate = dp.calculate_target_rate_by_category(data, column_name, target_column)
    
    # Giới hạn ở top N
    categories_to_plot = dist['categories'][:top_n]
    counts_to_plot = dist['counts'][:top_n]
    
    # Sắp xếp rates và counts từ target_rate theo thứ tự của categories_to_plot
    ordered_rates = [target_rate[cat]['target_rate'] for cat in categories_to_plot]
    ordered_counts = [target_rate[cat]['count'] for cat in categories_to_plot]

    fig, axes = plt.subplots(1, 2, figsize=(18, max(6, top_n * 0.4)))

    # Biểu đồ 1: Phân phối của đặc trưng
    sns.barplot(x=counts_to_plot, y=categories_to_plot, ax=axes[0], orient='h', hue=categories_to_plot, legend=False, palette='viridis')
    axes[0].set_title(f'Phân phối của {column_name} (Top {top_n})', fontsize=14)
    axes[0].set_xlabel('Số lượng', fontsize=12)
    axes[0].set_ylabel(column_name, fontsize=12)
    for i, v in enumerate(counts_to_plot):
        axes[0].text(v, i, f' {v}', color='black', va='center')

    # Biểu đồ 2: Tỷ lệ target theo đặc trưng
    # Sắp xếp lại theo tỷ lệ target để dễ so sánh
    sorted_indices = np.argsort(ordered_rates)[::-1]
    sorted_categories = [categories_to_plot[i] for i in sorted_indices]
    sorted_rates = [ordered_rates[i] for i in sorted_indices]
    sorted_counts = [ordered_counts[i] for i in sorted_indices]

    sns.barplot(x=sorted_rates, y=sorted_categories, ax=axes[1], orient='h', hue=sorted_categories, legend=False, palette='plasma')
    axes[1].set_title(f'Tỷ lệ Target=1 theo {column_name} (Top {top_n})', fontsize=14)
    axes[1].set_xlabel('Tỷ lệ Target=1', fontsize=12)
    axes[1].set_ylabel('')
    axes[1].axvline(x=np.nanmean(data[target_column]), color='r', linestyle='--', label='Tỷ lệ Target trung bình')
    axes[1].legend()

    for i, (rate, count) in enumerate(zip(sorted_rates, sorted_counts)):
        axes[1].text(rate, i, f' {rate:.2f} (n={count})', color='black', va='center')

    plt.tight_layout()
    plt.show()

def plot_missing_value_heatmap(data):
    """
    Creates a heatmap to visualize the pattern of missing values using NumPy.
    
    Args:
        data (np.ndarray): NumPy structured array.
    """
    # Create a boolean matrix for missing values
    missing_matrix = []
    for name in data.dtype.names:
        column = data[name]
        if np.issubdtype(column.dtype, np.number):
            missing_matrix.append(np.isnan(column))
        else:
            # For string types, an empty string denotes missing
            missing_matrix.append(column == '')
            
    # Transpose to have columns as x-axis and rows as y-axis
    missing_matrix = np.array(missing_matrix).T
    
    plt.figure(figsize=(15, 10))
    sns.heatmap(missing_matrix, cbar=False, cmap='viridis', yticklabels=False)
    plt.title('Heatmap của các giá trị thiếu', fontsize=16)
    plt.xlabel('Cột', fontsize=12)
    plt.ylabel(f'Dòng dữ liệu (tổng số {data.shape[0]})', fontsize=12)
    # Set x-axis labels to column names
    plt.xticks(ticks=np.arange(len(data.dtype.names)) + 0.5, labels=data.dtype.names, rotation=90)
    plt.show()

def plot_correlation_heatmap(data, numerical_cols):
    """
    Calculates and plots the correlation matrix for numerical columns using NumPy.
    
    Args:
        data (np.ndarray): NumPy structured array.
        numerical_cols (list): List of numerical column names.
    """
    # Create a 2D array from the numerical columns
    num_data = np.array([data[col] for col in numerical_cols], dtype=np.float64).T
    
    # Handle NaNs for correlation calculation. A simple approach is to replace NaNs with the column mean.
    # This is a simplification for visualization purposes.
    col_means = np.nanmean(num_data, axis=0)
    nan_indices = np.where(np.isnan(num_data))
    # Replace NaNs with the mean of their respective columns
    num_data[nan_indices] = np.take(col_means, nan_indices[1])

    # Calculate correlation matrix
    corr_matrix = np.corrcoef(num_data, rowvar=False)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5,
                xticklabels=numerical_cols, yticklabels=numerical_cols)
    plt.title('Heatmap tương quan giữa các biến số', fontsize=16)
    plt.show()