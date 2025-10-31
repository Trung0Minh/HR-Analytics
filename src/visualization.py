import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import data_processing as dp

def plot_missing_heatmap(missing_matrix, column_names):
    """
    Trực quan hóa các mẫu dữ liệu bị thiếu bằng heatmap.

    Args:
        missing_matrix (np.ndarray): Ma trận boolean biểu thị các giá trị bị thiếu.
        column_names (list): Danh sách các tên cột.
    """
    plt.figure(figsize=(12, 8))
    sns.heatmap(missing_matrix, cbar=False, yticklabels=False, cmap='viridis')
    plt.title('Sơ đồ nhiệt các giá trị bị thiếu', fontsize=16)
    plt.xlabel('Các đặc trưng', fontsize=12)
    plt.xticks(ticks=np.arange(len(column_names)) + 0.5, labels=column_names, rotation=90)
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

def plot_numerical_distribution(data, column_name, bins=30):
    """
    Tạo biểu đồ histogram với KDE overlay cho các đặc trưng số.

    Args:
        data (np.ndarray): Mảng cấu trúc NumPy chứa dữ liệu.
        column_name (str): Tên của cột số.
        bins (int): Số lượng bins cho histogram.
    """
    column = data[column_name][~np.isnan(data[column_name])]
    plt.figure(figsize=(10, 6))
    sns.histplot(column, bins=bins, kde=True)
    plt.axvline(np.mean(column), color='red', linestyle='--', label=f'Mean: {np.mean(column):.2f}')
    plt.axvline(np.median(column), color='green', linestyle='-', label=f'Median: {np.median(column):.2f}')
    plt.title(f'Phân phối của {column_name}', fontsize=16)
    plt.xlabel(column_name, fontsize=12)
    plt.ylabel('Tần suất', fontsize=12)
    plt.legend()
    plt.show()

def plot_categorical_distribution(categories, counts, title, top_n=None):
    """
    Tạo biểu đồ cột ngang cho các đặc trưng phân loại.

    Args:
        categories (list): Danh sách các loại.
        counts (list): Danh sách số lượng tương ứng.
        title (str): Tiêu đề của biểu đồ.
        top_n (int, optional): Chỉ hiển thị top N loại. Mặc định là None.
    """
    if top_n:
        categories = categories[:top_n]
        counts = counts[:top_n]
    
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x=counts, y=categories, orient='h')
    plt.title(title, fontsize=16)
    plt.xlabel('Số lượng', fontsize=12)
    plt.ylabel('Loại', fontsize=12)

    # Thêm nhãn phần trăm
    total = sum(counts)
    for i, v in enumerate(counts):
        ax.text(v + 3, i + .25, f'{v} ({(v/total)*100:.1f}%)', color='blue', fontweight='bold')

    plt.show()

def plot_target_by_category(categories, target_rates, counts, feature_name):
    """
    Tạo biểu đồ cột thể hiện tỷ lệ mục tiêu cho mỗi loại.

    Args:
        categories (list): Danh sách các loại.
        target_rates (list): Danh sách tỷ lệ mục tiêu tương ứng.
        counts (list): Danh sách số lượng tương ứng.
        feature_name (str): Tên của đặc trưng.
    """
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x=target_rates, y=categories, orient='h')
    plt.title(f'Tỷ lệ Target theo {feature_name}', fontsize=16)
    plt.xlabel('Tỷ lệ Target', fontsize=12)
    plt.ylabel(feature_name, fontsize=12)

    # Thêm nhãn số lượng
    for i, v in enumerate(counts):
        ax.text(target_rates[i] + 0.01, i + .25, f'n={v}', color='black', fontweight='light')

    plt.show()

def plot_correlation_heatmap(data, numerical_columns):
    """
    Tạo heatmap tương quan cho các đặc trưng số.

    Args:
        data (np.ndarray): Mảng cấu trúc NumPy chứa dữ liệu.
        numerical_columns (list): Danh sách các cột số.
    """
    # Chuyển đổi dữ liệu có cấu trúc sang mảng 2D thông thường
    numerical_data = np.array([data[col] for col in numerical_columns]).T
    # Tính toán ma trận tương quan, xử lý các giá trị nan
    correlation_matrix = np.ma.corrcoef(np.ma.masked_invalid(numerical_data), rowvar=False)
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', xticklabels=numerical_columns, yticklabels=numerical_columns)
    plt.title('Heatmap Tương quan', fontsize=16)
    plt.show()

def plot_numerical_vs_target(data, numerical_column, target_column='target'):
    """
    Tạo biểu đồ boxplot hoặc violin plot cho đặc trưng số so với mục tiêu.

    Args:
        data (np.ndarray): Mảng cấu trúc NumPy chứa dữ liệu.
        numerical_column (str): Tên của cột số.
        target_column (str): Tên của cột mục tiêu.
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=target_column, y=numerical_column, data={numerical_column: data[numerical_column], target_column: data[target_column]})
    plt.title(f'{numerical_column} vs {target_column}', fontsize=16)
    plt.xlabel(target_column, fontsize=12)
    plt.ylabel(numerical_column, fontsize=12)
    plt.show()

def plot_boxplot_with_outliers(data, column_name):
    """
    Tạo biểu đồ boxplot để xác định các giá trị ngoại lệ.

    Args:
        data (np.ndarray): Mảng cấu trúc NumPy chứa dữ liệu.
        column_name (str): Tên của cột.
    """
    plt.figure(figsize=(8, 6))
    sns.boxplot(y=data[column_name])
    plt.title(f'Biểu đồ Boxplot cho {column_name}', fontsize=16)
    plt.ylabel(column_name, fontsize=12)
    plt.show()

def plot_feature_interaction_heatmap(interaction_data, feature1_name, feature2_name):
    """
    Tạo heatmap thể hiện tương tác giữa hai đặc trưng.

    Args:
        interaction_data (np.ndarray): Dữ liệu tương tác (bảng chéo).
        feature1_name (str): Tên của đặc trưng thứ nhất.
        feature2_name (str): Tên của đặc trưng thứ hai.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(interaction_data, annot=True, fmt='d', cmap='YlGnBu')
    plt.title(f'Tương tác giữa {feature1_name} và {feature2_name}', fontsize=16)
    plt.xlabel(feature2_name, fontsize=12)
    plt.ylabel(feature1_name, fontsize=12)
    plt.show()

def create_subplots_grid(plot_data_list, nrows, ncols, figsize=(15, 10)):
    """
    Tạo một lưới các subplot.

    Args:
        plot_data_list (list): Danh sách các bộ dữ liệu để vẽ.
        nrows (int): Số hàng trong lưới.
        ncols (int): Số cột trong lưới.
        figsize (tuple): Kích thước của hình.
    """
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    for i, ax in enumerate(axes.flat):
        if i < len(plot_data_list):
            # Thêm logic vẽ biểu đồ cụ thể ở đây
            pass
    plt.tight_layout()
    plt.show()

def plot_categorical_analysis(data, column_name, target_column='target'):
    """
    Tạo hai biểu đồ cho một đặc trưng phân loại:
    1. Biểu đồ cột thể hiện phân phối của đặc trưng.
    2. Biểu đồ cột thể hiện tỷ lệ mục tiêu theo từng loại của đặc trưng.

    Args:
        data (np.ndarray): Mảng cấu trúc NumPy chứa dữ liệu.
        column_name (str): Tên của cột phân loại.
        target_column (str): Tên của cột mục tiêu.
    """
    # Lấy phân phối của đặc trưng
    dist = dp.get_categorical_distribution(data, column_name)
    
    # Lấy tỷ lệ target theo đặc trưng
    target_rate = dp.calculate_target_rate_by_category(data, column_name, target_column)
    
    # Sắp xếp categories, rates, và counts từ target_rate theo thứ tự của dist['categories']
    ordered_categories = dist['categories']
    ordered_rates = [target_rate[cat]['target_rate'] for cat in ordered_categories]
    ordered_counts = [target_rate[cat]['count'] for cat in ordered_categories]

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Biểu đồ 1: Phân phối của đặc trưng
    sns.barplot(x=dist['counts'], y=dist['categories'], ax=axes[0], orient='h')
    axes[0].set_title(f'Phân phối của {column_name}', fontsize=14)
    axes[0].set_xlabel('Số lượng', fontsize=12)
    axes[0].set_ylabel(column_name, fontsize=12)
    for i, v in enumerate(dist['counts']):
        axes[0].text(v + 3, i + .25, str(v), color='blue', fontweight='bold')

    # Biểu đồ 2: Tỷ lệ target theo đặc trưng
    sns.barplot(x=ordered_rates, y=ordered_categories, ax=axes[1], orient='h')
    axes[1].set_title(f'Tỷ lệ Target theo {column_name}', fontsize=14)
    axes[1].set_xlabel('Tỷ lệ Target', fontsize=12)
    axes[1].set_ylabel('')

    for i, v in enumerate(ordered_rates):
        axes[1].text(v + 0.01, i + .25, f'n={ordered_counts[i]}', color='black', fontweight='light')

    plt.tight_layout()
    plt.show()

def plot_top_n_cities(cities, counts, n=10):
    """
    Tạo biểu đồ cột cho top N thành phố.

    Args:
        cities (list): Danh sách tên thành phố.
        counts (list): Danh sách số lượng ứng viên.
        n (int): Số lượng thành phố hàng đầu để hiển thị.
    """
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x=counts, y=cities, orient='h')
    plt.title(f'Top {n} thành phố theo số lượng ứng viên', fontsize=16)
    plt.xlabel('Số lượng ứng viên', fontsize=12)
    plt.ylabel('Thành phố', fontsize=12)

    for i, v in enumerate(counts):
        ax.text(v + 3, i + .25, str(v), color='blue', fontweight='bold')

    plt.show()

def plot_city_development_index_by_city(city_dev_index):
    """
    Tạo biểu đồ cột thể hiện chỉ số phát triển của mỗi thành phố.

    Args:
        city_dev_index (dict): Từ điển với các thành phố làm khóa và chỉ số phát triển làm giá trị.
    """
    sorted_cities = sorted(city_dev_index.items(), key=lambda item: item[1], reverse=True)
    cities = [item[0] for item in sorted_cities]
    indices = [item[1] for item in sorted_cities]

    plt.figure(figsize=(12, 20))
    ax = sns.barplot(x=indices, y=cities, orient='h')
    plt.title('Chỉ số phát triển thành phố', fontsize=16)
    plt.xlabel('Chỉ số phát triển', fontsize=12)
    plt.ylabel('Thành phố', fontsize=12)

    for i, v in enumerate(indices):
        ax.text(v + 0.01, i + .25, f'{v:.3f}', color='blue', fontweight='bold')

    plt.show()

def plot_missing_value_correlation(data):
    """
    Trực quan hóa tương quan của các giá trị bị thiếu.

    Args:
        data (np.ndarray): Mảng cấu trúc NumPy chứa dữ liệu.
    """
    missing_corr = dp.get_missing_value_correlation(data)
    plt.figure(figsize=(12, 8))
    sns.heatmap(missing_corr, annot=True, cmap='coolwarm', xticklabels=data.dtype.names, yticklabels=data.dtype.names)
    plt.title('Tương quan giá trị thiếu', fontsize=16)
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
    plt.title(f'Tỷ lệ Target khi có/không có giá trị thiếu trong {column_name}', fontsize=14)
    plt.xlabel('Trạng thái giá trị thiếu', fontsize=12)
    plt.ylabel('Tỷ lệ Target', fontsize=12)
    plt.ylim(0, max(target_rates) * 1.2 if len(target_rates) > 0 else 0.5) # Đảm bảo trục y bắt đầu từ 0 và có khoảng trống

    for i, rate in enumerate(target_rates):
        ax.text(i, rate + 0.01, f'{rate:.2f} (n={counts[i]})', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.show()

def plot_categorical_missing_analysis(missing_info, title="Phân tích giá trị thiếu của các đặc trưng phân loại"):
    """
    Tạo biểu đồ cột thể hiện số lượng và tỷ lệ phần trăm giá trị thiếu cho các đặc trưng phân loại.

    Args:
        missing_info (dict): Từ điển chứa thông tin về giá trị thiếu cho mỗi cột.
                              (Output từ dp.analyze_categorical_missing)
        title (str): Tiêu đề của biểu đồ.
    """
    if not missing_info:
        print("Không có thông tin giá trị thiếu để hiển thị.")
        return

    columns = list(missing_info.keys())
    missing_counts = [missing_info[col]['missing_count'] for col in columns]
    missing_percentages = [missing_info[col]['missing_percentage'] for col in columns]

    # Sắp xếp theo tỷ lệ phần trăm giảm dần
    sorted_indices = np.argsort(missing_percentages)[::-1]
    columns = [columns[i] for i in sorted_indices]
    missing_counts = [missing_counts[i] for i in sorted_indices]
    missing_percentages = [missing_percentages[i] for i in sorted_indices]

    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x=missing_percentages, y=columns, orient='h')
    plt.title(title, fontsize=16)
    plt.xlabel('Tỷ lệ phần trăm thiếu (%)', fontsize=12)
    plt.ylabel('Đặc trưng phân loại', fontsize=12)

    for i, (count, percentage) in enumerate(zip(missing_counts, missing_percentages)):
        ax.text(percentage + 0.5, i, f'{count} ({percentage:.2f}%)', color='black', va='center')

    plt.xlim(right=max(missing_percentages) * 1.1 if missing_percentages else 10)
    plt.tight_layout()
    plt.show()

def plot_target_rate_by_outliers(outlier_comparison_data, column_name):
    """
    Tạo biểu đồ cột so sánh tỷ lệ mục tiêu cho các bản ghi có và không có ngoại lệ.

    Args:
        outlier_comparison_data (dict): Dữ liệu so sánh ngoại lệ (output từ dp.compare_target_rate_by_outliers).
        column_name (str): Tên của cột được phân tích.
    """
    labels = ['Có ngoại lệ', 'Không có ngoại lệ']
    target_rates = [outlier_comparison_data['outliers']['target_rate'],
                    outlier_comparison_data['not_outliers']['target_rate']]
    counts = [outlier_comparison_data['outliers']['count'],
              outlier_comparison_data['not_outliers']['count']]

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
    plt.title(f'Tỷ lệ Target khi có/không có ngoại lệ trong {column_name}', fontsize=14)
    plt.xlabel('Trạng thái ngoại lệ', fontsize=12)
    plt.ylabel('Tỷ lệ Target', fontsize=12)
    plt.ylim(0, max(target_rates) * 1.2 if len(target_rates) > 0 else 0.5)

    for i, rate in enumerate(target_rates):
        ax.text(i, rate + 0.01, f'{rate:.2f} (n={counts[i]})', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.show()

def plot_profile_comparison(profiles, numerical_cols, categorical_cols):
    """
    Trực quan hóa so sánh hồ sơ giữa các nhóm target=0 và target=1.

    Args:
        profiles (dict): Thông tin hồ sơ từ dp.compare_profiles_by_target.
        numerical_cols (list): Danh sách các cột số để trực quan hóa.
        categorical_cols (list): Danh sách các cột phân loại để trực quan hóa.
    """
    # Plot numerical summaries
    if numerical_cols:
        print("\n--- So sánh đặc trưng số ---")
        for col in numerical_cols:
            if col in profiles[0]['numerical_summary'] and col in profiles[1]['numerical_summary']:
                means = [profiles[0]['numerical_summary'][col]['mean'],
                         profiles[1]['numerical_summary'][col]['mean']]
                stds = [profiles[0]['numerical_summary'][col]['std'],
                        profiles[1]['numerical_summary'][col]['std']]
                labels = ['Target = 0', 'Target = 1']

                plt.figure(figsize=(7, 5))
                sns.barplot(x=labels, y=means, yerr=stds, capsize=0.1)
                plt.title(f'Trung bình của {col} theo Target', fontsize=14)
                plt.ylabel(f'Trung bình {col}', fontsize=12)
                plt.show()

    # Plot categorical distributions
    if categorical_cols:
        print("\n--- So sánh đặc trưng phân loại ---")
        for col in categorical_cols:
            if col in profiles[0]['categorical_distribution'] and col in profiles[1]['categorical_distribution']:
                dist_0 = profiles[0]['categorical_distribution'][col]
                dist_1 = profiles[1]['categorical_distribution'][col]

                categories = sorted(list(set(list(dist_0.keys()) + list(dist_1.keys()))))
                values_0 = [dist_0.get(cat, 0) for cat in categories]
                values_1 = [dist_1.get(cat, 0) for cat in categories]

                x = np.arange(len(categories))
                width = 0.35

                fig, ax = plt.subplots(figsize=(10, 6))
                rects1 = ax.bar(x - width/2, values_0, width, label='Target = 0')
                rects2 = ax.bar(x + width/2, values_1, width, label='Target = 1')

                ax.set_ylabel('Tỷ lệ phần trăm (%)', fontsize=12)
                ax.set_title(f'Phân phối của {col} theo Target', fontsize=14)
                ax.set_xticks(x)
                ax.set_xticklabels(categories, rotation=45, ha="right")
                ax.legend()
                plt.tight_layout()
                plt.show()    

def plot_multi_column_missing_patterns(missing_patterns, top_n=10):
    """
    Trực quan hóa các mẫu thiếu đa cột phổ biến nhất.

    Args:
        missing_patterns (dict): Từ điển các mẫu thiếu và số lượng của chúng 
                                 (output từ dp.analyze_multi_column_missing_patterns).
        top_n (int): Số lượng mẫu hàng đầu để hiển thị.
    """
    if not missing_patterns:
        print("Không tìm thấy mẫu thiếu đa cột nào.")
        return

    # Lấy top N mẫu
    top_patterns = dict(list(missing_patterns.items())[:top_n])
    
    pattern_labels = [' & '.join(cols) for cols in top_patterns.keys()]
    counts = list(top_patterns.values())

    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x=counts, y=pattern_labels, orient='h')
    plt.title(f'Top {top_n} mẫu thiếu đa cột phổ biến nhất', fontsize=16)
    plt.xlabel('Số lượng bản ghi', fontsize=12)
    plt.ylabel('Các cột bị thiếu', fontsize=12)

    for i, v in enumerate(counts):
        ax.text(v + 3, i + .25, str(v), color='blue', fontweight='bold')

    plt.tight_layout()
    plt.show()