import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
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
    ax.set_xlabel('Nhãn', fontsize=12)
    ax.set_ylabel('Số lượng', fontsize=12)

    for bar, percentage in zip(bars, percentages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height,
                f'{height}\n({percentage:.2f}%)',
                ha='center', va='bottom', fontsize=10)

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

def plot_categorical_analysis(data, column_name, target_column='target', top_n=None):
    """
    Tạo hai biểu đồ cho một đặc trưng phân loại:
    1. Biểu đồ cột thể hiện phân phối của đặc trưng.
    2. Biểu đồ cột thể hiện tỷ lệ mục tiêu theo từng loại của đặc trưng.

    Args:
        data (np.ndarray): Mảng cấu trúc NumPy chứa dữ liệu.
        column_name (str): Tên của cột phân loại.
        target_column (str): Tên của cột mục tiêu.
        top_n (int, optional): Chỉ hiển thị top N danh mục có tần suất cao nhất. Mặc định là None (hiển thị tất cả).
    """
    # Lấy phân phối của đặc trưng
    dist = dp.get_categorical_distribution(data, column_name)
    
    # Lấy tỷ lệ target theo đặc trưng
    target_rate = dp.calculate_target_rate_by_category(data, column_name, target_column)
    
    # Lấy tất cả hoặc top N danh mục
    categories_to_plot = dist['categories']
    counts_to_plot = dist['counts']
    
    if top_n is not None and top_n < len(categories_to_plot):
        categories_to_plot = categories_to_plot[:top_n]
        counts_to_plot = counts_to_plot[:top_n]

    # Sắp xếp rates và counts từ target_rate theo thứ tự của categories_to_plot
    ordered_rates = [target_rate[cat]['target_rate'] for cat in categories_to_plot]
    ordered_counts = [target_rate[cat]['count'] for cat in categories_to_plot]

    # Tạo một bảng màu nhất quán cho các danh mục
    palette = sns.color_palette('viridis', len(categories_to_plot))
    color_map = {cat: color for cat, color in zip(categories_to_plot, palette)}

    fig, axes = plt.subplots(1, 2, figsize=(18, max(6, len(categories_to_plot) * 0.4)))

    # Biểu đồ 1: Phân phối của đặc trưng
    axes[0].pie(counts_to_plot, labels=categories_to_plot, autopct='%1.1f%%', startangle=140, colors=[color_map[cat] for cat in categories_to_plot])
    axes[0].set_title(f'Phân phối của {column_name}', fontsize=14)

    # Biểu đồ 2: Tỷ lệ target theo đặc trưng
    # Sắp xếp lại theo tỷ lệ target để dễ so sánh
    sorted_indices = np.argsort(ordered_rates)[::-1]
    sorted_categories = [categories_to_plot[i] for i in sorted_indices]
    sorted_rates = [ordered_rates[i] for i in sorted_indices]
    sorted_counts = [ordered_counts[i] for i in sorted_indices]

    sns.barplot(x=sorted_rates, y=sorted_categories, ax=axes[1], orient='h', hue=sorted_categories, palette=color_map, legend=False)
    axes[1].set_title(f'Tỷ lệ Target=1 theo {column_name}', fontsize=14)
    axes[1].set_xlabel('Tỷ lệ Target=1', fontsize=12)
    axes[1].set_ylabel('')
    axes[1].axvline(x=np.nanmean(data[target_column]), color='r', linestyle='--', label='Tỷ lệ Target trung bình')
    axes[1].legend()

    for i, (rate, count) in enumerate(zip(sorted_rates, sorted_counts)):
        axes[1].text(rate, i, f' {rate:.2f} (n={count})', color='black', va='center')

    plt.tight_layout()
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

def plot_target_rate_by_missing_grid(data, columns_to_compare):
    """
    Tạo một lưới biểu đồ cột so sánh tỷ lệ mục tiêu cho các bản ghi có và không có giá trị thiếu.

    Args:
        data (np.ndarray): Mảng cấu trúc NumPy chứa dữ liệu.
        columns_to_compare (list): Danh sách các cột để phân tích.
    """
    num_plots = len(columns_to_compare)
    if num_plots == 0:
        print("Không có cột nào để hiển thị.")
        return

    # Xác định kích thước lưới
    cols = 3  # Số cột trong lưới
    rows = (num_plots + cols - 1) // cols  # Tính số hàng cần thiết

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5), squeeze=False)
    axes = axes.flatten() # Làm phẳng mảng axes để dễ dàng lặp

    for i, col_name in enumerate(columns_to_compare):
        ax = axes[i]
        missing_comparison_data = dp.compare_target_rate_by_missing(data, col_name)

        labels = ['Có giá trị thiếu', 'Không có giá trị thiếu']
        target_rates = [missing_comparison_data['missing']['target_rate'],
                        missing_comparison_data['not_missing']['target_rate']]
        counts = [missing_comparison_data['missing']['count'],
                  missing_comparison_data['not_missing']['count']]

        # Loại bỏ các giá trị NaN nếu không có dữ liệu
        valid_indices = ~np.isnan(target_rates)
        valid_labels = np.array(labels)[valid_indices]
        valid_target_rates = np.array(target_rates)[valid_indices]
        valid_counts = np.array(counts)[valid_indices]

        if len(valid_labels) == 0:
            ax.set_title(f'{col_name}\n(Không có dữ liệu)')
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        sns.barplot(x=valid_labels, y=valid_target_rates, ax=ax)
        ax.set_title(f'Tỷ lệ Target=1 trong {col_name}', fontsize=12)
        ax.set_xlabel('Trạng thái giá trị thiếu', fontsize=10)
        ax.set_ylabel('Tỷ lệ Target=1', fontsize=10)
        
        # Đảm bảo trục y bắt đầu từ 0 và có khoảng trống
        if len(valid_target_rates) > 0:
            ax.set_ylim(0, max(valid_target_rates) * 1.2)
        else:
            ax.set_ylim(0, 0.5)

        for j, rate in enumerate(valid_target_rates):
            ax.text(j, rate + 0.01, f'{rate:.2f}\n(n={valid_counts[j]})', ha='center', va='bottom', fontsize=9)

    # Ẩn các subplot không sử dụng
    for i in range(num_plots, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout(pad=3.0)
    plt.show()

def plot_numerical_vs_target_grid(data, numerical_columns, target_column='target'):
    """
    Tạo một lưới các biểu đồ boxplot cho các đặc trưng số so với mục tiêu.

    Args:
        data (np.ndarray): Mảng cấu trúc NumPy chứa dữ liệu.
        numerical_columns (list): Danh sách các cột số để vẽ.
        target_column (str): Tên của cột mục tiêu.
    """
    num_plots = len(numerical_columns)
    if num_plots == 0:
        print("Không có cột số nào để hiển thị.")
        return

    # Xác định kích thước lưới, ví dụ 2 cột
    cols = 2
    rows = (num_plots + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 8, rows * 6), squeeze=False)
    axes = axes.flatten()

    for i, num_col in enumerate(numerical_columns):
        ax = axes[i]
        
        # Convert to a dictionary that seaborn can understand for the specific plot
        plot_data = {
            num_col: data[num_col],
            target_column: data[target_column]
        }
        
        sns.boxplot(x=target_column, y=num_col, data=plot_data, ax=ax)
        ax.set_title(f'{num_col} vs {target_column}', fontsize=14)
        ax.set_xlabel(target_column, fontsize=12)
        ax.set_ylabel(num_col, fontsize=12)

    # Ẩn các subplot không sử dụng
    for i in range(num_plots, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout(pad=3.0)
    plt.show()

def plot_experience_interaction(interaction_data):
    """
    Plots the interaction between experience and another categorical variable.

    Args:
        interaction_data (dict): The output from dp.analyze_experience_interaction.
    """
    summary = interaction_data['summary']
    unique_exp_groups = interaction_data['exp_groups']
    unique_hue_values = interaction_data['hue_values']
    overall_mean = interaction_data['overall_mean']
    hue_column_name = interaction_data['hue_column_name']
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(unique_exp_groups))
    width = 0.15 # Width of each bar
    
    num_hues = len(unique_hue_values)

    # Draw the bars for each hue
    for i, (hue_val, rates) in enumerate(summary.items()):
        offset = width * (i - (num_hues - 1) / 2)
        ax.bar(x + offset, rates, width, label=hue_val)

    # Add the overall mean line
    ax.axhline(y=overall_mean, color='r', linestyle='--', label=f'Tỷ lệ trung bình ({overall_mean:.2f})')

    # Configure the plot
    ax.set_ylabel('Tỷ lệ Tìm việc (Target=1)')
    ax.set_title(f'Tương tác giữa Kinh nghiệm và {hue_column_name.replace("_", " ").title()} đến Tỷ lệ Tìm việc')
    ax.set_xticks(x)
    ax.set_xticklabels(unique_exp_groups, rotation=45, ha='right')
    ax.legend(title=hue_column_name.replace("_", " ").title())

    plt.tight_layout()
    plt.show()

def plot_bivariate_categorical(analysis_data):
    """
    Plots the interaction between two categorical variables from pre-analyzed data.

    Args:
        analysis_data (dict): The output from dp.analyze_bivariate_categorical.
    """
    summary = analysis_data['summary']
    x_values = analysis_data['x_values']
    hue_values = analysis_data['hue_values']
    overall_mean = analysis_data['overall_mean']
    x_col_name = analysis_data['x_col_name']
    hue_col_name = analysis_data['hue_col_name']
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(x_values))
    # Adjust width based on number of categories to avoid clutter
    num_hues = len(hue_values)
    width = 0.8 / (num_hues + 1)

    # Draw the bars for each hue value
    for i, (hue_val, rates) in enumerate(summary.items()):
        offset = width * (i - (num_hues - 1) / 2)
        # Convert rates to numpy array for nan handling
        rates_np = np.array(rates, dtype=float)
        ax.bar(x + offset, rates_np, width, label=hue_val)

    # Add the overall mean line
    ax.axhline(y=overall_mean, color='r', linestyle='--', label=f'Tỷ lệ trung bình ({overall_mean:.2f})')

    # Configure the plot
    ax.set_ylabel('Tỷ lệ Tìm việc (Target=1)')
    ax.set_title(f'Tương tác giữa {x_col_name.replace("_", " ").title()} và {hue_col_name.replace("_", " ").title()}')
    ax.set_xticks(x)
    ax.set_xticklabels(x_values, rotation=45, ha='right')
    ax.legend(title=hue_col_name.replace("_", " ").title())
    
    ax.set_ylim(0, max(ax.get_ylim()[1], overall_mean * 1.2)) # Ensure y-axis has some space

    plt.tight_layout()
    plt.show()

def plot_numerical_categorical_interaction(data, num_col, cat_col, target_col='target'):
    """
    Plots the interaction between a numerical and a categorical variable against a target.
    Uses seaborn's catplot for a faceted box plot view.

    Args:
        data (np.ndarray): The dataset.
        num_col (str): The numerical column.
        cat_col (str): The categorical column.
        target_col (str): The target column.
    """
    # Seaborn works best with pandas DataFrames
    import pandas as pd
    
    df = pd.DataFrame(data)
    
    # Clean up data for plotting
    df[target_col] = pd.to_numeric(df[target_col])
    df = df[df[cat_col] != ''] # Remove empty category
    
    # For experience, let's use the grouped version for clarity
    if cat_col == 'experience':
        exp_numeric = df['experience'].replace({'>20': '21', '<1': '0', '': '-1'}).astype(int)
        df['experience_group'] = pd.cut(exp_numeric, 
                                        bins=[-2, -1, 0, 5, 10, 20, 21], 
                                        labels=['Missing', 'Fresher (<1)', 'Junior (1-5)', 'Mid-level (6-10)', 'Senior (11-20)', 'Expert (>20)'])
        cat_col = 'experience_group'


    sns.catplot(data=df, x=target_col, y=num_col, col=cat_col, kind='box', col_wrap=4, height=4, aspect=1.2)
    
    plt.suptitle(f'Phân phối của {num_col.replace("_", " ").title()} theo {cat_col.replace("_", " ").title()} và Target', y=1.03, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

def _calculate_confusion_matrix(y_true, y_pred, classes):
    """Helper to calculate a confusion matrix using numpy."""
    num_classes = len(classes)
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    class_map = {cls: i for i, cls in enumerate(classes)}
    
    for i in range(len(y_true)):
        true_label = y_true[i]
        pred_label = y_pred[i]
        matrix[class_map[true_label], class_map[pred_label]] += 1
        
    return matrix

def plot_evaluation_visuals(y_true, y_pred, y_pred_proba, dataset_name=""):
    """
    Computes and plots a figure with two subplots: a confusion matrix and an ROC curve.
    
    Args:
        y_true (np.array): Ground truth labels.
        y_pred (np.array): Predicted labels.
        y_pred_proba (np.array): Predicted probabilities for each class.
        dataset_name (str): Name of the dataset to be used in titles.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # --- Subplot 1: Confusion Matrix ---
    classes = np.unique(np.concatenate((y_true, y_pred)))
    cm = _calculate_confusion_matrix(y_true, y_pred, classes)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes, ax=ax1)
    ax1.set_title(f'Confusion Matrix - {dataset_name}', fontsize=14)
    ax1.set_ylabel('Actual Label', fontsize=12)
    ax1.set_xlabel('Predicted Label', fontsize=12)

    # --- Subplot 2: ROC Curve ---
    positive_class = 1
    y_scores = y_pred_proba[:, positive_class]

    thresholds = np.linspace(0, 1, 101)
    tpr_list = []
    fpr_list = []

    for thresh in thresholds:
        y_pred_thresh = (y_scores >= thresh).astype(int)
        
        tp = np.sum((y_true == positive_class) & (y_pred_thresh == positive_class))
        fp = np.sum((y_true != positive_class) & (y_pred_thresh == positive_class))
        tn = np.sum((y_true != positive_class) & (y_pred_thresh != positive_class))
        fn = np.sum((y_true == positive_class) & (y_pred_thresh != positive_class))

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    # Calculate AUC
    auc_score = -np.trapz(tpr_list, fpr_list)

    # Plotting ROC
    ax2.plot(fpr_list, tpr_list, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate', fontsize=12)
    ax2.set_ylabel('True Positive Rate', fontsize=12)
    ax2.set_title(f'ROC Curve - {dataset_name}', fontsize=14)
    ax2.legend(loc="lower right")
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_train_test_comparison(train_data, test_data):
    """
    Replicates the comparison visualization from the prompt using 4x3 grid.
    Includes all categorical/ordinal features.
    """
    if isinstance(train_data, str):
        train_data = dp.load_data(train_data)
    if isinstance(test_data, str):
        test_data = dp.load_data(test_data)

    background_color = "#fbfbfb"
    fig = plt.figure(figsize=(22, 20), dpi=150)
    fig.patch.set_facecolor(background_color)
    gs = fig.add_gridspec(4, 3)
    gs.update(wspace=0.35, hspace=0.27)

    axes = []
    for i in range(4):
        for j in range(3):
            axes.append(fig.add_subplot(gs[i, j]))

    # Apply formatting to all axes
    for ax in axes:
        ax.set_facecolor(background_color)
        ax.tick_params(axis=u'both', which=u'both', length=0)
        for s in ["top", "right", "left"]:
            ax.spines[s].set_visible(False)

    def get_aligned_dist(train, test, col, sort_map=None):
        # Get all unique categories
        cats_tr = np.unique(train[col])
        cats_te = np.unique(test[col])
        all_cats = np.unique(np.concatenate([cats_tr, cats_te]))
        all_cats = all_cats[all_cats != ''] # Remove empty
        
        if sort_map:
             # Custom sort
             sorted_cats = sorted(all_cats, key=lambda x: sort_map.get(x, 999))
             all_cats = np.array(sorted_cats)
        else:
            all_cats.sort()
            
        def get_p(d, cats):
            vals, counts = np.unique(d[col], return_counts=True)
            mapping = dict(zip(vals, counts))
            total = sum([c for v,c in mapping.items() if v != ''])
            if total == 0: return np.zeros(len(cats))
            return np.array([mapping.get(c, 0)/total*100 for c in cats])

        return all_cats, get_p(train, all_cats), get_p(test, all_cats)

    # 0 - EDUCATION LEVEL
    cats, tr_p, te_p = get_aligned_dist(train_data, test_data, "education_level")
    x = np.arange(len(cats))
    axes[0].bar(x, height=tr_p, zorder=3, color="gray", width=0.05)
    axes[0].scatter(x, tr_p, zorder=3, s=200, color="gray")
    axes[0].bar(x + 0.4, height=te_p, zorder=3, color="#0e4f66", width=0.05)
    axes[0].scatter(x + 0.4, te_p, zorder=3, s=200, color="#0e4f66")
    axes[0].text(-0.5, max(tr_p.max(), te_p.max())*1.1, 'Education Level', fontsize=14, fontweight='bold', fontfamily='serif', color="#323232")
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter())
    axes[0].set_xticks(x + 0.2)
    axes[0].set_xticklabels(cats, rotation=0)

    # 1 - ENROLLED UNIVERSITY
    enroll_map = {'no_enrollment': 0, 'Part time course': 1, 'Full time course': 2}
    cats, tr_p, te_p = get_aligned_dist(train_data, test_data, "enrolled_university", enroll_map)
    axes[1].text(0, len(cats)-0.5, 'University Enrollment', fontsize=14, fontweight='bold', fontfamily='serif', color="#323232")
    axes[1].barh(cats, tr_p, color="gray", zorder=3, height=0.6)
    axes[1].barh(cats, te_p, color="#0e4f66", zorder=3, height=0.4)
    axes[1].xaxis.set_major_formatter(mtick.PercentFormatter())
    
    # 2 - GENDER
    cats, tr_p, te_p = get_aligned_dist(train_data, test_data, "gender")
    x = np.arange(len(cats))
    axes[2].text(-0.6, max(tr_p.max(), te_p.max())*1.1, 'Gender', fontsize=14, fontweight='bold', fontfamily='serif', color="#323232")
    axes[2].grid(color='gray', linestyle=':', axis='y', zorder=0, dashes=(1, 5))
    axes[2].bar(x, height=tr_p, zorder=3, color="gray", width=0.4)
    axes[2].bar(x + 0.4, height=te_p, zorder=3, color="#0e4f66", width=0.4)
    axes[2].set_xticks(x + 0.2)
    axes[2].set_xticklabels(cats)
    axes[2].yaxis.set_major_formatter(mtick.PercentFormatter())
    
    # 3 - CDI
    axes[3].grid(color='gray', linestyle=':', axis='y', zorder=0, dashes=(1, 5))
    sns.kdeplot(train_data["city_development_index"], ax=axes[3], color="gray", shade=True, label="Train")
    sns.kdeplot(test_data["city_development_index"], ax=axes[3], color="#0e4f66", shade=True, label="Test")
    axes[3].text(0.29, axes[3].get_ylim()[1]*0.9, 'City Development Index', fontsize=14, fontweight='bold', fontfamily='serif', color="#323232")
    axes[3].set_ylabel('')
    axes[3].set_xlabel('')
    
    # 4 - RELEVANT EXPERIENCE
    cats, tr_p, te_p = get_aligned_dist(train_data, test_data, "relevent_experience")
    x = np.arange(len(cats))
    axes[4].text(-0.4, max(tr_p.max(), te_p.max())*1.1, 'Relevant Experience', fontsize=14, fontweight='bold', fontfamily='serif', color="#323232")
    axes[4].grid(color='gray', linestyle=':', axis='y', zorder=0, dashes=(1, 5))
    axes[4].bar(x, height=tr_p, zorder=3, color="gray", width=0.4)
    axes[4].bar(x + 0.4, height=te_p, zorder=3, color="#0e4f66", width=0.4)
    axes[4].set_xticks(x + 0.2)
    axes[4].set_xticklabels(cats)
    axes[4].yaxis.set_major_formatter(mtick.PercentFormatter())
    
    # 5 - TRAINING HOURS
    tr_h = train_data["training_hours"]
    te_h = test_data["training_hours"]
    tr_h = tr_h[~np.isnan(tr_h)]
    te_h = te_h[~np.isnan(te_h)]
    
    y_vals = np.concatenate([tr_h, te_h])
    x_vals = np.array(["Train"] * len(tr_h) + ["Test"] * len(te_h))
    
    axes[5].text(-0.65, max(y_vals)*1.1 if len(y_vals)>0 else 100, 'Training Hours', fontsize=14, fontweight='bold', fontfamily='serif', color="#002d1d")
    sns.boxenplot(x=x_vals, y=y_vals, ax=axes[5], palette=["gray", "#0e4f66"])
    axes[5].set_xlabel("")
    axes[5].set_ylabel("")
    
    # 6 - EXPERIENCE
    def exp_key(val):
        if val == '<1': return 0
        if val == '>20': return 21
        try: return int(val)
        except: return -1
    
    unique_exp = np.unique(np.concatenate([train_data['experience'], test_data['experience']]))
    unique_exp = unique_exp[unique_exp != '']
    exp_map = {k: exp_key(k) for k in unique_exp}
    
    cats, tr_p, te_p = get_aligned_dist(train_data, test_data, "experience", exp_map)
    
    axes[6].grid(color='gray', linestyle=':', axis='y', zorder=0, dashes=(1, 5))
    axes[6].plot(cats, tr_p, zorder=3, color="gray", marker='o')
    axes[6].plot(cats, te_p, zorder=3, color="#0e4f66", marker='o')
    axes[6].text(-1.5, max(tr_p.max(), te_p.max())*1.1, 'Years Experience', fontsize=14, fontweight='bold', fontfamily='serif', color="#323232")
    axes[6].set_xticklabels(cats, rotation=90)
    
    # 7 - MAJOR DISCIPLINE
    cats, tr_p, te_p = get_aligned_dist(train_data, test_data, "major_discipline")
    axes[7].barh(np.arange(len(cats)), tr_p, zorder=3, color="gray", height=0.4)
    axes[7].barh(np.arange(len(cats)) + 0.4, te_p, zorder=3, color="#0e4f66", height=0.4)
    axes[7].text(-5, -0.8, 'Major Discipline', fontsize=14, fontweight='bold', fontfamily='serif', color="#323232")
    axes[7].xaxis.set_major_formatter(mtick.PercentFormatter())
    axes[7].set_yticks(np.arange(len(cats)) + 0.2)
    axes[7].set_yticklabels(cats)
    axes[7].invert_yaxis()

    # 8 - COMPANY SIZE
    size_map = {'<10': 0, '10/49': 1, '50-99': 2, '100-500': 3, '500-999': 4, '1000-4999': 5, '5000-9999': 6, '10000+': 7}
    cats, tr_p, te_p = get_aligned_dist(train_data, test_data, "company_size", size_map)
    x = np.arange(len(cats))
    axes[8].bar(x, height=tr_p, zorder=3, color="gray", width=0.4)
    axes[8].bar(x + 0.4, height=te_p, zorder=3, color="#0e4f66", width=0.4)
    axes[8].text(-0.5, max(tr_p.max(), te_p.max())*1.1, 'Company Size', fontsize=14, fontweight='bold', fontfamily='serif', color="#323232")
    axes[8].set_xticks(x + 0.2)
    axes[8].set_xticklabels(cats, rotation=45, ha='right')
    axes[8].yaxis.set_major_formatter(mtick.PercentFormatter())

    # 9 - COMPANY TYPE
    cats, tr_p, te_p = get_aligned_dist(train_data, test_data, "company_type")
    x = np.arange(len(cats))
    axes[9].bar(x, height=tr_p, zorder=3, color="gray", width=0.4)
    axes[9].bar(x + 0.4, height=te_p, zorder=3, color="#0e4f66", width=0.4)
    axes[9].text(-0.5, max(tr_p.max(), te_p.max())*1.1, 'Company Type', fontsize=14, fontweight='bold', fontfamily='serif', color="#323232")
    axes[9].set_xticks(x + 0.2)
    axes[9].set_xticklabels(cats, rotation=45, ha='right')
    axes[9].yaxis.set_major_formatter(mtick.PercentFormatter())

    # 10 - LAST NEW JOB
    job_map = {'never': 0, '1': 1, '2': 2, '3': 3, '4': 4, '>4': 5}
    cats, tr_p, te_p = get_aligned_dist(train_data, test_data, "last_new_job", job_map)
    x = np.arange(len(cats))
    axes[10].bar(x, height=tr_p, zorder=3, color="gray", width=0.4)
    axes[10].bar(x + 0.4, height=te_p, zorder=3, color="#0e4f66", width=0.4)
    axes[10].text(-0.5, max(tr_p.max(), te_p.max())*1.1, 'Last New Job', fontsize=14, fontweight='bold', fontfamily='serif', color="#323232")
    axes[10].set_xticks(x + 0.2)
    axes[10].set_xticklabels(cats)
    axes[10].yaxis.set_major_formatter(mtick.PercentFormatter())

    # Hide the last subplot (11)
    axes[11].set_visible(False)
    
    plt.show()
    