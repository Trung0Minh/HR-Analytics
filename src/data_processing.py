import numpy as np

def load_data(filepath):
    """
    Tải dữ liệu từ tệp CSV và trả về dưới dạng mảng cấu trúc NumPy.

    Args:
        filepath (str): Đường dẫn đến tệp CSV.

    Returns:
        np.ndarray: Mảng cấu trúc NumPy chứa dữ liệu.
    """
    # Tự động phát hiện kiểu dữ liệu, nhưng có thể cần điều chỉnh
    data = np.genfromtxt(filepath, delimiter=',', names=True, dtype=None, encoding='utf-8', autostrip=True)
    return data

def get_basic_info(data):
    """
    Lấy thông tin cơ bản về tập dữ liệu.

    Args:
        data (np.ndarray): Mảng cấu trúc NumPy chứa dữ liệu.

    Returns:
        dict: Một từ điển chứa thông tin cơ bản.
    """
    info = {
        'shape': data.shape,
        'columns': data.dtype.names,
        'dtypes': {name: data.dtype[name] for name in data.dtype.names},
        'memory_usage': data.nbytes
    }
    return info

def get_statistical_summary(data, column_name):
    """
    Lấy tóm tắt thống kê cho một cột cụ thể.

    Args:
        data (np.ndarray): Mảng cấu trúc NumPy chứa dữ liệu.
        column_name (str): Tên của cột cần phân tích.

    Returns:
        dict: Một từ điển chứa tóm tắt thống kê.
    """
    column = data[column_name]
    if np.issubdtype(column.dtype, np.number):
        # Xử lý cho cột số
        summary = {
            'min': np.nanmin(column),
            'max': np.nanmax(column),
            'mean': np.nanmean(column),
            'median': np.nanmedian(column),
            'std': np.nanstd(column),
            'quartiles': np.nanpercentile(column, [25, 50, 75])
        }
    else:
        # Xử lý cho cột phân loại
        unique_values, counts = np.unique(column, return_counts=True)
        most_frequent_index = np.argmax(counts)
        summary = {
            'unique_count': len(unique_values),
            'most_frequent': unique_values[most_frequent_index],
            'frequency': counts[most_frequent_index]
        }
    return summary

def get_missing_summary(data):
    """
    Tạo tóm tắt về các giá trị bị thiếu trong tập dữ liệu.

    Args:
        data (np.ndarray): Mảng cấu trúc NumPy chứa dữ liệu.

    Returns:
        dict: Một từ điển chứa số lượng và tỷ lệ phần trăm giá trị bị thiếu cho mỗi cột.
    """
    missing_summary = {}
    total_rows = len(data)
    for name in data.dtype.names:
        column = data[name]
        if np.issubdtype(column.dtype, np.number):
            missing_count = np.sum(np.isnan(column))
        else:
            missing_count = np.sum(column == '')
        missing_summary[name] = {
            'missing_count': missing_count,
            'missing_percentage': (missing_count / total_rows) * 100
        }
    # Sắp xếp theo tỷ lệ thiếu giảm dần
    sorted_missing = sorted(missing_summary.items(), key=lambda item: item[1]['missing_percentage'], reverse=True)
    return dict(sorted_missing)

def analyze_missing_patterns(data):
    """
    Phân tích các mẫu giá trị bị thiếu trong tập dữ liệu.

    Args:
        data (np.ndarray): Mảng cấu trúc NumPy chứa dữ liệu.

    Returns:
        np.ndarray: Một ma trận boolean biểu thị các giá trị bị thiếu.
                    Kích thước: (số dòng, số cột)
    """
    n_rows = len(data)
    n_cols = len(data.dtype.names)
    missing_matrix = np.zeros((n_rows, n_cols), dtype=bool)

    for i, name in enumerate(data.dtype.names):
        column = data[name]

        # Chuyển column sang numpy array 1D kiểu object để tránh tuple/list lồng nhau
        column = np.array(column, dtype=object)

        # Cột số: kiểm tra NaN
        if np.issubdtype(column.dtype, np.number):
            missing_matrix[:, i] = np.isnan(column)
        else:
            # Cột string: kiểm tra '' hoặc None
            missing_matrix[:, i] = np.vectorize(lambda x: x is None or x == '')(column)

    return missing_matrix

def get_target_distribution(data, target_column='target'):
    """
    Lấy phân phối của biến mục tiêu.

    Args:
        data (np.ndarray): Mảng cấu trúc NumPy chứa dữ liệu.
        target_column (str): Tên của cột mục tiêu.

    Returns:
        dict: Một từ điển chứa số lượng, tỷ lệ phần trăm và tỷ lệ mất cân bằng.
    """
    target = data[target_column]
    unique, counts = np.unique(target, return_counts=True)
    total = len(target)
    distribution = {
        'counts': dict(zip(unique, counts)),
        'percentages': {u: (c / total) * 100 for u, c in zip(unique, counts)},
        'imbalance_ratio': counts[0] / counts[1] if len(counts) == 2 else 1
    }
    return distribution

def get_categorical_distribution(data, column_name):
    """
    Lấy phân phối của một đặc trưng phân loại.

    Args:
        data (np.ndarray): Mảng cấu trúc NumPy chứa dữ liệu.
        column_name (str): Tên của cột phân loại.

    Returns:
        dict: Một từ điển chứa tần suất và tỷ lệ phần trăm của mỗi loại.
    """
    column = data[column_name]
    unique, counts = np.unique(column, return_counts=True)
    total = len(column)
    # Sắp xếp theo tần suất giảm dần
    sorted_indices = np.argsort(-counts)
    sorted_unique = unique[sorted_indices]
    sorted_counts = counts[sorted_indices]
    distribution = {
        'categories': sorted_unique,
        'counts': sorted_counts,
        'percentages': (sorted_counts / total) * 100
    }
    return distribution

def get_numerical_distribution(data, column_name):
    """
    Lấy phân phối của một đặc trưng số.

    Args:
        data (np.ndarray): Mảng cấu trúc NumPy chứa dữ liệu.
        column_name (str): Tên của cột số.

    Returns:
        dict: Một từ điển chứa thống kê phân phối và thông tin về ngoại lệ.
    """
    column = data[column_name]
    # Loại bỏ các giá trị nan
    column = column[~np.isnan(column)]
    q1, q3 = np.percentile(column, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    outliers = column[(column < lower_bound) | (column > upper_bound)]
    hist, bins = np.histogram(column, bins='auto')
    distribution = {
        'statistics': get_statistical_summary(data, column_name),
        'outliers': outliers,
        'histogram': (hist, bins)
    }
    return distribution

def calculate_target_rate_by_category(data, categorical_column, target_column='target'):
    """
    Tính tỷ lệ mục tiêu cho mỗi loại của một đặc trưng phân loại.

    Args:
        data (np.ndarray): Mảng cấu trúc NumPy chứa dữ liệu.
        categorical_column (str): Tên của cột phân loại.
        target_column (str): Tên của cột mục tiêu.

    Returns:
        dict: Một từ điển chứa tỷ lệ mục tiêu và số lượng cho mỗi loại.
    """
    categories = np.unique(data[categorical_column])
    rates = {} 
    for category in categories:
        category_mask = data[categorical_column] == category
        target_for_category = data[target_column][category_mask]
        target_rate = np.nanmean(target_for_category)
        rates[category] = {
            'target_rate': target_rate,
            'count': len(target_for_category)
        }
    return rates

def get_correlation_with_target(data, numerical_columns, target_column='target'):
    """
    Tính toán tương quan giữa các đặc trưng số và biến mục tiêu.

    Args:
        data (np.ndarray): Mảng cấu trúc NumPy chứa dữ liệu.
        numerical_columns (list): Danh sách các tên cột số.
        target_column (str): Tên của cột mục tiêu.

    Returns:
        dict: Một từ điển chứa các tương quan.
    """
    correlations = {}
    target = data[target_column]
    for name in numerical_columns:
        # Bỏ qua các giá trị nan trong cả hai cột
        valid_mask = ~np.isnan(data[name]) & ~np.isnan(target)
        correlations[name] = np.corrcoef(data[name][valid_mask], target[valid_mask])[0, 1]
    return correlations

def analyze_feature_interaction(data, feature1, feature2):
    """
    Phân tích tương tác giữa hai đặc trưng.

    Args:
        data (np.ndarray): Mảng cấu trúc NumPy chứa dữ liệu.
        feature1 (str): Tên của đặc trưng thứ nhất.
        feature2 (str): Tên của đặc trưng thứ hai.

    Returns:
        np.ndarray: Một bảng chéo (cross-tabulation) thể hiện tương tác.
    """
    unique_f1 = np.unique(data[feature1])
    unique_f2 = np.unique(data[feature2])
    crosstab = np.zeros((len(unique_f1), len(unique_f2)), dtype=int)
    for i, val1 in enumerate(unique_f1):
        for j, val2 in enumerate(unique_f2):
            mask = (data[feature1] == val1) & (data[feature2] == val2)
            crosstab[i, j] = np.sum(mask)
    return crosstab

def impute_missing_values(data, column_name, strategy='mean'):
    """
    Điền các giá trị bị thiếu trong một cột.

    Args:
        data (np.ndarray): Mảng cấu trúc NumPy chứa dữ liệu.
        column_name (str): Tên của cột cần xử lý.
        strategy (str): Chiến lược điền ('mean', 'median', 'mode').

    Returns:
        np.ndarray: Dữ liệu đã được điền.
    """
    column = data[column_name]
    if np.issubdtype(column.dtype, np.number):
        if strategy == 'mean':
            fill_value = np.nanmean(column)
        elif strategy == 'median':
            fill_value = np.nanmedian(column)
        else:
            fill_value = 0 # Mặc định
        column[np.isnan(column)] = fill_value
    else:
        if strategy == 'mode':
            unique, counts = np.unique(column, return_counts=True)
            fill_value = unique[np.argmax(counts)]
        else:
            fill_value = 'Unknown' # Mặc định
        column[column == ''] = fill_value
    return data

def one_hot_encode(data, column_name):
    """
    Thực hiện mã hóa one-hot cho một cột phân loại.

    Args:
        data (np.ndarray): Mảng cấu trúc NumPy chứa dữ liệu.
        column_name (str): Tên của cột cần mã hóa.

    Returns:
        np.ndarray: Dữ liệu đã được mã hóa.
    """
    column = data[column_name]
    unique_values = np.unique(column)
    new_cols = []
    for value in unique_values:
        new_col_name = f'{column_name}_{value}'
        new_col_data = (column == value).astype(int)
        new_cols.append((new_col_name, new_col_data))
    
    # Xóa cột cũ và thêm các cột mới
    new_dtype = data.dtype.descr
    new_dtype = [dt for dt in new_dtype if dt[0] != column_name]
    for new_col_name, _ in new_cols:
        new_dtype.append((new_col_name, 'i4'))
        
    new_data = np.empty(data.shape, dtype=new_dtype)
    for name in new_data.dtype.names:
        if name in data.dtype.names:
            new_data[name] = data[name]
        else:
            for new_col_name, new_col_data in new_cols:
                if name == new_col_name:
                    new_data[name] = new_col_data
                    break
    return new_data

def get_top_n_cities(data, n=10):
    """
    Lấy top N thành phố có số lượng ứng viên cao nhất.

    Args:
        data (np.ndarray): Mảng cấu trúc NumPy chứa dữ liệu.
        n (int): Số lượng thành phố hàng đầu cần lấy.

    Returns:
        tuple: Một tuple chứa hai danh sách: tên thành phố và số lượng ứng viên.
    """
    city_dist = get_categorical_distribution(data, 'city')
    return city_dist['categories'][:n], city_dist['counts'][:n]

def get_city_development_index_by_city(data):
    """
    Lấy chỉ số phát triển thành phố cho mỗi thành phố.

    Args:
        data (np.ndarray): Mảng cấu trúc NumPy chứa dữ liệu.

    Returns:
        dict: Một từ điển với các thành phố làm khóa và chỉ số phát triển làm giá trị.
    """
    city_dev_index = {}
    unique_cities = np.unique(data['city'])
    for city in unique_cities:
        indices = data['city_development_index'][data['city'] == city]
        if len(indices) > 0:
            # Giả sử chỉ số phát triển là nhất quán cho mỗi thành phố
            city_dev_index[city] = indices[0]
    return city_dev_index

def get_missing_value_correlation(data):
    """
    Phân tích tương quan của các giá trị bị thiếu giữa các cột.

    Args:
        data (np.ndarray): Mảng cấu trúc NumPy chứa dữ liệu.

    Returns:
        np.ndarray: Ma trận tương quan của các giá trị bị thiếu.
    """
    missing_matrix = analyze_missing_patterns(data).astype(int)
    correlation_matrix = np.corrcoef(missing_matrix, rowvar=False)
    return correlation_matrix

def compare_target_rate_by_missing(data, column_name, target_column='target'):
    """
    So sánh tỷ lệ mục tiêu giữa các bản ghi có và không có giá trị thiếu trong một cột cụ thể.

    Args:
        data (np.ndarray): Mảng cấu trúc NumPy chứa dữ liệu.
        column_name (str): Tên của cột cần phân tích giá trị thiếu.
        target_column (str): Tên của cột mục tiêu.

    Returns:
        dict: Một từ điển chứa tỷ lệ mục tiêu cho các bản ghi có và không có giá trị thiếu.
    """
    column = data[column_name]
    target = data[target_column]
    
    if np.issubdtype(column.dtype, np.number):
        missing_mask = np.isnan(column)
    else:
        missing_mask = np.vectorize(lambda x: x is None or x == '')(column)

    # Bản ghi có giá trị thiếu
    target_missing = target[missing_mask]
    target_rate_missing = np.nanmean(target_missing) if len(target_missing) > 0 else np.nan
    count_missing = len(target_missing)

    # Bản ghi không có giá trị thiếu
    target_not_missing = target[~missing_mask]
    target_rate_not_missing = np.nanmean(target_not_missing) if len(target_not_missing) > 0 else np.nan
    count_not_missing = len(target_not_missing)

    return {
        'missing': {'target_rate': target_rate_missing, 'count': count_missing},
        'not_missing': {'target_rate': target_rate_not_missing, 'count': count_not_missing}
    }

def analyze_categorical_missing(data, categorical_columns):
    """
    Phân tích các giá trị bị thiếu trong các cột phân loại được chỉ định.

    Args:
        data (np.ndarray): Mảng cấu trúc NumPy chứa dữ liệu.
        categorical_columns (list): Danh sách các tên cột phân loại cần phân tích.

    Returns:
        dict: Một từ điển chứa thông tin về giá trị thiếu cho mỗi cột.
              Mỗi cột sẽ có một từ điển con với 'missing_count' và 'missing_percentage'.
    """
    missing_info = {}
    total_rows = len(data)
    for col_name in categorical_columns:
        column = data[col_name]
        # Đối với các cột string, giá trị thiếu thường là rỗng ('') hoặc None
        missing_count = np.sum(np.vectorize(lambda x: x is None or x == '')(column))
        missing_percentage = (missing_count / total_rows) * 100
        missing_info[col_name] = {
            'missing_count': missing_count,
            'missing_percentage': missing_percentage
        }
    return missing_info

def compare_target_rate_by_outliers(data, numerical_column, target_column='target'):
    """
    So sánh tỷ lệ mục tiêu giữa các bản ghi có và không có giá trị ngoại lệ trong một cột số cụ thể.

    Args:
        data (np.ndarray): Mảng cấu trúc NumPy chứa dữ liệu.
        numerical_column (str): Tên của cột số cần phân tích ngoại lệ.
        target_column (str): Tên của cột mục tiêu.

    Returns:
        dict: Một từ điển chứa tỷ lệ mục tiêu cho các bản ghi có và không có ngoại lệ.
    """
    column = data[numerical_column]
    target = data[target_column]

    # Loại bỏ các giá trị NaN để tính toán IQR
    column_no_nan = column[~np.isnan(column)]

    if len(column_no_nan) == 0:
        return {
            'outliers': {'target_rate': np.nan, 'count': 0},
            'not_outliers': {'target_rate': np.nan, 'count': 0}
        }

    q1, q3 = np.percentile(column_no_nan, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)

    # Xác định các bản ghi là ngoại lệ
    outlier_mask = (column < lower_bound) | (column > upper_bound)

    # Bản ghi có ngoại lệ
    target_outliers = target[outlier_mask]
    target_rate_outliers = np.nanmean(target_outliers) if len(target_outliers) > 0 else np.nan
    count_outliers = len(target_outliers)

    # Bản ghi không có ngoại lệ
    target_not_outliers = target[~outlier_mask]
    target_rate_not_outliers = np.nanmean(target_not_outliers) if len(target_not_outliers) > 0 else np.nan
    count_not_outliers = len(target_not_outliers)

    return {
        'outliers': {'target_rate': target_rate_outliers, 'count': count_outliers},
        'not_outliers': {'target_rate': target_rate_not_outliers, 'count': count_not_outliers}
    }

def check_duplicate_enrollee_id(data):
    """
    Kiểm tra các giá trị trùng lặp trong cột 'enrollee_id'.

    Args:
        data (np.ndarray): Mảng cấu trúc NumPy chứa dữ liệu.

    Returns:
        np.ndarray: Mảng chứa các enrollee_id bị trùng lặp.
    """
    unique_ids, counts = np.unique(data['enrollee_id'], return_counts=True)
    duplicate_ids = unique_ids[counts > 1]
    return duplicate_ids

def identify_inconsistent_categories(data, column_name, expected_categories):
    """
    Xác định các danh mục không nhất quán trong một cột phân loại.

    Args:
        data (np.ndarray): Mảng cấu trúc NumPy chứa dữ liệu.
        column_name (str): Tên của cột phân loại cần kiểm tra.
        expected_categories (list): Danh sách các danh mục dự kiến.

    Returns:
        np.ndarray: Mảng chứa các danh mục không nhất quán.
    """
    actual_categories = np.unique(data[column_name])
    inconsistent_categories = [cat for cat in actual_categories if cat not in expected_categories]
    return np.array(inconsistent_categories)

def perform_logic_validation(data):
    """
    Thực hiện kiểm tra logic cho các mối quan hệ giữa các cột.
    Ví dụ: kinh nghiệm "<1" nhưng company_size lớn.

    Args:
        data (np.ndarray): Mảng cấu trúc NumPy chứa dữ liệu.

    Returns:
        dict: Một từ điển chứa các kết quả kiểm tra logic.
    """
    validation_results = {}

    # Logic check: experience="<1" but large company_size
    # Giả định company_size lớn là > 10000
    # Cần chuyển đổi company_size sang dạng số để so sánh
    # Đối với mục đích ví dụ, chúng ta sẽ kiểm tra các giá trị string của company_size
    # và experience

    # Chuyển đổi experience sang thứ tự để so sánh
    experience_mapping = {'<1': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, '11': 11, '12': 12, '13': 13, '14': 14, '15': 15, '16': 16, '17': 17, '18': 18, '19': 19, '20': 20, '>20': 21}
    # Tạo một mảng số từ cột experience
    experience_numeric = np.array([experience_mapping.get(exp, -1) for exp in data['experience']])

    # Giả định company_size lớn là '50000-99999' hoặc '100000+'
    large_company_sizes = ['50000-99999', '100000+']

    inconsistent_experience_company_size_mask = (
        (experience_numeric == experience_mapping['<1']) &
        np.isin(data['company_size'], large_company_sizes)
    )
    validation_results['inconsistent_experience_company_size'] = np.sum(inconsistent_experience_company_size_mask)

    return validation_results

def compare_profiles_by_target(data, target_column='target'):
    """
    So sánh hồ sơ của các nhóm target=1 và target=0.
    Tính toán thống kê mô tả cho các đặc trưng số và phân phối cho các đặc trưng phân loại.

    Args:
        data (np.ndarray): Mảng cấu trúc NumPy chứa dữ liệu.
        target_column (str): Tên của cột mục tiêu.

    Returns:
        dict: Một từ điển chứa thông tin hồ sơ cho mỗi nhóm target.
    """
    profiles = {
        0: {'numerical_summary': {}, 'categorical_distribution': {}},
        1: {'numerical_summary': {}, 'categorical_distribution': {}}
    }

    for target_value in [0, 1]:
        target_mask = data[target_column] == target_value
        subset_data = data[target_mask]

        for col_name in data.dtype.names:
            if col_name == target_column or col_name == 'enrollee_id':
                continue

            column = subset_data[col_name]

            if np.issubdtype(column.dtype, np.number):
                # Numerical features: mean, median, std
                profiles[target_value]['numerical_summary'][col_name] = {
                    'mean': np.nanmean(column),
                    'median': np.nanmedian(column),
                    'std': np.nanstd(column)
                }
            else:
                # Categorical features: distribution
                unique, counts = np.unique(column, return_counts=True)
                total = len(column)
                distribution = {u: (c / total) * 100 for u, c in zip(unique, counts)}
                profiles[target_value]['categorical_distribution'][col_name] = distribution

    return profiles

def rank_features_by_p_value(p_values):
    """
    Xếp hạng các đặc trưng dựa trên p-value (thấp hơn là quan trọng hơn).

    Args:
        p_values (dict): Một từ điển với tên đặc trưng là khóa và p-value là giá trị.

    Returns:
        list: Danh sách các tuple (tên đặc trưng, p-value) được sắp xếp theo p-value tăng dần.
    """
    sorted_features = sorted(p_values.items(), key=lambda item: item[1])
    return sorted_features

def analyze_multi_column_missing_patterns(data, columns_to_check):
    """
    Phân tích các mẫu thiếu đa cột để xác định các kết hợp cột thường bị thiếu cùng nhau.

    Args:
        data (np.ndarray): Mảng cấu trúc NumPy chứa dữ liệu.
        columns_to_check (list): Danh sách các cột để kiểm tra mẫu thiếu.

    Returns:
        dict: Một từ điển trong đó các khóa là các bộ tên cột bị thiếu 
              và các giá trị là số lượng bản ghi có mẫu thiếu đó.
    """
    missing_patterns = {}
    
    # Tạo một ma trận boolean cho các cột được chỉ định
    missing_matrix = np.zeros((len(data), len(columns_to_check)), dtype=bool)
    for i, col_name in enumerate(columns_to_check):
        column = data[col_name]
        if np.issubdtype(column.dtype, np.number):
            missing_matrix[:, i] = np.isnan(column)
        else:
            missing_matrix[:, i] = np.vectorize(lambda x: x is None or x == '')(column)
            
    # Lặp qua từng hàng để xác định các mẫu thiếu
    for row in missing_matrix:
        missing_cols_indices = np.where(row)[0]
        if len(missing_cols_indices) > 1: # Chỉ quan tâm đến các mẫu thiếu đa cột
            missing_cols = tuple(columns_to_check[i] for i in missing_cols_indices)
            if missing_cols in missing_patterns:
                missing_patterns[missing_cols] += 1
            else:
                missing_patterns[missing_cols] = 1
                
    # Sắp xếp các mẫu theo số lượng giảm dần
    sorted_patterns = sorted(missing_patterns.items(), key=lambda item: item[1], reverse=True)
    
    return dict(sorted_patterns)