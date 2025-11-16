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
        'shape': (data.shape[0], len(data.dtype.names)),
        'columns': data.dtype.names,
        'dtypes': {name: data.dtype[name] for name in data.dtype.names},
        'memory_usage': data.nbytes
    }
    return info

def check_duplicates(data):
    """
    Checks for, counts, and returns representative duplicated rows 
    in a NumPy structured array, excluding 'enrollee_id'.
    
    Args:
        data (np.ndarray): NumPy structured array.
        
    Returns:
        dict: A dictionary containing counts and representative data of duplicated rows.
    """
    from collections import Counter

    # Define columns to check for duplicates, excluding the ID
    cols_to_check = [name for name in data.dtype.names if name != 'enrollee_id']
    
    # Create a new view of the array with only the columns to be checked
    data_subset = data[cols_to_check]

    # Convert each row to a tuple to make them hashable
    rows_as_tuples = [tuple(row) for row in data_subset]
    
    # Count occurrences of each unique row tuple
    row_counts = Counter(rows_as_tuples)
    
    # Filter for row tuples that appear more than once
    duplicated_row_counts = {item: count for item, count in row_counts.items() if count > 1}
    
    # The total number of "extra" rows (duplicates)
    num_duplicates = sum(count - 1 for count in duplicated_row_counts.values())
    total_rows = len(rows_as_tuples)
    
    # Get a single representative row for each group of duplicates
    representatives = []
    found_tuples = set()
    for i, row_tuple in enumerate(rows_as_tuples):
        # Check if this row is a duplicate and we haven't processed this group yet
        if row_tuple in duplicated_row_counts and row_tuple not in found_tuples:
            original_row = data[i]
            count = duplicated_row_counts[row_tuple]
            representatives.append({'row': original_row, 'count': count})
            found_tuples.add(row_tuple)

    return {
        'total_rows': total_rows,
        'duplicated_rows': num_duplicates,
        'duplicated_percentage': (num_duplicates / total_rows) * 100 if total_rows > 0 else 0,
        'representative_duplicates': representatives
    }

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

def get_numerical_summary(data, numerical_cols):
    """
    Tính toán thống kê mô tả cho các cột số được chỉ định.

    Args:
        data (np.ndarray): Mảng cấu trúc NumPy chứa dữ liệu.
        numerical_cols (list): Danh sách các tên cột số.

    Returns:
        dict: Một từ điển chứa thống kê mô tả cho mỗi cột.
    """
    summary = {}
    for col in numerical_cols:
        col_data = data[col][~np.isnan(data[col])]  # Loại bỏ NaN
        if col_data.size == 0:
            summary[col] = {
                'mean': np.nan, 'median': np.nan, 'std': np.nan,
                'min': np.nan, 'max': np.nan, '25%': np.nan,
                '50%': np.nan, '75%': np.nan
            }
            continue
        summary[col] = {
            'count': len(col_data),
            'mean': np.mean(col_data),
            'std': np.std(col_data),
            'min': np.min(col_data),
            '25%': np.percentile(col_data, 25),
            '50%': np.median(col_data),
            '75%': np.percentile(col_data, 75),
            'max': np.max(col_data)
        }
    return summary

def count_row_completeness(data):
    """
    Counts the number of completely full rows and completely empty rows.

    Args:
        data (np.ndarray): NumPy structured array.

    Returns:
        dict: A dictionary with counts of complete, empty, and partial rows.
    """
    n_rows = data.shape[0]
    
    complete_rows_count = 0
    empty_rows_count = 0
    
    # Check all columns except the ID for emptiness/completeness
    cols_to_check = [name for name in data.dtype.names if name != 'enrollee_id']
    
    for i in range(n_rows):
        is_complete = True
        is_empty = True
        
        for name in cols_to_check:
            field = data[i][name]
            
            field_is_missing = False
            if np.issubdtype(data.dtype[name], np.number):
                if np.isnan(field):
                    field_is_missing = True
            else:
                if field == '':
                    field_is_missing = True

            if field_is_missing:
                is_complete = False # At least one field is missing, so not complete
            else:
                is_empty = False # At least one field is not missing, so not empty
        
        if is_complete:
            complete_rows_count += 1
        if is_empty:
            empty_rows_count += 1
            
    partial_rows_count = n_rows - complete_rows_count - empty_rows_count
    
    return {
        'total_rows': n_rows,
        'complete_rows': complete_rows_count,
        'empty_rows': empty_rows_count,
        'partial_rows': partial_rows_count
    }
