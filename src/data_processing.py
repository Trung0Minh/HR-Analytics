import numpy as np
from numpy.lib import recfunctions as rfn

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
    Kiểm tra, đếm và trả về các hàng trùng lặp đại diện
    trong mảng cấu trúc NumPy, loại trừ 'enrollee_id'.
    
    Args:
        data (np.ndarray): Mảng cấu trúc NumPy.
        
    Returns:
        dict: Một từ điển chứa số lượng và dữ liệu đại diện của các hàng trùng lặp.
    """
    from collections import Counter

    # Xác định các cột cần kiểm tra trùng lặp, loại trừ ID
    cols_to_check = [name for name in data.dtype.names if name != 'enrollee_id']
    
    # Tạo một view mới của mảng chỉ với các cột cần kiểm tra
    data_subset = data[cols_to_check]

    # Chuyển đổi mỗi hàng thành một tuple để có thể băm (hashable)
    rows_as_tuples = [tuple(row) for row in data_subset]
    
    # Đếm số lần xuất hiện của mỗi tuple hàng duy nhất
    row_counts = Counter(rows_as_tuples)
    
    # Lọc các tuple hàng xuất hiện nhiều hơn một lần
    duplicated_row_counts = {item: count for item, count in row_counts.items() if count > 1}
    
    # Tổng số hàng "thừa" (trùng lặp)
    num_duplicates = sum(count - 1 for count in duplicated_row_counts.values())
    total_rows = len(rows_as_tuples)
    
    # Lấy một hàng đại diện duy nhất cho mỗi nhóm trùng lặp
    representatives = []
    found_tuples = set()
    for i, row_tuple in enumerate(rows_as_tuples):
        # Kiểm tra xem hàng này có phải là hàng trùng lặp và chúng ta chưa xử lý nhóm này không
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
    Đếm số lượng hàng đầy đủ hoàn toàn và hàng trống hoàn toàn.

    Args:
        data (np.ndarray): Mảng cấu trúc NumPy.

    Returns:
        dict: Một từ điển với số lượng các hàng đầy đủ, trống và một phần.
    """
    n_rows = data.shape[0]
    
    complete_rows_count = 0
    empty_rows_count = 0
    
    # Kiểm tra tất cả các cột ngoại trừ ID về độ trống/đầy đủ
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
                is_complete = False # Ít nhất một trường bị thiếu, vì vậy không đầy đủ
            else:
                is_empty = False # Ít nhất một trường không bị thiếu, vì vậy không trống
        
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

def analyze_city_and_index_relationship(data):
    """
    Phân tích mối quan hệ giữa 'city' và 'city_development_index'.

    Hàm này tính toán:
    1. Số lượng thành phố duy nhất và chỉ số phát triển thành phố duy nhất.
    2. Số lượng thành phố ánh xạ đến một chỉ số phát triển duy nhất.
    3. Số lượng chỉ số phát triển được chia sẻ bởi nhiều thành phố.

    Args:
        data (np.ndarray): Mảng cấu trúc với 'city' và 'city_development_index'.

    Returns:
        dict: Một bản tóm tắt chứa bốn chỉ số chính.
    """
    # Lọc bỏ các giá trị bị thiếu cho thành phố và chỉ số
    valid_cities = data['city'][data['city'] != '']
    valid_indices = data['city_development_index'][~np.isnan(data['city_development_index'])]

    # 1. Số lượng giá trị duy nhất
    num_unique_cities = len(np.unique(valid_cities))
    num_unique_indices = len(np.unique(valid_indices))

    # Tạo ánh xạ từ thành phố đến các chỉ số phát triển duy nhất, không bị thiếu của nó
    city_to_indices = {}
    for city in np.unique(valid_cities):
        indices_for_city = data[data['city'] == city]['city_development_index']
        non_nan_indices = indices_for_city[~np.isnan(indices_for_city)]
        city_to_indices[city] = np.unique(non_nan_indices)

    # 2. Số lượng thành phố chỉ ánh xạ đến một chỉ số
    cities_with_one_index = sum(1 for indices in city_to_indices.values() if len(indices) == 1)

    # Tạo ánh xạ từ chỉ số đến các thành phố duy nhất, không bị thiếu của nó
    index_to_cities = {}
    for index_val in np.unique(valid_indices):
        cities_for_index = data[data['city_development_index'] == index_val]['city']
        non_empty_cities = cities_for_index[cities_for_index != '']
        index_to_cities[index_val] = np.unique(non_empty_cities)

    # 3. Số lượng chỉ số được chia sẻ bởi nhiều thành phố
    shared_indices = sum(1 for cities in index_to_cities.values() if len(cities) > 1)

    return {
        "num_unique_cities": num_unique_cities,
        "num_unique_indices": num_unique_indices,
        "cities_mapping_to_one_index": cities_with_one_index,
        "indices_shared_by_multiple_cities": shared_indices,
    }

def remove_columns(data, columns_to_remove):
    """
    Loại bỏ các cột được chỉ định khỏi mảng cấu trúc NumPy.

    Args:
        data (np.ndarray): Mảng cấu trúc đầu vào.
        columns_to_remove (list): Một danh sách các tên cột cần loại bỏ.

    Returns:
        np.ndarray: Một mảng mới với các cột được chỉ định đã bị loại bỏ.
    """
    # Đảm bảo tất cả các cột cần xóa đều tồn tại trong dữ liệu
    valid_columns_to_remove = [col for col in columns_to_remove if col in data.dtype.names]
    return rfn.drop_fields(data, valid_columns_to_remove)

def remove_duplicates(data):
    """
    Loại bỏ các hàng trùng lặp khỏi mảng cấu trúc NumPy, dựa trên tất cả các cột ngoại trừ 'enrollee_id'.

    Args:
        data (np.ndarray): Mảng cấu trúc đầu vào.

    Returns:
        np.ndarray: Một mảng mới với các hàng trùng lặp đã bị loại bỏ.
    """
    # Lấy các cột để xem xét tính duy nhất. Nếu 'enrollee_id' đã biến mất, hãy sử dụng tất cả các cột.
    if 'enrollee_id' in data.dtype.names:
        key_columns = [name for name in data.dtype.names if name != 'enrollee_id']
        data_subset = data[key_columns]
    else:
        data_subset = data
    
    # np.unique trả về các chỉ số của lần xuất hiện đầu tiên của các hàng duy nhất
    _, unique_indices = np.unique(data_subset, return_index=True)
    
    # Sắp xếp các chỉ số để duy trì thứ tự ban đầu càng nhiều càng tốt
    sorted_unique_indices = np.sort(unique_indices)
    
    # Trả về các hàng duy nhất từ mảng dữ liệu ban đầu
    return data[sorted_unique_indices]