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

def analyze_city_and_index_relationship(data):
    """
    Analyzes the relationship between 'city' and 'city_development_index'.

    This function calculates:
    1. The number of unique cities and unique city development indices.
    2. The number of cities that map to a single development index.
    3. The number of development indices that are shared by multiple cities.

    Args:
        data (np.ndarray): Structured array with 'city' and 'city_development_index'.

    Returns:
        dict: A summary containing the four key metrics.
    """
    # Filter out missing values for cities and indices
    valid_cities = data['city'][data['city'] != '']
    valid_indices = data['city_development_index'][~np.isnan(data['city_development_index'])]

    # 1. Number of unique values
    num_unique_cities = len(np.unique(valid_cities))
    num_unique_indices = len(np.unique(valid_indices))

    # Create a mapping from city to its unique, non-missing development indices
    city_to_indices = {}
    for city in np.unique(valid_cities):
        indices_for_city = data[data['city'] == city]['city_development_index']
        non_nan_indices = indices_for_city[~np.isnan(indices_for_city)]
        city_to_indices[city] = np.unique(non_nan_indices)

    # 2. Number of cities mapping to only one index
    cities_with_one_index = sum(1 for indices in city_to_indices.values() if len(indices) == 1)

    # Create a mapping from index to its unique, non-missing cities
    index_to_cities = {}
    for index_val in np.unique(valid_indices):
        cities_for_index = data[data['city_development_index'] == index_val]['city']
        non_empty_cities = cities_for_index[cities_for_index != '']
        index_to_cities[index_val] = np.unique(non_empty_cities)

    # 3. Number of indices shared by multiple cities
    shared_indices = sum(1 for cities in index_to_cities.values() if len(cities) > 1)

    return {
        "num_unique_cities": num_unique_cities,
        "num_unique_indices": num_unique_indices,
        "cities_mapping_to_one_index": cities_with_one_index,
        "indices_shared_by_multiple_cities": shared_indices,
    }

def analyze_experience_interaction(data, hue_column='company_type', target_column='target'):
    """
    Analyzes the interaction between experience groups and another categorical variable.

    Args:
        data (np.ndarray): The dataset.
        hue_column (str): The column to interact with experience (e.g., 'company_type').
        target_column (str): The name of the target variable.

    Returns:
        dict: A dictionary containing the data needed for plotting.
    """
    # Convert experience to a numerical format
    exp_numeric = data['experience'].copy()
    exp_numeric[exp_numeric == '>20'] = '21'
    exp_numeric[exp_numeric == '<1'] = '0'
    exp_numeric[exp_numeric == ''] = '-1' # Missing values
    exp_numeric = exp_numeric.astype(int)

    # Create experience groups
    experience_groups = np.select(
        [
            (exp_numeric == -1),
            (exp_numeric == 0),
            (exp_numeric >= 1) & (exp_numeric <= 5),
            (exp_numeric >= 6) & (exp_numeric <= 10),
            (exp_numeric >= 11) & (exp_numeric <= 20),
            (exp_numeric > 20)
        ],
        ['Missing', 'Fresher (<1)', 'Junior (1-5)', 'Mid-level (6-10)', 'Senior (11-20)', 'Expert (>20)'],
        default='N/A'
    )

    # Get necessary columns
    hue_values = data[hue_column]
    targets = data[target_column]

    # Get unique categories to analyze
    unique_exp_groups = ['Fresher (<1)', 'Junior (1-5)', 'Mid-level (6-10)', 'Senior (11-20)', 'Expert (>20)']
    unique_hue_values = [h for h in np.unique(hue_values) if h != '' and h != 'Missing']

    # Calculate interaction summary
    interaction_summary = {}
    for hue_val in unique_hue_values:
        rates = []
        for exp_group in unique_exp_groups:
            mask = (experience_groups == exp_group) & (hue_values == hue_val)
            targets_in_group = targets[mask]
            
            if len(targets_in_group) > 0:
                rate = np.mean(targets_in_group)
            else:
                rate = 0
            rates.append(rate)
        interaction_summary[hue_val] = rates
        
    overall_mean = np.mean(targets)

    return {
        'summary': interaction_summary,
        'exp_groups': unique_exp_groups,
        'hue_values': unique_hue_values,
        'overall_mean': overall_mean,
        'hue_column_name': hue_column
    }

def analyze_bivariate_categorical(data, x_col, hue_col, target_col='target', x_order=None):
    """
    Analyzes the interaction between two categorical variables against a target.

    Args:
        data (np.ndarray): The dataset.
        x_col (str): The column for the x-axis.
        hue_col (str): The column for the color grouping (hue).
        target_col (str): The name of the target variable.
        x_order (list, optional): A specific order for the x-axis categories.

    Returns:
        dict: A dictionary containing the data needed for plotting.
    """
    # Get necessary columns
    x_values = data[x_col]
    hue_values = data[hue_col]
    targets = data[target_col]

    # Get unique categories to analyze, excluding empty strings
    unique_x_values = np.unique(x_values[x_values != ''])
    if x_order:
        unique_x_values = [x for x in x_order if x in unique_x_values]

    unique_hue_values = np.unique(hue_values[hue_values != ''])
    
    # Calculate interaction summary
    interaction_summary = {}
    for hue_val in unique_hue_values:
        rates = []
        for x_val in unique_x_values:
            mask = (x_values == x_val) & (hue_values == hue_val)
            targets_in_group = targets[mask]
            
            if len(targets_in_group) > 0:
                rate = np.mean(targets_in_group)
            else:
                rate = np.nan # Use NaN for no data to avoid plotting a zero point
            rates.append(rate)
        interaction_summary[hue_val] = rates
        
    overall_mean = np.mean(targets)

    return {
        'summary': interaction_summary,
        'x_values': unique_x_values,
        'hue_values': unique_hue_values,
        'overall_mean': overall_mean,
        'x_col_name': x_col,
        'hue_col_name': hue_col
    }

def remove_columns(data, columns_to_remove):
    """
    Removes specified columns from a NumPy structured array.

    Args:
        data (np.ndarray): The input structured array.
        columns_to_remove (list): A list of column names to remove.

    Returns:
        np.ndarray: A new array with the specified columns removed.
    """
    # Ensure all columns to remove exist in the data
    valid_columns_to_remove = [col for col in columns_to_remove if col in data.dtype.names]
    return rfn.drop_fields(data, valid_columns_to_remove)

def remove_duplicates(data):
    """
    Removes duplicate rows from a NumPy structured array, based on all columns except 'enrollee_id'.

    Args:
        data (np.ndarray): The input structured array.

    Returns:
        np.ndarray: A new array with duplicate rows removed.
    """
    # Get columns to consider for uniqueness. If 'enrollee_id' is already gone, use all columns.
    if 'enrollee_id' in data.dtype.names:
        key_columns = [name for name in data.dtype.names if name != 'enrollee_id']
        data_subset = data[key_columns]
    else:
        data_subset = data
    
    # np.unique returns the indices of the first occurrences of unique rows
    _, unique_indices = np.unique(data_subset, return_index=True)
    
    # Sort indices to maintain original order as much as possible
    sorted_unique_indices = np.sort(unique_indices)
    
    # Return the unique rows from the original data array
    return data[sorted_unique_indices]

def fill_missing_categorical(data, columns, fill_value='Missing'):
    """
    Replaces empty strings in specified categorical columns with a fill value.
    This function creates a new array with updated dtypes to accommodate the fill value,
    preventing truncation.

    Args:
        data (np.ndarray): The input structured array.
        columns (list): A list of column names to process.
        fill_value (str): The string to replace empty values with.

    Returns:
        np.ndarray: A new array with missing values filled and dtypes adjusted.
    """
    new_dtype_descr = list(data.dtype.descr)
    
    # Create a lookup map for faster access
    dtype_map = {name: i for i, (name, _) in enumerate(new_dtype_descr)}

    # Update dtypes for the specified columns to ensure they can hold the fill_value
    for col in columns:
        # Check if column exists
        if col not in dtype_map:
            continue

        # Get current dtype and length
        idx = dtype_map[col]
        dtype_str = new_dtype_descr[idx][1]

        # Process only string types
        if 'U' in dtype_str:
            try:
                # Correctly extract length from string like '<U15'
                current_len = int(dtype_str[2:]) 
                if len(fill_value) > current_len:
                    new_dtype_descr[idx] = (col, f'<U{len(fill_value)}')
            except (ValueError, IndexError):
                # Fallback for dtypes not in the expected format
                pass

    # Create a new array with the new dtype
    new_data = np.empty(data.shape, dtype=np.dtype(new_dtype_descr))

    # Copy data and fill missing values
    for name in data.dtype.names:
        if name in columns:
            original_col = data[name]
            # Use np.where to efficiently fill missing values
            new_data[name] = np.where(original_col == '', fill_value, original_col)
        else:
            new_data[name] = data[name]
            
    return new_data

def encode_ordinal(data, column_name, category_mapping):
    """
    Encodes a categorical column using a specified ordinal mapping.
    Any value not in the mapping will be set to -1.

    Args:
        data (np.ndarray): The input structured array.
        column_name (str): The name of the column to encode.
        category_mapping (dict): A dictionary mapping categories to integers.

    Returns:
        np.ndarray: A new array with the column encoded and dtype updated.
    """
    if column_name not in data.dtype.names:
        return data

    # Create a new array for the encoded column, defaulting to -1
    encoded_column = np.full(len(data), -1, dtype=np.int32)

    # Apply mapping
    for i, value in enumerate(data[column_name]):
        encoded_column[i] = category_mapping.get(value, -1)

    # Drop the old field and append the new one to handle dtype changes
    temp_data = rfn.drop_fields(data, column_name)
    new_data = rfn.append_fields(temp_data, column_name, encoded_column, usemask=False)
    
    # Reorder fields to maintain original order
    original_order = list(data.dtype.names)
    current_order = list(new_data.dtype.names)
    # Move the new column to its original position
    current_order.insert(original_order.index(column_name), current_order.pop(current_order.index(column_name)))
    
    return new_data[current_order]