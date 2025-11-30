import numpy as np
from collections import Counter
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

class Node:
    """
    Một nút đơn trong cây quyết định.
    """
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class DecisionTree:
    """
    Triển khai Decision Tree.
    """
    def __init__(self, min_samples_split=2, max_depth=100, n_feats=None, min_info_gain=0):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.min_info_gain = min_info_gain
        self.root = None

    def fit(self, X, y):
        """
        Xây dựng cây quyết định.
        """
        self.n_feats = self.n_feats if self.n_feats is not None else X.shape[1]
        self.feature_importances_ = np.zeros(X.shape[1])
        self.root = self._grow_tree(X, y)

    def predict(self, X):
        """
        Thực hiện dự đoán cho một tập hợp các mẫu.
        """
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _grow_tree(self, X, y, depth=0):
        """
        Phát triển cây quyết định một cách đệ quy.
        """
        n_samples, n_features = X.shape
        if n_samples == 0:
            return None
        
        n_labels = len(np.unique(y))

        # Tiêu chí dừng
        if (depth >= self.max_depth
                or n_labels == 1
                or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        n_feats_to_sample = min(self.n_feats, n_features)
        
        # Chọn ngẫu nhiên các đặc trưng
        feat_idxs = np.random.choice(n_features, n_feats_to_sample, replace=False)

        # Tìm phép chia tốt nhất
        best_feat, best_thresh, best_gain = self._best_criteria(X, y, feat_idxs)
        
        if best_feat is not None:
            self.feature_importances_[best_feat] += best_gain * n_samples
            left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
            if len(left_idxs) > 0 and len(right_idxs) > 0:
                left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
                right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
                return Node(best_feat, best_thresh, left, right)
        
        leaf_value = self._most_common_label(y)
        return Node(value=leaf_value)

    def _best_criteria(self, X, y, feat_idxs):
        """
        Tìm đặc trưng và ngưỡng tốt nhất cho một phép chia.
        """
        best_gain = self.min_info_gain
        split_idx, split_thresh = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            # Tối ưu hóa: Nếu quá nhiều ngưỡng, chỉ lấy các phân vị (Quantile Binning)
            if len(thresholds) > 20:
                p = np.linspace(0, 100, 21)
                thresholds = np.unique(np.percentile(X_column, p))

            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold

        return split_idx, split_thresh, best_gain

    def _information_gain(self, y, X_column, split_thresh):
        """
        Tính toán information gain của một phép chia.
        """
        parent_gini = self._gini(y)
        left_idxs, right_idxs = self._split(X_column, split_thresh)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        g_l, g_r = self._gini(y[left_idxs]), self._gini(y[right_idxs])
        child_gini = (n_l / n) * g_l + (n_r / n) * g_r
        return parent_gini - child_gini

    def _split(self, X_column, split_thresh):
        """
        Chia một cột dựa trên một ngưỡng.
        """
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _gini(self, y):
        """
        Tính toán Gini impurity.
        Sử dụng np.einsum để tối ưu hóa tính toán tổng bình phương.
        """
        if len(y) == 0:
            return 0
        counts = np.array(list(Counter(y).values()))
        proportions = counts / len(y)
        # Sử dụng Einstein summation convention thay vì np.sum(proportions**2)
        # i,i-> nghĩa là nhân vector với chính nó và tính tổng (dot product)
        return 1 - np.einsum('i,i->', proportions, proportions)

    def _most_common_label(self, y):
        """
        Tìm nhãn phổ biến nhất trong một tập hợp các nhãn.
        """
        if len(y) == 0:
            return None
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def _traverse_tree(self, x, node):
        """
        Duyệt cây để dự đoán nhãn cho một mẫu đơn.
        """
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)


def _train_single_tree(args):
    """
    Hàm hỗ trợ để huấn luyện một cây đơn (cho xử lý song song).
    """
    X, y, min_samples_split, max_depth, n_feats, min_info_gain, seed = args
    
    # Thiết lập seed cho tiến trình này để đảm bảo tính tái lập
    np.random.seed(seed)
    
    tree = DecisionTree(
        min_samples_split=min_samples_split,
        max_depth=max_depth,
        n_feats=n_feats,
        min_info_gain=min_info_gain
    )
    
    # Lấy mẫu Bootstrap (Sampling with replacement)
    n_samples = X.shape[0]
    idxs = np.random.choice(n_samples, n_samples, replace=True)
    X_sample, y_sample = X[idxs], y[idxs]
    
    tree.fit(X_sample, y_sample)
    return tree


class RandomForest:
    """
    Thuật toán Random Forest
    """
    def __init__(self, n_trees=100, min_samples_split=2, max_depth=100, 
                 n_feats=None, min_info_gain=0, n_jobs=-1, random_state=None):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.min_info_gain = min_info_gain
        self.n_jobs = n_jobs if n_jobs > 0 else mp.cpu_count()
        self.random_state = random_state # Thêm tham số seed
        self.trees = []

    def fit(self, X, y):
        """
        Huấn luyện rừng cây bằng cách sử dụng xử lý song song.
        """
        self.trees = []
        num_features = X.shape[1]
        n_feats_to_use = self.n_feats if self.n_feats is not None else int(np.sqrt(num_features))
        
        # Khởi tạo bộ sinh số ngẫu nhiên để kiểm soát seed cho từng cây
        rng = np.random.RandomState(self.random_state)
        # Sinh ra danh sách seed cố định cho n cây
        tree_seeds = rng.randint(0, 1e9, size=self.n_trees)
        
        # Chuẩn bị các tham số cho xử lý song song
        args_list = [
            (X, y, self.min_samples_split, self.max_depth, 
             n_feats_to_use, self.min_info_gain, tree_seeds[i])
            for i in range(self.n_trees)
        ]
        
        # Huấn luyện các cây song song
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = [executor.submit(_train_single_tree, args) for args in args_list]
            
            # Thu thập kết quả với thanh tiến trình
            for future in tqdm(as_completed(futures), total=self.n_trees, desc="Training Forest"):
                self.trees.append(future.result())

        # Tính toán Feature Importance trung bình
        self.feature_importances_ = np.zeros(X.shape[1])
        for tree in self.trees:
            self.feature_importances_ += tree.feature_importances_
        
        self.feature_importances_ /= self.n_trees
        
        # Chuẩn hóa để tổng bằng 1
        sum_importances = np.sum(self.feature_importances_)
        if sum_importances > 0:
            self.feature_importances_ /= sum_importances

    def predict(self, X):
        """
        Thực hiện dự đoán bằng cách sử dụng tổng hợp tối ưu.
        """
        n_samples = X.shape[0]
        
        # Thu thập dự đoán từ tất cả các cây
        all_predictions = np.zeros((self.n_trees, n_samples))
        
        for i, tree in enumerate(self.trees):
            all_predictions[i] = tree.predict(X)
        
        # Bỏ phiếu tối ưu bằng cách sử dụng mode
        final_predictions = self._fast_mode(all_predictions)
        return final_predictions
    
    def _fast_mode(self, predictions):
        """
        Tính toán mode nhanh cho các dự đoán trên các cây.
        """
        n_samples = predictions.shape[1]
        result = np.zeros(n_samples)
        
        for i in range(n_samples):
            # Lấy tất cả các dự đoán cho mẫu này
            sample_preds = predictions[:, i]
            unique_vals, counts = np.unique(sample_preds, return_counts=True)
            result[i] = unique_vals[np.argmax(counts)]
        
        return result

    def predict_proba(self, X):
        """
        Dự đoán xác suất lớp bằng cách tính tỷ lệ phiếu bầu.
        """
        n_samples = X.shape[0]
        all_predictions = np.zeros((self.n_trees, n_samples))
        
        for i, tree in enumerate(self.trees):
            all_predictions[i] = tree.predict(X)
        
        # Lấy các lớp duy nhất
        classes = np.unique(all_predictions)
        n_classes = len(classes)
        
        # Tính toán xác suất
        probas = np.zeros((n_samples, n_classes))
        for i in range(n_samples):
            sample_preds = all_predictions[:, i]
            for j, cls in enumerate(classes):
                probas[i, j] = np.sum(sample_preds == cls) / self.n_trees
        
        return probas

def get_features_and_target(data, target_column='target'):
    if data is None:
        return None, None
    feature_names = [name for name in data.dtype.names if name != target_column]
    X = np.array(data[feature_names].tolist(), dtype=np.float64)
    
    y = None
    if target_column in data.dtype.names:
        y = data[target_column].astype(int)
        
    return X, y

def train_val_split(X, y, val_size=0.2, random_state=42):
    """Chia tập train/val với random_state để tái lập kết quả."""
    if random_state is not None:
        np.random.seed(random_state)
        
    n_samples = X.shape[0]
    shuffled_indices = np.random.permutation(n_samples)
    val_samples = int(n_samples * val_size)
    val_indices = shuffled_indices[:val_samples]
    train_indices = shuffled_indices[val_samples:]
    return X[train_indices], X[val_indices], y[train_indices], y[val_indices]

def calculate_metrics_per_class(y_true, y_pred, class_label):
    """Tính toán precision, recall, và f1-score cho một lớp cụ thể."""
    tp = np.sum((y_true == class_label) & (y_pred == class_label))
    fp = np.sum((y_true != class_label) & (y_pred == class_label))
    fn = np.sum((y_true == class_label) & (y_pred != class_label))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

def evaluate_model_detailed(clf, X, y, dataset_name):
    """Thực hiện dự đoán và in ra các chỉ số đánh giá chi tiết."""
    print(f"-- Đánh giá trên tập {dataset_name} --")
    y_pred = clf.predict(X)
    
    overall_accuracy = np.sum(y == y_pred) / len(y)
    print(f"Overall accuracy: {overall_accuracy:.4f}")
    
    print("Chỉ số theo từng lớp:")
    for cls in np.unique(y):
        pre, rec, f1 = calculate_metrics_per_class(y, y_pred, cls)
        print(f"  Class {cls}: Precision={pre:.4f}, Recall={rec:.4f}, F1-Score={f1:.4f}")
    print("------------------------------------")

def random_oversample(X, y, random_state=42):
    """
    Thực hiện Random Oversampling bằng NumPy để cân bằng dữ liệu.
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Xác định các lớp và số lượng mẫu
    classes, counts = np.unique(y, return_counts=True)
    max_counts = np.max(counts)
    
    X_resampled = X.copy()
    y_resampled = y.copy()
    
    for cls in classes:
        # Lấy chỉ số của các mẫu thuộc lớp hiện tại
        cls_indices = np.where(y == cls)[0]
        n_samples = len(cls_indices)
        
        # Nếu số lượng mẫu ít hơn lớp đa số, tiến hành nhân bản ngẫu nhiên
        if n_samples < max_counts:
            n_diff = max_counts - n_samples
            # Chọn ngẫu nhiên các chỉ số để nhân bản
            random_indices = np.random.choice(cls_indices, n_diff, replace=True)
            
            # Nối các mẫu nhân bản vào dữ liệu gốc
            X_resampled = np.vstack((X_resampled, X[random_indices]))
            y_resampled = np.hstack((y_resampled, y[random_indices]))
            
    # Xáo trộn lại dữ liệu để tránh thứ tự lớp
    perm = np.random.permutation(len(y_resampled))
    return X_resampled[perm], y_resampled[perm]

def _stratified_k_fold_split(X, y, k, random_state=42):
    """Triển khai phân chia k-fold phân tầng."""
    if random_state is not None:
        np.random.seed(random_state)

    # Lấy chỉ số cho từng lớp
    class_indices = [np.where(y == cls)[0] for cls in np.unique(y)]
    
    # Khởi tạo các fold
    folds = [[] for _ in range(k)]
    
    # Phân phối chỉ số của từng lớp vào các fold
    for indices in class_indices:
        np.random.shuffle(indices)
        for i, index in enumerate(indices):
            folds[i % k].append(index)
            
    # Tạo các phần chia train/test
    for i in range(k):
        test_indices = np.array(folds[i])
        train_indices = np.concatenate([folds[j] for j in range(k) if i != j])
        yield train_indices, test_indices

def calculate_roc_auc(y_true, y_prob):
    """
    Tính chỉ số ROC-AUC sử dụng NumPy.
    
    Args:
        y_true: Nhãn thực tế (0 hoặc 1).
        y_prob: Xác suất dự đoán của lớp 1.
    """
    # Sắp xếp theo xác suất giảm dần
    desc_score_indices = np.argsort(y_prob)[::-1]
    y_true = y_true[desc_score_indices]
    y_prob = y_prob[desc_score_indices]

    # Tính True Positives và False Positives tích lũy
    tps = np.cumsum(y_true)
    fps = np.cumsum(1 - y_true)
    
    # Tổng số dương và âm
    n_pos = tps[-1]
    n_neg = fps[-1]
    
    if n_pos == 0 or n_neg == 0:
        return 0.5 # Trường hợp biên

    # Tính TPR và FPR
    tpr = tps / n_pos
    fpr = fps / n_neg
    
    # Thêm điểm (0,0) vào đầu để tính diện tích chính xác
    tpr = np.concatenate([[0], tpr])
    fpr = np.concatenate([[0], fpr])
    
    # Tính diện tích dưới đường cong bằng quy tắc hình thang
    auc = np.trapz(tpr, fpr)
    return auc

def cross_validation(clf_class, clf_params, X, y, k=5, random_state=42):
    """
    Thực hiện k-fold cross-validation, bao gồm tính AUC.
    """
    print(f"--- Bắt đầu {k}-Fold Cross-Validation ---")
    
    # Lưu trữ điểm số cho mỗi fold, thêm 'auc'
    all_scores = {
        'accuracy': [],
        'auc': [],
        'precision_0': [], 'recall_0': [], 'f1_0': [],
        'precision_1': [], 'recall_1': [], 'f1_1': []
    }
    
    # Thêm random_state vào params nếu chưa có
    if 'random_state' not in clf_params:
        clf_params['random_state'] = random_state

    # Phân chia K-Fold phân tầng
    for fold, (train_idx, val_idx) in enumerate(_stratified_k_fold_split(X, y, k, random_state=random_state)):
        print(f"  Fold {fold+1}/{k}...")
        
        # Lấy dữ liệu cho fold này
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Khởi tạo và huấn luyện bộ phân loại
        clf = clf_class(**clf_params)
        clf.fit(X_train, y_train)
        
        # 1. Dự đoán nhãn (cho Accuracy, Precision, Recall, F1)
        y_pred = clf.predict(X_val)
        
        # 2. Dự đoán xác suất (cho AUC)
        # Lấy cột 1 (xác suất của lớp Positive)
        y_prob = clf.predict_proba(X_val)[:, 1]
        
        # Tính toán các chỉ số
        accuracy = np.sum(y_val == y_pred) / len(y_val)
        auc = calculate_roc_auc(y_val, y_prob) # Sử dụng hàm đã viết trước đó
        
        all_scores['accuracy'].append(accuracy)
        all_scores['auc'].append(auc)
        
        for cls in [0, 1]:
            pre, rec, f1 = calculate_metrics_per_class(y_val, y_pred, cls)
            all_scores[f'precision_{cls}'].append(pre)
            all_scores[f'recall_{cls}'].append(rec)
            all_scores[f'f1_{cls}'].append(f1)
            
    # --- In kết quả trung bình ---
    print("\n--- Kết quả Cross-Validation (Trung bình) ---")
    
    # In Accuracy và AUC
    mean_acc = np.mean(all_scores['accuracy'])
    std_acc = np.std(all_scores['accuracy'])
    mean_auc = np.mean(all_scores['auc'])
    std_auc = np.std(all_scores['auc'])
    
    print(f"  Average Accuracy: {mean_acc:.4f} (std: {std_acc:.4f})")
    print(f"  Average AUC:      {mean_auc:.4f} (std: {std_auc:.4f})")
    
    for cls in [0, 1]:
        print(f"  Class {cls}:")
        mean_pre = np.mean(all_scores[f'precision_{cls}'])
        std_pre = np.std(all_scores[f'precision_{cls}'])
        mean_rec = np.mean(all_scores[f'recall_{cls}'])
        std_rec = np.std(all_scores[f'recall_{cls}'])
        mean_f1 = np.mean(all_scores[f'f1_{cls}'])
        std_f1 = np.std(all_scores[f'f1_{cls}'])
        
        print(f"    - Precision: {mean_pre:.4f} (std: {std_pre:.4f})")
        print(f"    - Recall:    {mean_rec:.4f} (std: {std_rec:.4f})")
        print(f"    - F1-Score:  {mean_f1:.4f} (std: {std_f1:.4f})")
        
    print("------------------------------------------")