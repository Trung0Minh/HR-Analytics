import numpy as np
from collections import Counter
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

class Node:
    """
    A class representing a single node in a decision tree.
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
    An optimized Decision Tree classifier implementation.
    """
    def __init__(self, min_samples_split=2, max_depth=100, n_feats=None, min_info_gain=0):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.min_info_gain = min_info_gain
        self.root = None

    def fit(self, X, y):
        """
        Builds the decision tree.
        """
        self.n_feats = self.n_feats if self.n_feats is not None else X.shape[1]
        self.root = self._grow_tree(X, y)

    def predict(self, X):
        """
        Makes predictions for a set of samples.
        """
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _grow_tree(self, X, y, depth=0):
        """
        Recursively grows the decision tree.
        """
        n_samples, n_features = X.shape
        if n_samples == 0:
            return None
        
        n_labels = len(np.unique(y))

        # Stopping criteria
        if (depth >= self.max_depth
                or n_labels == 1
                or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        n_feats_to_sample = min(self.n_feats, n_features)
        feat_idxs = np.random.choice(n_features, n_feats_to_sample, replace=False)

        # Find the best split
        best_feat, best_thresh = self._best_criteria(X, y, feat_idxs)
        
        if best_feat is not None:
            left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
            if len(left_idxs) > 0 and len(right_idxs) > 0:
                left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
                right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
                return Node(best_feat, best_thresh, left, right)
        
        leaf_value = self._most_common_label(y)
        return Node(value=leaf_value)

    def _best_criteria(self, X, y, feat_idxs):
        """
        Finds the best feature and threshold for a split.
        """
        best_gain = self.min_info_gain
        split_idx, split_thresh = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            if len(thresholds) > 20:
                p = np.linspace(0, 100, 21)
                thresholds = np.unique(np.percentile(X_column, p))

            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold

        return split_idx, split_thresh

    def _information_gain(self, y, X_column, split_thresh):
        """
        Calculates the information gain of a split.
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
        Splits a column based on a threshold.
        """
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _gini(self, y):
        """
        Calculates the Gini impurity of a set of labels.
        """
        if len(y) == 0:
            return 0
        counts = np.array(list(Counter(y).values()))
        proportions = counts / len(y)
        return 1 - np.sum(proportions**2)

    def _most_common_label(self, y):
        """
        Finds the most common label in a set of labels.
        """
        if len(y) == 0:
            return None
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def _traverse_tree(self, x, node):
        """
        Traverses the tree to predict a label for a single sample.
        """
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)


def _train_single_tree(args):
    """
    Helper function to train a single tree (for parallel processing).
    """
    X, y, min_samples_split, max_depth, n_feats, min_info_gain, seed = args
    np.random.seed(seed)
    
    tree = DecisionTree(
        min_samples_split=min_samples_split,
        max_depth=max_depth,
        n_feats=n_feats,
        min_info_gain=min_info_gain
    )
    
    # Bootstrap sampling
    n_samples = X.shape[0]
    idxs = np.random.choice(n_samples, n_samples, replace=True)
    X_sample, y_sample = X[idxs], y[idxs]
    
    tree.fit(X_sample, y_sample)
    return tree


class RandomForest:
    """
    An optimized Random Forest classifier with parallel training and faster predictions.
    """
    def __init__(self, n_trees=100, min_samples_split=2, max_depth=100, 
                 n_feats=None, min_info_gain=0, n_jobs=-1):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.min_info_gain = min_info_gain
        self.n_jobs = n_jobs if n_jobs > 0 else mp.cpu_count()
        self.trees = []

    def fit(self, X, y):
        """
        Trains the forest using parallel processing for speed.
        """
        self.trees = []
        num_features = X.shape[1]
        n_feats_to_use = self.n_feats if self.n_feats is not None else int(np.sqrt(num_features))
        
        # Prepare arguments for parallel processing
        args_list = [
            (X, y, self.min_samples_split, self.max_depth, 
             n_feats_to_use, self.min_info_gain, np.random.randint(0, 1e9))
            for _ in range(self.n_trees)
        ]
        
        # Train trees in parallel
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = [executor.submit(_train_single_tree, args) for args in args_list]
            
            # Collect results with progress bar
            for future in tqdm(as_completed(futures), total=self.n_trees, desc="Training Forest"):
                self.trees.append(future.result())

    def predict(self, X):
        """
        Makes predictions using optimized aggregation.
        """
        n_samples = X.shape[0]
        
        # Collect predictions from all trees
        all_predictions = np.zeros((self.n_trees, n_samples))
        
        for i, tree in enumerate(self.trees):
            all_predictions[i] = tree.predict(X)
        
        # Optimized voting using scipy's mode or manual implementation
        final_predictions = self._fast_mode(all_predictions)
        return final_predictions
    
    def _fast_mode(self, predictions):
        """
        Fast mode calculation for predictions across trees.
        Uses numpy operations instead of Counter for each sample.
        """
        n_samples = predictions.shape[1]
        result = np.zeros(n_samples)
        
        for i in range(n_samples):
            # Get all predictions for this sample
            sample_preds = predictions[:, i]
            # Find unique values and their counts
            unique_vals, counts = np.unique(sample_preds, return_counts=True)
            # Return the value with maximum count
            result[i] = unique_vals[np.argmax(counts)]
        
        return result

    def predict_proba(self, X):
        """
        Predicts class probabilities by computing the proportion of votes.
        """
        n_samples = X.shape[0]
        all_predictions = np.zeros((self.n_trees, n_samples))
        
        for i, tree in enumerate(self.trees):
            all_predictions[i] = tree.predict(X)
        
        # Get unique classes
        classes = np.unique(all_predictions)
        n_classes = len(classes)
        
        # Calculate probabilities
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

def train_val_split(X, y, val_size=0.2):
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
    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    
    print("Metrics per class:")
    for cls in np.unique(y):
        pre, rec, f1 = calculate_metrics_per_class(y, y_pred, cls)
        print(f"  Class {cls}: Precision={pre:.4f}, Recall={rec:.4f}, F1-Score={f1:.4f}")
    print("------------------------------------")

def _stratified_k_fold_split(X, y, k):
    """A from-scratch implementation of stratified k-fold splitting."""
    # Get indices for each class
    class_indices = [np.where(y == cls)[0] for cls in np.unique(y)]
    
    # Initialize folds
    folds = [[] for _ in range(k)]
    
    # Distribute indices of each class into folds
    for indices in class_indices:
        np.random.shuffle(indices)
        for i, index in enumerate(indices):
            folds[i % k].append(index)
            
    # Create train/test splits
    for i in range(k):
        test_indices = np.array(folds[i])
        train_indices = np.concatenate([folds[j] for j in range(k) if i != j])
        yield train_indices, test_indices

def cross_validate_model(clf_class, clf_params, X, y, k=5):
    """
    Performs k-fold cross-validation for a given classifier.
    
    Args:
        clf_class: The classifier class (e.g., RandomForest).
        clf_params (dict): Parameters to initialize the classifier.
        X (np.array): Feature data.
        y (np.array): Target labels.
        k (int): Number of folds.
    """
    print(f"--- Bắt đầu {k}-Fold Cross-Validation ---")
    
    # Store scores for each fold
    all_scores = {
        'accuracy': [],
        'precision_0': [], 'recall_0': [], 'f1_0': [],
        'precision_1': [], 'recall_1': [], 'f1_1': []
    }
    
    # Stratified K-Fold
    for fold, (train_idx, val_idx) in enumerate(_stratified_k_fold_split(X, y, k)):
        print(f"  Fold {fold+1}/{k}...")
        
        # Get data for this fold
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Initialize and train the classifier
        clf = clf_class(**clf_params)
        clf.fit(X_train, y_train)
        
        # Make predictions
        y_pred = clf.predict(X_val)
        
        # Calculate and store metrics
        accuracy = np.sum(y_val == y_pred) / len(y_val)
        all_scores['accuracy'].append(accuracy)
        
        for cls in [0, 1]:
            pre, rec, f1 = calculate_metrics_per_class(y_val, y_pred, cls)
            all_scores[f'precision_{cls}'].append(pre)
            all_scores[f'recall_{cls}'].append(rec)
            all_scores[f'f1_{cls}'].append(f1)
            
    # --- In kết quả trung bình ---
    print("\n--- Kết quả Cross-Validation (Trung bình) ---")
    print(f"  Average Accuracy: {np.mean(all_scores['accuracy']):.4f} (std: {np.std(all_scores['accuracy']):.4f})")
    
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
    
