# HR Analytics: Job Change of Data Scientists

## 1. Mô tả
Dự án xây dựng mô hình học máy dự đoán khả năng thay đổi công việc của ứng viên sau khóa đào tạo. Điểm đặc biệt của dự án là **KHÔNG sử dụng Pandas hay Scikit-learn models**. Toàn bộ quy trình từ xử lý dữ liệu (Data Manipulation) đến xây dựng thuật toán (Random Forest) đều được cài đặt từ đầu sử dụng **NumPy**.

---

## 2. Mục lục
- [HR Analytics: Job Change of Data Scientists](#hr-analytics-job-change-of-data-scientists)
  - [1. Mô tả](#1-mô-tả)
  - [2. Mục lục](#2-mục-lục)
  - [3. Giới thiệu](#3-giới-thiệu)
    - [Bài toán](#bài-toán)
    - [Mục tiêu](#mục-tiêu)
    - [Động lực \& Ứng dụng](#động-lực--ứng-dụng)
  - [4. Dataset](#4-dataset)
  - [5. Method](#5-method)
    - [5.1. Data Processing (Không Pandas)](#51-data-processing-không-pandas)
    - [5.2. Thuật toán: Random Forest](#52-thuật-toán-random-forest)
  - [6. Installation \& Setup](#6-installation--setup)
  - [7. Usage](#7-usage)
  - [8. Results](#8-results)
  - [9. Project Structure](#9-project-structure)
  - [10. Challenges \& Solutions](#10-challenges--solutions)
  - [11. Future Improvements](#11-future-improvements)
  - [12. Contributors](#12-contributors)
  - [13. License](#13-license)
---

## 3. Giới thiệu

### Bài toán
Một công ty hoạt động trong lĩnh vực Big Data và Data Science muốn tuyển dụng các nhà khoa học dữ liệu từ những ứng viên đã hoàn thành các khóa đào tạo do công ty tổ chức. Tuy nhiên, nhiều ứng viên sau khi đào tạo xong lại tìm kiếm việc làm ở công ty khác.

### Mục tiêu
Xây dựng mô hình dự báo xác suất một ứng viên sẽ tìm việc mới (`target = 1`) hay ở lại làm việc cho công ty (`target = 0`).

### Động lực & Ứng dụng
*   **Tối ưu chi phí:** Giúp công ty giảm chi phí tuyển dụng và đào tạo bằng cách tập trung vào nhóm ứng viên có khả năng gắn bó cao.
*   **Thách thức kỹ thuật:** Dự án này được thực hiện để chứng minh khả năng hiểu sâu về **toán học**, **thuật toán** và kỹ thuật **tối ưu hóa code** bằng cách loại bỏ sự phụ thuộc vào các thư viện high-level như Pandas.

---

## 4. Dataset

*   **Nguồn dữ liệu:** [HR Analytics: Job Change of Data Scientists](https://www.kaggle.com/arashnic/hr-analytics-job-change-of-data-scientists) (Kaggle).
*   **Kích thước:** ~19,000 dòng, 14 cột.
*   **Đặc điểm nổi bật:**
    *   Dữ liệu hỗn hợp (Numerical & Categorical).
    *   Tỷ lệ giá trị thiếu (Missing values) cao ở một số cột quan trọng (ví dụ: `company_type` thiếu ~32%).
    *   Mất cân bằng dữ liệu (Imbalance Class): Tỷ lệ 75:25.
*   **Các đặc trưng chính:**
    *   `city_development_index`: Chỉ số phát triển của thành phố.
    *   `education_level`, `experience`: Trình độ học vấn và thâm niên.
    *   `company_size`, `company_type`: Thông tin về công ty hiện tại.

---

## 5. Method

Dự án tuân thủ nghiêm ngặt quy tắc **chỉ NumPy**:

### 5.1. Data Processing (Không Pandas)
*   **Loading:** Sử dụng `np.genfromtxt` để đọc dữ liệu vào các mảng cấu trúc (Structured Arrays).
*   **Manipulation:** Sử dụng `numpy.lib.recfunctions` để thao tác trên các trường dữ liệu.
*   **Preprocessing:**
    *   Tự viết hàm Ordinal Encoding và One-Hot Encoding.
    *   Xử lý Missing Values bằng cách tạo category riêng ("Missing") để giữ lại tín hiệu dự báo.
    *   Oversampling: Tự cài đặt thuật toán **Random Oversampling** để cân bằng dữ liệu.

### 5.2. Thuật toán: Random Forest
Mô hình được xây dựng từ đầu tại `src/models.py`:
*   **Decision Tree:** Cài đặt thuật toán CART sử dụng **Gini Impurity**.
    *   Tối ưu hóa tính toán Gini bằng `np.einsum`.
    *   Tăng tốc tìm điểm cắt (threshold) bằng phương pháp **Quantile Binning** (dùng `np.percentile`).
*   **Random Forest:**
    *   Sử dụng kỹ thuật **Bagging** (Bootstrap Aggregating).
    *   Triển khai huấn luyện song song đa luồng (**Parallel Processing**) sử dụng `ProcessPoolExecutor` để tận dụng đa nhân CPU.

---

## 6. Installation & Setup

Dự án yêu cầu Python 3.8+.

1.  **Clone repository:**
    ```bash
    git clone https://github.com/yourusername/hr-analytics-numpy.git
    cd hr-analytics-numpy
    ```

2.  **Cài đặt các thư viện cần thiết:**
    (Lưu ý: Chỉ sử dụng các thư viện cơ bản và visualization)
    ```bash
    pip install -r requirements.txt
    ```

---

## 7. Usage

Dự án được tổ chức thành 3 notebook chạy theo thứ tự:

1.  **Khám phá dữ liệu:**
    ```bash
    jupyter notebook notebooks/01_data_exploration.ipynb
    ```
    *Phân tích đơn biến, đa biến và tìm ra các insight quan trọng.*

2.  **Tiền xử lý:**
    ```bash
    jupyter notebook notebooks/02_preprocessing.ipynb
    ```
    *Làm sạch, mã hóa và chuẩn bị dữ liệu cho mô hình.*

3.  **Huấn luyện & Đánh giá:**
    ```bash
    jupyter notebook notebooks/03_modeling.ipynb
    ```
    *Huấn luyện Random Forest tự viết, đánh giá bằng Cross-Validation.*

---

## 8. Results

*   **Metric đánh giá:** F1-Score, Precision, Recall, ROC-AUC.
*   **Hiệu năng:** Mô hình Random Forest tự xây dựng đạt độ chính xác tương đương với Scikit-learn nhưng cho phép tùy biến sâu hơn.
*   **Feature Importance:**
    *   `city_development_index` là yếu tố quan trọng nhất.
    *   Việc thiếu thông tin công ty (`company_size_is_missing`) là một chỉ báo mạnh cho việc ứng viên muốn nghỉ việc.

*(Xem chi tiết biểu đồ ROC và Confusion Matrix trong notebook 03)*

---

## 9. Project Structure

```
├── data/
│   ├── raw/                   # Dữ liệu gốc (aug_train.csv, aug_test.csv)
│   └── processed/             # Dữ liệu sau khi xử lý
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── __init__.py
│   ├── data_processing.py     # Module xử lý dữ liệu (NumPy only)
│   ├── models.py              # Cài đặt Random Forest & Decision Tree
│   └── visualization.py       # Các hàm vẽ biểu đồ (Matplotlib/Seaborn)
├── README.md
└── requirements.txt
```

---

## 10. Challenges & Solutions

| Thách thức | Giải pháp |
| :--- | :--- |
| **Không có Pandas DataFrame** | Sử dụng **NumPy Structured Arrays** và `recfunctions` để quản lý dữ liệu dạng bảng với nhiều kiểu dữ liệu khác nhau. |
| **Tốc độ huấn luyện chậm** | 1. **Vectorization**: Thay thế vòng lặp bằng broadcasting.<br>2. **Parallelization**: Đa luồng hóa quá trình xây dựng cây.<br>3. **Math Opt**: Dùng `np.einsum` cho các phép tính tổng. |
| **Tìm điểm cắt tối ưu** | Thay vì duyệt qua hàng ngàn giá trị, sử dụng **Percentiles** để chọn ra các ngưỡng tiềm năng, giảm độ phức tạp tính toán. |

---

## 11. Future Improvements

*   Cài đặt thêm thuật toán **Gradient Boosting** từ đầu.
*   Tối ưu hóa hơn nữa việc sử dụng bộ nhớ cho tập dữ liệu lớn hơn.
*   Triển khai Grid Search tự động để tinh chỉnh tham số (Hyperparameter Tuning).

---

## 12. Contributors

**Hoàng Minh Trung** - 23TNT

*   MSSV: *23122014*
*   Email: *23122014@student.hcmus.edu.vn*
*   [LinkedIn](https://www.linkedin.com/in/trung-ho%C3%A0ng-minh-b83216215/)
*   [GitHub](https://github.com/Trung0Minh)

---

## 13. License

Dự án này được cấp phép theo giấy phép [MIT License](LICENSE).