# HR Analytics: Job Change of Data Scientists

## 1. Mô tả
Dự án xây dựng mô hình học máy dự đoán khả năng thay đổi công việc của ứng viên sau khóa đào tạo. Toàn bộ quy trình từ xử lý dữ liệu đến xây dựng thuật toán (Random Forest) đều được cài đặt từ đầu sử dụng **NumPy**.

[Link GitHub](https://github.com/Trung0Minh/HR-Analytics.git)

---

## 2. Mục lục
- [HR Analytics: Job Change of Data Scientists](#hr-analytics-job-change-of-data-scientists)
  - [1. Mô tả](#1-mô-tả)
  - [2. Mục lục](#2-mục-lục)
  - [3. Giới thiệu](#3-giới-thiệu)
    - [Bối cảnh \& Bài toán](#bối-cảnh--bài-toán)
    - [Mục tiêu](#mục-tiêu)
    - [Động lực \& Ứng dụng](#động-lực--ứng-dụng)
  - [4. Dataset](#4-dataset)
    - [Bảng đặc trưng](#bảng-đặc-trưng)
  - [5. Phương pháp](#5-phương-pháp)
    - [5.1. Xử lý dữ liệu](#51-xử-lý-dữ-liệu)
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

### Bối cảnh & Bài toán
Một công ty hoạt động trong lĩnh vực Big Data và Data Science tổ chức các khóa đào tạo và muốn tuyển dụng Data Scientist từ chính nguồn học viên này. Rất nhiều người đã đăng ký tham gia đào tạo.

Vấn đề đặt ra là công ty muốn phân loại và xác định xem ứng viên nào thực sự muốn làm việc cho công ty sau khóa học, và ứng viên nào đang tìm kiếm cơ hội việc làm mới nói chung. Việc dự đoán chính xác giúp công ty:
*   Giảm thiểu chi phí và thời gian tuyển dụng.
*   Nâng cao chất lượng đào tạo và quy hoạch các khóa học phù hợp.
*   Phân loại ứng viên hiệu quả hơn.

### Mục tiêu
1.  **Dự báo:** Xây dựng mô hình sử dụng dữ liệu nhân khẩu học, giáo dục và kinh nghiệm để dự đoán xác suất một ứng viên sẽ tìm kiếm công việc mới (`target = 1`) hay không (`target = 0`).
2.  **Interpretability:** Xác định các yếu tố nào ảnh hưởng lớn nhất đến quyết định thay đổi công việc của nhân viên, phục vụ cho các nghiên cứu nhân sự (HR Research).

### Động lực & Ứng dụng
Dự án này không chỉ giải quyết bài toán phân loại nhị phân mà còn hướng tới việc **tối ưu hóa quy trình quản trị nhân sự (HR Analytics)** thông qua dữ liệu.

---

## 4. Dataset

*   **Nguồn dữ liệu:** [HR Analytics: Job Change of Data Scientists](https://www.kaggle.com/datasets/arashnic/hr-analytics-job-change-of-data-scientists/data) (Kaggle).
*   **Kích thước:** Tập huấn luyện (~19,158 dòng) và tập kiểm tra (~2,129 dòng), 14 cột.
*   **Đặc điểm nổi bật:**
    *   Dữ liệu hỗn hợp (Numerical & Categorical).
    *   Tỷ lệ giá trị thiếu cao ở một số cột quan trọng (ví dụ: `company_type` thiếu ~32%).
    *   Mất cân bằng nhãn: Tỷ lệ 3:1.

### Bảng đặc trưng

| Tên cột | Mô tả |
| :--- | :--- |
| **`enrollee_id`** | ID duy nhất định danh cho từng ứng viên. |
| **`city`** | Mã thành phố nơi ứng viên sinh sống. |
| **`city_development_index`** | Chỉ số phát triển của thành phố (đã được chuẩn hóa/scaled). |
| **`gender`** | Giới tính của ứng viên. |
| **`relevent_experience`** | Kinh nghiệm làm việc liên quan đến Data Science. |
| **`enrolled_university`** | Loại khóa học đại học mà ứng viên đang theo học (nếu có). |
| **`education_level`** | Trình độ học vấn cao nhất của ứng viên. |
| **`major_discipline`** | Chuyên ngành học chính của ứng viên. |
| **`experience`** | Tổng số năm kinh nghiệm làm việc. |
| **`company_size`** | Số lượng nhân viên trong công ty hiện tại của ứng viên. |
| **`company_type`** | Loại hình doanh nghiệp của công ty hiện tại. |
| **`last_new_job`** | Số năm chênh lệch giữa công việc trước đó và công việc hiện tại. |
| **`training_hours`** | Số giờ đào tạo đã hoàn thành. |
| **`target`** | **Biến mục tiêu:** <br> `0` – Không tìm kiếm việc làm mới. <br> `1` – Đang tìm kiếm việc làm mới. |

---

## 5. Phương pháp

### 5.1. Xử lý dữ liệu
*   **Đọc dữ liệu:** Sử dụng `np.genfromtxt` để tải dữ liệu CSV vào các mảng cấu trúc (Structured Arrays), giúp quản lý dữ liệu hỗn hợp hiệu quả mà không cần Pandas DataFrame.
*   **Thao tác (Manipulation):** Sử dụng thư viện `numpy.lib.recfunctions` để thực hiện các thao tác phức tạp như thêm cột mới, xóa cột, hoặc nối mảng cấu trúc.
*   **Tiền xử lý:**
    *   **Mã hóa:** Chuyển đổi các biến phân loại sang dạng số dựa trên thứ tự logic.
    *   **Xử lý dữ liệu thiếu (Imputation):** Điền giá trị thiếu bằng **trung vị** kết hợp với việc tạo thêm các cột chỉ báo (Indicator columns) để mô hình học được mẫu hình của dữ liệu bị khuyết.
    *   **Cân bằng dữ liệu:** Cài đặt thuật toán **Random Oversampling** (`random_oversample`) để nhân bản ngẫu nhiên các mẫu thuộc lớp thiểu số, giải quyết vấn đề mất cân bằng dữ liệu.

### 5.2. Thuật toán: Random Forest

*   **Decision Tree - CART (Classification And Regression Tree):**
    *   Sử dụng tiêu chí **Gini Impurity** để phân chia nút.
    *   Sử dụng `np.einsum` để tính toán tổng bình phương vector nhanh thay vì dùng vòng lặp hay phép cộng thông thường.
    *   **Tối ưu tốc độ:** Dùng **Quantile Binning** (sử dụng `np.percentile`) để giảm số lượng điểm cắt cần duyệt, giúp tăng tốc độ huấn luyện lên nhiều lần trên dữ liệu liên tục.
*   **Random Forest:**
    *   Triển khai **Bagging** (Bootstrap Aggregating) kết hợp với **Random Feature Selection**.
    *   **Xử lý song song:** Sử dụng `ProcessPoolExecutor` để huấn luyện đa luồng (Multi-processing), tận dụng tối đa sức mạnh của đa nhân CPU, giúp tăng tốc độ huấn luyện.

---

## 6. Installation & Setup

Dự án yêu cầu Python 3.11.5

1.  **Clone repository:**
    ```bash
    git clone https://github.com/Trung0Minh/HR-Analytics.git
    cd HR-Analytics
    ```

2.  **Cài đặt các thư viện cần thiết:**
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
    *Huấn luyện mô hình Random Forest. Đánh giá bằng Cross-Validation.*

Hoặc đơn giản là vào từng notebook và nhấn `Run all`.

---

# 8. Results

Mô hình Random Forest tự xây dựng đạt kết quả khả quan với **ROC-AUC ~ 0.80** trên tập kiểm thử (Validation), chứng minh khả năng phân loại tốt trong bối cảnh dữ liệu mất cân bằng.

| Tập dữ liệu | Accuracy | AUC | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 84.89% | 0.94 | 0.85 | 0.85 | 0.85 |
| **Validation** | 78.93% | 0.80 | 0.73 | 0.76 | 0.74 |

**Phân tích nhanh:**
*   **Chiến lược:** Mô hình ưu tiên **Recall** (phát hiện được nhiều nhất số người muốn nghỉ việc) thay vì Precision, phù hợp với bài toán quản trị rủi ro nhân sự.
*   **Top feature:** `city_development_index` là yếu tố dự báo mạnh nhất.

---

## 9. Project Structure

```
├── data/
│   ├── raw/                            # Dữ liệu gốc
|       ├── aug_train.csv
|       └── aug_test.csv
│   └── processed/                      # Dữ liệu sau khi xử lý
|       ├── aug_train_processed.csv
|       └── aug_test_processed.csv
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── __init__.py
│   ├── data_processing.py              # Module xử lý dữ liệu
│   ├── visualization.py                # Các hàm vẽ biểu đồ
│   └── models.py                       # Cài đặt Random Forest
├── README.md
├── LICENSE
└── requirements.txt                    # Các thư viện cần thiết
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

---

## 12. Contributors

**Hoàng Minh Trung** - 23122014 - 23TNT

*   Email: *23122014@student.hcmus.edu.vn*
*   [LinkedIn](https://www.linkedin.com/in/trung-ho%C3%A0ng-minh-b83216215/)
*   [GitHub](https://github.com/Trung0Minh)

---

## 13. License

Dự án này được cấp phép theo giấy phép [MIT License](LICENSE).