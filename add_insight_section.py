import json

notebook_path = "/home/trungminh/HK1/KHDL/TH/HW02/HR Analytics/notebooks/01_data_exploration.ipynb"

# Read the notebook content
with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook_content = json.load(f)

# Define the new section to be added
new_section_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 12. Kết luận nâng cao với thông tin chi tiết về kinh doanh"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 12.1. Tại sao `city_development_index` thấp hơn lại tương quan với tỷ lệ Target cao hơn?"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "**Giải thích:**\n",
            "- **Cơ hội hạn chế:** Các thành phố có chỉ số phát triển thấp hơn thường có ít cơ hội việc làm hơn, đặc biệt là trong lĩnh vực công nghệ. Điều này có thể thúc đẩy các lập trình viên tìm kiếm cơ hội ở những nơi khác.\n",
            "- **Mức lương thấp hơn:** Mức lương ở các thành phố kém phát triển có thể thấp hơn, khiến các lập trình viên có động lực chuyển đến các thành phố lớn hơn hoặc các công ty trả lương cao hơn.\n",
            "- **Thiếu tiện ích/phát triển nghề nghiệp:** Môi trường làm việc và cơ hội phát triển nghề nghiệp ở các thành phố này có thể không hấp dẫn bằng, dẫn đến mong muốn thay đổi môi trường.\n",
            "- **Chất lượng cuộc sống:** Các yếu tố về chất lượng cuộc sống, giáo cũng có thể đóng vai trò trong quyết định chuyển đi.\n",
            "**Ý nghĩa kinh doanh:** Các công ty muốn giữ chân nhân tài ở các thành phố này cần đầu tư vào việc cải thiện môi trường làm việc, cung cấp các gói phúc lợi cạnh tranh và cơ hội phát triển nghề nghiệp rõ ràng."
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 12.2. Giải thích về \"không có kinh nghiệm liên quan\" với nhiều năm kinh nghiệm"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "**Giải thích:**\n",
            "- **Chuyển đổi nghề nghiệp:** Một số lập trình viên có thể đã có nhiều năm kinh nghiệm trong một lĩnh vực khác trước khi chuyển sang lập trình. Do đó, họ có nhiều năm kinh nghiệm làm việc tổng thể nhưng không có kinh nghiệm \"liên quan\" trực tiếp đến lĩnh vực hiện tại.\n",
            "- **Kinh nghiệm không chính thức:** Kinh nghiệm có thể được tích lũy thông qua các dự án cá nhân, học tập tự túc hoặc làm việc freelance mà không được coi là \"kinh nghiệm liên quan\" chính thức.\n",
            "- **Định nghĩa \"liên quan\" không rõ ràng:** Định nghĩa về \"kinh nghiệm liên quan\" có thể khác nhau giữa các công ty hoặc trong ngữ cảnh của tập dữ liệu.\n",
            "**Ý nghĩa kinh doanh:** Các nhà tuyển dụng nên xem xét kỹ lưỡng hồ sơ của những ứng viên này. Kinh nghiệm từ các lĩnh vực khác có thể mang lại những kỹ năng và góc nhìn độc đáo. Việc đánh giá chỉ dựa trên \"kinh nghiệm liên quan\" có thể bỏ qua những tài năng tiềm năng."
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### 12.3. Phân tích sâu các đặc trưng có mẫu bất ngờ"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "**Ví dụ (nếu có):**\n",
            "- Nếu có một đặc trưng nào đó cho thấy mối quan hệ ngược lại với kỳ vọng hoặc có phân phối rất bất thường, cần phải điều tra thêm.\n",
            "- Ví dụ, nếu `training_hours` rất cao nhưng tỷ lệ chuyển việc cũng cao, điều này có thể chỉ ra rằng các khóa đào tạo không đủ để giữ chân nhân viên hoặc họ đang được đào tạo để chuyển sang một công việc khác.\n",
            "**Ý nghĩa kinh doanh:** Việc hiểu rõ các mẫu bất ngờ có thể tiết lộ những vấn đề tiềm ẩn trong chính sách nhân sự, môi trường làm việc hoặc thị trường lao động. Điều này đòi hỏi sự hợp tác chặt chẽ với các chuyên gia HR để đưa ra các giải pháp phù hợp."
        ]
    }
]

# Find the index of the last "## 7. Tổng kết và Nhận xét" section
insert_index = -1
for i, cell in enumerate(notebook_content['cells']):
    if cell['cell_type'] == 'markdown' and "## 7. Tổng kết và Nhận xét" in "".join(cell['source']):
        insert_index = i
        break

if insert_index != -1:
    # Insert the new section before the existing "Tổng kết và Nhận xét"
    notebook_content['cells'][insert_index:insert_index] = new_section_cells
else:
    # If "Tổng kết và Nhận xét" is not found, append to the end
    notebook_content['cells'].extend(new_section_cells)

# Write the modified content back to the notebook file
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook_content, f, indent=1, ensure_ascii=False)

print(f"Successfully added 'Section 12: Enhanced conclusions with business insights' to {notebook_path}.")
