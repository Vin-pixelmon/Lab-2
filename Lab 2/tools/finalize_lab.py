from pathlib import Path
import json

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "lab2_customer_segmentation.ipynb"


markdown_cells = [
    """# Lab 2 - Phân khúc khách hàng bằng Clustering

**Mục tiêu:** phân nhóm khách hàng theo tuổi, thu nhập hằng năm và điểm chi tiêu; đánh giá kết quả bằng Silhouette, Davies-Bouldin và Calinski-Harabasz theo Chapter 3.

> **Lưu ý quan trọng:** đề bài là **clustering (học không giám sát)**, không phải classification. Phần classification ở cuối notebook là phần mở rộng: mô hình học nhãn cụm do K-Means tạo ra (pseudo-label), không phải nhãn thật.

**Dữ liệu:** do đề không cung cấp tệp CSV, notebook dùng bộ dữ liệu khách hàng mô phỏng 200 dòng, được sinh với `random seed = 42` để có thể tái lập. Khi có dữ liệu thật, chỉ cần thay `data/input/mall_customers.csv` và giữ nguyên tên cột.""",
    """## Step 1 - Import thư viện và cấu hình

Notebook tự tìm thư mục gốc dự án nên có thể chạy bằng Jupyter Notebook, JupyterLab hoặc VS Code.""",
    """## Step 2 - Đọc và khám phá dữ liệu (EDA)

Kiểm tra kích thước, kiểu dữ liệu, giá trị thiếu, dòng trùng, thống kê mô tả và ngoại lệ theo quy tắc IQR. `CustomerID` chỉ là mã định danh nên không được dùng làm đặc trưng phân cụm.""",
    """## Step 3 - Tiền xử lý và chuẩn hóa

Sử dụng `StandardScaler` để đưa `Age`, `Annual Income (k$)` và `Spending Score (1-100)` về cùng thang đo. Biến `Gender` được giữ để mô tả cụm nhưng không đưa vào K-Means chính, đúng theo yêu cầu chuẩn hóa ba biến số trong đề.""",
    """## Step 4 - Thử nghiệm K-Means với k = 2..10

Chọn số cụm dựa trên nhiều tiêu chí: Silhouette cao, Davies-Bouldin thấp, Calinski-Harabasz cao. Inertia được dùng cho phương pháp Elbow.""",
    """## Step 5 - So sánh các thuật toán

So sánh K-Means, Agglomerative Clustering (Ward) và DBSCAN. Với DBSCAN, nhãn `-1` là nhiễu; các chỉ số nội bộ chỉ được tính trên những điểm không phải nhiễu khi tồn tại ít nhất hai cụm hợp lệ.""",
    """## Step 6 - Mô hình cuối, trực quan hóa và hồ sơ từng cụm

Huấn luyện K-Means với số cụm tốt nhất, giảm chiều bằng PCA để vẽ 2D, sau đó tính tuổi, thu nhập, chi tiêu và tỷ lệ nữ trung bình của mỗi cụm.""",
    """## Step 7 - Classification mở rộng

Random Forest dự đoán **nhãn cụm K-Means** cho khách hàng mới. Đây là bài toán phân loại trên pseudo-label. Accuracy chỉ đo khả năng bắt chước K-Means, không chứng minh các phân khúc là nhãn đúng tuyệt đối trong thực tế.""",
    """## Step 8 - Kết luận

- Đã hoàn thành EDA, kiểm tra missing/outlier, scaling, thử tham số, so sánh thuật toán và đánh giá nội bộ.
- Chọn K-Means theo Silhouette; kết quả chi tiết nằm trong `data/output/kmeans_metrics.csv`.
- Hồ sơ cụm hỗ trợ đề xuất chiến lược marketing; tên phân khúc là diễn giải nghiệp vụ, không phải nhãn có sẵn.
- Classification là phần mở rộng để gán nhanh khách hàng mới vào các segment đã học.
- Kết quả chỉ mang tính minh họa vì dữ liệu hiện tại là dữ liệu mô phỏng.""",
]

nb = nbformat.read(NB_PATH, as_version=4)
md_indexes = [i for i, cell in enumerate(nb.cells) if cell.cell_type == "markdown"]
if len(md_indexes) != len(markdown_cells):
    raise RuntimeError(f"Số ô Markdown không như dự kiến: {len(md_indexes)}")
for index, source in zip(md_indexes, markdown_cells):
    nb.cells[index].source = source

# Việt hóa nhãn biểu đồ và bổ sung lưu báo cáo classification.
replacements = {
    "Distribution: {col}": "Phân phối: {col}",
    "Elbow method": "Phương pháp Elbow",
    "Silhouette by k": "Silhouette theo k",
    "Customer segments - KMeans k={best_k} (PCA 2D)": "Phân khúc khách hàng - K-Means k={best_k} (PCA 2D)",
    "Confusion matrix - du doan nhan cum": "Ma trận nhầm lẫn - dự đoán nhãn cụm",
    "Pseudo-label classification accuracy:": "Độ chính xác classification trên pseudo-label:",
    "Best k by Silhouette:": "Số cụm tốt nhất theo Silhouette:",
}
for cell in nb.cells:
    if cell.cell_type == "code":
        for old, new in replacements.items():
            cell.source = cell.source.replace(old, new)

# Bổ sung ví dụ dự đoán rõ ràng ở cuối ô classification.
cls_cell = next(cell for cell in nb.cells if cell.cell_type == "code" and "classifier.fit" in cell.source)
cls_cell.source += """

report_text = classification_report(y_test, pred, zero_division=0)
(OUTPUT / 'classification_report.txt').write_text(
    f'Accuracy: {accuracy_score(y_test, pred):.4f}\\n\\n{report_text}', encoding='utf-8')

khach_hang_moi = pd.DataFrame([{
    'Gender': 'Female', 'Age': 30,
    'Annual Income (k$)': 70, 'Spending Score (1-100)': 80
}])
cum_du_doan = int(classifier.predict(khach_hang_moi)[0])
print('Ví dụ khách hàng mới được gán vào cụm:', cum_du_doan)
"""

nbformat.write(nb, NB_PATH)

(ROOT / "README.md").write_text("""# Lab 2 - Machine Learning

Bài thực hành phân khúc khách hàng theo Chapter 3 - Unsupervised Learning.

- Hướng dẫn: `docs/HUONG_DAN.md`
- Báo cáo kết quả: `docs/BAO_CAO_KET_QUA.md`
- Notebook đã chạy: `notebooks/lab2_customer_segmentation.ipynb`

Chạy notebook bằng Python 3 và chọn **Run All**. Dữ liệu hiện tại là dữ liệu mô phỏng có seed cố định vì đề không kèm CSV gốc.
""", encoding="utf-8")

(ROOT / "docs" / "HUONG_DAN.md").write_text("""# Hướng dẫn Lab 2

## Phân biệt yêu cầu

Đề trong ảnh là **clustering**: dữ liệu không có nhãn đích và cần tìm các nhóm khách hàng tự nhiên. Classification cần nhãn có sẵn nên không thể thay thế bước phân cụm. Notebook vẫn có một bước classification mở rộng để dự đoán nhãn cụm (pseudo-label) cho khách hàng mới.

## Các step

1. Đọc và khám phá dữ liệu: kiểm tra missing, duplicate, thống kê, ngoại lệ IQR và histogram.
2. Chọn `Age`, `Annual Income (k$)`, `Spending Score (1-100)`; loại `CustomerID`; chuẩn hóa bằng StandardScaler.
3. Thử K-Means với `k` từ 2 đến 10.
4. Đánh giá bằng Silhouette (cao tốt), Davies-Bouldin (thấp tốt), Calinski-Harabasz (cao tốt) và Elbow.
5. So sánh K-Means với Agglomerative Clustering và DBSCAN.
6. Trực quan PCA 2D, tạo hồ sơ và đặt tên các phân khúc.
7. Mở rộng classification: Random Forest học nhãn cụm để gán segment cho khách hàng mới.
8. Lưu bảng và biểu đồ vào `data/output/`.

## Cách chạy

1. Cài thư viện: `pip install -r requirements.txt`.
2. Mở `notebooks/lab2_customer_segmentation.ipynb`.
3. Chọn kernel Python 3 và **Run All**.
4. Xem kết quả trong `data/output/`.

## Cấu trúc thư mục

- `docs/`: hướng dẫn và báo cáo.
- `data/input/`: CSV đầu vào, ảnh đề và file bài giảng.
- `data/output/`: kết quả CSV, TXT và PNG.
- `notebooks/`: notebook `.ipynb` đã chạy sẵn.
- `src/`: mã Python có thể tái sử dụng.
- `tools/`: script dựng và hoàn thiện dự án.

## Ghi chú về dữ liệu

Đề không kèm CSV nên `mall_customers.csv` là dữ liệu mô phỏng 200 khách hàng với seed 42. Không dùng kết quả này để đưa ra quyết định kinh doanh thật. Khi giảng viên cung cấp dữ liệu, thay tệp CSV nhưng giữ nguyên các tên cột để chạy lại toàn bộ notebook.
""", encoding="utf-8")

print(json.dumps({"notebook": str(NB_PATH), "localized": True}, ensure_ascii=False))
