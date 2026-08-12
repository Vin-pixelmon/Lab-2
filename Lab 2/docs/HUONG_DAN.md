# Hướng dẫn Lab 2

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
