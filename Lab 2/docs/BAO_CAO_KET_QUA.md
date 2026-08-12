# Báo cáo kết quả Lab 2

## 1. Bài toán

Phân khúc 200 khách hàng của cửa hàng bán lẻ dựa trên `Age`, `Annual Income (k$)` và `Spending Score (1-100)`. Theo Chapter 3, đây là bài toán học không giám sát (clustering) vì dữ liệu không có nhãn đích.

Do đề bài không cung cấp CSV, bài làm dùng dữ liệu mô phỏng có seed 42. Kết quả nhằm minh họa đúng quy trình kỹ thuật, không phải kết luận kinh doanh từ dữ liệu thật.

## 2. Quy trình

1. Kiểm tra missing, duplicate, phân phối và ngoại lệ theo IQR.
2. Loại `CustomerID` khỏi tập đặc trưng và chuẩn hóa ba biến số bằng StandardScaler.
3. Thử K-Means với `k = 2..10`.
4. Đánh giá bằng Silhouette, Davies-Bouldin, Calinski-Harabasz và Elbow.
5. So sánh K-Means với Agglomerative Clustering và DBSCAN.
6. Trực quan hóa bằng PCA 2D, lập hồ sơ từng cụm và đề xuất tên segment.
7. Mở rộng bằng Random Forest classification để dự đoán nhãn cụm cho khách hàng mới.

## 3. Kết quả chính

K-Means tốt nhất tại `k = 5` theo Silhouette:

| Chỉ số | Giá trị |
|---|---:|
| Silhouette | 0.5067 |
| Davies-Bouldin | 0.7107 |
| Calinski-Harabasz | 223.7252 |
| Inertia | 107.3492 |

DBSCAN đạt Silhouette 0.5463 trên những điểm không bị xem là nhiễu, nhưng loại 32/200 điểm (16%) thành nhiễu. Vì mục tiêu cần gán segment cho mọi khách hàng, K-Means 5 cụm dễ diễn giải và triển khai hơn.

## 4. Diễn giải 5 phân khúc

| Cụm | Số KH | Tuổi TB | Thu nhập TB | Chi tiêu TB | Diễn giải |
|---:|---:|---:|---:|---:|---|
| 0 | 40 | 29.20 | 28.75 | 73.45 | Chi tiêu cao - nhạy với ưu đãi |
| 1 | 35 | 56.97 | 52.77 | 43.54 | Trưởng thành - chi tiêu ổn định |
| 2 | 46 | 33.35 | 34.33 | 35.46 | Phổ thông - cần nuôi dưỡng |
| 3 | 42 | 32.24 | 74.76 | 77.55 | Giá trị cao - ưu tiên chăm sóc |
| 4 | 37 | 40.73 | 76.22 | 20.46 | Thu nhập cao - tiềm năng kích hoạt |

## 5. Lưu ý về classification

Classification trong notebook học nhãn do K-Means tạo ra (pseudo-label), không phải nhãn thực tế. Nó hữu ích để gán nhanh khách hàng mới vào segment đã có, nhưng accuracy chỉ đo khả năng bắt chước K-Means và không chứng minh segment là chân lý nghiệp vụ.

## 6. Kết luận

Mô hình K-Means với 5 cụm đáp ứng đầy đủ đề bài: khám phá dữ liệu, scaling, thử tham số, đánh giá nội bộ và trực quan hóa. Cụm 3 nên được ưu tiên giữ chân; cụm 4 phù hợp với chiến dịch kích hoạt chi tiêu; cụm 0 phù hợp với chương trình ưu đãi giá hợp lý. Các đề xuất này cần được kiểm chứng lại khi có dữ liệu thật.
