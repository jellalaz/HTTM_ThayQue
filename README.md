# Knowledge Graph và RAG

Bài tiểu luận Chương 4 của **Kiều Trường Giang - B23DCCN254**, học phần Phát triển các hệ thống thông minh.

## Cách chạy

Mở `Tieuluan02_06_Nhom2_Giangkt.ipynb` và chọn **Run All**. Notebook tự thực hiện toàn bộ quy trình:

1. tải và xử lý dữ liệu tiểu đường, cổ phiếu FPT.VN và vàng GC=F;
2. tuning mô hình Random Forest bằng GridSearchCV 5-fold;
3. xây Knowledge Graph cho ba miền;
4. chạy Vector RAG và GraphRAG;
5. tạo bảng, biểu đồ, ảnh code và dashboard;
6. sinh báo cáo `Tieuluan02_06_Nhom2_Giangkt.pdf`.

Nếu Yahoo Finance tạm thời không truy cập được, notebook chuyển sang dữ liệu dự phòng có seed cố định và ghi rõ nguồn trong báo cáo.

## Cấu trúc

- `Tieuluan02_06_Nhom2_Giangkt.ipynb`: tệp duy nhất cần chạy.
- `Tieuluan02_06_Nhom2_Giangkt.pdf`: báo cáo được sinh từ notebook.
- `artifacts_giangkt/data/`: dữ liệu sau xử lý và bộ ba tri thức.
- `artifacts_giangkt/figures/`: biểu đồ, đồ thị và ảnh code.
- `vercel_app/`: dashboard tĩnh để triển khai Vercel.

## Liên kết

- GitHub: https://github.com/jellalaz/RAG_HTTM
- Vercel account: https://vercel.com/trng-giang

> Dữ liệu tài chính và y tế trong dự án phục vụ mục đích học tập.
