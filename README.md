# Knowledge Graph và RAG

Bài tiểu luận Chương 4 của **Kiều Trường Giang - B23DCCN254**, học phần Phát triển các hệ thống thông minh.

## Cách chạy

Mở `Tieuluan02_06_Nhom2_Giangkt.ipynb` và chọn **Run All**. Notebook tự thực hiện toàn bộ quy trình:

1. tải và xử lý dữ liệu tiểu đường, cổ phiếu FPT.VN và vàng GC=F;
2. so sánh và tuning mô hình cho cả ba miền: K-fold cho tiểu đường, TimeSeriesSplit cho FPT và vàng;
3. xây Knowledge Graph cho ba miền;
4. chạy Vector RAG và GraphRAG;
5. xuất tham số mô hình sang ứng dụng web, tạo ba biểu mẫu dự đoán và chatbot RAG sức khỏe;
6. sinh báo cáo `Tieuluan02_06_Nhom2_Giangkt.pdf`.

Nếu Yahoo Finance tạm thời không truy cập được, notebook chuyển sang dữ liệu dự phòng có seed cố định và ghi rõ nguồn trong báo cáo.

## Cấu trúc

- `Tieuluan02_06_Nhom2_Giangkt.ipynb`: tệp duy nhất cần chạy.
- `Tieuluan02_06_Nhom2_Giangkt.pdf`: báo cáo được sinh từ notebook.
- `artifacts_giangkt/data/`: dữ liệu sau xử lý và bộ ba tri thức.
- `artifacts_giangkt/figures/`: biểu đồ, đồ thị và ảnh code.
- `vercel_app/`: ứng dụng web tương tác để triển khai Vercel. Chatbot nhận thông số sức khỏe, chạy mô hình tiểu đường, giải thích kết quả, đưa gợi ý và tiếp tục hỏi đáp từ kho tri thức WHO/CDC/ADA; FPT và vàng có khu dự báo riêng.

## Liên kết

- GitHub: https://github.com/jellalaz/RAG_HTTM
- Web demo: https://rag-httm-chatbot.vercel.app
- Vercel account: https://vercel.com/trng-giang

> Dữ liệu tài chính và y tế trong dự án phục vụ mục đích học tập.
