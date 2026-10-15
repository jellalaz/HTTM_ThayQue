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

Nếu Yahoo Finance tạm thời không truy cập được, notebook đọc snapshot đã lưu và ghi rõ ngày/nguồn. Không tự thay dữ liệu thật bằng giá mô phỏng. Nếu không có snapshot, notebook dừng và yêu cầu chạy lại khi có mạng.

## Insight Lab 2.0

- Ba tab **Tiểu đường / FPT / Vàng** cùng khu Dự đoán; chatbot sức khỏe nằm riêng. Kết quả chỉ được đưa sang chatbot khi người dùng chọn chia sẻ.
- So sánh 7 họ mô hình sức khỏe và 5 cấu hình thị trường, có baseline; lựa chọn bằng CV MAE. Thị trường có mô hình dự báo lợi suất, shrinkage và `TimeSeriesSplit(gap=1)`.
- Train, calibration, test tách riêng. Khoảng dự báo mục tiêu 90%, công bố độ bao phủ thực tế; không dùng R² làm phần trăm chính xác.
- Chatbot trích xuất dựa trên BM25 + bằng chứng KG: nhớ thông số trong phiên, hỏi tiếp đơn vị/thời điểm đo, quy đổi mmol/L, nhận xét HbA1c/BMI, phân biệt sau ăn và OGTT, xử lý đường huyết thấp trước.
- Kiểm tra đầu vào, cảnh báo ngoài phạm vi train, không thực thi HTML từ câu hỏi. Không dùng API key hoặc LLM bên ngoài.
- Dữ liệu không cập nhật trực tiếp trên web; chạy lại notebook để cập nhật.

`src/` và `scripts/sync_notebook.py` chỉ phục vụ bảo trì. Notebook phát hành chứa toàn bộ mã nguồn và không phụ thuộc các tệp đó khi Run All. `node tests/engine.test.cjs` kiểm tra logic hội thoại và đối chiếu suy luận JavaScript/Python.

## Cấu trúc

- `Tieuluan02_06_Nhom2_Giangkt.ipynb`: tệp duy nhất cần chạy.
- `Tieuluan02_06_Nhom2_Giangkt.pdf`: báo cáo được sinh từ notebook.
- `artifacts_giangkt/data/`: dữ liệu sau xử lý và bộ ba tri thức.
- `artifacts_giangkt/figures/`: biểu đồ, đồ thị và ảnh code.
- `vercel_app/`: ứng dụng Insight Lab để triển khai Vercel. Ba mô hình dùng chung khu dự đoán; chatbot sức khỏe độc lập dùng nguồn NIDDK, WHO và CDC.

## Liên kết

- GitHub: https://github.com/jellalaz/RAG_HTTM
- Web demo: https://rag-httm-chatbot.vercel.app
- Vercel account: https://vercel.com/trng-giang

> Dữ liệu tài chính và y tế trong dự án phục vụ mục đích học tập.
