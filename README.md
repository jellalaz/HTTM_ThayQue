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

## Insight Lab 2.2

- Ba tab **Tiểu đường / FPT / Vàng** cùng khu Dự đoán; chatbot sức khỏe nằm riêng. Kết quả chỉ được đưa sang chatbot khi người dùng chọn chia sẻ.
- So sánh 8 họ mô hình sức khỏe và 9 họ mô hình thị trường, gồm SVR RBF, Huber, Ridge, Random Forest, Gradient Boosting và baseline; lựa chọn bằng CV MAE. Thị trường tải lịch sử 5 năm, thử cửa sổ học gần nhất, shrinkage và `TimeSeriesSplit(gap=1, test_size=63)`.
- Train, calibration, test tách riêng. Khoảng dự báo mục tiêu 90%, công bố độ bao phủ thực tế; không dùng R² làm phần trăm chính xác.
- Giữ nguyên giai đoạn test của bản 2.0: FPT 04/06–13/10/2026, vàng 09/06–13/10/2026. Không chọn cấu hình theo điểm test. Bản web fit lại với cấu hình đã chọn, giữ riêng 63 phiên gần nhất để hiệu chỉnh; chỉ số test lịch sử và khoảng dự báo bản refit được phân biệt rõ.
- Trên cùng test, vàng giảm MAE từ 50,47 còn 49,44 USD/oz nhưng chưa thắng baseline 48,27. FPT vẫn chọn Persistence (MAE 728,13 VND), sức khỏe vẫn chọn Ridge (MAE 42,94). Không ép mô hình phức tạp nếu CV kém hơn. Bootstrap theo khối cho biết độ bất định của mức cải thiện; không dùng khoảng này để tuning.
- Chatbot trích xuất dựa trên BM25 + bằng chứng KG: nhớ thông số trong phiên, hỏi tiếp đơn vị/thời điểm đo, quy đổi mmol/L, nhận xét HbA1c/BMI, phân biệt sau ăn và OGTT, xử lý đường huyết thấp trước.
- Khi nhập số vượt giới hạn như 150 mmol/L, chatbot giữ số đo để sửa đơn vị hoặc nhập lại, không tự đổi thành 150 mg/dL hay 15 mmol/L.
- Chatbot hiểu toàn bộ hồ sơ form: tuổi, BMI, huyết áp trung bình, cholesterol toàn phần, LDL, HDL, s4/s5, mã giới tính, glucose và HbA1c; bổ sung huyết áp tâm thu/tâm trương và triglyceride gốc. Hỏi lại khi chỉ có “huyết áp 90” hoặc “cholesterol 5,2”; xử lý nhiều chỉ số với đơn vị riêng trong một tin nhắn.
- Kho tri thức 20 chủ đề có AHA, MedlinePlus, Mayo Clinic và các nguồn hiện có; notebook xuất nội dung và danh mục nguồn ra `artifacts_giangkt/data/health_knowledge.json`. Đây là bổ sung tri thức và logic hội thoại, không phải train một LLM mới.
- Khi chia sẻ form, nhận đủ trường người dùng cung cấp; không coi các trung vị thay thế là xét nghiệm. Không tự diễn giải mã giới tính nghiên cứu để chọn ngưỡng HDL, không tự suy ngược s5 thành triglyceride xét nghiệm. Ngưỡng huyết áp theo AHA, không áp cho MAP; thai kỳ/trẻ em cần cách đánh giá riêng.
- Kiểm tra đầu vào, cảnh báo ngoài phạm vi train, không thực thi HTML từ câu hỏi. Không dùng API key hoặc LLM bên ngoài.
- Dữ liệu không cập nhật trực tiếp trên web; chạy lại notebook để cập nhật.

`src/` và `scripts/sync_notebook.py` chỉ phục vụ bảo trì. Notebook phát hành chứa toàn bộ mã nguồn và không phụ thuộc các tệp đó hay Node.js khi Run All. `node tests/engine.test.cjs` kiểm tra logic hội thoại và đối chiếu suy luận JavaScript/Python; `node tests/clinical.test.cjs` kiểm tra chỉ số bổ sung, đơn vị và chia sẻ hồ sơ.

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
