# Tiểu luận 03 — AI, ML, CNN, RNN, Knowledge Graph và RAG

Kiều Trường Giang — B23DCCN254. Học phần **Phát triển các hệ thống thông minh**, giảng viên **Thầy Trần Đình Quế**. Hai bìa PDF gốc không ghi tên lớp, nên báo cáo không tự suy ra thông tin này.

## Sản phẩm chính

- `tieuluan03_06_02_giangkt.254.docx`: Word chỉnh sửa được, 5 chương, mục lục TOC và công thức OMML.
- `tieuluan03_06_02_giangkt.254.pdf`: xuất từ chính bản Word, giữ màu đen/đỏ.
- `main.ipynb`: notebook chính duy nhất, đã chạy **Run All** chế độ full.
- `CHANGELOG.md`, `EXPERIMENT_RESULTS.md`, `FINAL_VERIFICATION.md`: thay đổi, số liệu và kết quả kiểm tra.
- [`PROJECT_REPORT.md`](PROJECT_REPORT.md): phân tích chi tiết các bài toán trên web, liên kết demo trực tiếp và hình từ dữ liệu/checkpoint.

## Chạy thực nghiệm

Mở `main.ipynb` bằng Jupyter/VS Code hoặc môi trường notebook tương thích, chọn kernel Python 3.10+ rồi **Run All**. Khuyến nghị Python 3.12 trở lên. Notebook tự kiểm tra/cài thư viện, tải dataset công khai, chuẩn bị snapshot, huấn luyện, đánh giá, lưu checkpoint, tạo biểu đồ và build web. Không phải mở notebook khác hoặc chạy thủ công file Python phụ.

Mặc định `MODE="full"`, `REUSE_CACHE=True`, `REFRESH_MARKET=False`, seed 42 và CPU. Full gồm 50.000 dòng nguy cơ, 30.229 dòng giá nhà, 48.000 ảnh train CNN với 8 epoch và sáu mạng chuỗi với 35 epoch/mạng. Lần đầu cần Internet cho Kaggle, MNIST và các gói còn thiếu; dữ liệu đã tải được cache. Snapshot thị trường TL02 được giữ cố định để tái lập số liệu. Bật `REFRESH_MARKET` tạo trạng thái dữ liệu khác, cần đánh giá và xuất lại báo cáo trước khi dùng số liệu mới.

`MODE="quick"` dùng ít dữ liệu/epoch cho thực nghiệm mới, ghi vào `tmp/quick03`, không thay model/web/số liệu full. `REUSE_CACHE=False` buộc huấn luyện lại. Cache kiểm tra code, seed, phiên bản thư viện, snapshot và hash artifact. `requirements.txt` mô tả môi trường hỗ trợ; `requirements-verified.txt` ghi phiên bản đã kiểm chứng. Không phụ thuộc torchvision. CPU đã chạy; phát hiện GPU không đồng nghĩa đã kiểm thử GPU.

## Website và API

Production: https://rag-httm-chatbot.vercel.app/

GitHub: https://github.com/jellalaz/HTTM_ThayQue

Giữ giao diện chính Insight Lab, ba tab dự đoán và chatbot sức khỏe cũ. Khu **Thực nghiệm Tiểu luận 03** bổ sung giá nhà, nhãn nguy cơ trên dataset tổng hợp, CNN nhận dạng chữ số và dự báo chuỗi FPT/vàng. Demo mới gọi `POST /api/predict`; dữ liệu nhập ở khu này được gửi đến server inference. Chatbot sức khỏe cũ vẫn chạy trong trình duyệt và chỉ nhận hồ sơ khi người dùng chọn chia sẻ.

API suy luận bằng Python standard library và tham số thật từ model đã train. Không đóng gói PyTorch/scikit-learn vào Vercel, không dùng API key/LLM ngoài. Bundle model nén khoảng 5,1 MB. `GET /api/predict` là health check; `POST` nhận `task`: `housing`, `diabetes_classifier`, `cnn` hoặc `sequence`.

Bảo trì web cục bộ sau khi chạy notebook: `python scripts/serve03.py`, mở http://127.0.0.1:8765/. Đây là server tùy chọn cho web, không phải bước bắt buộc để chạy notebook.

## Phạm vi kết quả

Classifier nguy cơ dùng nhãn dữ liệu tổng hợp, không phải xác suất mắc bệnh/chẩn đoán. Hồi quy diabetes cũ dùng 442 người đã mắc bệnh, đầu ra là điểm tiến triển nghiên cứu. Giá nhà dùng tin đăng 2024, không phải định giá hiện thời. GC=F là vàng futures, khác XAU/USD spot và SJC.

Mô hình chuỗi chọn bằng validation, có Persistence tham gia; không đổi lựa chọn theo test. Test Chương 4 ghi **ngày nhãn**, khác ngày đặc trưng của mô hình cũ Chương 5. Không so hai bảng như cùng tập test. RAG dùng TF-IDF/NetworkX và deterministic generator; smoke test không phải faithfulness của LLM.

ASL, e-commerce và ba bản Scratch/Keras/PyTorch của tác giả TL01 không được nhận là thực nghiệm mới. Không dùng checkpoint hoặc biểu đồ huấn luyện của người khác làm thành quả của sinh viên.

## Màu và báo cáo

Lý thuyết giữ từ hai nguồn mặc định đen; bổ sung/sửa kiến thức, kết quả mới và tính năng mới màu `#FF0000`. Sửa nhỏ được đánh dấu theo từ/cụm khi có thể. Chương 5 giữ phần lớn nội dung gốc màu đen. Báo cáo chỉ có bìa, mục lục và 5 chương, không có tài liệu tham khảo/citation số.

Nguồn chỉ có PDF nên văn bản được trích xuất và ghép lại; toàn bộ công thức hiển thị độc lập của ML/CNN/RNN đã được dựng thành Word Equation/OMML để đọc rõ và chỉnh sửa được. BPTT được hiệu chỉnh phép cộng qua các đường thời gian; giải thích riêng hai quy ước GRU của phần lý thuyết và PyTorch. `scripts/build_report03.py` là builder bảo trì; `scripts/populate_toc03.py` điền cache TOC từ PDF đã render. Word giữ TOC, bookmark và PAGEREF để cập nhật Ctrl+A → F9 sau khi chỉnh sửa, rồi xuất lại PDF.

Quy trình bảo trì báo cáo: build Word → `normalize_word03.py` → render Word/PDF → `populate_toc03.py` → render lại đến khi TOC ổn định → `finalize_pdf03.py` → `audit_report03.py`. Bộ xuất LibreOffice bỏ màu của OMML; bước finalize khôi phục toán tử màu của các công thức mới trong PDF vector, tự đếm từ bản Word, không thay văn bản, glyph hoặc tọa độ. Bản Word vẫn giữ công thức OMML chỉnh sửa được và màu đỏ trong XML. PNG render được rà từng trang. Không cần các bước này để chạy thực nghiệm qua notebook.

## Cấu trúc và kiểm thử

- `src/experiments03.py`: thực nghiệm mới, được nhúng trong notebook.
- `src/analysis03.py`: missingness, sai số phân khúc, feature map/kernel thật, bootstrap theo khối và truy xuất thật; được nhúng trong notebook.
- `artifacts_tl03/data/figure_lineage.json`: tệp đầu vào và SHA-256 của 12 hình phân tích/sơ đồ mới. Sơ đồ cơ chế được phân biệt với ảnh thực nghiệm.
- `src/inference03.py`: suy luận từ model export.
- `src/experiments03.html`, `src/experiments03.js`: phần giao diện bổ sung.
- `artifacts_tl03/models/`: checkpoint và tham số.
- `artifacts_tl03/data/`: metrics, predictions, history, split, manifest, parity và QA.
- `artifacts_tl03/legacy/`: kết quả tái lập riêng TL02.
- `artifacts_tl03/source/`: cấu trúc trích xuất PDF và hình nguồn để ghép.
- `vercel_app/`: gói triển khai gồm API và model nén.

Kiểm thử bảo trì: `python tests/test_inference03.py`, `node tests/engine.test.cjs`, `node tests/clinical.test.cjs`. `tests/check_production03.py` gọi production bằng mẫu công khai/tổng hợp. Token không đưa vào notebook/repository. Raw dataset tải tự động và Git ignore.

PDF/notebook TL01/TL02 được giữ như đầu vào lưu trữ; **main.ipynb là entrypoint duy nhất** của bản 03.
