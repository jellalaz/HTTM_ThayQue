# Nhật ký ghép và cải tiến Tiểu luận 03

Ngày thực hiện: 23/10/2026 (Asia/Ho_Chi_Minh).

## Giữ từ nguồn — màu đen

| Phần | Nguồn chính | Cách sử dụng |
|---|---|---|
| Chương 1 | TL01 trang 11–20 | Giữ diễn giải lịch sử, phân loại AI, hệ chuyên gia, ML/DL, thách thức và hai minh họa |
| Chương 2 | TL01 trang 21–28 | Giữ khái niệm, các dạng học, quy trình, gradient descent, hồi quy, phân loại, cây/rừng/SVM/KNN, điều chuẩn, DNN |
| Chương 3 | TL01 trang 37–38 | Giữ động cơ CNN, convolution, pooling và các tầng CNN |
| Chương 4 | TL01 trang 45–46 | Giữ động cơ xử lý chuỗi, RNN, hidden state, LSTM và GRU |
| Chương 5 | TL02 trang 4–37 | Giữ các đoạn KG/RAG, ontology, ba miền dữ liệu, mô hình, hội thoại, deployment và giới hạn |
| Bảng tổng hợp KG/RAG | TL02 trang 38–40 | Chuyển phụ lục nguồn vào 5.8.1/5.8.2, giữ đen |

Đổi chương 4→5, sửa mục trùng, định dạng, TOC, đánh số hình/bảng/phương trình và chuyển công thức sang OMML là chuẩn hóa trình bày. Không tự tô đỏ vì chuyển vị trí; không tô đỏ cả nguồn TL01 hoặc Chương 5.

## Cải tiến kiến thức — màu đỏ

- Chương 1: hiệu chỉnh quan hệ tính toán Turing/tư duy, mốc Dartmouth, khẳng định LSTM giải quyết triệt để gradient, cách nói đặt tên Deep Learning và đơn vị năng lượng; bổ sung liên hệ AI biểu tượng → học máy → học sâu → KG/RAG.
- Chương 2: sửa nhãn hồi quy thuộc số thực, entropy dùng p_k, chi phí thuật toán; bổ sung fit preprocessing chỉ trên train, CV/tuning và các metrics hồi quy/phân loại. Sửa nhỏ trong đoạn nguồn dùng diff theo từ/cụm khi đủ tương đồng.
- Chương 3: phân biệt equivariance/bất biến cục bộ; sửa LeNet-5 năm 1998, bỏ Top-5 không cùng benchmark; bổ sung backprop, loss, Adam, transfer learning và augmentation.
- Chương 4: phân biệt CNN 1D với mạng có trạng thái; bổ sung BPTT/Jacobian, clipping, các kiểu đầu vào/đầu ra, ngày nhãn và quy ước GRU của PyTorch.
- Chương 5: làm rõ TF-IDF vector thưa, chưa có vector database/dense embedding production; chunking và giới hạn điểm hybrid; notebook, snapshot và API mới. Cập nhật GitHub ở đoạn liên quan.
- Kết luận Chương 5: thay câu lặp về xuất PDF tự động bằng câu ngắn về artifact dùng chung; notebook xuất dữ liệu/model/web, còn báo cáo dùng builder Word riêng.

## Thực nghiệm xây lại — màu đỏ

1. Diabetes Risk 50k: tải lại, chọn tám đầu vào, loại ID/cột tiết lộ nhãn, stratified split, sáu họ mô hình, test và confusion matrix. Sửa mô tả thành 41 **cột**, dữ liệu tổng hợp và mất cân bằng.
2. Vietnam Housing 2024: tải lại, chia nhóm địa chỉ, preprocessing trong CV, log-target, bốn họ mô hình, test và biểu đồ đơn vị tỷ VND.
3. CNN MNIST: PyTorch train từ đầu, 48k/12k/10k, augmentation dịch một pixel, 8 epoch, checkpoint, learning curve, confusion matrix và lỗi test.
4. RNN/LSTM/GRU: sáu checkpoint trên FPT/GC=F snapshot, 35 epoch/mạng, ngày nhãn, Persistence baseline và chọn deployment trên validation.

Không nhận accuracy/RMSE, checkpoint hoặc biểu đồ của tác giả TL01 là kết quả mới. ASL, e-commerce và so sánh ba framework chưa chạy lại được nêu đúng giới hạn, không có số liệu giả.

## Tái lập TL02

Đã chạy lại dữ liệu, CV/tuning ba miền, KG, TF-IDF/GraphRAG, sáu câu hỏi kiểm tra, biểu đồ và export. MAE chính khớp nguồn trong sai số số học: 42,9364; 728,1284; 49,4405. Lưu riêng `artifacts_tl03/legacy` để bảo toàn snapshot/sản phẩm cũ.

## Notebook và web

`main.ipynb` tự chứa mã và tự tạo module/HTML/CSS/JS. Đã Run All full qua Jupyter, 17 code cells, không lỗi. Có quick/full, seed, hash cache và đối chiếu metrics từ predictions.

Giữ theme, hero, ba tab cũ, chatbot và khu chất lượng. Thêm demo ML/CNN/RNN, API `/api/predict`, validation, loading/error và model thật. API có parity với sklearn/PyTorch. Production đã deploy và kiểm tra; xem `FINAL_VERIFICATION.md` và `production_verified.json`.

## Dấu vết

`report_provenance.json` ghi đoạn/hình/công thức, chương, màu và trang nguồn. `report_structure.json` ghi cấu trúc; `report_audit.json` kiểm tra TOC, màu, công thức và trang. Các tệp QA không đưa thành phụ lục báo cáo.
