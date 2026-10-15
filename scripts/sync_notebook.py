"""Đóng gói mã bảo trì vào notebook; notebook được phát hành luôn tự chứa mọi mã nguồn."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
path=ROOT/'Tieuluan02_06_Nhom2_Giangkt.ipynb'
n=json.loads(path.read_text())
def setcell(i,source):
    n['cells'][i]['source']=source.splitlines(keepends=True)
    if n['cells'][i]['cell_type']=='code':
        n['cells'][i]['outputs']=[]
        n['cells'][i]['execution_count']=None
def src(name):return (ROOT/'src'/name).read_text()
setcell(3,src('data.py'))
setcell(4,src('training.py'))
template=src('web.html').replace('__STYLES__',src('styles.css')).replace('__ENGINE__',src('engine.js')).replace('__APP__',src('app.js'))
webcode=src('export_models.py')+'\nhtml_template = '+repr(template)+'\n'
webcode+='''html=html_template.replace("__MODEL_CONFIG__",json.dumps(web_config,ensure_ascii=False,allow_nan=False))
(WEB_DIR/"index.html").write_text(html,encoding="utf-8")
(WEB_DIR/"vercel.json").write_text(json.dumps({"rewrites":[{"source":"/(.*)","destination":"/index.html"}],"headers":[{"source":"/(.*)","headers":[{"key":"X-Content-Type-Options","value":"nosniff"},{"key":"Referrer-Policy","value":"strict-origin-when-cross-origin"}]}]},indent=2),encoding="utf-8")
print("Đã xuất Insight Lab 2.0:",{k:v["name"] for k,v in web_config["models"].items()})

# Sơ đồ bố cục minh họa, có nhãn rõ ràng; không giả làm ảnh chụp màn hình.
fig,ax=plt.subplots(figsize=(12,7),dpi=240)
ax.axis("off"); ax.set_xlim(0,12);ax.set_ylim(0,7)
def layout_box(x,y,w,h,title,subtitle):
    ax.add_patch(plt.Rectangle((x,y),w,h,facecolor="#f5f5f5",edgecolor="black",lw=.8))
    ax.text(x+.2,y+h-.35,title,fontweight="bold",fontsize=11,color="black")
    ax.text(x+.2,y+h-.75,subtitle,fontsize=9,color="black",va="top")
ax.text(.25,6.75,"INSIGHT LAB 2.0 - SƠ ĐỒ BỐ CỤC GIAO DIỆN",fontsize=15,fontweight="bold",color="black")
for x,title in [(0.25,"TIỂU ĐƯỜNG"),(4.15,"FPT.VN"),(8.05,"VÀNG GC=F")]:
    layout_box(x,5.4,3.7,.95,title,"Ba tab trong cùng khu Dự đoán")
layout_box(.25,3.4,5.55,1.75,"ĐẦU VÀO","Đơn vị rõ ràng · kiểm tra dữ liệu\\nThông số bổ sung · bối cảnh xét nghiệm")
layout_box(6,3.4,5.75,1.75,"KẾT QUẢ","Dự báo · khoảng sai số · nhận xét\\nSo sánh baseline · ngoài phạm vi train")
layout_box(.25,.35,7.4,2.75,"CHATBOT SỨC KHỎE ĐỘC LẬP","Truy xuất BM25 + bằng chứng KG\\nHỏi tiếp đơn vị / thời điểm đo\\nNhớ ngữ cảnh trong phiên · dẫn nguồn\\nChỉ nhận hồ sơ khi người dùng chọn chia sẻ")
layout_box(7.9,.35,3.85,2.75,"NGỮ CẢNH","Glucose · HbA1c · BMI\\nThời điểm đo · thai kỳ\\nGợi ý câu hỏi tiếp theo")
fig.tight_layout();fig.savefig(FIG_DIR/"web_app_preview.png",dpi=240,bbox_inches="tight");plt.close(fig)
'''
setcell(13,webcode)
setcell(12,'## 5. Sinh ứng dụng và báo cáo PDF\n\nBa mô hình dùng chung khu Dự đoán. Chatbot sức khỏe có khu vực riêng, truy xuất BM25, bằng chứng KG và bộ nhớ hội thoại trong phiên. Mô hình, pipeline tiền xử lý và khoảng dự báo được xuất từ Python sang JavaScript. Notebook chứa toàn bộ HTML/CSS/JS nên không cần chạy tệp khác.\n')
report=''.join(n['cells'][14]['source'])
# Loại phần chèn cũ khi chạy lại script.
marker='# INSIGHT_LAB_REPORT_UPDATES'
if marker in report:
    start=report.index(marker); end=report.index('for title, paragraphs, figure in pages:',start)
    report=report[:start]+report[end:]
updates='''# INSIGHT_LAB_REPORT_UPDATES
page_updates = {
"4.4. KB VÀ RAG CHO BỆNH TIỂU ĐƯỜNG": [
    "Dữ liệu diabetes gồm 442 người đã mắc bệnh, 10 thuộc tính nền và điểm tiến triển sau một năm. Đầu ra là hồi quy nghiên cứu, không phải xác suất mắc tiểu đường.",
    f"Dữ liệu được chia thành {len(X_train)} mẫu train, {len(X_cal)} mẫu hiệu chỉnh khoảng dự báo và {len(X_test)} mẫu test. K-fold 5 phần chỉ chạy trên train; so sánh Median baseline, Linear, Ridge, ElasticNet, Huber, Random Forest và Gradient Boosting.",
    f"Mô hình được chọn theo CV: {diabetes_model_name}; CV MAE {diabetes_metrics['CV_MAE']:.2f}; test MAE {diabetes_metrics['MAE']:.2f}; R² {diabetes_metrics['R2']:.3f}. MAE của dự báo trung vị là {diabetes_metrics['Baseline_MAE']:.2f}. Khoảng dự báo mục tiêu 90% có độ bao phủ test {diabetes_interval['test_coverage']:.1%}."],
"4.5.2. HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH FPT": [
    "So sánh Persistence, Ridge dự báo mức giá và ba mô hình dự báo lợi suất: Ridge, Random Forest, Gradient Boosting. Các cấu hình co dự báo lợi suất về 0 giúp hạn chế biến động quá mức. Lựa chọn theo CV MAE bằng TimeSeriesSplit 5 phần, gap=1.",
    f"Chia theo thời gian: {stock_model_info['train_size']} train, {stock_model_info['calibration_size']} hiệu chỉnh, {stock_model_info['test_size']} test; bỏ một quan sát tại mỗi ranh giới. Mô hình {stock_model_info['Model']} đạt MAE {stock_model_info['MAE']:.2f}, R² {stock_model_info['R2']:.3f}; baseline MAE {stock_model_info['Persistence_MAE']:.2f}. Cải thiện trên test {stock_model_info['Improvement_vs_Persistence_pct']:.2f}%.",
    f"Khoảng sai số tương đối được hiệu chỉnh trên block riêng, không dùng test. Bao phủ thực tế: {market_details['Chứng khoán FPT']['interval']['test_coverage']:.1%}. Dữ liệu chuỗi thời gian không thỏa giả định trao đổi như mẫu độc lập nên mức 90% chỉ là mục tiêu, không phải bảo đảm cho tương lai."],
"4.6.2. HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH VÀNG": [
    "Vàng dùng cùng quy trình lựa chọn có baseline và mô hình lợi suất như FPT. Không ép chọn mô hình phức tạp nếu Persistence có CV MAE thấp hơn. Train, hiệu chỉnh và test nằm theo đúng thứ tự thời gian, có khoảng cách một phiên ở ranh giới.",
    f"Mô hình được chọn: {gold_model_info['Model']}; CV MAE {gold_model_info['CV_MAE']:.3f}; test MAE {gold_model_info['MAE']:.3f}; R² {gold_model_info['R2']:.3f}. Baseline MAE {gold_model_info['Persistence_MAE']:.3f}; cải thiện {gold_model_info['Improvement_vs_Persistence_pct']:.2f}%.",
    f"Độ bao phủ của khoảng dự báo trên test: {market_details['Giá vàng GC=F']['interval']['test_coverage']:.1%}. Nếu chọn Persistence, dự báo bằng giá đóng cửa đầu vào và không tạo tín hiệu chiều tăng/giảm. GC=F là hợp đồng vàng quốc tế, không phải vàng SJC."],
"4.7. DEPLOYMENT": [
    "Insight Lab 2.0 chia thành khu Dự đoán với ba tab Tiểu đường, FPT, Vàng và khu Trợ lý sức khỏe riêng. Người dùng xem kết quả tại từng tab; khi muốn thảo luận, chủ động chọn chia sẻ kết quả sức khỏe cho chatbot.",
    "Web hiển thị biểu đồ lịch sử giá, dự báo điểm, khoảng sai số hiệu chỉnh và so sánh baseline. Đầu vào trống, sai đơn vị hoặc vượt giới hạn được kiểm tra trước suy luận. Giá trị ngoài phạm vi train được đánh dấu để tránh hiểu sai mức tin cậy.",
    f"Web: <link href='{DEPLOY_URL}' color='black'>{DEPLOY_URL}</link>. GitHub: <link href='{GITHUB_URL}' color='black'>{GITHUB_URL}</link>. Sơ đồ dưới đây minh họa bố cục, không phải ảnh chụp màn hình."],
"4.7.1. SUY LUẬN MÔ HÌNH TRÊN TRÌNH DUYỆT": [
    "Notebook xuất cấu trúc estimator được chọn: scaler và hệ số tuyến tính, cây Random Forest/Gradient Boosting, hoặc công thức Persistence/lợi suất. JavaScript dùng cùng thứ tự đặc trưng và tham số, không hard-code kết quả dự báo. Các vector đối chiếu Python/JavaScript được lưu để kiểm tra sai số đóng gói.",
    "Mô hình sức khỏe có sáu trường chính và bốn trường bổ sung. Trường bổ sung trống dùng trung vị tập train và được công bố ở kết quả. HbA1c và bối cảnh đo dùng cho nhận xét xét nghiệm riêng, không bị đưa nhầm vào mô hình chưa học chúng. Điểm tiến triển không được dùng quyết định chẩn đoán hoặc cấp cứu.",
    "Mô hình thị trường dùng đủ 11 đặc trưng. Lợi suất hiện tại tính nhất quán từ hai giá đóng cửa; biểu mẫu tách đơn vị tiền và phần trăm. Ngày dữ liệu được hiển thị rõ; chỉ cập nhật khi chạy notebook. Web không gửi thông số sức khỏe đến dịch vụ bên ngoài."],
"4.7.2. CHATBOT DỰ ĐOÁN, TƯ VẤN VÀ HỎI ĐÁP": [
    "Chatbot hoạt động độc lập với biểu mẫu. Người dùng có thể nhập đường huyết, HbA1c, BMI trực tiếp. Bộ nhớ trong phiên giữ các thông số và câu hỏi đang chờ; nếu thiếu đơn vị hoặc thời điểm đo, chatbot hỏi tiếp rồi mới nhận xét. mg/dL và mmol/L được quy đổi minh bạch.",
    "Các ngưỡng xét nghiệm tham khảo NIDDK/ADA. Đường huyết thấp được xử lý trước bảng phân loại đường cao. Sau bữa ăn thông thường không bị đánh đồng với nghiệm pháp 75 g glucose. Thai kỳ và trẻ em cần bối cảnh riêng. Câu hỏi kê liều thuốc được chuyển thành hướng dẫn trao đổi với bác sĩ.",
    "Kho tri thức sức khỏe gồm 12 chủ đề. BM25 xếp hạng đoạn văn, bổ sung điểm cụm từ và đường quan hệ KG tương ứng. Câu trả lời dạng trích xuất có dẫn nguồn; không kết nối LLM bên ngoài. Câu hỏi không đủ bằng chứng được thông báo rõ thay vì trả lời một đoạn bất kỳ. Hội thoại bị xóa khi tải lại trang hoặc chọn cuộc trò chuyện mới."],
}
pages = [(title,page_updates.get(title,paras),fig) for title,paras,fig in pages]
'''
report=report.replace('for title, paragraphs, figure in pages:',updates+'\nfor title, paragraphs, figure in pages:')
report=report.replace('"[14] Vercel Documentation, Static Deployments and CLI.",','"[14] Vercel Documentation, Static Deployments and CLI.",\n    "[17] NIDDK, Diabetes Tests and Diagnosis, https://www.niddk.nih.gov/health-information/diabetes/overview/tests-diagnosis.",') if '[17] NIDDK' not in report else report
setcell(14,report)
# Cập nhật ảnh code để khớp triển khai thực tế.
figcode=''.join(n['cells'][11]['source'])
figcode=figcode.replace('lines = source.splitlines()', 'lines = [part for line in source.splitlines() for part in (textwrap.wrap(line, width=86, subsequent_indent="    ", replace_whitespace=False, drop_whitespace=False) or [""])]')
figcode=figcode.replace('0.31 * len(lines) + 1.25','0.28 * len(lines) + 1.0').replace('max(5.0,','max(3.0,')
figcode=figcode.replace('figsize=(12.8, height)','figsize=(10.0, height)').replace('fontsize=8.6, linespacing=1.35','fontsize=11.0, linespacing=1.25')
start=figcode.index('render_code_card("code_diabetes_training.png"')
end=figcode.index('render_code_card("code_kg_core.png"',start)
figcode=figcode[:start]+'''render_code_card("code_diabetes_training.png", "CV + calibration riêng cho mô hình sức khỏe", r\'\'\'
X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=.2, random_state=42)
X_train, X_cal, y_train, y_cal = train_test_split(
    X_dev, y_dev, test_size=.2, random_state=42)
name, search, rows = select_model(
    diabetes_candidates, X_train, y_train,
    KFold(5, shuffle=True, random_state=42), "Tiểu đường")
model = search.best_estimator_
interval = calibrate(model, X_cal, y_cal, X_test, y_test)
pred = model.predict(X_test)
\'\'\')
render_code_card("code_market_training.png", "TimeSeriesSplit có gap và mô hình baseline", r\'\'\'
a, b = int(len(data)*.64), int(len(data)*.80)
train, cal, test = data.iloc[:a-1], data.iloc[a:b-1], data.iloc[b:]
cv = TimeSeriesSplit(n_splits=5, gap=1)
name, search, rows = select_model(candidates, Xtr, ytr, cv, domain)
model = search.best_estimator_
interval = calibrate(model, Xcal, ycal, Xte, yte, relative=True)
pred = model.predict(Xte)
baseline = Xte["Close"].to_numpy()
\'\'\')

'''+figcode[end:]
start=figcode.index('render_code_card("code_deployment.png"')
end=figcode.index('print("Đã tạo"',start)
figcode=figcode[:start]+'''render_code_card("code_deployment.png", "Xuất estimator và web tự chứa", r\'\'\'
web_config["models"]["diabetes"] = {
    "estimator": export_estimator(diabetes_model),
    "features": list(X.columns),
    "metrics": diabetes_metrics,
    "interval": diabetes_interval,
}
model_json = json.dumps(web_config, ensure_ascii=False, allow_nan=False)
html = html_template.replace("__MODEL_CONFIG__", model_json)
(WEB_DIR / "index.html").write_text(html, encoding="utf-8")
\'\'\')
'''+figcode[end:]
setcell(11,figcode)
path.write_text(json.dumps(n,ensure_ascii=False,indent=1)+'\n')
print('Notebook tự chứa đã đồng bộ:',path)
