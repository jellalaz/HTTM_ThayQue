const assert=require('node:assert/strict');
const fs=require('node:fs');
const {HealthChat,glucoseAssessment,a1cAssessment,parseNumber,esc,infer,retrieve}=require('../src/engine.js');
let count=0;
function test(name,fn){fn();count++;console.log('PASS',name)}
test('Hạ đường huyết phải ưu tiên trước nhãn bình thường',()=>{
 for(const ctx of ['fasting','ogtt','postmeal','random','unknown']){
  assert.equal(glucoseAssessment(69,ctx).low,true);
  assert.match(glucoseAssessment(54,ctx).title,/rất thấp/);
 }
 assert.equal(glucoseAssessment(70,'fasting').level,'good');
});
test('Các ranh giới xét nghiệm và sau bữa ăn',()=>{
 assert.equal(glucoseAssessment(99,'fasting').level,'good');
 assert.equal(glucoseAssessment(100,'fasting').level,'watch');
 assert.equal(glucoseAssessment(126,'fasting').level,'high');
 assert.equal(glucoseAssessment(140,'ogtt').level,'watch');
 assert.equal(glucoseAssessment(200,'ogtt').level,'high');
 assert.match(glucoseAssessment(150,'postmeal').text,/không tương đương/);
 assert.equal(a1cAssessment(5.7).level,'watch');
 assert.equal(a1cAssessment(6.5).level,'high');
});
test('Hội thoại hỏi đơn vị rồi hỏi thời điểm và nhớ kết quả',()=>{
 const c=new HealthChat();
 assert.match(c.respond('Đường huyết 150').html,/mg\/dL hay mmol\/L/);
 assert.match(c.respond('mg/dL').html,/thời điểm nào/);
 assert.match(c.respond('Lúc đói ít nhất 8 giờ').html,/Đạt ngưỡng/);
 assert.equal(c.profile.glucose,150);
 assert.equal(c.profile.context,'fasting');
 assert.match(c.respond('Gợi ý một bữa ăn').html,/150 mg\/dL/);
});
test('Số thập phân tiếng Việt và quy đổi mmol/L',()=>{
 const c=new HealthChat();const r=c.respond('Đường huyết 7,2 mmol/L lúc đói');
 assert.ok(Math.abs(c.profile.glucose-129.73104)<1e-5);
 assert.match(r.html,/Đạt ngưỡng/);
 assert.equal(parseNumber('6,7'),6.7);
 assert.ok(Number.isNaN(parseNumber('')));
 assert.ok(Number.isNaN(parseNumber('15abc')));
});
test('Không áp ngưỡng người lớn cho thai kỳ và trẻ em',()=>{
 const c=new HealthChat();
 assert.match(c.respond('Tôi đang mang thai, đường huyết 150 mg/dL lúc đói').html,/ngưỡng đánh giá riêng/);
 const child=new HealthChat();assert.match(child.respond('Bé 10 tuổi, đường huyết 150 mg/dL lúc đói').html,/ngưỡng đánh giá riêng/);
});
test('Thay đổi số đo không mượn bối cảnh cũ',()=>{
 const c=new HealthChat();c.respond('Đường huyết 150 mg/dL lúc đói');
 assert.match(c.respond('Đường huyết 180').html,/đơn vị|Đơn vị/);
 c.respond('mg/dL');assert.equal(c.pending.context,null);
});
test('Thấp vẫn được xử lý khi chưa biết thời điểm đo',()=>{
 const c=new HealthChat();assert.match(c.respond('Đường huyết 3,2 mmol/L').html,/Đường huyết thấp/);
 assert.match(c.respond('Tôi nên ăn gì?').html,/15 g/);
});
test('Câu hỏi ngoài miền không rơi về thực đơn mặc định',()=>{
 const c=new HealthChat();assert.match(c.respond('Viết code Python sắp xếp danh sách').html,/chưa tìm được bằng chứng/);
 assert.match(c.respond('Metformin uống mấy viên?').html,/không thể quyết định liều/);
});
test('Cấp cứu, phủ định và câu hỏi kiến thức',()=>{
 assert.match(new HealthChat().respond('Tôi khó thở và đang lú lẫn').html,/Cần trợ giúp y tế ngay/);
 assert.doesNotMatch(new HealthChat().respond('Tôi không khó thở, đường huyết 110 mg/dL lúc đói').html,/<h4>Cần trợ giúp y tế ngay/);
 assert.doesNotMatch(new HealthChat().respond('Khi nào cần cấp cứu?').html,/<h4>Cần trợ giúp y tế ngay/);
});
test('Reset xóa ngữ cảnh; lời người dùng không được coi là HTML',()=>{
 const c=new HealthChat();c.respond('HbA1c 6,7%');c.reset();assert.deepEqual(c.profile,{});
 assert.equal(esc('<img onerror="x">'),'&lt;img onerror=&quot;x&quot;&gt;');
});
test('Thông số hội thoại mới không giữ điểm mô hình của hồ sơ cũ',()=>{
 const c=new HealthChat();c.share({glucose:100,context:'fasting',score:140,bmi:25});
 c.respond('Đường huyết 7,2 mmol/L lúc đói');assert.equal(c.profile.score,null);
});
test('Truy xuất theo chủ đề có bằng chứng phù hợp',()=>{
 for(const [q,id] of [['Nên ăn tinh bột thế nào?','carb'],['Tập thể dục bao nhiêu phút?','activity'],['HbA1c là gì?','a1c'],['Có nên tự tăng liều insulin?','medicine']])assert.equal(retrieve(q)[0].id,id);
});
const configPath='artifacts_giangkt/data/web_model_config.json';
if(fs.existsSync(configPath))test('Suy luận JavaScript khớp Python trên 36 vector giữ lại',()=>{
 const config=JSON.parse(fs.readFileSync(configPath));
 const samples=JSON.parse(fs.readFileSync('artifacts_giangkt/data/model_parity.json'));
 let maxError=0;
 for(const r of samples){const y=infer(config.models[r.kind].estimator,r.x);const error=Math.abs(y-r.expected);maxError=Math.max(error,maxError);assert.ok(error<Math.max(1e-7,Math.abs(r.expected)*1e-8),`${r.kind}: ${error}`)}
 console.log('MAX PARITY ERROR',maxError);
});
console.log(`${count} test groups passed`);
