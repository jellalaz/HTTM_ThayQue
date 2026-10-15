const assert=require('node:assert/strict');
const {HealthChat,retrieve}=require('../src/engine.js');
const clinical=require('../src/clinical.js');
let count=0;
function test(name,fn){fn();count++;console.log('PASS',name)}
test('Câu trong ảnh hỏi lại loại huyết áp và giữ số 90',()=>{
 const c=new HealthChat();assert.match(c.respond('Huyết áp 90 có cao không?').html,/tâm thu.*tâm trương.*trung bình/);
 assert.equal(c.pendingClinical[0].value,90);assert.equal(c.profile.bp,undefined);
 assert.match(c.respond('Tâm trương (mmHg)').html,/Cần thêm tâm thu/);
 assert.match(c.respond('Tâm thu 120 mmHg').html,/độ 2/);
 assert.equal(c.profile.sys,120);assert.equal(c.profile.dia,90);
});
test('Cặp huyết áp thiếu đơn vị chỉ hỏi một lần',()=>{
 const c=new HealthChat();assert.match(c.respond('Huyết áp 120/90').html,/Đơn vị/);
 assert.match(c.respond('mmHg').html,/độ 2/);assert.equal(c.pendingClinical.length,0);
});
test('Ngưỡng huyết áp, thành phần cao hơn quyết định; MAP không dùng ngưỡng đó',()=>{
 for(const [sys,dia,rx] of [[119,79,/khoảng tham chiếu/],[120,79,/tăng so/],[129,80,/độ 1/],[140,70,/độ 2/],[120,90,/độ 2/],[180,110,/rất cao/],[181,80,/rất cao/],[100,120,/rất cao/],[85,55,/thấp/]])assert.match(clinical.pressureAssessment({sys,dia}).title,rx);
 const c=new HealthChat();const r=c.respond('Huyết áp động mạch trung bình 93 mmHg');assert.match(r.html,/Không áp ngưỡng/);assert.doesNotMatch(r.html,/độ 1|độ 2/);
 assert.match(c.respond('Huyết áp của tôi thế nào?').html,/93/);
});
test('Không ghép thành phần của hai lần đo khác nhau',()=>{
 const c=new HealthChat();c.respond('Huyết áp 120/80 mmHg');c.respond('Tâm thu 140 mmHg');assert.equal(c.profile.dia,undefined);
 c.respond('Tâm trương 90 mmHg');assert.equal(c.profile.sys,140);assert.equal(c.profile.dia,90);
 const prev={...c.profile};assert.match(c.respond('Huyết áp 80/120 mmHg').html,/phải lớn hơn/);assert.deepEqual(c.profile,prev);
});
test('HDL thiếu đơn vị, tham chiếu giới được xác nhận riêng',()=>{
 const c=new HealthChat();assert.match(c.respond('HDL 47').html,/Đơn vị/);assert.equal(c.profile.s3,undefined);
 assert.match(c.respond('mg/dL').html,/đối chiếu giới/);
 assert.match(c.respond('Tôi là nữ').html,/HDL thấp/);
 assert.match(c.respond('Tôi là nam').html,/không nằm dưới/);
 assert.equal(c.profile.glucose,undefined);
});
test('Hỏi loại cholesterol, quy đổi đúng nồng độ và triglyceride',()=>{
 const c=new HealthChat();assert.match(c.respond('Cholesterol 5,2').html,/toàn phần, LDL hay HDL/);
 c.respond('Cholesterol toàn phần');c.respond('mmol/L');assert.ok(Math.abs(c.profile.s1-201.084)<1e-6);
 c.respond('LDL 3 mmol/L');assert.ok(Math.abs(c.profile.s2-116.01)<1e-6);
 c.respond('Triglyceride 2 mmol/L');assert.ok(Math.abs(c.profile.tg-177.14)<1e-6);
 assert.match(c.respond('Triglyceride 550 mg/dL').html,/rất cao/);
});
test('Đọc nhiều chỉ số, giữ từng đơn vị, không lấy đơn vị HDL cho glucose',()=>{
 const c=new HealthChat();const r=c.respond('Tuổi 50; huyết áp trung bình 93 mmHg; cholesterol toàn phần 185 mg/dL; HDL 1,2 mmol/L; LDL 120 mg/dL; tỷ lệ cholesterol/HDL 3,9; log triglyceride 4,8; glucose 150 mg/dL lúc đói');
 for(const key of ['age','bp','s1','s2','s3','s4','s5','glucose'])assert.ok(Number.isFinite(c.profile[key]),key);
 assert.equal(c.profile.glucose,150);assert.ok(Math.abs(c.profile.s3-46.404)<1e-6);assert.match(r.html,/Không tự quy đổi s5/);
});
test('Đổi chủ đề không làm đơn vị của lipid đi vào glucose đang chờ',()=>{
 const c=new HealthChat();c.respond('Đường huyết 150');c.respond('LDL 3');c.respond('mmol/L');assert.equal(c.profile.glucose,undefined);assert.ok(c.profile.s2>116);
 c.respond('HDL 47');c.respond('Đường huyết 150');c.respond('mg/dL');c.respond('lúc đói');assert.equal(c.profile.glucose,150);assert.equal(c.profile.s3,undefined);
});
test('Nhiều chỉ số cùng thiếu đơn vị được hỏi tuần tự, không mất số đo',()=>{
 const c=new HealthChat();const r=c.respond('HDL 47; đường huyết 150');assert.match(r.html,/mg\/dL hay mmol\/L/);
 c.respond('mg/dL');const follow=c.respond('Lúc đói ít nhất 8 giờ');
 assert.equal(c.profile.glucose,150);assert.match(follow.html,/HDL cholesterol: 47/);
 c.respond('mg/dL');assert.equal(c.profile.s3,47);assert.equal(c.profile.glucose,150);
});
test('Chia sẻ đủ form; không coi trung vị thay thế là xét nghiệm thật',()=>{
 const c=new HealthChat();c.share({age:50,bmi:25.75,bp:93,s1:185,s2:120,s3:47,s4:4,s5:4.8,sex:2,glucose:92,context:'fasting',a1c:null,score:120,missing:['s2','s4','s5','sex']});
 assert.equal(c.profile.bp,93);assert.equal(c.profile.s1,185);assert.equal(c.profile.s3,47);
 for(const k of ['s2','s4','s5','sex'])assert.equal(c.profile[k],undefined);
 assert.equal(c.profile.labSex,undefined);assert.match(c.summaryReply().html,/Chưa cung cấp/);
 assert.match(c.respond('HDL của tôi thế nào?').html,/47/);
 c.respond('LDL 100 mg/dL');assert.equal(c.profile.score,null);assert.ok(!c.profile.missing.includes('s2'));
 c.reset();assert.deepEqual(c.profile,{});assert.deepEqual(c.pendingClinical,[]);
});
test('Trẻ em và thai kỳ không áp máy móc bảng người lớn',()=>{
 const c=new HealthChat();assert.match(c.respond('Bé 10 tuổi HDL 47 mg/dL').html,/khoảng tham chiếu riêng/);
 assert.match(new HealthChat().respond('Tôi mang thai huyết áp 130/90 mmHg').html,/đánh giá riêng/);
 assert.match(new HealthChat().respond('Tôi mang thai huyết áp 190/125 mmHg').html,/rất cao/);
 assert.match(new HealthChat().respond('Tôi mang thai huyết áp 160/110 mmHg').html,/sản khoa ngay/);
 assert.match(new HealthChat().respond('Huyết áp 140/50 mmHg').html,/thành phần cao và thành phần thấp/);
});
test('Giới hạn dữ liệu, từ chối đơn vị sai và không suy diễn xét nghiệm log',()=>{
 for(const q of ['LDL -10 mg/dL','LDL 500 mmol/L','HDL 47 g/L','Huyết áp 120/80 kPa','Tuổi 150','mã giới tính 3']){
  const c=new HealthChat();const r=c.respond(q);assert.match(r.html,/kiểm tra|Kiểm tra|hỗ trợ|lớn hơn 0/);assert.equal(c.profile.s2,undefined);assert.equal(c.profile.bp,undefined);
 }
 const c=new HealthChat();c.respond('Log triglyceride 4,8');assert.equal(c.profile.tg,undefined);assert.match(c.respond('Tỷ lệ cholesterol/HDL 4').html,/Không có kết luận bệnh/);
});
test('Cấp cứu và phủ định triệu chứng',()=>{
 assert.match(new HealthChat().respond('Huyết áp 190/125 mmHg, đang đau ngực').html,/Cần trợ giúp y tế ngay/);
 assert.doesNotMatch(new HealthChat().respond('Không đau ngực, huyết áp 120/80 mmHg').html,/<h4>Cần trợ giúp y tế ngay/);
});
test('Kho tri thức mở rộng truy xuất đúng chủ đề',()=>{
 for(const [q,id] of [['HDL là gì?','hdl'],['LDL là gì?','ldl'],['Huyết áp trung bình là gì?','meanpressure'],['Log triglyceride là gì?','triglyceride'],['Mã giới tính là gì?','demographics']])assert.equal(retrieve(q)[0].id,id,q);
});
console.log(`${count} clinical test groups passed`);
