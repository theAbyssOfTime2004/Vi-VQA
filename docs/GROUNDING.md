# Knowledge grounding

## Vấn đề

Mỗi record trong `Viet-ViTextVQA-gemini-VQA` có ba phần: ảnh, các lượt hỏi–đáp,
và một trường **`description`** dài trung bình ~558 ký tự.

Gemini 1.5 Flash sinh câu trả lời **từ description đó**. Hệ quả: rất nhiều câu
trả lời khẳng định những điều không đọc được từ pixel — năm xây dựng một ngôi
đền, quận của một khu chợ, tên đầy đủ của một thương hiệu.

Trước đợt refactor này, không dòng code nào trong repo đụng tới `description`.
Pipeline chỉ đưa (ảnh, câu hỏi) vào model và bắt nó tự nhớ ra kiến thức mà nó
chưa từng được cho xem. Phần loss không giảm được của bài toán đó không phải do
model yếu — mà do thiếu thông tin.

Grounding đưa description trở lại prompt, biến bài toán từ "hãy nhớ về di tích
này" thành "hãy đọc ngữ cảnh này" — thứ mà một VLM 8B học được thật.

## Cách bật

```bash
# Chuẩn bị dataset có grounding
vivqa prepare --set data.grounding.enabled=true

# Hoặc sửa hẳn trong config/config.yaml
```

```yaml
data:
  grounding:
    enabled: true
    mode: "prefix"      # prefix | system
    max_chars: 1200
```

Grounding được áp **lúc chuẩn bị dữ liệu**, nên nó nằm sẵn trong
`train.json`/`val.json`. Cùng một hàm `apply_grounding` được dùng cho cả
training lẫn `vivqa chat`, nên prompt lúc infer khớp với prompt lúc train.
Điều này quan trọng hơn vẻ ngoài của nó: một model fine-tune trên prompt có
grounding mà lúc infer lại nhận prompt trần sẽ trả lời tệ đi rõ rệt, và triệu
chứng trông y hệt một lỗi mô hình.

Riêng khi eval, `vivqa eval` phát lại **nguyên văn** prompt đã lưu trong file
split, không áp grounding lần nữa — nếu không sẽ thành grounding hai lần.

## Hai chế độ

| Mode | Cấu trúc | Khi nào dùng |
|------|----------|--------------|
| `prefix` | Ngữ cảnh + câu hỏi gộp trong một lượt user | Mặc định. Chạy với mọi loader. |
| `system` | Ngữ cảnh thành một lượt `system` riêng | Sạch hơn về mặt cấu trúc, nhưng **trainer phải hiểu role `system`** |

`mode: system` sinh ra lượt `{"from": "system", ...}` trong file dữ liệu. Tôi
chưa kiểm chứng được `2U1/Qwen-VL-Series-Finetune` có xử lý lượt này hay bỏ qua
nó. Trước khi chuyển sang `system`, hãy chạy `vivqa prepare --limit 20` rồi kiểm
tra trainer có nạp đúng không. `prefix` không có rủi ro đó, nên nó là mặc định.

`max_chars` cắt description theo **ranh giới câu**: cắt giữa câu để lại một mệnh
đề lơ lửng mà model có xu hướng viết tiếp thay vì trả lời.

## Đo tác động

Grounding là một giả thuyết, không phải một cải tiến hiển nhiên — nó có thể dạy
model dựa dẫm vào ngữ cảnh mà bỏ qua ảnh. Hãy đo:

```bash
# Nhánh A — không grounding
vivqa prepare --set data.data_dir=./data/plain
vivqa train  --set data.data_dir=./data/plain \
             --set training.output_dir=./checkpoints/plain
vivqa eval   --model-path ./checkpoints/plain/checkpoint-XXXX \
             --set data.data_dir=./data/plain \
             --output ./results/plain.json

# Nhánh B — có grounding
vivqa prepare --set data.data_dir=./data/grounded \
              --set data.grounding.enabled=true
vivqa train  --set data.data_dir=./data/grounded \
             --set training.output_dir=./checkpoints/grounded
vivqa eval   --model-path ./checkpoints/grounded/checkpoint-XXXX \
             --set data.data_dir=./data/grounded \
             --set data.grounding.enabled=true \
             --output ./results/grounded.json
```

Cả hai nhánh dùng chung `seed`, nên `assign_splits` chia ảnh giống hệt nhau và
hai con số so sánh được với nhau.

Khi đọc kết quả, lưu ý:

- **CIDEr chỉ so sánh được trên cùng một tập mẫu.** IDF được tính từ chính tập
  reference đang eval, nên hãy giữ `evaluation.num_samples` cố định.
- **Nhánh B dễ thắng một cách không công bằng.** Nó được cho ngữ cảnh chứa sẵn
  câu trả lời. Con số cao hơn chứng minh model biết đọc ngữ cảnh, *không* chứng
  minh nó nhìn ảnh giỏi hơn.

Muốn biết grounding có giá trị thật lúc triển khai hay không, phải trả lời được
câu: lúc chạy thật, description ở đâu ra? Có ba đường:

1. **Không có** — grounding chỉ là công cụ lúc train, lúc deploy prompt trần.
   Kiểm chứng bằng cách eval checkpoint B trên dữ liệu *không* grounding: nếu
   vẫn hơn A, model đã học được kiến thức thật, không chỉ học đọc.
2. **Truy hồi (RAG)** — index description của 9.594 ảnh, lúc infer tìm entry gần
   nhất rồi chèn vào. Đây là hướng mở rộng tự nhiên; `apply_grounding` đã là chỗ
   nối sẵn cho nó.
3. **Người dùng cung cấp** — `vivqa chat` đã hỏi ngữ cảnh sau mỗi câu hỏi khi
   grounding bật.

Đợt refactor này làm phần (1) và dọn đường cho (2).
