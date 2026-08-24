# FVQA — quyết định thiết kế và nhật ký thay đổi

Tài liệu hướng dẫn sử dụng nằm ở [README.md](README.md). File này ghi lại *vì
sao* project trông như hiện tại.

---

## Xuất phát điểm: Vi-VQA

Project khởi đầu là VQA tiếng Việt trên `Viet-ViTextVQA-gemini-VQA` (9.594
ảnh, 31.420 câu hỏi, đáp án do Gemini sinh). Sau khi đo baseline zero-shot,
phát hiện: **một dòng chỉ dẫn văn phong đưa exact_match từ 0,5% lên 23,5%**,
không train gì. Đọc tay 30 mẫu cho thấy ~87% câu trả lời zero-shot vốn đã
đúng nội dung — metric bề mặt sai lệch gần 170 lần so với thực tế. Kết luận:
fine-tune trên benchmark đó chủ yếu mua văn phong, không mua năng lực.

Đồng thời nhận ra: dữ liệu Vi-VQA không hợp để làm knowledge base — phần lớn
"tri thức" trong trường `description` là fact hyperlocal (số điện thoại một
cửa hàng cụ thể, địa chỉ một quán ăn) xuất hiện đúng một lần trong toàn
dataset, không tái sử dụng được cho câu hỏi khác. Không phải encyclopedic
knowledge, chỉ là caption gắn với một tấm ảnh.

**Quyết định: bỏ Vi-VQA, chuyển sang dataset thật sự hỗ trợ knowledge-graph
traversal.**

## Vì sao FVQA, không phải các ứng viên khác

Đã kiểm chứng bằng cách **tải và đọc thật** dữ liệu (không chỉ tin mô tả),
vì nhiều ứng viên tưởng khớp hoá ra không:

| Ứng viên | Vấn đề phát hiện được sau khi tải |
|---|---|
| KVQA (AAAI'19) | Khớp nhất về khái niệm (multi-hop trên Wikidata thật) nhưng **link tải chết** (503 ở cả 4 URL, không mirror) |
| M3-VQA (ACL'26) | Tải được, nhưng "evidence" là đoạn văn Wikipedia, **không có field triple** dù mô tả nói "khai thác Wikidata triples" |
| WikiVQABench | Tải được, nhưng chỉ `{image, question, correct, wrongs}` — trắc nghiệm trơn, <1.000 mẫu |
| Encyclopedic-VQA (ICCV'23) | Tải được, có cấu trúc 2-hop thật, nhưng KB là Wikipedia text — muốn graph tổng quát phải tự bóc hyperlink, mà **schema KB không giữ link** (verify qua tài liệu WikiWeb2M) |
| FVQA 2.0 | Không phải bản mở rộng — chỉ là lớp phụ hẹp (474 ảnh, toàn câu hỏi so sánh), phụ thuộc FVQA 1.0 làm nền |
| **FVQA 1.0** | **Tải được** (Dropbox, 451MB, verify sống), **225.434 triple thật** `(e1, relation, e2)` từ DBpedia/ConceptNet/WebChild — graph có sẵn, dùng ngay |

FVQA thắng không phải vì khớp nhất về mặt lý thuyết, mà vì là ứng viên
**duy nhất tải được ngay và có graph thật sẵn sàng**, không cần giải pháp
vòng (gọi API sống, tự bóc link, chờ server hồi phục).

## Dữ liệu thật khác README của chính FVQA

Tải và soi trực tiếp (2026-08-24), phát hiện README của FVQA cũng sai:

| | README ghi | Thực tế |
|---|---|---|
| Số fact | 193.449 | **225.434** |
| Số câu hỏi | 5.286 | **5.826** |
| `fact` | 1 id | **list** (luôn đúng 1 phần tử, verify 5.826/5.826) |
| `e1`/`e2` | "unique id" | **3 định dạng khác nhau**: ConceptNet URI, DBpedia URL, WebChild plain word |

Code viết theo dữ liệu thật, không theo tài liệu.

## Kiến trúc

```
data/fvqa.py         loader: parse JSON gốc, giữ nguyên fold test chính thức
data/fvqa_graph.py    KnowledgeGraph: adjacency, find_entities, BFS, shortest_path
data/grounding.py     oracle-fact grounding (tái dùng, chỉ đổi nguồn text)
data/samples.py       IMAGE_TOKEN / assign_splits / write_split — dùng chung
```

`KnowledgeGraph` là graph traversal đúng nghĩa CS — build từ triple thật,
BFS/shortest-path tự viết, không phải embedding retrieval giả lập. Verify
trên toàn bộ 225.434 triple: build 3,78s, `find_entities("trumpet")` → BFS
1-hop → tìm đúng fact oracle của một câu hỏi thật.

Hai điều kiện grounding tách biệt, đo hai câu hỏi khác nhau:
- **Oracle fact**: model được cho thẳng đúng fact — đo cận trên
- **Graph retrieval**: model tự tìm fact bằng `KnowledgeGraph.bfs` — đo cái
  graph traversal thật sự đóng góp được

## Đổi tên package: `vivqa` → `fvqa`

Sau khi bỏ Vi-VQA hoàn toàn, rà lại toàn bộ codebase tìm chỗ đặt tên theo
"Vi-VQA": không có class/module lẻ nào bị đặt tên vậy (đã xoá từ đợt refactor
đầu) — chỗ duy nhất là **tên package** (`pyproject.toml`, 59 dòng import,
lệnh CLI, biến môi trường `$VIVQA_CONFIG`). Đổi toàn bộ sang `fvqa`.

Nhân dịp dọn, tách phần dùng chung ra khỏi module Vi-VQA-specific đã xoá:
`IMAGE_TOKEN`, `assign_splits`, `write_split` chuyển sang `data/samples.py`
— ba hàm này không phụ thuộc dataset nào, chỉ tình cờ từng sống chung file
với logic parse HuggingFace/Gemini đã bị xoá.

## Trạng thái

Đã verify chạy thật, không chỉ viết xong:

- `fvqa prepare` chạy trên dữ liệu thật: 5.826 câu hỏi → 2.914 train /
  134 val / 2.778 test
- `KnowledgeGraph` chạy trên 225.434 triple thật, BFS tìm đúng fact
- `fvqa train --dry-run` build lệnh đúng trên output đã chuẩn bị
- `fvqa eval` đọc đúng sample FVQA (chỉ chạy được điều kiện oracle-fact)
- Test suite chạy không cần GPU

**Chưa làm:**

1. **Vision-seeding** — chưa có bước "model tự nhìn ảnh → ra tên thực thể"
   để seed graph. Hiện `find_entities()` cần bạn tự gõ tên thực thể.
2. **Graph-retrieval trong `fvqa eval`** — `KnowledgeGraph` đứng độc lập,
   gọi tay được, nhưng chưa có flag tự động chạy so sánh oracle vs
   graph-retrieval như một eval mode.
3. **Train thật** — cần GPU, chưa chạy (chỉ mới xác nhận dry-run đúng lệnh).

## Hướng tiếp theo

1. Hỏi Qwen3-VL "vật thể/cảnh/hành động chính trong ảnh là gì" → seed cho
   `find_entities` — tận dụng model đã có trong `model.py`
2. Nối graph-retrieval vào `evaluation/runner.py` như một điều kiện eval
   thứ ba (không context / oracle fact / graph retrieval)
3. Chạy 3 điều kiện so sánh, đo graph traversal đóng góp được bao nhiêu so
   với biết trước đáp án
4. Train thật, đối chứng với baseline zero-shot
