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

## Lộ trình 4 milestone

Từ đây làm tuần tự, không song song, vì M2 phụ thuộc model loading (M1) và
M4 phụ thuộc pipeline local chạy đúng: **M1 train/checkpoint chắc chắn →
M2 graph retrieval (text seed) → M3 vision-seeding → M4 Modal + CI**.

### M1 — PR1: pin trainer revision, CI, fix flag contract (xong)

- `config.trainer.{repo_url,revision}` mới — pin `2U1/Qwen-VL-Series-Finetune`
  vào đúng commit (`70c7b2f`), không còn track HEAD ngầm. `ensure_trainer_repo`
  fetch đúng SHA (shallow, fallback full clone nếu server từ chối), cảnh báo
  (không tự ý checkout đè) nếu thư mục đã có sẵn ở revision khác.
- `scripts/check_trainer_flags.py` — đối chiếu **mọi** flag mà
  `build_train_command()` có thể sinh ra với field thật trong
  `ModelArguments`/`DataArguments`/`TrainingArguments` của trainer (đọc bằng
  `ast`, không cần cài torch/transformers/trl). Chạy thử ngay lập tức bắt
  được lỗi thật: **`command.py` gửi `--eval_data_path`, nhưng field thật của
  trainer là `eval_path`** — mọi lần train có kèm validation split (tức là
  mặc định, vì `fvqa prepare` luôn ra `val.json`) sẽ chết vài phút sau khi
  chạy vì `HfArgumentParser` từ chối flag lạ. Đã sửa.
- `.github/workflows/ci.yml` — CPU-only, Python 3.10-3.12, `pytest` +
  `compileall` + `fvqa config`, chạy mọi PR.
- `.github/workflows/integration.yml` — `workflow_dispatch` + weekly, clone
  trainer đúng revision pin, verify entrypoint/deepspeed config, chạy
  `check_trainer_flags.py`, tải FVQA thật rồi `fvqa train --dry-run`.
- 172 test (từ 162), gồm `test_train_runner.py` mới (dùng git repo cục bộ
  làm "remote" giả, không cần mạng) và case cho `TrainerConfig` trong
  `test_config.py`.

### M1 — PR2: checkpoint loader (xong)

`model.py` luôn gọi `from_pretrained` cho mọi `--model-path`. Đúng với
model id và full-weights checkpoint, **sai với đúng thứ trainer thật sự
ghi ra khi bật LoRA**: thư mục adapter chỉ chứa delta, không có base
weight để cộng vào.

`src/fvqa/checkpoint.py` mới phân loại `hf_id` / `full` / `peft`. Hai chi
tiết đọc được từ code trainer thật, không phải từ giả định:

1. **LoRA checkpoint của trainer này cũng có `config.json`** (ghi bởi
   `model.config.save_pretrained` cạnh adapter) → phải check
   `adapter_config.json` **trước**, nếu không adapter bị route nhầm sang
   nhánh full-weights.
2. **`non_lora_state_dict.bin` chứa weight đã train nhưng không nằm trong
   adapter** — mọi param `requires_grad` không phải LoRA tensor. Với config
   hiện tại (`freeze_merger: false`) đó là vision-language merger. Chỉ load
   `adapter_model.safetensors` sẽ **giữ nguyên merger chưa train và không
   báo lỗi gì cả**: model chạy bình thường, chỉ là điểm thấp hơn run đã tạo
   ra nó. Giờ load vào base model trước khi PEFT bọc, dùng đúng logic strip
   prefix của trainer, và key không khớp thì raise thay vì bị `strict=False`
   nuốt mất.

### M1 — PR3: `tuning_method` enum (xong)

Hai boolean `lora.enabled` / `qlora.enabled` mô tả 4 trạng thái mà chỉ 3
có nghĩa — và config **cấm đúng cái thật sự là QLoRA** (cả hai true), nên
cách duy nhất để xin QLoRA là `lora=false, qlora=true`, sinh ra
`--lora_enable False --bits 4`: base 4-bit, **không gắn gì trainable lên
đó**. Run không train gì và không báo gì.

`model.tuning_method: full | lora | qlora` thay cả hai. QLoRA giờ emit cả
`--lora_enable True` lẫn `--bits 4`.

`model.qlora` → `model.quantization`, chỉ giữ hai field trainer thật sự
nhận (`--quant_type`, `--double_quant`). Bỏ `bnb_4bit_compute_dtype`:
trainer tự suy từ `training.bf16/fp16`, không có flag nào để forward.

Thêm `training.max_steps` + `scripts/smoke_gpu.sh` — 2 optimizer step rồi
reload lại adapter vừa ghi, phủ được: model load, decode ảnh, forward,
backward, ghi checkpoint, reload adapter, generate.

### M2 — PR4: retrieval core (xong)

`src/fvqa/retrieval/`: `GraphRetriever`, `LexicalRanker`, `RetrievedFact`/
`EntityCandidate`/`RetrievalResult`, `format_facts`.

Luồng: `seed text → find_entities → bfs_with_hops → rank(question) → top-k`.
Nhận **seed dạng text, không nhận ảnh** — để test được retrieval độc lập
với chất lượng vision-seeding. Điểm kém end-to-end có hai nguyên nhân khả
dĩ (graph đi sai chỗ / model nhìn sai vật), tách ra mới quy trách nhiệm được.

`max_hops` chuyển từ `data` sang section `retrieval` mới — nó là tham số
thuật toán, không phải thuộc tính dataset. Toàn bộ setting được ghi vào
`RetrievalResult.settings` để hai run khác hop không lẫn vào nhau.

**Đo thật trên 225.434 triple + 981 câu hỏi thật** (oracle seed = entity
của supporting fact *không phải* đáp án — seed bằng đáp án thì không đo gì
cả), 1 hop:

| | recall@1 | recall@5 | recall@10 |
|---|---|---|---|
| lexical ranker | 37,9% | 59,4% | 66,3% |

Đây là **cận trên của nhánh graph-retrieval khi seed đã đúng**: vision-seeding
chỉ có thể tệ hơn con số này. Nếu oracle-fact đạt X% thì phần mất do
traversal+ranking đã đo được ngay từ bây giờ, chưa cần GPU.

Đo luôn để chọn default thay vì đoán: `max_candidate_facts` 50/100/300/5000
cho recall@5 58,0% / 59,4% / 61,1% / 62,7%; cap 300 tốn 1,6 ms/câu — không
đáng kể so với một lần generate của VLM → default 300.

Hạn chế đã ghi nhận, không giấu: câu hỏi kiểu "What can a dog do?" sau khi
bỏ stopword chỉ còn đúng seed, mọi fact của `dog` điểm bằng nhau, tie-break
theo fact id — deterministic nhưng vô nghĩa. Đây chính là chỗ embedding
hoặc LLM reranker sẽ ăn điểm.

### M2 — PR5: eval conditions (xong)

`fvqa eval --condition {stored | no-context | style | oracle-fact |
oracle-seed-graph}`. Mỗi condition **chỉ khác nhau đúng một thứ** — phần
context đặt trước câu hỏi — nên hiệu số điểm giữa chúng đọc được:

```
oracle-fact − oracle-seed-graph   = mất mát do traversal + ranking
oracle-seed-graph − vision-seed   = mất mát do vision-seeding  (M3)
vision-seed − no-context          = giá trị thật của graph retrieval
```

Retrieval chạy **lúc eval**, không bake vào split JSON — đúng như đề xuất.
Split giữ `fvqa_question`/`fvqa_answer` gốc, condition tự dựng prompt. Nhờ
vậy đổi `max_hops` hay đổi ranker không phải `prepare` lại, và metadata
không lệch khỏi prompt thật sự đã chấm.

Mọi condition có fact đều đi qua chung `apply_grounding` → hai condition
chỉ khác **fact đến từ đâu**, không khác hình dạng prompt.

Provenance ghi từng sample (seed nào, resolve ra entity nào, fact nào vào
prompt, supporting fact có sống sót không) + tổng hợp mức run. Recall này
là **cận trên của phần grounding có thể đóng góp** — model không thể dùng
fact mà retrieval chưa bao giờ đưa cho nó.

**Bug bắt được khi tự chạy thử, không phải từ test:**

1. `apply_grounding` trả về câu hỏi trần khi `grounding.enabled=false` (mặc
   định). Nghĩa là `--condition oracle-fact` sẽ **âm thầm không đưa fact
   nào** rồi báo điểm như thể đã đưa. Sửa: condition tự bật grounding —
   `data.grounding.enabled` quyết định `prepare` có bake fact vào split
   không, còn chọn condition là quyết định riêng, đã nằm ở tên condition.
2. `oracle-fact` báo "supporting fact retrieved: 0/11" vì provenance thiếu
   key — đọc thành "retrieval fail 100%" trong khi condition này theo định
   nghĩa luôn đưa đúng fact. Sửa thành `True` khi fact resolve được.
3. `oracle-fact` in ra "Retrieval: 2 hop(s), top-5, lexical" — mô tả việc
   chưa hề xảy ra. Tách `needs_graph` (cần load facts) khỏi `traverses`
   (thật sự đi graph).
4. `--condition style` với `inference.system_prompt` rỗng giống hệt
   `no-context` → giờ log warning thay vì im lặng.

`vision-seed-graph` raise `NotImplementedError` với thông báo rõ, không
lặng lẽ tụt về condition khác rồi báo số cho thí nghiệm chưa từng chạy.

**Đo thật, 282 câu val:**

| hops | top-5 | top-10 |
|---|---|---|
| 1 | 52,8% | 61,3% |
| 2 | 52,8% | 61,3% |
| 3 | 52,8% | 61,3% |

`max_hops` **không ảnh hưởng gì cả** — và đây là chuyện cấu trúc, không
phải kết luận về graph: oracle seed lấy từ chính endpoint của supporting
fact, nên fact đó luôn cách 1 hop; đi sâu hơn chỉ thêm nhiễu xếp hạng thấp
hơn. Multi-hop chỉ có ý nghĩa cho **vision-seeding** (M3), khi model đoán
ra một entity *hàng xóm* của entity đúng chứ không phải entity đúng. Default
vẫn để 2 vì đó mới là pipeline thật sự nhắm tới — nhưng ghi rõ ở đây để
sau này không ai đọc nhầm "tăng hop vô dụng" thành kết luận về dataset.

### M3 — PR6: vision-seeding (xong)

`src/fvqa/retrieval/seeds.py`: `SeedProvider` protocol,
`ManualSeedProvider`, `QwenVisionSeedProvider`, `SeedCache`.

**Guardrail quan trọng nhất:** interface chỉ nhận `(image_path, question)`.
Không phải thiếu sót — nếu seed provider nhìn được supporting fact hay đáp
án thì nó có thể trả về chính đáp án dưới dạng "đoán", và điều kiện
vision-seed sẽ đạt điểm cao trong khi **không đo gì cả**. Chặn bằng cấu
trúc đáng giá hơn là tự nhắc mình đừng dùng.

(Oracle seed *không* phải một `SeedProvider` — nó suy từ supporting fact
chứ không từ ảnh, nên để nguyên trong `conditions.py` nơi phép suy đó nhìn
thấy được. Gói nó thành SeedProvider sẽ che mất việc nó không hề nhìn ảnh.)

Parse phản hồi model chịu được thực tế: JSON trong code fence, có preamble,
JSON hỏng → fallback tách dòng. Mất hẳn seed list biến thành retrieval
failure trông như lỗi graph, nên khôi phục tạm bợ vẫn hơn không khôi phục.

`SeedCache` khoá theo `(model, image, question)` — seed không phụ thuộc gì
ở hạ nguồn, nên mọi thí nghiệm đổi hop/ranker/top-k lẽ ra phải chạy lại VLM
trên cả split để ra đúng kết quả cũ.

**Fallback ladder, đo trước khi giữ:** `seed_variants()` thử lần lượt dạng
chuẩn hoá → số ít → head noun → head noun số ít. Mô phỏng cách VLM thật
phát âm entity (thêm article, số nhiều, viết hoa) trên 3.200 cách diễn đạt
từ dữ liệu thật:

| | tỉ lệ resolve được |
|---|---|
| chỉ normalize (code cũ) | 86,6% |
| full ladder | **99,8%** |

`singularize()` cố tình thô sơ, không kéo thư viện morphology về cho một
luật: đoán sai chỉ tốn một lần thử không ra gì, không bao giờ ra đáp án sai.

**Chuỗi đo hoàn chỉnh** (152 câu val thật, stub VLM đoán rộng rãi — model
thật sẽ tệ hơn):

| condition | supporting fact vào được prompt |
|---|---|
| `oracle-fact` | 100% (theo định nghĩa) |
| `oracle-seed-graph` | 50,7% |
| `vision-seed-graph` | 25,7% |

```
100   − 50,7 = 49,3pp   mất do traversal + ranking
50,7  − 25,7 = 25,0pp   mất do vision-seeding
```

Đây là **cận trên** của phần grounding đóng góp được ở mỗi tầng — model
không dùng được fact mà retrieval chưa từng đưa cho nó. Cột metric trong
lần chạy này vô nghĩa (stub luôn trả "trumpet"); cột recall mới là kết quả.

### M4 — PR7: Modal smoke pipeline + ruff (xong)

`modal run scripts/train_on_modal.py --step smoke [--smoke-level N]` —
4 mức, rẻ trước, **dừng ở mức đầu tiên fail** (mức fail làm mọi mức trên
nó vô nghĩa). Mỗi mức có timeout riêng ngắn, không mượn timeout 24h của
hàm train: smoke test treo được một ngày thì không còn là smoke test.

| mức | kiểm tra | GPU | timeout |
|---|---|---|---|
| 1 | import + config + mini-graph | không | 10 phút |
| 2 | tải + prepare + kiểm tra format split | không | 30 phút |
| 3 | base model load + generate | có | 30 phút |
| 4 | 2 optimizer step + **reload lại adapter** | có | 60 phút |

Mức 4 quan trọng ở nửa sau: train ghi ra *một* thư mục thì chứng minh
được rất ít; lỗi thật sự cần bắt là adapter checkpoint **không load lại
được** — thứ chỉ lộ ra khi ai đó eval một run thật vài ngày sau.

Sửa luôn mấy chỗ Modal-specific dễ âm thầm hỏng:

- `PYTHONPATH=/root/src` tường minh, thay vì trông chờ working directory
  tình cờ import được `fvqa`.
- `secrets=[HF_SECRET]` gắn cho **cả** `evaluate_model` và các smoke mức
  3-4, không chỉ `train_model`. Trước đó repo gated sẽ chỉ fail ở nhánh
  eval/baseline.
- `retrieval.seed_cache_dir` trỏ vào data volume + `data_volume.commit()`
  sau eval — nếu không, vision seed bị vứt cùng container và lần chạy sau
  trả lại toàn bộ VLM call cho đúng kết quả cũ.
- `evaluate_model` nhận `condition` và `base_model`, ghi condition vào tên
  file kết quả để 4 điều kiện không đè lên nhau.
- `--smoke-level 0` trước đây thoát vòng lặp ngay rồi vẫn in "✅ passed" —
  báo thành công trong khi không chạy gì. Giờ reject.

`ruff` vào CI (`F,E,W,I,B,UP`, bỏ E501 vì đã có line-length). Cố tình gọn:
bắt lỗi thật — tên chưa định nghĩa, import thừa, mutable default — chứ
không đem chuyện format ra review lại.

### Còn lại

**M3**: `SeedProvider` interface, Qwen3-VL vision-seeding có cache, fallback
khi `find_entities` không match, nối `vision-seed-graph`.

**M4**: Modal smoke test theo 4 mức (import → prepare → baseline GPU →
train 2-step GPU), lint (`ruff`) thêm sau CI cơ bản.

Chi tiết đầy đủ từng PR nằm trong lịch sử hội thoại — tài liệu này chỉ ghi
quyết định và trạng thái đã verify.
