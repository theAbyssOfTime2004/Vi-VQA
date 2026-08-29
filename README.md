# FVQA: Fact-based VQA với knowledge-graph traversal, trên Qwen3-VL

Fine-tune / eval **Qwen3-VL-8B-Instruct** trên
[FVQA](https://github.com/wangpengnorman/FVQA) — 2.190 ảnh (COCO + ImageNet),
5.826 câu hỏi, mỗi câu neo vào đúng một fact trong **225.434 triple**
`(e1, relation, e2)` thật từ DBpedia/ConceptNet/WebChild.

Khác với các VQA dataset dùng free text làm "kiến thức", FVQA ship một
**đồ thị tri thức thật** — đủ để tự dựng graph và viết BFS/DFS/shortest-path
lên đó, không phải retrieval bằng embedding.

## Cài đặt

```bash
git clone <repo-url> && cd Vi-VQA
python3 -m venv .venv && source .venv/bin/activate

pip install -e '.[train]'
pip install flash-attn --no-build-isolation   # tuỳ chọn, nhanh hơn đáng kể
```

`transformers>=4.57.0` là yêu cầu bắt buộc: `Qwen3VLForConditionalGeneration`
không tồn tại ở bản thấp hơn.

### Tải dữ liệu

FVQA không có trên HuggingFace — tải và giải nén thủ công:

```bash
curl -L -o fvqa.zip "https://www.dropbox.com/s/iyz6l7jhbt6jb7q/new_dataset_release.zip?dl=1"
unzip fvqa.zip -d data/fvqa
# ra: data/fvqa/Name_Lists/  và  data/fvqa/new_dataset_release/
```

## Dùng

```bash
fvqa config                               # in ra cấu hình đã resolve
fvqa prepare                              # local release -> train/val/test JSON
fvqa train                                # fine-tune LoRA
fvqa eval --model-path ./checkpoints/qwen3vl-fvqa/checkpoint-1500
fvqa chat --model-path ./checkpoints/qwen3vl-fvqa/checkpoint-1500
```

```bash
# Ví dụ override
fvqa prepare --limit 100                            # thử nhanh trên 100 câu hỏi
fvqa prepare --set data.fold=2                      # đổi fold train/test (0-4)
fvqa train --dry-run                                # in lệnh train, không chạy
fvqa eval --model-path <ckpt> --num-samples -1 --output results.json
```

## Baseline zero-shot

`--model-path` nhận thẳng HF model id, chấm được model **chưa fine-tune**
mà không cần train gì — luôn chạy cái này trước khi bắt đầu một run nhiều giờ:

```bash
fvqa eval --model-path Qwen/Qwen3-VL-8B-Instruct \
          --num-samples 200 --output ./results/baseline.json
```

Chạy trên Modal:

```bash
modal secret create huggingface-secret HF_TOKEN=hf_...
modal run scripts/train_on_modal.py --step all
modal run scripts/train_on_modal.py::check_status
```

Modal và local dùng **chung** module sinh lệnh train, nên hai đường chạy
không thể lệch nhau.

## Hai kiểu grounding — đo hai câu hỏi khác nhau

**Oracle fact** — model được cho thẳng đúng fact hỗ trợ câu hỏi
(`fact_surface`), tách biệt với việc tự đi tìm fact đó:

```bash
fvqa prepare --set data.grounding.enabled=true
```

Đo *"biết trước fact đúng thì giúp được bao nhiêu"* — cận trên, không phải
kết quả pipeline thật.

**Graph retrieval** — model (hoặc bạn) tự đi tìm fact bằng cách duyệt đồ thị,
không được cho biết trước đáp án:

```python
from fvqa.data import KnowledgeGraph

g = KnowledgeGraph.from_facts_file("data/fvqa/new_dataset_release/all_fact_triples_release.json")
seeds = g.find_entities("trumpet")           # ['/c/en/trumpet', '/c/en/trumpet/n', ...]
facts = g.bfs(seeds, max_hops=1)              # BFS thật, không phải embedding search
path = g.shortest_path(entity_a, entity_b)    # None nếu không nối được
```

Đo *"tự tìm fact đúng bằng graph thì còn lại bao nhiêu"* — hiệu số giữa hai
điều kiện là thước đo thật cho việc graph có đáng hay không.

**Chưa nối vào `fvqa eval`**: hiện tại eval chỉ chạy được điều kiện oracle.
Đường graph-retrieval — seed từ thực thể model tự nhận diện trong ảnh, BFS,
rank fact, nhét vào prompt — `KnowledgeGraph` đã có sẵn, chỉ chưa lắp vào
`evaluation/runner.py`.

## Cấu trúc

```
config/config.yaml          # nguồn cấu hình duy nhất cho mọi đường chạy
src/fvqa/
  config.py                 # load + validate config thành dataclass
  cli.py                    # fvqa prepare | train | eval | chat | config
  model.py                  # nạp Qwen3-VL, sinh câu trả lời
  data/
    fvqa.py                 # loader FVQA (không qua datasets.load_dataset)
    fvqa_graph.py            # KnowledgeGraph: BFS/DFS/shortest-path thật
    grounding.py             # nhét fact vào prompt (oracle-fact grounding)
    samples.py                # IMAGE_TOKEN, assign_splits, write_split — dùng chung
  train/
    command.py               # config -> argv cho trainer
    runner.py                # clone trainer repo và chạy
  evaluation/
    metrics.py                # exact match, similarity, BLEU, ROUGE-L, CIDEr
    runner.py                 # sinh dự đoán và chấm điểm
scripts/
  train_qwen3vl.sh           # wrapper mỏng cho chạy local
  train_on_modal.py          # phần thuộc về Modal, không chứa logic dataset
tests/                       # chạy không cần GPU
```

## Kiến trúc

**Model:** Qwen3-VL-8B-Instruct — vision encoder ViT, LLM 8B, projector
vision-language.

**Fine-tune:** LoRA rank 128 / alpha 256 trên LLM và projector, đóng băng
vision encoder. Batch hiệu dụng 16 (1 × 16 gradient accumulation), bf16,
gradient checkpointing — vừa 24GB VRAM.

**Split:** FVQA có sẵn 5 fold train/test chính thức (0-4) — `test` giữ
nguyên theo đúng fold gốc để so được với kết quả trong paper; `val` được cắt
từ `train` bằng `assign_splits` (chia theo *ảnh*, không phải theo câu hỏi,
để câu hỏi về cùng một ảnh không rơi vào cả hai phía ranh giới).

Training thực tế được giao cho
[`2U1/Qwen-VL-Series-Finetune`](https://github.com/2U1/Qwen-VL-Series-Finetune);
repo này lo dữ liệu, cấu hình và đánh giá.

## Đánh giá

```
exact_match   % trùng khít sau khi chuẩn hoá (NFC, lowercase, bỏ dấu câu)
similarity    % tương đồng mức ký tự
bleu          BLEU-4 mức corpus, có smoothing
rouge_l       F-measure trên LCS, beta=1.2
cider         CIDEr-D, thang 0–10
```

Chỉ dùng exact match không đủ cho câu trả lời tự do: một câu đúng nhưng diễn
đạt khác vẫn được 0 điểm. Các metric còn lại chấm điểm từng phần.

## Phát triển

```bash
pip install -e '.[dev]'
pytest                       # không cần torch/transformers
```

Tầng config, data và metrics cố tình không import torch, nên test chạy
trong vài giây trên máy không GPU.

## Các điều kiện đánh giá

```bash
fvqa eval --model-path Qwen/Qwen3-VL-8B-Instruct --condition no-context
fvqa eval --model-path Qwen/Qwen3-VL-8B-Instruct --condition style
fvqa eval --model-path Qwen/Qwen3-VL-8B-Instruct --condition oracle-fact
fvqa eval --model-path Qwen/Qwen3-VL-8B-Instruct --condition oracle-seed-graph
```

| Condition | Model nhận được |
|---|---|
| `no-context` | ảnh + câu hỏi. Sàn để so mọi thứ khác |
| `style` | thêm system prompt chỉ nói *cách* trả lời, không nói nội dung |
| `oracle-fact` | thêm đúng supporting fact — cận trên của grounding |
| `oracle-seed-graph` | thêm entity xuất phát đúng, **không** cho fact ID: phải tự BFS + rank để tìm |
| `vision-seed-graph` | model tự nhìn ảnh đoán entity rồi mới BFS — pipeline thật |
| `stored` | phát lại đúng prompt trong split file (mặc định; dùng cho checkpoint đã fine-tune) |

Mỗi condition chỉ khác nhau đúng phần context, nên hiệu số điểm chỉ ra lỗi
nằm ở đâu:

```
oracle-fact − oracle-seed-graph   mất mát do traversal + ranking
oracle-seed-graph − vision-seed   mất mát do vision-seeding
vision-seed − no-context          giá trị thật của graph retrieval
```

Bước vision-seeding chỉ nhận `(ảnh, câu hỏi)` — không thấy supporting fact
hay đáp án. Đó là chặn bằng cấu trúc: một seed provider nhìn được annotation
có thể trả về chính đáp án dưới dạng "đoán", và điều kiện này sẽ đạt điểm
cao mà không đo gì cả. Seed được cache theo `(model, ảnh, câu hỏi)` nên đổi
hop/ranker/top-k không phải chạy lại VLM.

Retrieval chạy lúc eval, không bake vào split JSON — đổi `max_hops` hay
ranker không cần `prepare` lại. Result JSON ghi đủ provenance: seed nào,
resolve ra entity nào, fact nào vào prompt, supporting fact có sống sót
không. Recall đó là cận trên của phần grounding có thể đóng góp — model
không dùng được fact mà retrieval chưa từng đưa cho nó.

## Chạy trên Modal

```bash
pip install modal && modal token new
modal secret create huggingface-secret HF_TOKEN=hf_...

modal run scripts/train_on_modal.py --step smoke      # 4 mức, rẻ trước
modal run scripts/train_on_modal.py --step smoke --smoke-level 2   # chỉ mức không cần GPU
modal run scripts/train_on_modal.py --step prepare
modal run scripts/train_on_modal.py --step train
modal run scripts/train_on_modal.py --step baseline --condition no-context
```

| smoke | kiểm tra | GPU |
|---|---|---|
| 1 | import + config + mini-graph | không |
| 2 | tải dataset + prepare + format split | không |
| 3 | base model load + generate | có |
| 4 | 2 optimizer step + reload lại adapter | có |

Dừng ở mức đầu tiên fail — mức fail làm mọi mức trên nó vô nghĩa.

## Yêu cầu phần cứng

| GPU | `model.tuning_method` | Thời gian (2 epoch) |
|-----|----------------------|---------------------|
| RTX 3080 (10GB) | `qlora` (4-bit) | vài giờ |
| RTX 3090 (24GB) | `lora` | vài giờ |
| A100 (40GB) | `lora` | 1-2 giờ |

Trước khi chạy thật, nên chạy smoke test — 2 optimizer step rồi reload lại
adapter vừa lưu, đủ để biết model load được, ảnh decode được, forward và
backward chạy, checkpoint ghi được và `fvqa eval` đọc lại được:

```bash
bash scripts/smoke_gpu.sh
```

FVQA nhỏ hơn nhiều VQA dataset khác (2.190 ảnh, 5.826 câu hỏi) — train
nhanh hơn hẳn so với dataset cỡ chục nghìn ảnh. Hết VRAM thì giảm
`image_max_pixels` trước, rồi mới tới batch size:

```bash
fvqa train --set model.image_max_pixels=589824 \
           --set training.gradient_accumulation_steps=32
```

## Tham khảo

- [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
- [FVQA — GitHub](https://github.com/wangpengnorman/FVQA)
- [FVQA paper — arXiv:1606.05433](https://arxiv.org/abs/1606.05433)
- [Qwen-VL-Series-Finetune](https://github.com/2U1/Qwen-VL-Series-Finetune)
- [LoRA — arXiv:2106.09685](https://arxiv.org/abs/2106.09685)

## Giấy phép

MIT
