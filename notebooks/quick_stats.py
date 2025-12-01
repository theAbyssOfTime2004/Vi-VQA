"""Quick statistics script - chạy nhanh để lấy insights"""
import sys
import os
sys.path.append('..')

from datasets import load_dataset
from huggingface_hub import login
from collections import Counter

# Login - set HF_TOKEN environment variable or use: huggingface-cli login
hf_token = os.environ.get('HF_TOKEN')
if hf_token:
    login(token=hf_token)
else:
    login()  # Will use cached credentials from huggingface-cli login

print('Loading dataset...')
dataset = load_dataset('5CD-AI/Viet-ViTextVQA-gemini-VQA', split='train')
print(f'✓ Loaded {len(dataset)} samples\n')

# 1. Conversation statistics
num_turns = [len(item['conversations']) for item in dataset]
num_qa_pairs = [len(item['conversations']) // 2 for item in dataset]

print("="*80)
print("1. CONVERSATION STATISTICS")
print("="*80)
print(f"Số turns trung bình: {sum(num_turns) / len(num_turns):.2f}")
print(f"Min turns: {min(num_turns)}, Max turns: {max(num_turns)}")
print(f"Số QA pairs trung bình: {sum(num_qa_pairs) / len(num_qa_pairs):.2f}")
print(f"Tổng số QA pairs: {sum(num_qa_pairs)}")

# 2. Length statistics
question_lengths = []
answer_lengths = []

for item in dataset:
    convs = item['conversations']
    for i in range(0, len(convs), 2):
        if i < len(convs) and convs[i]['role'] in ['user', 'human']:
            question_lengths.append(len(convs[i]['content']))
        if i+1 < len(convs) and convs[i+1]['role'] in ['assistant', 'gpt']:
            answer_lengths.append(len(convs[i+1]['content']))

print(f"\n{'='*80}")
print("2. LENGTH STATISTICS")
print("="*80)
print(f"Độ dài câu hỏi: avg={sum(question_lengths)/len(question_lengths):.1f}, min={min(question_lengths)}, max={max(question_lengths)}")
print(f"Độ dài câu trả lời: avg={sum(answer_lengths)/len(answer_lengths):.1f}, min={min(answer_lengths)}, max={max(answer_lengths)}")

# 3. Answer vocabulary
all_answers = []
for item in dataset:
    convs = item['conversations']
    for turn in convs:
        if turn['role'] in ['assistant', 'gpt']:
            all_answers.append(turn['content'])

answer_counter = Counter(all_answers)

print(f"\n{'='*80}")
print("3. ANSWER VOCABULARY")
print("="*80)
print(f"Tổng số câu trả lời: {len(all_answers)}")
print(f"Số câu trả lời unique: {len(answer_counter)}")
print(f"\nTop 20 câu trả lời phổ biến nhất:")
for i, (answer, count) in enumerate(answer_counter.most_common(20), 1):
    display = answer[:60] + "..." if len(answer) > 60 else answer
    print(f"  {i:2d}. ({count:4d}x) {display}")

# 4. Coverage analysis
answer_frequencies = list(answer_counter.values())
answer_frequencies.sort(reverse=True)
total_answers = sum(answer_frequencies)

print(f"\n{'='*80}")
print("4. VOCABULARY COVERAGE")
print("="*80)
print("Coverage khi lấy top K answers:")
for k in [100, 500, 1000, 2000, 5000, 10000]:
    if k <= len(answer_frequencies):
        coverage = sum(answer_frequencies[:k]) / total_answers * 100
        print(f"  Top {k:5d}: {coverage:6.2f}%")

# 5. Description statistics
description_lengths = [len(item['description']) for item in dataset]

print(f"\n{'='*80}")
print("5. DESCRIPTION STATISTICS")
print("="*80)
print(f"Độ dài description: avg={sum(description_lengths)/len(description_lengths):.1f}, min={min(description_lengths)}, max={max(description_lengths)}")

print(f"\n{'='*80}")
print("DONE!")
print("="*80)
