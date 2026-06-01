import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CATEGORY = 'Beauty'  # change if needed
path = f'cache/AmazonReviews2014/{CATEGORY}/processed/all_item_seqs.json'

with open(path) as f:
    all_item_seqs = json.load(f)

seq_lens = [len(seq) for seq in all_item_seqs.values()]
print(f"Number of users: {len(seq_lens)}")
print(f"Number of items: {len(all_item_seqs)}")
print(f"Average sequence length: {sum(seq_lens) / len(seq_lens)}")
print(f"Minimum sequence length: {min(seq_lens)}")
print(f"Maximum sequence length: {max(seq_lens)}")

plt.figure(figsize=(8, 5))
plt.hist(seq_lens, bins=100, edgecolor='black', alpha=0.7)
plt.xlabel('Sequence length')
plt.ylabel('Number of users')
plt.xticks(range(0, max(seq_lens), 10))
plt.xlim(0, max(seq_lens))
plt.title(f'Sequence length distribution — {CATEGORY}')
plt.savefig(f'figure/{CATEGORY}_seq_len_distribution.png', dpi=150)
print(f'Saved {CATEGORY}_seq_len_distribution.png ({len(seq_lens)} users)')