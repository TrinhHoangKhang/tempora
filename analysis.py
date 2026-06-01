import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CATEGORY = 'Beauty'  # change if needed
path = f'cache/AmazonReviews2014/{CATEGORY}/processed/all_item_seqs.json'

with open(path) as f:
    all_item_seqs = json.load(f)

seq_lens = [len(seq) for seq in all_item_seqs.values()]

plt.figure(figsize=(8, 5))
plt.hist(seq_lens, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Sequence length')
plt.ylabel('Number of users')
plt.title(f'Sequence length distribution — {CATEGORY}')
plt.savefig('seq_len_hist.png', dpi=150)
print(f'Saved seq_len_hist.png ({len(seq_lens)} users)')