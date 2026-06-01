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

# Make a bar plot of the sequence length distribution
plt.figure(figsize=(8, 5))
plt.bar(range(len(seq_lens)), seq_lens)
plt.xlabel('User')
plt.ylabel('Sequence length')
plt.title(f'Sequence length distribution — {CATEGORY}')
plt.savefig('seq_len_bar.png', dpi=150)
print(f'Saved seq_len_bar.png ({len(seq_lens)} users)')