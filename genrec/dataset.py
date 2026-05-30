from logging import getLogger
import numpy as np
from datasets import Dataset


class AbstractDataset:
    def __init__(self, config: dict):
        self.config = config
        self.accelerator = self.config['accelerator']
        self.logger = getLogger()

        self.all_item_seqs = {}
        self.id_mapping = {
            'user2id': {'[PAD]': 0},
            'item2id': {'[PAD]': 0},
            'id2user': ['[PAD]'],
            'id2item': ['[PAD]']
        }
        self.item2meta = None
        self.split_data = None

    def __str__(self) -> str:
        return f'[Dataset] {self.__class__.__name__}\n' \
                f'\tNumber of users: {self.n_users}\n' \
                f'\tNumber of items: {self.n_items}\n' \
                f'\tNumber of interactions: {self.n_interactions}\n' \
                f'\tAverage item sequence length: {self.avg_item_seq_len}'

    @property
    def n_users(self):
        """
        Returns the number of users in the dataset.

        Returns:
            int: The number of users in the dataset.
        """
        return len(self.user2id)

    @property
    def n_items(self):
        """
        Returns the total number of items in the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        return len(self.item2id)

    @property
    def n_interactions(self):
        """
        Returns the total number of interactions in the dataset.

        Returns:
            int: The total number of interactions.
        """
        n_inters = 0
        for user in self.all_item_seqs:
            n_inters += len(self.all_item_seqs[user])
        return n_inters

    @property
    def avg_item_seq_len(self):
        """
        Returns the average length of item sequences in the dataset.

        Returns:
            float: The average length of item sequences.
        """
        return self.n_interactions / self.n_users

    @property
    def user2id(self):
        """
        Returns the user-to-id mapping.

        Returns:
            dict: The user-to-id mapping.
        """
        return self.id_mapping['user2id']

    @property
    def item2id(self):
        """
        Returns the item-to-id mapping.

        Returns:
            dict: The item-to-id mapping.
        """
        return self.id_mapping['item2id']

    def _download_and_process_raw(self):
        """
        This method should be implemented in the subclass.
        It is responsible for downloading and processing the raw data.
        """
        raise NotImplementedError('This method should be implemented in the subclass')

    def _leave_one_out(self):
        """
        Splits the dataset into train, validation, and test sets using the leave-one-out strategy.

        Returns:
            dict: A dictionary containing the train, validation, and test datasets.
                  Each dataset is represented as a dictionary with 'user' and 'item_seq' keys.
                  The 'user' key contains a list of users, and the 'item_seq' key contains a list of item sequences.
        """
        self.log('[DATASET] Building leave-one-out train/val/test splits...')

        datasets = {'train': {'user': [], 'item_seq': []},
                    'val': {'user': [], 'item_seq': []},
                    'test': {'user': [], 'item_seq': []}}

        # Optionally subsample users for faster debugging/tuning.
        # Stratifies by sequence length so short/medium/long sequences are
        # all equally represented — avoids the bias of pure random sampling.
        users = list(self.all_item_seqs.keys())
        n_subset = self.config.get('debug_subset_users', None)
        if n_subset is not None:
            n_buckets = self.config.get('debug_n_buckets', 10)
            rng = np.random.default_rng(seed=42)

            # Sort users by sequence length and divide into equal-width buckets
            users_sorted = sorted(users, key=lambda u: len(self.all_item_seqs[u]))
            bucket_size = max(1, len(users_sorted) // n_buckets)
            buckets = [users_sorted[i:i + bucket_size] for i in range(0, len(users_sorted), bucket_size)]

            # Sample evenly from each bucket; carry remainder to last bucket
            per_bucket = n_subset // len(buckets)
            sampled = []
            for bucket in buckets:
                k = min(per_bucket, len(bucket))
                sampled.extend(rng.choice(bucket, size=k, replace=False).tolist())
            # Top up to exactly n_subset if rounding left us short
            remaining = [u for u in users if u not in set(sampled)]
            shortfall = n_subset - len(sampled)
            if shortfall > 0 and remaining:
                sampled.extend(rng.choice(remaining, size=min(shortfall, len(remaining)), replace=False).tolist())
            users = sampled

            seq_lens = [len(self.all_item_seqs[u]) for u in users]
            self.log(
                f'[DATASET] debug_subset_users={n_subset}: using {len(users)} users '
                f'(stratified, seq_len min={min(seq_lens)} avg={sum(seq_lens)/len(seq_lens):.1f} max={max(seq_lens)})'
            )
        else:
            self.log(f'[DATASET] Using all {len(users)} users for split')

        for user in users:
            datasets['test']['user'].append(user)
            datasets['test']['item_seq'].append(self.all_item_seqs[user])
            if len(self.all_item_seqs[user]) > 1:
                datasets['val']['user'].append(user)
                datasets['val']['item_seq'].append(self.all_item_seqs[user][:-1])
            if len(self.all_item_seqs[user]) > 2:
                datasets['train']['user'].append(user)
                datasets['train']['item_seq'].append(self.all_item_seqs[user][:-2])

        self.log('[DATASET] Split info:')
        for split_name in ('train', 'val', 'test'):
            split = datasets[split_name]
            n_users = len(split['user'])
            if n_users == 0:
                self.log(f'[DATASET]   {split_name}: 0 users')
                continue
            seq_lens = [len(seq) for seq in split['item_seq']]
            n_inters = sum(seq_lens)
            avg_len = n_inters / n_users
            self.log(
                f'[DATASET]   {split_name}: {n_users} users, '
                f'{n_inters} interactions, avg seq length {avg_len:.1f} '
                f'(min={min(seq_lens)}, max={max(seq_lens)})'
            )
        self.log('[DATASET] Converting splits to HuggingFace Dataset objects...')
        for split in datasets:
            datasets[split] = Dataset.from_dict(datasets[split])
        return datasets

    def split(self):
        """
        Split the dataset into train, validation, and test sets based on the specified split strategy.

        Returns:
            datasets (dict): A dictionary containing the train and test datasets.
        """
        if self.split_data is not None:
            self.log('[DATASET] Using cached train/val/test splits')
            return self.split_data

        split_strategy = self.config['split']
        self.log(f'[DATASET] Split strategy: {split_strategy}')
        if split_strategy in ['leave_one_out', 'last_out']:
            datasets = self._leave_one_out()
        else:
            raise NotImplementedError(f'Split strategy [{split_strategy}] not implemented.')

        self.split_data = datasets
        self.log('[DATASET] ========== Split complete ==========')
        for split_name in ('train', 'val', 'test'):
            self.log(f'[DATASET]   {split_name}: {len(self.split_data[split_name])} users')
        return self.split_data

    def log(self, message, level='info'):
        from genrec.utils import log
        return log(message, self.config['accelerator'], self.logger, level=level)
