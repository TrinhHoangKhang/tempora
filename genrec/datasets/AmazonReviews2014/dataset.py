import os
import gzip
import json
from tqdm import tqdm
from collections import defaultdict
import numpy as np
from typing import Optional

from genrec.dataset import AbstractDataset
from genrec.utils import download_file, clean_text


class AmazonReviews2014(AbstractDataset):

    def __init__(self, config: dict):
        super(AmazonReviews2014, self).__init__(config)

        self.category = config['category']
        self._check_available_category()
        self.log(
            f'[DATASET] Amazon Reviews 2014 for category: {self.category}'
        )

        self.cache_dir = os.path.join(
            config['cache_dir'], 'AmazonReviews2014', self.category
        )
        self._download_and_process_raw()

    def _check_available_category(self):
        # Checks if the `self.category` is available in the dataset.
        
        available_categories = [
            'Books',
            'Electronics',
            'Movies_and_TV',
            'CDs_and_Vinyl',
            'Clothing_Shoes_and_Jewelry',
            'Home_and_Kitchen',
            'Kindle_Store',
            'Sports_and_Outdoors',
            'Cell_Phones_and_Accessories',
            'Health_and_Personal_Care',
            'Toys_and_Games',
            'Video_Games',
            'Tools_and_Home_Improvement',
            'Beauty',
            'Apps_for_Android',
            'Office_Products',
            'Pet_Supplies',
            'Automotive',
            'Grocery_and_Gourmet_Food',
            'Patio_Lawn_and_Garden',
            'Baby',
            'Digital_Music',
            'Musical_Instruments',
            'Amazon_Instant_Video'
        ]
        assert self.category in available_categories, \
            f'Category "{self.category}" not available. ' \
            f'Available categories: {available_categories}'

    def _download_raw(self, path: str, type: str = 'reviews') -> str:
        # Downloads the raw data file from the specified URL and saves it locally.
        
        url = f'https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/{type}_{self.category}{"_5" if type == "reviews" else ""}.json.gz'
        base_name = os.path.basename(url)
        local_filepath = os.path.join(path, base_name)
        if not os.path.exists(local_filepath):
            self.log(f'[DATASET] Downloading {type} from {url}...')
            download_file(url, local_filepath)
            self.log(f'[DATASET] Downloaded {type} to {local_filepath}')
        else:
            self.log(f'[DATASET] Raw data file already exists: {local_filepath}')
        return local_filepath

    def _parse_gz(self, path: str):
        # Parse a gzipped file and yield each line as a dict.
        
        g = gzip.open(path, 'r')
        for l in g:
            l = l.replace(b'true', b'True').replace(b'false', b'False')
            yield eval(l)

    def _load_reviews(self, path: str) -> list:
        # Load reviews from a given path.
        
        self.log(f'[DATASET] Loading reviews from {path}...')
        reviews = []
        for inter in self._parse_gz(path):
            user = inter['reviewerID']
            item = inter['asin']
            time = inter['unixReviewTime']
            reviews.append((user, item, int(time)))
        self.log(f'[DATASET] Loaded {len(reviews)} reviews')
        return reviews

    def _get_item_seqs(self, reviews: list[tuple]) -> dict:
        # Group the reviews by user and sort the items by time.
        
        self.log('[DATASET] Grouping reviews by user and sorting by time...')
        # Group reviews by user
        item_seqs = defaultdict(list)
        for data in reviews:
            user, item, time = data
            item_seqs[user].append((item, time))

        # Sort items by time
        for user, item_time in item_seqs.items():
            item_time.sort(key=lambda x: x[1])
            item_seqs[user] = [_[0] for _ in item_time]
        self.log(f'[DATASET] successfully built sequences for {len(item_seqs)} users')
        return item_seqs

    def _remap_ids(self, item_seqs: dict) -> tuple[dict, dict]:
        
        # Remaps the user and item IDs in the given item sequences dictionary.

        # Args:
        #     item_seqs (dict): A dictionary containing user-item sequences, where the keys are the users and the values are lists of items sorted by time.

        # Returns:
        #     all_item_seqs (dict): A dictionary containing the user-item sequences.
        #     id_mapping (dict): A dictionary containing the mapping between raw and remapped user and item IDs.
        #         - user2id (dict): A dictionary mapping raw user IDs to remapped user IDs.
        #         - item2id (dict): A dictionary mapping raw item IDs to remapped item IDs.
        #         - id2user (list): A list mapping remapped user IDs to raw user IDs.
        #         - id2item (list): A list mapping remapped item IDs to raw item IDs.

        # Note:
        #     The remapped user and item IDs start from 1. The ID 0 is reserved for padding `[PAD]`.
        
        self.log('[DATASET] Remapping user and item IDs...')
        for user, items in item_seqs.items():
            if user not in self.id_mapping['user2id']:
                self.id_mapping['user2id'][user] = len(self.id_mapping['id2user'])
                self.id_mapping['id2user'].append(user)
            iids = []           # item id lists
            for item in items:
                if item not in self.id_mapping['item2id']:
                    self.id_mapping['item2id'][item] = len(self.id_mapping['id2item'])
                    self.id_mapping['id2item'].append(item)
                iids.append(item)
            self.all_item_seqs[user] = iids
        n_items = len(self.id_mapping['id2item']) - 1  # exclude padding
        self.log(
            f'[DATASET] Remapping complete: {len(self.all_item_seqs)} users, '
            f'{n_items} items'
        )
        return self.all_item_seqs, self.id_mapping

    def _process_reviews(self,
        input_path: str,
        output_path: str
    ) -> tuple[dict, dict]:
        
        # Process the reviews from the input path and save the data to the output path.
        # Returns:
        #     all_item_seqs (dict): A dictionary containing the user-item sequences.
        #     id_mapping (dict): A dictionary containing data maps.
        
        # Check if the processed data already exists
        seq_file = os.path.join(output_path, 'all_item_seqs.json')
        id_mapping_file = os.path.join(output_path, 'id_mapping.json')
        if os.path.exists(seq_file) and os.path.exists(id_mapping_file):
            self.log('[DATASET] Reviews have been processed...')
            self.log(f'[DATASET] Loading processed reviews from {seq_file}...')
            self.log(f'[DATASET] Loading processed id mapping from {id_mapping_file}...')
            with open(seq_file, 'r') as f:
                all_item_seqs = json.load(f)
            with open(id_mapping_file, 'r') as f:
                id_mapping = json.load(f)
            n_items = len(id_mapping['id2item']) - 1  # exclude padding
            self.log(
                f'[DATASET] Loaded {len(all_item_seqs)} user sequences, '
                f'{n_items} items'
            )
            return all_item_seqs, id_mapping

        self.log('[DATASET] Processing reviews...')

        # Load reviews
        reviews = self._load_reviews(input_path)
        self.log(f'[DATASET] >> GET ITEM SEQUENCES...')
        item_seqs = self._get_item_seqs(reviews)
        self.log(f'[DATASET] >> REMAP IDS...')
        all_item_seqs, id_mapping = self._remap_ids(item_seqs)

        n_items = len(id_mapping['id2item']) - 1  # exclude padding
        self.log(
            f'[DATASET] Review processing complete: {len(all_item_seqs)} users, '
            f'{n_items} items'
        )

        # Save data
        self.log(f'[DATASET] Saving item sequences to {seq_file}...')
        with open(seq_file, 'w') as f:
            json.dump(all_item_seqs, f)
        self.log(f'[DATASET] Saving id mapping to {id_mapping_file}...')
        with open(id_mapping_file, 'w') as f:
            json.dump(id_mapping, f)
        return all_item_seqs, id_mapping

    def _load_metadata(
        self,
        path: str,
        item2id: dict
    ) -> dict:
        # Load metadata from a given path and filter it based on the provided data maps.
        self.log('[DATASET] Loading metadata...')
        data = {}
        item_asins = set(item2id.keys())
        for info in tqdm(self._parse_gz(path)):
            if info['asin'] not in item_asins:
                continue
            data[info['asin']] = info
        self.log(f'[DATASET] Loaded metadata for {len(data)} items')
        return data

    def _sent_process(self, raw: str) -> str:
        # Process the raw input according to the raw data type and return a processed sentence.
        sentence = ""
        if isinstance(raw, float):
            sentence += str(raw)
            sentence += '.'
        elif len(raw) > 0 and isinstance(raw[0], list):
            for v1 in raw:
                for v in v1:
                    sentence += clean_text(v)[:-1]
                    sentence += ', '
            sentence = sentence[:-2]
            sentence += '.'
        elif isinstance(raw, list):
            for v1 in raw:
                sentence += clean_text(v1)
        else:
            sentence = clean_text(raw)
        return sentence + ' '

    def _extract_meta_sentences(
        self,
        metadata: dict
    ) -> dict:
        # Extracts meta sentences from the given metadata dictionary.
        self.log('[DATASET] Extracting meta sentences...')
        item2meta = {}
        for item, meta in tqdm(metadata.items()):
            meta_sentence = ''
            keys = set(meta.keys())
            features_needed = [
                'title', 'price', 'brand',
                'feature', 'categories', 'description'
            ]
            for feature in features_needed:
                if feature in keys:
                    meta_sentence += self._sent_process(meta[feature])
            item2meta[item] = meta_sentence
        self.log(f'[DATASET] Extracted meta sentences for {len(item2meta)} items')
        return item2meta

    def _process_meta(
        self,
        input_path: str,
        output_path: str
    ) -> Optional[dict]:
        #  Process metadata based on the specified process type.

        process_mode = self.config['metadata']
        meta_file = os.path.join(output_path, f'metadata.{process_mode}.json')
        if os.path.exists(meta_file):
            self.log('[DATASET] Metadata has been processed...')
            self.log(f'[DATASET] Loading processed metadata from {meta_file}...')
            with open(meta_file, 'r') as f:
                item2meta = json.load(f)
            self.log(f'[DATASET] Loaded metadata for {len(item2meta)} items')
            return item2meta

        self.log(f'[DATASET] Processing metadata, mode: {process_mode}')

        if process_mode == 'none':
            self.log('[DATASET] Metadata processing disabled (mode: none)')
            return None

        item2meta = self._load_metadata(
            path=input_path,
            item2id=self.item2id
        )
        if process_mode == 'raw':
            pass
        if process_mode == 'sentence':
            # Extract sentences from metadata
            item2meta = self._extract_meta_sentences(metadata=item2meta)
        else:
            raise NotImplementedError('Metadata processing type not implemented.')

        self.log(f'[DATASET] Saving metadata to {meta_file}...')
        with open(meta_file, 'w') as f:
            json.dump(item2meta, f)
        self.log(f'[DATASET] Metadata processing complete ({len(item2meta)} items)')
        return item2meta

    def _download_and_process_raw(self):
        self.log('[DATASET] ========== DOWNLOADING RAW DATA FILES... ==========')
        raw_data_path = os.path.join(self.cache_dir, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        with self.accelerator.main_process_first(): # only download once when ddp
            self.log('[DATASET] Attempting to download reviews...')
            reviews_localpath = self._download_raw(
                path=raw_data_path,
                type='reviews'
            )
            self.log('[DATASET] Attempting to download metadata...')
            meta_localpath = self._download_raw(
                path=raw_data_path,
                type='meta'
            )

        # Following https://github.com/RUCAIBox/CIKM2020-S3Rec/blob/master/data/data_process.py
        np.random.seed(12345)

        # Process raw data
        self.log('[DATASET] ============= PROCESSING RAW DATA... ==============')
        processed_data_path = os.path.join(self.cache_dir, 'processed')
        os.makedirs(processed_data_path, exist_ok=True)

        self.log('[DATASET] ========== PROCESS REVIEWS... ==========')
        self.all_item_seqs, self.id_mapping = self._process_reviews(
            input_path=reviews_localpath,
            output_path=processed_data_path
        )

        self.log('[DATASET] ========== PROCESS METADATA... ==========')
        self.item2meta = self._process_meta(
            input_path=meta_localpath,
            output_path=processed_data_path
        )
