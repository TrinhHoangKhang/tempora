import torch


class Evaluator:
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.metric2func = {
            'recall': self.recall_at_k,
            'ndcg': self.ndcg_at_k
        }

        self.eos_token = self.tokenizer.eos_token
        self.maxk = max(config['topk'])
        self.debug_flag = False
        
    def calculate_pos_index(self, preds, labels):
        preds = preds.detach().cpu()
        labels = labels.detach().cpu()
        
        # Extract true labels: labels shape is (B, max_seq_len) with -100 as padding
        # Find the non-padding label for each example
        label_mask = labels != -100
        true_labels = []
        for i in range(labels.shape[0]):
            valid_idx = torch.where(label_mask[i])[0]
            if len(valid_idx) > 0:
                # Take the last valid label (the actual target)
                true_labels.append(labels[i, valid_idx[-1]].item())
            else:
                # Fallback if no valid label found
                true_labels.append(-100)
        true_labels = torch.tensor(true_labels)
        
        # Debug
        if self.debug_flag:
            print(f"DEBUG FROM EVALUATOR.CALCULATE_POS_INDEX():")
            print(f"preds.shape: {preds.shape}, true_labels.shape: {true_labels.shape}")
            print(f"true_labels[0]: {true_labels[0].tolist()}")
            self.debug_flag = False
            
        
        # preds: (B, n_return_sequences=maxk)
        # true_labels: (B,) - single true label per example
        # pos_index: (B, maxk) boolean tensor indicating whether each prediction is correct
        
        assert preds.shape[1] == self.maxk, f"preds.shape[1] = {preds.shape[1]} != {self.maxk}"
        
        pos_index = torch.zeros((preds.shape[0], self.maxk), dtype=torch.bool)
        for i in range(preds.shape[0]):
            cur_label = true_labels[i].item()
                
            for j in range(self.maxk):
                cur_pred = preds[i, j].item()
                if cur_pred == cur_label:
                    pos_index[i, j] = True
                    break
                
        return pos_index

    def recall_at_k(self, pos_index, k):        
        return pos_index[:, :k].sum(dim=1).cpu().float()

    def ndcg_at_k(self, pos_index, k):
        # Assume only one ground truth item per example
        ranks = torch.arange(1, pos_index.shape[-1] + 1).to(pos_index.device)
        dcg = 1.0 / torch.log2(ranks + 1)
        dcg = torch.where(pos_index, dcg, 0)
        return dcg[:, :k].sum(dim=1).cpu().float()

    def calculate_metrics(self, preds, labels):
        if isinstance(preds, tuple):
            preds, n_visited_items = preds
        else:
            n_visited_items = torch.FloatTensor([len(self.tokenizer.item2tokens)] * preds.shape[0])
        results = {}
        pos_index = self.calculate_pos_index(preds, labels)
        for metric in self.config['metrics']:
            for k in self.config['topk']:
                results[f"{metric}@{k}"] = self.metric2func[metric](pos_index, k)
        results['n_visited_items'] = n_visited_items
        return results
