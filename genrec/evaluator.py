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

    def calculate_pos_index(self, preds, labels):
        preds = preds.detach().cpu()
        labels = labels.detach().cpu()
        assert preds.shape[1] == self.maxk, f"preds.shape[1] = {preds.shape[1]} != {self.maxk}"

        # ==== DEBUG ====
        print(f"preds: {preds}")
        print(f"Shape of preds: {preds.shape}")
        print(f"labels: {labels}")
        print(f"Shape of labels: {labels.shape}")
        
        
        pos_index = torch.zeros((preds.shape[0], self.maxk), dtype=torch.bool)
        for i in range(preds.shape[0]):
            cur_label = labels[i].tolist()
            
            print(f"Example {i}:")
            print(f"  Original label: {cur_label}")
            
            if self.eos_token in cur_label:
                eos_pos = cur_label.index(self.eos_token)
                cur_label = cur_label[:eos_pos]
                print(f"Found EOS token at position {eos_pos}, truncated label: {cur_label}")
                
            for j in range(self.maxk):
                cur_pred = preds[i, j].tolist()
                print(f"  Predicted item at rank {j+1}: {cur_pred}")

                if cur_pred == cur_label:
                    pos_index[i, j] = True
                    break
        return pos_index

    def recall_at_k(self, pos_index, k):
        print(f"Calculating Recall@{k} with pos_index:\n{pos_index}")
        value = pos_index[:, :k].sum(dim=1).cpu().float()
        print(f"Recall@{k} values for each example: {value}")
        
        return value

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
        
        print("RESULTS:")
        for key, value in results.items():
            print(f"  {key}: {value}")  
        return results
