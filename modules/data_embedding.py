from transformers import AutoTokenizer, AutoModel
import torch

class TextEmbedder:
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name)
        self.embedding_dim = self.model.embeddings.position_embeddings.embedding_dim
        self.max_seq_length = self.model.embeddings.position_embeddings.num_embeddings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

    def generate_embeddings(self, text_list, max_length=500):
        self.model.to(self.device)
        tokenized_inputs = self.tokenizer(text_list, padding=True, truncation=True, return_tensors='pt', max_length=max_length)
        tokenized_inputs = {key: tensor.to(self.device) for key, tensor in tokenized_inputs.items()}
        print("Tokenized inputs moved to device")
        with torch.no_grad():
            model_outputs = self.model(**tokenized_inputs)
        embeddings = self.apply_mean_pooling(model_outputs, tokenized_inputs['attention_mask'])
        normalized_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return normalized_embeddings.cpu().numpy()

    def apply_mean_pooling(self, model_outputs, attention_mask):
        token_embeddings = model_outputs.last_hidden_state.to(self.device)
        attention_mask = attention_mask.to(self.device)
        expanded_mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * expanded_mask, 1)
        sum_mask = torch.clamp(expanded_mask.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def calculate_similarity(self, embedding1, embedding2):
        return embedding1 @ embedding2
