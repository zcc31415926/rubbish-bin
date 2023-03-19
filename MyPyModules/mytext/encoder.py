import numpy as np
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class TextEncoder(nn.Module):
    def __init__(self, bert_config=None, output_dim=384):
        super().__init__()
        bert_dict = {
            '384': ['microsoft/Multilingual-MiniLM-L12-H384', 'microsoft/MiniLM-L12-H384-uncased'],
        }
        bert_config = bert_dict[str(output_dim)][0] if bert_config is None else bert_config
        self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_config)
        self.bert_model = AutoModel.from_pretrained(bert_config)

    def forward(self, text):
        inputs = self.bert_tokenizer(text, max_length=512, truncation=True, return_tensors='pt')
        outputs = self.bert_model(**inputs)
        return outputs['pooler_output'][0].detach().cpu().numpy().astype(np.float32)

