from transformers import BartTokenizer
from src.model.utterance_rewriter_model import UtteranceRewriterModel, BartConfig


tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
ARTICLE_TO_SUMMARIZE = "My friends are cool but they eat too many carbs."
inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=15, return_tensors='pt')
config_bart = BartConfig.from_pretrained('../../data/pretrained_model/roberta/config.json')
model = UtteranceRewriterModel(config=config_bart)
model(input_ids=inputs['input_ids'])
pass