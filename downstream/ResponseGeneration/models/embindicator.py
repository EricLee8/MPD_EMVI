import torch
import torch.nn as nn
from transformers import EncoderDecoderModel
from utils.config import *


class GenerationModel(nn.Module):
    def __init__(self, encoder, decoder, decoder_start_token_id, pad_token_id, eos_token_id):
        super().__init__()
        self.model = EncoderDecoderModel(encoder=encoder, decoder=decoder)
        self.model.config.decoder_start_token_id = decoder_start_token_id
        self.model.config.decoder.decoder_start_token_id = decoder_start_token_id
        self.model.config.encoder.decoder_start_token_id = decoder_start_token_id
        self.model.config.bos_token_id = decoder_start_token_id
        self.model.config.decoder.bos_token_id = decoder_start_token_id
        self.model.config.encoder.bos_token_id = decoder_start_token_id
        self.model.config.pad_token_id = pad_token_id
        self.model.config.eos_token_id = eos_token_id
        self.model.config.decoder.eos_token_id = eos_token_id
        self.model.config.encoder.eos_token_id = eos_token_id
        self.indicator_embs = nn.Embedding(2, self.model.encoder.config.hidden_size)

    def forward(
        self,
        input_ids=None,
        indicator_ids=None,
        attention_mask=None,
        decoder_attention_mask=None,
        labels=None,
        training=True,
        output_attentions=None,
    ):
        embedding_table = self.model.encoder.get_input_embeddings()
        inputs_embeds = embedding_table(input_ids) + self.indicator_embs(indicator_ids)
        if training:
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                decoder_attention_mask=decoder_attention_mask,
                labels=labels,
                output_attentions=output_attentions
            )
            return outputs

        else: # inference
            summary_ids = self.model.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, num_beams=args.num_beams,\
                    max_length=50, min_length=5, no_repeat_ngram_size=5, early_stopping=True)
            return summary_ids
