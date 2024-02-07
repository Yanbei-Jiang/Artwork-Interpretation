from functools import partial
from models.vit import VisionTransformer
from models.modeling_mplug import BertConfig, BertModel, BertPrefixModel, FusionModel
from models.visual_transformers import initialize_clip
from models.predictor import TextGenerator

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

class MPLUG(nn.Module):
    def __init__(self,                 
                 tokenizer = None,
                 config = None,     
                 ):
        super().__init__()
        
        self.tokenizer = tokenizer 
        self.module_setting(config)
        self.visual_encoder, _ = initialize_clip(config)
        self.text_encoder = BertModel.from_pretrained(config['text_encoder'], config=self.config_encoder, add_pooling_layer=False)  
        self.fusion_encoder = FusionModel.from_pretrained(config['text_encoder'], config=self.config_fusion, add_pooling_layer=False)  
        self.text_decoder = BertPrefixModel.from_pretrained(config['text_decoder'], config=self.config_decoder)    
        # self.init_distill(config)
        self.beam_generator = TextGenerator(config, self.text_decoder) 
            
        
    def forward(self, image, metadata, caption=None, alpha=0, k=None, weights=None, train=True):
        image = image.to(dtype=next(self.parameters()).dtype) 
        image_embeds = self.visual_encoder.visual(image, skip_last_layer=True, use_checkpoint=self.use_checkpoint)
        if self.large:
            image_embeds = self.dropout(self.visn_layer_norm(self.visn_fc(image_embeds)))
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        
        if train:               
            caption_targets = caption.input_ids.masked_fill(caption.input_ids == self.tokenizer.pad_token_id, -100)

            target_values = torch.tensor([4,5,6,7,8,9]).to(image.device)
            indices = torch.isin(metadata.input_ids, target_values).to(image.device)
            indices = torch.nonzero(indices, as_tuple=False)[:, 1].to(image.device)
            indices = indices.view(metadata.input_ids.size(0), target_values.size(0)).to(image.device)

            text_output = self.text_encoder(metadata.input_ids, attention_mask=metadata.attention_mask, return_dict=True)
            text_embeds = text_output.last_hidden_state

            text_embeds = text_embeds[torch.arange(text_embeds.size(0)).unsqueeze(1), indices, :].to(image.device)
            new_attention_mask = metadata.attention_mask[torch.arange(metadata.attention_mask.size(0)).unsqueeze(1), indices].to(image.device)
            fusion_output = self.fusion_encoder(encoder_embeds=text_embeds, 
                                                attention_mask = new_attention_mask, 
                                                encoder_hidden_states = image_embeds,
                                                encoder_attention_mask = image_atts, return_dict=False)
            
            image_output, metadata_output = fusion_output
            
            metadata_output = torch.cat([image_output, metadata_output], 1)
            merge_text_attention = torch.cat([image_atts, new_attention_mask], 1)

            caption_output = self.text_decoder(caption.input_ids,
                                              attention_mask = caption.attention_mask,
                                              encoder_hidden_states = metadata_output,
                                              encoder_attention_mask = merge_text_attention,
                                              labels = caption_targets,
                                              return_dict = True,
                                              reduction = 'none',
                                             )
            loss = caption_output.loss

            return loss
            

        else: 
            target_values = torch.tensor([4,5,6,7,8,9]).to(image.device)
            indices = torch.isin(metadata.input_ids, target_values).to(image.device)
            indices = torch.nonzero(indices, as_tuple=False)[:, 1].to(image.device)
            indices = indices.view(metadata.input_ids.size(0), target_values.size(0)).to(image.device)

            text_output = self.text_encoder(metadata.input_ids, attention_mask=metadata.attention_mask,
                                                return_dict=True)
            text_embeds = text_output.last_hidden_state
            text_embeds = text_embeds[torch.arange(text_embeds.size(0)).unsqueeze(1), indices, :].to(image.device)
            new_attention_mask = metadata.attention_mask[torch.arange(metadata.attention_mask.size(0)).unsqueeze(1), indices].to(image.device)
            fusion_output = self.fusion_encoder(encoder_embeds=text_embeds, 
                                                attention_mask = new_attention_mask, 
                                                encoder_hidden_states = image_embeds,
                                                encoder_attention_mask = image_atts,                             
                                                return_dict = False) 
            image_output, metadata_output = fusion_output 
            metadata_output = torch.cat([image_output, metadata_output], 1)
            merge_text_attention = torch.cat([image_atts, new_attention_mask], 1)
            topk_ids, topk_probs = self.generation(metadata_output, merge_text_attention) 
            return topk_ids, topk_probs
 

    def module_setting(self, config):
        self.config_encoder = BertConfig.from_json_file(config['bert_config'])   
        self.config_encoder.num_hidden_layers = self.config_encoder.text_encoder_layers
        self.config_fusion = BertConfig.from_json_file(config['bert_config'])   
        self.config_decoder = BertConfig.from_json_file(config['bert_config'])
        self.config_decoder.add_cross_attention = True
        self.config_decoder.num_hidden_layers = self.config_decoder.text_decode_layers
        self.large = False
        if self.config_encoder.hidden_size != config['vision_width']:
            self.visn_fc = nn.Linear(config['vision_width'], self.config_encoder.hidden_size)
            self.visn_layer_norm = nn.LayerNorm(self.config_encoder.hidden_size, eps=1e-12)
            self.dropout = nn.Dropout(self.config_encoder.hidden_dropout_prob)
            self.large = True
        self.use_checkpoint = config["use_checkpoint"] if "use_checkpoint" in config else True

    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
                
    def generation(self, states, atts):
        encoder_inputs = [states, atts]
        topk_ids, topk_scores = self.beam_generator.translate_batch(encoder_inputs)  
        return topk_ids, topk_scores

