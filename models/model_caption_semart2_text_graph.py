from functools import partial
from models.vit import VisionTransformer
from models.modeling_mplug import BertConfig, BertModel, BertPrefixModel, FusionModel
from models.visual_transformers import initialize_clip
from models.predictor import TextGenerator

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
from torch_geometric.nn import HANConv
import re
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import FastText

class HAN(nn.Module):
    def __init__(self, hidden_channels, out_channels, graph):
        super(HAN, self).__init__()

        # Define initial transformation layers for each node type
        self.artwork_transform = nn.Linear(graph['artwork'].x.shape[1], hidden_channels)
        self.author_transform = nn.Linear(graph['author'].x.shape[1], hidden_channels)
        self.school_transform = nn.Linear(graph['school'].x.shape[1], hidden_channels)
        self.type_transform = nn.Linear(graph['artwork_type'].x.shape[1], hidden_channels)
        self.timeframe_transform = nn.Linear(graph['timeframe'].x.shape[1], hidden_channels)
        self.technique_transform = nn.Linear(graph['technique'].x.shape[1], hidden_channels)
        self.title_cluster_transform = nn.Linear(graph['title_cluster'].x.shape[1], hidden_channels)
        self.ngram_transform = nn.Linear(graph['ngram'].x.shape[1], hidden_channels)

        # Define HAN convolutional layers
        self.conv1 = HANConv(hidden_channels, hidden_channels, graph.metadata(), heads=8)
        self.conv2 = HANConv(hidden_channels, out_channels, graph.metadata(), heads=4)
        self.graph = graph
        self.LayerNorm = nn.LayerNorm(hidden_channels, eps=1e-12)

    def forward(self, x_dict, edge_index_dict):
        # Transform node features to a common size
        x_dict['artwork'] = self.LayerNorm(self.artwork_transform(x_dict['artwork']))
        x_dict['author'] = self.LayerNorm(self.author_transform(x_dict['author']))
        x_dict['school'] = self.LayerNorm(self.school_transform(x_dict['school']))
        x_dict['artwork_type'] = self.LayerNorm(self.type_transform(x_dict['artwork_type']))
        x_dict['timeframe'] = self.LayerNorm(self.timeframe_transform(x_dict['timeframe']))
        x_dict['technique'] = self.LayerNorm(self.technique_transform(x_dict['technique']))
        x_dict['title_cluster'] = self.LayerNorm(self.title_cluster_transform(x_dict['title_cluster']))
        x_dict['ngram'] = self.LayerNorm(self.ngram_transform(x_dict['ngram']))
            
        identity1 = x_dict

        # Apply HAN convolutional layers
        x = self.conv1(x_dict, edge_index_dict)
        x = {key: self.LayerNorm(x[key]) + identity1[key] for key in x}  # Apply LayerNorm and add residual
        
        identity2 = x
        x = self.conv2(x, edge_index_dict)
        x = {key: self.LayerNorm(x[key]) + identity2[key] for key in x}  # Apply LayerNorm and add residual
        return x
    
class MetadataImageLoss(nn.Module):
    def __init__(self, metadata_emb_size, image_emb_size, kernel_size):
        super(MetadataImageLoss, self).__init__()
        # Linear layer to transform concatenated metadata embeddings
        self.linear = nn.Linear(metadata_emb_size, image_emb_size)
        self.max_pool = nn.MaxPool1d(kernel_size=kernel_size)
        self.cosine_similarity = nn.CosineSimilarity(dim=1)
        
    def forward(self, metadata_embeddings, image_embeddings):
        # Concatenate metadata embeddings
        concatenated_metadata = metadata_embeddings.view(metadata_embeddings.shape[0], -1)  # Concatenate along the feature dimension
        # Transform to match image embedding size
        transformed_metadata = self.linear(concatenated_metadata)
        image_embeddings = image_embeddings.transpose(1, 2)
        pooled_embeddings = self.max_pool(image_embeddings).squeeze(2)
        # Compute the loss (e.g., Mean Squared Error)
        # loss = F.mse_loss(transformed_metadata, pooled_embeddings)
        loss = 1 - self.cosine_similarity(transformed_metadata, pooled_embeddings)
        # loss = loss.mean(dim=1)
        return loss.mean()

class MPLUG(nn.Module):
    def __init__(self,    
                 graph,
                 mappings,             
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
        self.beam_generator = TextGenerator(config, self.text_decoder) 
            
        self.graph = graph.cuda()
        self.mappings = mappings
        self.han_model = HAN(768, 768, graph)
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        self.loss_module = MetadataImageLoss(11*768, 768, 577)

        self.fasttext = FastText.load_fasttext_format("cc.en.300.bin")
        
    def forward(self, image, metadata, caption=None, metadata_words=None, train=True, out_size=5, scst=False):
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

            extracted_embeddings_tensor = self.graph_encoder(self.graph, self.metadata_words, train=True)
            
            fusion_embeddings = torch.cat([text_embeds, extracted_embeddings_tensor], 1)

            fusion_attention_mask = torch.ones(fusion_embeddings.size()[:-1], dtype=torch.long).to(image.device)
            
            fusion_output = self.fusion_encoder(encoder_embeds = fusion_embeddings, 
                                                attention_mask = fusion_attention_mask, 
                                                encoder_hidden_states = image_embeds,
                                                encoder_attention_mask = image_atts, return_dict=False)
            
            image_output, metadata_output = fusion_output
            
            metadata_output = torch.cat([image_output, metadata_output], 1)
            merge_text_attention = torch.cat([image_atts, fusion_attention_mask], 1)
            caption_output = self.text_decoder(caption.input_ids, 
                                                  attention_mask = caption.attention_mask, 
                                                  encoder_hidden_states = metadata_output,
                                                  encoder_attention_mask = merge_text_attention,                  
                                                  labels = caption_targets,
                                                  return_dict = True,   
                                                  reduction = 'none',
                                                 )                      
            loss = caption_output.loss         
            metadata_image_loss = self.loss_module(extracted_embeddings_tensor[:, 1:, :], image_embeds)
            combined_loss = 0.8*loss + (1-0.8)*metadata_image_loss
            return combined_loss
            

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
            
            # batch_size*12*768
            extracted_embeddings_tensor = self.graph_encoder(self.graph, self.metadata_words, train=False)

            fusion_embeddings = torch.cat([text_embeds, extracted_embeddings_tensor], 1)

            fusion_attention_mask = torch.ones(fusion_embeddings.size()[:-1], dtype=torch.long).to(image.device)
            
            fusion_output = self.fusion_encoder(encoder_embeds=fusion_embeddings, 
                                                attention_mask = fusion_attention_mask, 
                                                encoder_hidden_states = image_embeds,
                                                encoder_attention_mask = image_atts,                             
                                                return_dict = False) 
            image_output, metadata_output = fusion_output 
            metadata_output = torch.cat([image_output, metadata_output], 1)
            merge_text_attention = torch.cat([image_atts, fusion_attention_mask], 1)

            topk_ids_content, topk_probs_content, topk_ids_form, topk_probs_form, topk_ids_context, topk_probs_context = self.generation(metadata_output, merge_text_attention) 
            return topk_ids_content, topk_probs_content, topk_ids_form, topk_probs_form, topk_ids_context, topk_probs_context

    def graph_encoder(graph, metadata_words, train):
        if train:
            graph_embeds = self.han_model(graph.x_dict, graph.edge_index_dict)

            extracted_embeddings_tensor = []
            for metadata_word in metadata_words:
                extracted_embeddings = []
                for i, (metadata_name, metadata) in enumerate(metadata_word.items()):
                    metadata_indices = dict(self.mappings[i])
                    if metadata_name == "title":
                        title_embeddding = torch.tensor(self.sentence_transformer.encode(metadata)).to(image.device)
                        title_embeddding = self.han_model.title_cluster_transform(title_embeddding).to(image.device)
                        input_embedding_norm = F.normalize(title_embeddding.unsqueeze(0), p=2, dim=1)
                        embeddings_list_norm = F.normalize(torch.tensor(graph_embeds["title_cluster"]).to(image.device), p=2, dim=1)

                        # Compute cosine similarity
                        cosine_similarities = torch.mm(input_embedding_norm, embeddings_list_norm.t()).squeeze(0)

                        # Find the index of the most similar embedding
                        closest_idx = torch.argmax(cosine_similarities).item()
                        extracted_embeddings.append(graph_embeds["title_cluster"][closest_idx])
                        
                        vectorizer = CountVectorizer(ngram_range=(1, 3), stop_words='english')
                        try: 
                            vec = vectorizer.fit([metadata])
                            title_ngrams = set(vec.vocabulary_.keys())
                        except ValueError:
                            title_ngrams = []
                            
                        ngrams_index = [index for ngram, index in metadata_indices.items() if ngram in title_ngrams]
                        max_ngrams = 5
                        i = 0
                        for index in ngrams_index:
                            if i < 5:
                                extracted_embeddings.append(graph_embeds["ngram"][index])
                                i += 1
                        if i < 5:
                            for j in range(max_ngrams-i):
                                extracted_embeddings.append(torch.zeros(768))
                    elif metadata_name == "technique":
                        metadata = metadata.split(",")[0].split(".")[0]
                        pattern = r'\d+\s*x\s*\d+.*'
                        processed_metadata = re.sub(pattern, '', metadata)
                        for metadata_type, index in metadata_indices.items():
                            if processed_metadata in metadata_type:
                                extracted_embeddings.append(graph_embeds[metadata_name][index])
                                break
                    else:
                        for metadata_type, index in metadata_indices.items():
                            if metadata == metadata_type:
                                extracted_embeddings.append(graph_embeds[metadata_name][index])
                                break
                            
                extracted_embeddings = [emb.to(image.device) for emb in extracted_embeddings]
                    
                extracted_embeddings_tensor.append(torch.stack(extracted_embeddings))

            # batch_size*12*768
            return torch.stack(extracted_embeddings_tensor).to(image.device)
        else:
            
            graph_embeds = self.han_model(graph.x_dict, graph.edge_index_dict)
            extracted_embeddings_tensor = []
            for metadata_word in metadata_words:
                extracted_embeddings = []
                for i, (metadata_name, metadata) in enumerate(metadata_word.items()):
                    metadata_indices = dict(self.mappings[i])
                    if metadata_name == "artwork":
                        continue
                    elif metadata_name == "title":
                        title_embeddding = torch.tensor(self.sentence_transformer.encode(metadata)).to(image.device)
                        title_embeddding = self.han_model.title_cluster_transform(title_embeddding).to(image.device)
                        input_embedding_norm = F.normalize(title_embeddding.unsqueeze(0), p=2, dim=1)
                        embeddings_list_norm = F.normalize(torch.tensor(graph_embeds["title_cluster"]).to(image.device), p=2, dim=1)

                        # Compute cosine similarity
                        cosine_similarities = torch.mm(input_embedding_norm, embeddings_list_norm.t()).squeeze(0)

                        # Find the index of the most similar embedding
                        closest_idx = torch.argmax(cosine_similarities).item()
                        extracted_embeddings.append(graph_embeds["title_cluster"][closest_idx])

                        vectorizer = CountVectorizer(ngram_range=(1, 3), stop_words='english')
                        try: 
                            vec = vectorizer.fit([metadata])
                            title_ngrams = set(vec.vocabulary_.keys())
                        except ValueError:
                            title_ngrams = []
                            
                        ngrams_index = [index for ngram, index in metadata_indices.items() if ngram in title_ngrams]
                        max_ngrams = 5
                        i = 0
                        for index in ngrams_index:
                            if i < 5:
                                extracted_embeddings.append(graph_embeds["ngram"][index])
                                i += 1
                        if i < 5:
                            for j in range(max_ngrams-i):
                                extracted_embeddings.append(torch.zeros(768))
                    elif metadata_name == "technique":
                        metadata = metadata.split(",")[0].split(".")[0]
                        pattern = r'\d+\s*x\s*\d+.*'
                        processed_metadata = re.sub(pattern, '', metadata)
                        flag = 0
                        for metadata_type, index in metadata_indices.items():
                            if processed_metadata in metadata_type:
                                extracted_embeddings.append(graph_embeds[metadata_name][index])
                                flag = 1
                                break
                        if flag == 0:
                            transformed_embedding = torch.tensor(self.fasttext.wv[processed_metadata]).to(image.device)
                            extracted_embeddings.append(self.han_model.technique_transform(transformed_embedding))
                    else:
                        flag = 0
                        for metadata_type, index in metadata_indices.items():
                            if metadata == metadata_type:
                                extracted_embeddings.append(graph_embeds[metadata_name][index])
                                flag = 1
                                break
                        if flag == 0:
                            if metadata_name == "author":
                                transformed_embedding = torch.tensor(self.fasttext.wv[metadata]).to(image.device)
                                extracted_embeddings.append(self.han_model.author_transform(transformed_embedding))
                            elif metadata_name == "artwork_type":
                                extracted_embeddings.append(torch.zeros(768).to(image.device))
                            elif metadata_name == "school":
                                extracted_embeddings.append(torch.zeros(768).to(image.device))
                            elif metadata_name == "timeframe":
                                extracted_embeddings.append(torch.zeros(768).to(image.device))
                                
                extracted_embeddings = [emb.to(image.device) for emb in extracted_embeddings]
                
                extracted_embeddings_tensor.append(torch.stack(extracted_embeddings))

            # batch_size*12*768
            return torch.stack(extracted_embeddings_tensor).to(image.device)


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
        print("use_checkpoint: ", self.use_checkpoint)

    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
                
    def generation(self, states, atts, out_size=1):
        encoder_inputs = [states, atts]
        topk_ids_content, topk_probs_content = self.beam_generator.translate_batch_scst(encoder_inputs,out_size=out_size, new_semart=True, type="content")
        topk_ids_form, topk_probs_form = self.beam_generator.translate_batch_scst(encoder_inputs,out_size=out_size, new_semart=True, type="form") 
        topk_ids_context, topk_probs_context = self.beam_generator.translate_batch_scst(encoder_inputs,out_size=out_size, new_semart=True, type="context")   
        return topk_ids_content, topk_probs_content, topk_ids_form, topk_probs_form, topk_ids_context, topk_probs_context

