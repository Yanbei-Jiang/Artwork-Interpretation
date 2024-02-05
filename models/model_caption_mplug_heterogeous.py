from functools import partial
from models.vit import VisionTransformer
from models.modeling_mplug import BertConfig, BertModel, BertPrefixModel, FusionModel, BertLMHeadModel
from models.visual_transformers import initialize_clip
from models.predictor import TextGenerator
from torch_geometric.nn import HANConv
import torch
from torch import nn
import torch.nn.functional as F
import re
import numpy as np
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
        x_dict['artwork'] = self.artwork_transform(x_dict['artwork'])
        x_dict['author'] = self.author_transform(x_dict['author'])
        x_dict['school'] = self.school_transform(x_dict['school'])
        x_dict['artwork_type'] = self.type_transform(x_dict['artwork_type'])
        x_dict['timeframe'] = self.timeframe_transform(x_dict['timeframe'])
        x_dict['technique'] = self.technique_transform(x_dict['technique'])
        x_dict['title_cluster'] = self.title_cluster_transform(x_dict['title_cluster'])
        x_dict['ngram'] = self.ngram_transform(x_dict['ngram'])

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
        # loss = loss.mean(dim=1)
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
        # self.init_distill(config)
        self.beam_generator = TextGenerator(config, self.text_decoder) 
        self.graph = graph.cuda()
        self.mappings = mappings
        self.han_model = HAN(768, 768, graph)
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        self.loss_module = MetadataImageLoss(11*768, 768, 577)

        self.fasttext = FastText.load_fasttext_format("cc.en.300.bin")
        
    def forward(self, image, question, answer=None, metadata_words=None, alpha=0, k=None, weights=None, train=True):
        image = image.to(dtype=next(self.parameters()).dtype) 
        image_embeds = self.visual_encoder.visual(image, skip_last_layer=True, use_checkpoint=self.use_checkpoint)
        if self.large:
            image_embeds = self.dropout(self.visn_layer_norm(self.visn_fc(image_embeds)))
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        
        if train:               
            '''
            k: number of answers for each question
            weights: weight for each answer
            '''          
            answer_targets = answer.input_ids.masked_fill(answer.input_ids == self.tokenizer.pad_token_id, -100)

            target_values = torch.tensor([4,5,6,7,8,9]).to(image.device)
            indices = torch.isin(question.input_ids, target_values).to(image.device)
            indices = torch.nonzero(indices, as_tuple=False)[:, 1].to(image.device)
            indices = indices.view(question.input_ids.size(0), target_values.size(0)).to(image.device)

            text_output = self.text_encoder(question.input_ids, attention_mask=question.attention_mask, return_dict=True)
            text_embeds = text_output.last_hidden_state

            text_embeds = text_embeds[torch.arange(text_embeds.size(0)).unsqueeze(1), indices, :].to(image.device)
            new_attention_mask = question.attention_mask[torch.arange(question.attention_mask.size(0)).unsqueeze(1), indices].to(image.device)
            
            graph_embeds = self.han_model(self.graph.x_dict, self.graph.edge_index_dict)

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
            extracted_embeddings_tensor = torch.stack(extracted_embeddings_tensor).to(image.device)
            fusion_embeddings = torch.cat([text_embeds, extracted_embeddings_tensor], 1)

            fusion_attention_mask = torch.ones(fusion_embeddings.size()[:-1], dtype=torch.long).to(image.device)
            
            fusion_output = self.fusion_encoder(encoder_embeds=fusion_embeddings, 
                                                attention_mask = fusion_attention_mask, 
                                                encoder_hidden_states = image_embeds,
                                                encoder_attention_mask = image_atts, return_dict=False)
            
            image_output, question_output = fusion_output
            
            question_output = torch.cat([image_output, question_output], 1)
            merge_text_attention = torch.cat([image_atts, fusion_attention_mask], 1)

            answer_output = self.text_decoder(answer.input_ids,
                                              attention_mask = answer.attention_mask,
                                              encoder_hidden_states = question_output,
                                              encoder_attention_mask = merge_text_attention,
                                              labels = answer_targets,
                                              return_dict = True,
                                              reduction = 'none',
                                             )
            loss = answer_output.loss

            metadata_image_loss = self.loss_module(extracted_embeddings_tensor[:, 1:, :], image_embeds)
            combined_loss = 0.8*loss + (1-0.8)*metadata_image_loss
            return loss
            

        else: 
            target_values = torch.tensor([4,5,6,7,8,9]).to(image.device)
            indices = torch.isin(question.input_ids, target_values).to(image.device)
            indices = torch.nonzero(indices, as_tuple=False)[:, 1].to(image.device)
            indices = indices.view(question.input_ids.size(0), target_values.size(0)).to(image.device)

            text_output = self.text_encoder(question.input_ids, attention_mask=question.attention_mask,
                                                return_dict=True)
            text_embeds = text_output.last_hidden_state
            text_embeds = text_embeds[torch.arange(text_embeds.size(0)).unsqueeze(1), indices, :].to(image.device)
            new_attention_mask = question.attention_mask[torch.arange(question.attention_mask.size(0)).unsqueeze(1), indices].to(image.device)
            
            graph_embeds = self.han_model(self.graph.x_dict, self.graph.edge_index_dict)

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
            extracted_embeddings_tensor = torch.stack(extracted_embeddings_tensor)
            fusion_embeddings = torch.cat([text_embeds, extracted_embeddings_tensor], 1)

            fusion_attention_mask = torch.ones(fusion_embeddings.size()[:-1], dtype=torch.long).to(image.device)
            
            fusion_output = self.fusion_encoder(encoder_embeds=fusion_embeddings, 
                                                attention_mask = fusion_attention_mask, 
                                                encoder_hidden_states = image_embeds,
                                                encoder_attention_mask = image_atts,                             
                                                return_dict = False) 
            image_output, question_output = fusion_output 
            question_output = torch.cat([image_output, question_output], 1)
            merge_text_attention = torch.cat([image_atts, fusion_attention_mask], 1)
            topk_ids, topk_probs = self.generation(question_output, merge_text_attention) 
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
    # def init_distill(self, config):
    #     self.distill = config['distill']
    #     if self.distill:
    #         self.visual_encoder_m, _ = initialize_clip(config)
    #         self.text_encoder_m = BertModel.from_pretrained(config['text_encoder'], config=self.config_encoder, add_pooling_layer=False)
    #         self.fusion_encoder_m = FusionModel.from_pretrained(config['text_encoder'], config=self.config_fusion, add_pooling_layer=False)
    #         self.text_decoder_m = BertLMHeadModel.from_pretrained(config['text_decoder'], config=self.config_decoder)
    #         self.model_pairs = [[self.visual_encoder,self.visual_encoder_m],
    #                             [self.text_encoder,self.text_encoder_m],
    #                             [self.text_decoder,self.text_decoder_m],
    #                            ]
    #         if self.config_encoder.hidden_size != config['vision_width']:
    #             self.visn_fc_m = nn.Linear(config['vision_width'], self.config_encoder.hidden_size)
    #             self.visn_layer_norm_m = nn.LayerNorm(self.config_encoder.hidden_size, eps=1e-12)
    #             self.dropout_m = nn.Dropout(self.config_encoder.hidden_dropout_prob)
    #             self.model_pairs.extend([[self.visn_fc, self.visn_fc_m], [self.visn_layer_norm, self.visn_layer_norm_m]])
    #         self.copy_params()
    #         self.momentum = 0.995

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
                
    def generation(self, question_states, question_atts):
        encoder_inputs = [question_states, question_atts]
        topk_ids, topk_scores = self.beam_generator.translate_batch(encoder_inputs)  
        return topk_ids, topk_scores

    def rank_answer(self, question_states, question_atts, answer_ids, answer_atts, k):
        
        num_ques = question_states.size(0)
        start_ids = answer_ids[0,0].repeat(num_ques,1) # bos token
        
        start_output = self.text_decoder(start_ids, 
                                         encoder_hidden_states = question_states,
                                         encoder_attention_mask = question_atts,                                      
                                         return_dict = True,
                                         reduction = 'none')              
        logits = start_output.logits[:,0,:] # first token's logit
        
        # topk_probs: top-k probability 
        # topk_ids: [num_question, k]        
        answer_first_token = answer_ids[:,1]
        prob_first_token = F.softmax(logits,dim=1).index_select(dim=1, index=answer_first_token) 
        topk_probs, topk_ids = prob_first_token.topk(k,dim=1) 
        
        # answer input: [num_question*k, answer_len]                 
        input_ids = []
        input_atts = []
        for b, topk_id in enumerate(topk_ids):
            input_ids.append(answer_ids.index_select(dim=0, index=topk_id))
            input_atts.append(answer_atts.index_select(dim=0, index=topk_id))
        input_ids = torch.cat(input_ids,dim=0)  
        input_atts = torch.cat(input_atts,dim=0)  

        targets_ids = input_ids.masked_fill(input_ids == self.tokenizer.pad_token_id, -100)

        # repeat encoder's output for top-k answers
        question_states = tile(question_states, 0, k)
        question_atts = tile(question_atts, 0, k)
        
        output = self.text_decoder(input_ids, 
                                   attention_mask = input_atts, 
                                   encoder_hidden_states = question_states,
                                   encoder_attention_mask = question_atts,     
                                   labels = targets_ids,
                                   return_dict = True, 
                                   reduction = 'none')                 

        answer_loss = output.loss 
        answer_loss = answer_loss.view(input_ids.size(0),-1)
        
        # topk_prob: first token probability
        topk_probs = topk_probs.view(-1,1)
        log_probs = torch.cat([topk_probs.log(), -answer_loss],dim=1)

        # re-calculate log probabilities for the answer sequences using chain rule
        log_probs_sum = log_probs.sum(1)
        log_probs_sum = log_probs_sum.view(num_ques,k)

        topk_probs = F.softmax(log_probs_sum, dim=-1)
        # get top-k after re-ranking
        topk_probs, rerank_id = topk_probs.topk(k,dim=1) 
        topk_ids = torch.gather(topk_ids, 1, rerank_id)    

        return topk_ids, topk_probs
    
def tile(x, dim, n_tile):
    init_dim = x.size(dim)
    repeat_idx = [1] * x.dim()
    repeat_idx[dim] = n_tile
    x = x.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(x, dim, order_index.to(x.device))    