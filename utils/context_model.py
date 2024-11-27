import torch
from torch import nn
from transformers import RobertaModel, HubertModel, AutoModel, Data2VecAudioModel
from utils.cross_attn_encoder import CMELayer, BertConfig



# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# English text model + context
class roberta_en_context(nn.Module):
            
    def __init__(self):        
        super().__init__() 
        self.roberta_model = AutoModel.from_pretrained('roberta-large') # with context, we can improve using a larger model
        self.classifier = nn.Linear(1024*2, 1)    
   
    def forward(self, input_ids, attention_mask, context_input_ids, context_attention_mask):        
        raw_output = self.roberta_model(input_ids, attention_mask, return_dict=True)        
        input_pooler = raw_output["pooler_output"]    # Shape is [batch_size, 1024]

        context_output = self.roberta_model(context_input_ids, context_attention_mask, return_dict=True)
        context_pooler = context_output["pooler_output"]   # Shape is [batch_size, 1024]

        pooler = torch.cat((input_pooler, context_pooler), dim=1)
        output = self.classifier(pooler)                    # Shape is [batch_size, 1]
        return output
    

# English text+audio model + context
class rob_d2v_cc_context(nn.Module):            
    def __init__(self, config):        
        super().__init__()
        self.roberta_model = RobertaModel.from_pretrained('roberta-large')
        self.data2vec_model = Data2VecAudioModel.from_pretrained("facebook/data2vec-audio-base")
        
        self.T_output_layers = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(1024*2, 1)
           )           
        self.A_output_layers = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(768*2, 1)
          )
        self.fused_output_layers = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(1024*2+768*2, 1024*2),
            nn.ReLU(),
            nn.Linear(1024*2, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )
        self.device = torch.device("cuda:"+config.device if torch.cuda.is_available() else "cpu")
        
        
    def forward(self, text_inputs, text_mask, text_context_inputs, text_context_mask, audio_inputs, audio_mask, audio_context_inputs, audio_context_mask):
        # text feature extraction
        raw_output = self.roberta_model(text_inputs, text_mask, return_dict=True)        
        input_pooler = raw_output["pooler_output"]    # Shape is [batch_size, 1024]

        # text context feature extraction
        raw_output_context = self.roberta_model(text_context_inputs, text_context_mask, return_dict=True)
        context_pooler = raw_output_context["pooler_output"]    # Shape is [batch_size, 1024]

        # audio feature extraction
        audio_out = self.data2vec_model(audio_inputs, audio_mask, output_attentions=True)
        A_hidden_states = audio_out.last_hidden_state
        ## average over unmasked audio tokens
        A_features = []
        audio_mask_idx_new = []
        for batch in range(A_hidden_states.shape[0]):
            layer = 0
            while layer<12:
                try:
                    padding_idx = sum(audio_out.attentions[layer][batch][0][0]!=0)
                    audio_mask_idx_new.append(padding_idx)
                    break
                except:
                    layer += 1
            truncated_feature = torch.mean(A_hidden_states[batch][:padding_idx],0) #Shape is [768]
            A_features.append(truncated_feature)
        A_features = torch.stack(A_features,0).to(self.device)   # Shape is [batch_size, 768]
        
        # audio context feature extraction
        audio_context_out = self.data2vec_model(audio_context_inputs, audio_context_mask, output_attentions=True)
        A_context_hidden_states = audio_context_out.last_hidden_state
        ## average over unmasked audio tokens
        A_context_features = []
        audio_context_mask_idx_new = []
        for batch in range(A_context_hidden_states.shape[0]):
            layer = 0
            while layer<12:
                try:
                    padding_idx = sum(audio_context_out.attentions[layer][batch][0][0]!=0)
                    audio_context_mask_idx_new.append(padding_idx)
                    break
                except:
                    layer += 1
            truncated_feature = torch.mean(A_context_hidden_states[batch][:padding_idx],0) #Shape is [768]
            A_context_features.append(truncated_feature)
        A_context_features = torch.stack(A_context_features,0).to(self.device)   # Shape is [batch_size, 768]

        T_features = torch.cat((input_pooler, context_pooler), dim=1)    # Shape is [batch_size, 1024*2]
        A_features = torch.cat((A_features, A_context_features), dim=1)  # Shape is [batch_size, 768*2]
        T_output = self.T_output_layers(T_features)                    # Shape is [batch_size, 1]
        A_output = self.A_output_layers(A_features)                    # Shape is [batch_size, 1]
        
        fused_features = torch.cat((T_features, A_features), dim=1)    # Shape is [batch_size, 1024*2+768*2]
        fused_output = self.fused_output_layers(fused_features)        # Shape is [batch_size, 1]

        return {
                'T': T_output, 
                'A': A_output, 
                'M': fused_output
        }


class rob_d2v_cme_context(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.version = config.cme_version

        self.device = torch.device("cuda:" + config.device if torch.cuda.is_available() else "cpu")

        self.roberta_model = RobertaModel.from_pretrained('../roberta-base')

        self.data2vec_model = Data2VecAudioModel.from_pretrained("../facebook/data2vec-audio-base")

        self.T_output_layers = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(768, 1)
        )
        self.A_output_layers = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(768, 1)
        )
        # cls embedding layers
        self.text_cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=768)
        self.audio_cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=768)
        # if self.version == 'v3':
        #     self.text_mixed_cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=768 * 2)
        #     self.audio_mixed_cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=768 * 2)
        # else:
        #     self.text_mixed_cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=768)
        #     self.audio_mixed_cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=768)

        # CME layers
        Bert_config = BertConfig(num_hidden_layers=config.num_hidden_layers, hidden_size=768)
        self.CME_layers = nn.ModuleList(
            [CMELayer(Bert_config) for _ in range(Bert_config.num_hidden_layers)]
        )

        # # fused method V2
        # self.text_mixed_layer = nn.Sequential(
        #     nn.Dropout(config.dropout),
        #     nn.Linear(768 * 2, 768),
        #     nn.ReLU()
        # )
        # self.audio_mixed_layer = nn.Sequential(
        #     nn.Dropout(config.dropout),
        #     nn.Linear(768 * 2, 768),
        #     nn.ReLU()
        # )
        #
        # # fusion method V3
        # if self.version == 'v3':
        #     encoder_layer = nn.TransformerEncoderLayer(d_model=768 * 2, nhead=12, batch_first=True)
        #     # self.text_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3,enable_nested_tensor=False)
        #     self.text_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        #     encoder_layer = nn.TransformerEncoderLayer(d_model=768 * 2, nhead=12, batch_first=True)
        #     # self.audio_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3,enable_nested_tensor=False)
        #     self.audio_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        # # else:
        # #     encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=12, batch_first=True)
        # #     # self.text_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3,enable_nested_tensor=False)
        # #     self.text_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        # #     encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=12, batch_first=True)
        # #     # self.audio_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3,enable_nested_tensor=False)
        # #     self.audio_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        #
        # if self.version == 'v3':
        #     self.fused_output_layers = nn.Sequential(
        #         nn.Dropout(config.dropout),
        #         nn.Linear(768 * 4, 768),
        #         nn.ReLU(),
        #         nn.Linear(768, 512),
        #         nn.ReLU(),
        #         nn.Linear(512, 1)
        #     )
        # else:
        #     self.fused_output_layers = nn.Sequential(
        #         nn.Dropout(config.dropout),
        #         nn.Linear(768 * 2, 768),
        #         nn.ReLU(),
        #         nn.Linear(768, 512),
        #         nn.ReLU(),
        #         nn.Linear(512, 1)
        #     )
        self.fused_output_layers = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(768 * 2, 768),
            nn.ReLU(),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def prepend_cls(self, inputs, masks, layer_name):
        if layer_name == 'text':
            embedding_layer = self.text_cls_emb
        elif layer_name == 'audio':
            embedding_layer = self.audio_cls_emb
        elif layer_name == 'text_mixed':
            embedding_layer = self.text_mixed_cls_emb
        elif layer_name == 'audio_mixed':
            embedding_layer = self.audio_mixed_cls_emb
        index = torch.LongTensor([0]).to(device=inputs.device)
        cls_emb = embedding_layer(index)
        cls_emb = cls_emb.expand(inputs.size(0), 1, inputs.size(2))
        outputs = torch.cat((cls_emb, inputs), dim=1)

        cls_mask = torch.ones(inputs.size(0), 1).to(device=inputs.device)
        masks = torch.cat((cls_mask, masks), dim=1)
        return outputs, masks

    def forward(self, text_inputs, text_mask, text_context_inputs, text_context_mask, text_and_context_inputs, text_and_context_mask,
                audio_inputs, audio_mask, audio_context_inputs, audio_context_mask, audio_and_context_inputs, audio_and_context_mask):
        # text feature extraction
        raw_output = self.roberta_model(text_inputs, text_mask)
        t_hidden_states = raw_output.last_hidden_state
        input_pooler = raw_output["pooler_output"]  # Shape is [batch_size, 1024]

        # # text context feature extraction
        # raw_output_context = self.roberta_model(text_context_inputs, text_context_mask)
        # t_context_hidden_states = raw_output_context.last_hidden_state
        # context_pooler = raw_output_context["pooler_output"]  # Shape is [batch_size, 1024]

        # text and context feature extraction
        raw_output_text_and_context = self.roberta_model(text_and_context_inputs, text_and_context_mask)
        t_and_context_hidden_states = raw_output_text_and_context.last_hidden_state
        text_context_pooler = raw_output_text_and_context["pooler_output"]  # Shape is [batch_size, 1024]

        # audio feature extraction
        audio_out = self.data2vec_model(audio_inputs, audio_mask, output_attentions=True)
        A_hidden_states = audio_out.last_hidden_state
        ## average over unmasked audio tokens
        A_features = []
        audio_mask_idx_new = []
        for batch in range(A_hidden_states.shape[0]):
            layer = 0
            while layer < 12:
                try:
                    padding_idx = sum(audio_out.attentions[layer][batch][0][0] != 0)
                    audio_mask_idx_new.append(padding_idx)
                    break
                except:
                    layer += 1
            truncated_feature = torch.mean(A_hidden_states[batch][:padding_idx], 0)  # Shape is [1024]
            A_features.append(truncated_feature)
        A_features = torch.stack(A_features, 0).to(self.device)  # Shape is [batch_size, 1024]
        ## create new audio mask
        audio_mask_new = torch.zeros(A_hidden_states.shape[0], A_hidden_states.shape[1]).to(self.device)
        for batch in range(audio_mask_new.shape[0]):
            audio_mask_new[batch][:audio_mask_idx_new[batch]] = 1

        # # audio context feature extraction
        # audio_context_out = self.data2vec_model(audio_context_inputs, audio_context_mask, output_attentions=True)
        # A_context_hidden_states = audio_context_out.last_hidden_state
        # ## average over unmasked audio tokens
        # A_context_features = []
        # audio_context_mask_idx_new = []
        # for batch in range(A_context_hidden_states.shape[0]):
        #     layer = 0
        #     while layer < 12:
        #         try:
        #             padding_idx = sum(audio_context_out.attentions[layer][batch][0][0] != 0)
        #             audio_context_mask_idx_new.append(padding_idx)
        #             break
        #         except:
        #             layer += 1
        #     truncated_feature = torch.mean(A_context_hidden_states[batch][:padding_idx], 0)  # Shape is [768]
        #     A_context_features.append(truncated_feature)
        # A_context_features = torch.stack(A_context_features, 0).to(self.device)  # Shape is [batch_size, 768]
        # ## create new audio mask
        # audio_context_mask_new = torch.zeros(A_context_hidden_states.shape[0], A_context_hidden_states.shape[1]).to(self.device)
        # for batch in range(audio_context_mask_new.shape[0]):
        #     audio_context_mask_new[batch][:audio_context_mask_idx_new[batch]] = 1

        # audio and context feature extraction
        audio_and_context_out = self.data2vec_model(audio_and_context_inputs, audio_and_context_mask, output_attentions=True)
        A_and_context_hidden_states = audio_and_context_out.last_hidden_state
        ## average over unmasked audio tokens
        A_and_context_features = []
        audio_and_context_mask_idx_new = []
        for batch in range(A_and_context_hidden_states.shape[0]):
            layer = 0
            while layer < 12:
                try:
                    padding_idx = sum(audio_and_context_out.attentions[layer][batch][0][0] != 0)
                    audio_and_context_mask_idx_new.append(padding_idx)
                    break
                except:
                    layer += 1
            truncated_feature = torch.mean(A_and_context_hidden_states[batch][:padding_idx], 0)  # Shape is [768]
            A_and_context_features.append(truncated_feature)
        A_and_context_features = torch.stack(A_and_context_features, 0).to(self.device)  # Shape is [batch_size, 768]
        ## create new audio mask
        audio_and_context_mask_new = torch.zeros(A_and_context_hidden_states.shape[0], A_and_context_hidden_states.shape[1]).to(
            self.device)
        for batch in range(audio_and_context_mask_new.shape[0]):
            audio_and_context_mask_new[batch][:audio_and_context_mask_idx_new[batch]] = 1

        # T_features = torch.cat((input_pooler, context_pooler), dim=1)  # Shape is [batch_size, 768*2]
        # A_features = torch.cat((A_features, A_context_features), dim=1)  # Shape is [batch_size, 768*2]
        T_output = self.T_output_layers(text_context_pooler)  # Shape is [batch_size, 1]
        A_output = self.A_output_layers(A_and_context_features)  # Shape is [batch_size, 1]

        # CME layers
        ## prepend cls tokens
        t_inputs, t_attn_mask = self.prepend_cls(t_and_context_hidden_states, text_and_context_mask, 'text')  # add cls token
        a_inputs, a_attn_mask = self.prepend_cls(A_and_context_hidden_states, audio_and_context_mask_new, 'audio')  # add cls token

        # position encoding
        # pos_enc_text = Summer(PositionalEncodingPermute1D(text_inputs.shape[1]))
        # text_inputs = pos_enc_text(text_inputs)
        # pos_enc_audio = Summer(PositionalEncodingPermute1D(audio_inputs.shape[1]))
        # audio_inputs = pos_enc_audio(audio_inputs)

        # pass through CME layers
        for layer_module in self.CME_layers:
            t_inputs, a_inputs = layer_module(t_inputs, t_attn_mask,
                                                     a_inputs, a_attn_mask)

        # different fusion methods
        if self.version == 'v1':
            # fused features
            fused_hidden_states = torch.cat((t_inputs[:, 0, :], a_inputs[:, 0, :]),
                                            dim=1)  # Shape is [batch_size, 768*2]
        # elif self.version == 'v2':
        #     # concatenate original features with fused features
        #     text_concat_features = torch.cat((text_context_pooler, t_inputs[:, 0, :]), dim=1)  # Shape is [batch_size, 768*2]
        #     audio_concat_features = torch.cat((A_and_context_features, a_inputs[:, 0, :]),
        #                                       dim=1)  # Shape is [batch_size, 768*2]
        #     text_mixed_features = self.text_mixed_layer(text_concat_features)  # Shape is [batch_size, 768]
        #     audio_mixed_features = self.audio_mixed_layer(audio_concat_features)  # Shape is [batch_size, 768]
        #     # fused features
        #     fused_hidden_states = torch.cat((text_mixed_features, audio_mixed_features),
        #                                     dim=1)  # Shape is [batch_size, 768*2]
        # elif self.version == 'v3':
        #     # concatenate original features with fused features
        #     text_concat_features = torch.cat((text_context_pooler, t_inputs[:, 1:, :]),
        #                                      dim=2)  # Shape is [batch_size, text_length, 768*2]
        #     audio_concat_features = torch.cat((A_and_context_features, a_inputs[:, 1:, :]),
        #                                       dim=2)  # Shape is [batch_size, audio_length, 768*2]
        #     text_concat_features, text_attn_mask = self.prepend_cls(text_concat_features, text_and_context_mask,
        #                                                             'text_mixed')  # add cls token
        #     audio_concat_features, audio_attn_mask = self.prepend_cls(audio_concat_features, audio_and_context_mask_new,
        #                                                               'audio_mixed')  # add cls token
        #     text_mixed_features = self.text_encoder(text_concat_features,
        #                                             src_key_padding_mask=(1 - text_attn_mask).bool())
        #     audio_mixed_features = self.audio_encoder(audio_concat_features,
        #                                               src_key_padding_mask=(1 - audio_attn_mask).bool())
        #     # fused features
        #     fused_hidden_states = torch.cat((text_mixed_features[:, 0, :], audio_mixed_features[:, 0, :]),
        #                                     dim=1)  # Shape is [batch_size, 768*4]

        # last linear output layer
        fused_output = self.fused_output_layers(fused_hidden_states)  # Shape is [batch_size, 2]

        return {
            'T': T_output,
            'A': A_output,
            'M': fused_output
        }




class rob_hub_cme_context(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.version = config.cme_version

        self.device = torch.device("cuda:" + config.device if torch.cuda.is_available() else "cpu")

        # load text pre-trained model
        self.roberta_model = AutoModel.from_pretrained("hfl/chinese-roberta-wwm-ext")

        # load audio pre-trained model
        self.hubert_model = AutoModel.from_pretrained('TencentGameMate/chinese-hubert-base')

        self.T_output_layers = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(768, 1)
        )
        self.A_output_layers = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(768, 1)
        )
        # cls embedding layers
        self.text_cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=768)
        self.audio_cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=768)
        if self.version == 'v3':
            self.text_mixed_cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=768 * 2)
            self.audio_mixed_cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=768 * 2)
        else:
            self.text_mixed_cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=768)
            self.audio_mixed_cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=768)

        # CME layers
        Bert_config = BertConfig(num_hidden_layers=config.num_hidden_layers)
        self.CME_layers = nn.ModuleList(
            [CMELayer(Bert_config) for _ in range(Bert_config.num_hidden_layers)]
        )

        # fused method V2
        self.text_mixed_layer = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(768 * 2, 768),
            nn.ReLU()
        )
        self.audio_mixed_layer = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(768 * 2, 768),
            nn.ReLU()
        )

        # fusion method V3
        if self.version == 'v3':
            encoder_layer = nn.TransformerEncoderLayer(d_model=768 * 2, nhead=12, batch_first=True)
            # self.text_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3,enable_nested_tensor=False)
            self.text_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
            encoder_layer = nn.TransformerEncoderLayer(d_model=768 * 2, nhead=12, batch_first=True)
            # self.audio_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3,enable_nested_tensor=False)
            self.audio_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        # else:
        #     encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=12, batch_first=True)
        #     # self.text_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3,enable_nested_tensor=False)
        #     self.text_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        #     encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=12, batch_first=True)
        #     # self.audio_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3,enable_nested_tensor=False)
        #     self.audio_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)

        if self.version == 'v3':
            self.fused_output_layers = nn.Sequential(
                nn.Dropout(config.dropout),
                nn.Linear(768 * 4, 768),
                nn.ReLU(),
                nn.Linear(768, 512),
                nn.ReLU(),
                nn.Linear(512, 1)
            )
        else:
            self.fused_output_layers = nn.Sequential(
                nn.Dropout(config.dropout),
                nn.Linear(768 * 2, 768),
                nn.ReLU(),
                nn.Linear(768, 512),
                nn.ReLU(),
                nn.Linear(512, 1)
            )

    def prepend_cls(self, inputs, masks, layer_name):
        if layer_name == 'text':
            embedding_layer = self.text_cls_emb
        elif layer_name == 'audio':
            embedding_layer = self.audio_cls_emb
        elif layer_name == 'text_mixed':
            embedding_layer = self.text_mixed_cls_emb
        elif layer_name == 'audio_mixed':
            embedding_layer = self.audio_mixed_cls_emb
        index = torch.LongTensor([0]).to(device=inputs.device)
        cls_emb = embedding_layer(index)
        cls_emb = cls_emb.expand(inputs.size(0), 1, inputs.size(2))
        outputs = torch.cat((cls_emb, inputs), dim=1)

        cls_mask = torch.ones(inputs.size(0), 1).to(device=inputs.device)
        masks = torch.cat((cls_mask, masks), dim=1)
        return outputs, masks

    def forward(self, text_inputs, text_mask, text_context_inputs, text_context_mask, text_and_context_inputs,
                text_and_context_mask,
                audio_inputs, audio_mask, audio_context_inputs, audio_context_mask, audio_and_context_inputs,
                audio_and_context_mask):
        # text feature extraction
        raw_output = self.roberta_model(text_inputs, text_mask)
        t_hidden_states = raw_output.last_hidden_state
        input_pooler = raw_output["pooler_output"]  # Shape is [batch_size, 768]

        # text context feature extraction
        raw_output_context = self.roberta_model(text_context_inputs, text_context_mask)
        t_context_hidden_states = raw_output_context.last_hidden_state
        context_pooler = raw_output_context["pooler_output"]  # Shape is [batch_size, 768]

        # text and context feature extraction
        raw_output_text_and_context = self.roberta_model(text_and_context_inputs, text_and_context_mask)
        t_and_context_hidden_states = raw_output_text_and_context.last_hidden_state
        text_context_pooler = raw_output_text_and_context["pooler_output"]  # Shape is [batch_size, 768]

        # audio feature extraction
        audio_out = self.hubert_model(audio_inputs, audio_mask, output_attentions=True)
        A_hidden_states = audio_out.last_hidden_state
        ## average over unmasked audio tokens
        A_features = []
        audio_mask_idx_new = []
        for batch in range(A_hidden_states.shape[0]):
            layer = 0
            while layer < 12:
                try:
                    padding_idx = sum(audio_out.attentions[layer][batch][0][0] != 0)
                    audio_mask_idx_new.append(padding_idx)
                    break
                except:
                    layer += 1
            truncated_feature = torch.mean(A_hidden_states[batch][:padding_idx], 0)  # Shape is [768]
            A_features.append(truncated_feature)
        A_features = torch.stack(A_features, 0).to(self.device)  # Shape is [batch_size, 768]
        ## create new audio mask
        audio_mask_new = torch.zeros(A_hidden_states.shape[0], A_hidden_states.shape[1]).to(self.device)
        for batch in range(audio_mask_new.shape[0]):
            audio_mask_new[batch][:audio_mask_idx_new[batch]] = 1

        # audio context feature extraction
        audio_context_out = self.hubert_model(audio_context_inputs, audio_context_mask, output_attentions=True)
        A_context_hidden_states = audio_context_out.last_hidden_state
        ## average over unmasked audio tokens
        A_context_features = []
        audio_context_mask_idx_new = []
        for batch in range(A_context_hidden_states.shape[0]):
            layer = 0
            while layer < 12:
                try:
                    padding_idx = sum(audio_context_out.attentions[layer][batch][0][0] != 0)
                    audio_context_mask_idx_new.append(padding_idx)
                    break
                except:
                    layer += 1
            truncated_feature = torch.mean(A_context_hidden_states[batch][:padding_idx], 0)  # Shape is [768]
            A_context_features.append(truncated_feature)
        A_context_features = torch.stack(A_context_features, 0).to(self.device)  # Shape is [batch_size, 768]
        ## create new audio mask
        audio_context_mask_new = torch.zeros(A_context_hidden_states.shape[0], A_context_hidden_states.shape[1]).to(
            self.device)
        for batch in range(audio_context_mask_new.shape[0]):
            audio_context_mask_new[batch][:audio_context_mask_idx_new[batch]] = 1

        # audio and context feature extraction
        audio_and_context_out = self.hubert_model(audio_and_context_inputs, audio_and_context_mask,
                                                    output_attentions=True)
        A_and_context_hidden_states = audio_and_context_out.last_hidden_state
        ## average over unmasked audio tokens
        A_and_context_features = []
        audio_and_context_mask_idx_new = []
        for batch in range(A_and_context_hidden_states.shape[0]):
            layer = 0
            while layer < 12:
                try:
                    padding_idx = sum(audio_and_context_out.attentions[layer][batch][0][0] != 0)
                    audio_and_context_mask_idx_new.append(padding_idx)
                    break
                except:
                    layer += 1
            truncated_feature = torch.mean(A_and_context_hidden_states[batch][:padding_idx], 0)  # Shape is [768]
            A_and_context_features.append(truncated_feature)
        A_and_context_features = torch.stack(A_and_context_features, 0).to(self.device)  # Shape is [batch_size, 768]
        ## create new audio mask
        audio_and_context_mask_new = torch.zeros(A_and_context_hidden_states.shape[0],
                                                 A_and_context_hidden_states.shape[1]).to(
            self.device)
        for batch in range(audio_and_context_mask_new.shape[0]):
            audio_and_context_mask_new[batch][:audio_and_context_mask_idx_new[batch]] = 1

        # T_features = torch.cat((input_pooler, context_pooler), dim=1)  # Shape is [batch_size, 768*2]
        # A_features = torch.cat((A_features, A_context_features), dim=1)  # Shape is [batch_size, 768*2]
        T_output = self.T_output_layers(text_context_pooler)  # Shape is [batch_size, 1]
        A_output = self.A_output_layers(A_and_context_features)  # Shape is [batch_size, 1]

        # CME layers
        ## prepend cls tokens
        t_inputs, t_attn_mask = self.prepend_cls(t_and_context_hidden_states, text_and_context_mask,
                                                 'text')  # add cls token
        a_inputs, a_attn_mask = self.prepend_cls(A_and_context_hidden_states, audio_and_context_mask_new,
                                                 'audio')  # add cls token

        # position encoding
        # pos_enc_text = Summer(PositionalEncodingPermute1D(text_inputs.shape[1]))
        # text_inputs = pos_enc_text(text_inputs)
        # pos_enc_audio = Summer(PositionalEncodingPermute1D(audio_inputs.shape[1]))
        # audio_inputs = pos_enc_audio(audio_inputs)

        # pass through CME layers
        for layer_module in self.CME_layers:
            t_inputs, a_inputs = layer_module(t_inputs, t_attn_mask,
                                              a_inputs, a_attn_mask)

        # different fusion methods
        if self.version == 'v1':
            # fused features
            fused_hidden_states = torch.cat((t_inputs[:, 0, :], a_inputs[:, 0, :]),
                                            dim=1)  # Shape is [batch_size, 768*2]
        elif self.version == 'v2':
            # concatenate original features with fused features
            text_concat_features = torch.cat((text_context_pooler, t_inputs[:, 0, :]),
                                             dim=1)  # Shape is [batch_size, 768*2]
            audio_concat_features = torch.cat((A_and_context_features, a_inputs[:, 0, :]),
                                              dim=1)  # Shape is [batch_size, 768*2]
            text_mixed_features = self.text_mixed_layer(text_concat_features)  # Shape is [batch_size, 768]
            audio_mixed_features = self.audio_mixed_layer(audio_concat_features)  # Shape is [batch_size, 768]
            # fused features
            fused_hidden_states = torch.cat((text_mixed_features, audio_mixed_features),
                                            dim=1)  # Shape is [batch_size, 768*2]
        elif self.version == 'v3':
            # concatenate original features with fused features
            text_concat_features = torch.cat((text_context_pooler, t_inputs[:, 1:, :]),
                                             dim=2)  # Shape is [batch_size, text_length, 768*2]
            audio_concat_features = torch.cat((A_and_context_features, a_inputs[:, 1:, :]),
                                              dim=2)  # Shape is [batch_size, audio_length, 768*2]
            text_concat_features, text_attn_mask = self.prepend_cls(text_concat_features, text_and_context_mask,
                                                                    'text_mixed')  # add cls token
            audio_concat_features, audio_attn_mask = self.prepend_cls(audio_concat_features, audio_and_context_mask_new,
                                                                      'audio_mixed')  # add cls token
            text_mixed_features = self.text_encoder(text_concat_features,
                                                    src_key_padding_mask=(1 - text_attn_mask).bool())
            audio_mixed_features = self.audio_encoder(audio_concat_features,
                                                      src_key_padding_mask=(1 - audio_attn_mask).bool())
            # fused features
            fused_hidden_states = torch.cat((text_mixed_features[:, 0, :], audio_mixed_features[:, 0, :]),
                                            dim=1)  # Shape is [batch_size, 768*4]

        # last linear output layer
        fused_output = self.fused_output_layers(fused_hidden_states)  # Shape is [batch_size, 2]

        return {
            'T': T_output,
            'A': A_output,
            'M': fused_output
        }