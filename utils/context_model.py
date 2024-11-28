import torch
from torch import nn
from transformers import RobertaModel, HubertModel, AutoModel, Data2VecAudioModel
from utils.cross_attn_encoder import CMELayer, BertConfig



# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class rob_d2v_cme_context(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.version = config.cme_version

        self.device = torch.device("cuda:" + config.device if torch.cuda.is_available() else "cpu")

        self.roberta_model = RobertaModel.from_pretrained('roberta-base')

        # CME layers
        Bert_config = BertConfig(num_hidden_layers=config.num_hidden_layers, hidden_size=768)
        self.CME_layers = nn.ModuleList(
            [CMELayer(Bert_config) for _ in range(Bert_config.num_hidden_layers)]
        )
        self.con_projector = ProEncoder(Bert_config)
        self.text_projector = ProEncoder(Bert_config)
        self.audio_projector = ProEncoder(Bert_config)
        self.viison_projector = ProEncoder(Bert_config)

        self.T_output_layers = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(768*2, 1)
        )
        self.A_output_layers = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(768*2, 1)
        )
        self.V_output_layers = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(768*2, 1)
        )
        # cls embedding layers
        self.text_cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=768)
        self.audio_cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=768)
        self.vision_cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=768)

        self.heat = config.MOSI.downStream.const_heat

        self.ntxent_loss = cont_NTXentLoss(temperature=self.heat)
        self.mono_decoder = BaseClassifier(input_size=uni_fea_dim,
                                           hidden_size=hidden_size[2:],
                                           output_size=1, drop_out=drop_out,
                                           name='TVAMonoRegClassifier', )


        self.fused_output_layers = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(768 * 6, 768),
            nn.ReLU(),
            nn.Linear(768, 1)
        )

    def prepend_cls(self, inputs, masks, layer_name):
        if layer_name == 'text':
            embedding_layer = self.text_cls_emb
        elif layer_name == 'audio':
            embedding_layer = self.audio_cls_emb
        elif layer_name == 'vision':
            embedding_layer = self.vision_cls_emb
        index = torch.LongTensor([0]).to(device=inputs.device)
        cls_emb = embedding_layer(index)
        cls_emb = cls_emb.expand(inputs.size(0), 1, inputs.size(2))
        outputs = torch.cat((cls_emb, inputs), dim=1)

        cls_mask = torch.ones(inputs.size(0), 1).to(device=inputs.device)
        masks = torch.cat((cls_mask, masks), dim=1)
        return outputs, masks

    def forward(self, text_inputs, text_mask, text_context_inputs, text_context_mask,
                audio_inputs, audio_mask, audio_context_inputs, audio_context_mask,
                vision_inputs, vision_mask, vision_context_inputs, vision_context_mask,
                sample2, targets=None, return_loss=True,):

        raw_output = self.roberta_model(text_inputs, text_mask)
        t_hidden_states = raw_output.last_hidden_state
        input_pooler = raw_output["pooler_output"]  # Shape is [batch_size, 1024]

        # text context feature extraction
        raw_output_context = self.roberta_model(text_context_inputs, text_context_mask)
        t_context_hidden_states = raw_output_context.last_hidden_state
        context_pooler = raw_output_context["pooler_output"]  # Shape is [batch_size, 1024]


        # T_features = torch.cat((input_pooler, context_pooler), dim=1)    # Shape is [batch_size, 1024*2]
        # A_features = torch.cat((audio_inputs, audio_context_inputs), dim=1)  # Shape is [batch_size, 1024*2]
        # V_features = torch.cat((vision_inputs, vision_context_inputs), dim=1)  # Shape is [batch_size, 1024*2]

        # CME layers
        text_inputs, text_attn_mask = self.prepend_cls(raw_output, text_mask, 'text')  # add cls token
        audio_inputs, audio_attn_mask = self.prepend_cls(audio_inputs, audio_mask, 'audio')  # add cls token
        vision_inputs, vision_attn_mask = self.prepend_cls(vision_inputs, audio_mask, 'audio')  # add cls token
        text_context_inputs, text_context_attn_mask = self.prepend_cls(t_context_hidden_states, text_context_mask, 'text')  # add cls token
        audio_context_inputs, audio_context_attn_mask = self.prepend_cls(audio_inputs, audio_context_mask, 'audio')  # add cls token
        vision_context_inputs, vision_context_attn_mask = self.prepend_cls(vision_context_inputs, audio_context_mask, 'audio')  # add cls token

        for layer_module in self.CME_layers:
            text_inputs, audio_inputs, vision_inputs = layer_module(text_inputs, text_attn_mask,
                                                     audio_inputs, audio_attn_mask,
                                                     vision_inputs, vision_attn_mask
                                                     )
        for layer_module in self.CME_layers:
            text_context_inputs, audio_context_inputs, vision_context_inputs = layer_module(text_context_inputs, text_context_attn_mask,
                                                                     audio_context_inputs, audio_context_attn_mask,
                                                                    vision_context_inputs,vision_context_attn_mask)

        x_t_simi1 = self.con_projector(text_inputs)
        x_v_simi1 = self.con_projector(vision_inputs)
        x_a_simi1 = self.con_projector(audio_inputs)
        x_t_dissimi1 = self.text_projector(text_context_inputs)
        x_v_dissimi1 = self.viison_projector(vision_context_inputs)
        x_a_dissimi1 = self.audio_projector(audio_context_inputs)

        x1_s = torch.cat((x_t_simi1, x_v_simi1, x_a_simi1), dim=-1)
        x1_ds = torch.cat((x_t_dissimi1, x_v_dissimi1, x_a_dissimi1), dim=-1)
        x1_all = torch.cat((x1_s, x1_ds), dim=-1)
        x1_sds = torch.cat((x_t_simi1, x_v_simi1, x_a_simi1, x_t_dissimi1, x_v_dissimi1, x_a_dissimi1,
                            ), dim=0)
        x_sds = x1_sds
        x2 = None
        x = x1_all
        if sample2 is not None:
            text2 = sample2['raw_text']
            vision2 = sample2['vision'].clone().detach().float()
            audio2 = sample2['audio'].clone().detach().float()
            label2 = sample2['regression_labels'].clone().detach().float()  # .squeeze()
            label_T2 = sample2['regression_labels'].clone().detach().float()  # .squeeze()
            label_V2 = sample2['regression_labels'].clone().detach().float()  # .squeeze()
            label_A2 = sample2['regression_labels'].clone().detach().float()  # .squeeze()
            key_padding_mask_V2, key_padding_mask_A2 = (sample2['vision_padding_mask'].clone().detach(),
                                                        sample2['audio_padding_mask'].clone().detach())

            x_t_embed2 = self.text_encoder(text2).squeeze()
            x_v_embed2 = self.vision_encoder(vision2, key_padding_mask=key_padding_mask_V2).squeeze()
            x_a_embed2 = self.audio_encoder(audio2, key_padding_mask=key_padding_mask_A2).squeeze()

            x_t_simi2 = self.T_simi_proj(x_t_embed2)
            x_v_simi2 = self.V_simi_proj(x_v_embed2)
            x_a_simi2 = self.A_simi_proj(x_a_embed2)
            x_t_dissimi2 = self.T_dissimi_proj(x_t_embed2)
            x_v_dissimi2 = self.V_dissimi_proj(x_v_embed2)
            x_a_dissimi2 = self.A_dissimi_proj(x_a_embed2)

            x2_s = torch.cat((x_t_simi2, x_v_simi2, x_a_simi2), dim=-1)
            x2_ds = torch.cat((x_t_dissimi2, x_v_dissimi2, x_a_dissimi2), dim=-1)
            x2_all = torch.cat((x2_s, x2_ds), dim=-1)
            x2_sds = torch.cat((x_t_simi2, x_v_simi2, x_a_simi2, x_t_dissimi2, x_v_dissimi2, x_a_dissimi2,
                                ), dim=0)

            if return_loss:
                sup_const_loss = 0
                # sds_loss = 0
                if sample2 is not None:
                    # For sequence [Ts,T1s,T2s...T6s, Vs,V1s.....,As,A1s,...], construct corresponding positive and negative embedding pairs, which is use for contrastive learning later.
                    t1, p, t2, n = torch.tensor([0, 0, 7, 7, 14, 14,
                                                 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]), \
                        torch.tensor([1, 2, 8, 9, 15, 16,
                                      7, 14, 8, 15, 9, 16, 10, 17, 11, 18, 12, 19, 13, 20]), \
                        torch.tensor([0, 0, 0, 0, 7, 7, 7, 7, 14, 14, 14, 14,
                                      0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6]), \
                        torch.tensor([3, 4, 5, 6, 10, 11, 12, 13, 17, 18, 19, 20,
                                      21, 28, 35, 22, 29, 36, 23, 30, 37, 24, 31, 38, 25, 32, 39, 26, 33, 40, 27,
                                      34, 41])

                    indices_tuple = (t1, p, t2, n)
                    pre_sample_label = torch.tensor([0, 0, 0, 1, 2, 3, 4, 0, 0, 0, 1, 2, 3, 4, 0, 0, 0, 1, 2, 3, 4,
                                                     5, 5, 5, 6, 7, 8, 9, 5, 5, 5, 6, 7, 8, 9, 5, 5, 5, 6, 7, 8, 9, ])
                    for i in range(len(x1_all)):
                        pre_sample_x = []
                        for fea1, fea2 in zip(
                                [x_t_simi1, x_v_simi1, x_a_simi1, x_t_dissimi1, x_v_dissimi1, x_a_dissimi1, ],
                                [x_t_simi2, x_v_simi2, x_a_simi2, x_t_dissimi2, x_v_dissimi2,
                                 x_a_dissimi2, ]):
                            pre_sample_x.append(torch.cat((fea1[i].unsqueeze(0), fea2[6 * i:6 * (i + 1)]), dim=0))

                        sup_const_loss += self.ntxent_loss(torch.cat(pre_sample_x, dim=0), pre_sample_label,
                                                           # pre_sample_x=list{6},list{i}=tensor(7,384)
                                                           indices_tuple=indices_tuple)

                    sup_const_loss /= len(x1_all)

        uni_output = self.mono_decoder(x_sds)
        fused_output = self.fused_output_layers(x)  # Shape is [batch_size, 1]

        return {
            'uni': uni_output,
            'M': fused_output
        }
        # return None




class rob_hub_cme_context(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.version = config.cme_version

        self.device = torch.device("cuda:" + config.device if torch.cuda.is_available() else "cpu")

        # load text pre-trained model
        self.roberta_model = AutoModel.from_pretrained("hfl/chinese-roberta-wwm-ext")

        # CME layers
        Bert_config = BertConfig(num_hidden_layers=config.num_hidden_layers, hidden_size=768)
        self.CME_layers = nn.ModuleList(
            [CMELayer(Bert_config) for _ in range(Bert_config.num_hidden_layers)]
        )
        self.con_projector = ProEncoder(Bert_config)
        self.text_projector = ProEncoder(Bert_config)
        self.audio_projector = ProEncoder(Bert_config)
        self.viison_projector = ProEncoder(Bert_config)

        self.T_output_layers = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(768 * 2, 1)
        )
        self.A_output_layers = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(768 * 2, 1)
        )
        self.V_output_layers = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(768 * 2, 1)
        )
        # cls embedding layers
        self.text_cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=768)
        self.audio_cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=768)
        self.vision_cls_emb = nn.Embedding(num_embeddings=1, embedding_dim=768)

        self.heat = config.MOSI.downStream.const_heat

        self.ntxent_loss = cont_NTXentLoss(temperature=self.heat)
        self.mono_decoder = BaseClassifier(input_size=uni_fea_dim,
                                           hidden_size=hidden_size[2:],
                                           output_size=1, drop_out=drop_out,
                                           name='TVAMonoRegClassifier', )

        self.fused_output_layers = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(768 * 6, 768),
            nn.ReLU(),
            nn.Linear(768, 1)
        )

    def prepend_cls(self, inputs, masks, layer_name):
        if layer_name == 'text':
            embedding_layer = self.text_cls_emb
        elif layer_name == 'audio':
            embedding_layer = self.audio_cls_emb
        elif layer_name == 'vision':
            embedding_layer = self.vision_cls_emb
        index = torch.LongTensor([0]).to(device=inputs.device)
        cls_emb = embedding_layer(index)
        cls_emb = cls_emb.expand(inputs.size(0), 1, inputs.size(2))
        outputs = torch.cat((cls_emb, inputs), dim=1)

        cls_mask = torch.ones(inputs.size(0), 1).to(device=inputs.device)
        masks = torch.cat((cls_mask, masks), dim=1)
        return outputs, masks

    def forward(self, text_inputs, text_mask, text_context_inputs, text_context_mask,
                audio_inputs, audio_mask, audio_context_inputs, audio_context_mask,
                vision_inputs, vision_mask, vision_context_inputs, vision_context_mask,
                sample2, targets=None, return_loss=True, ):

        raw_output = self.roberta_model(text_inputs, text_mask)
        t_hidden_states = raw_output.last_hidden_state
        input_pooler = raw_output["pooler_output"]  # Shape is [batch_size, 1024]

        # text context feature extraction
        raw_output_context = self.roberta_model(text_context_inputs, text_context_mask)
        t_context_hidden_states = raw_output_context.last_hidden_state
        context_pooler = raw_output_context["pooler_output"]  # Shape is [batch_size, 1024]

        # T_features = torch.cat((input_pooler, context_pooler), dim=1)    # Shape is [batch_size, 1024*2]
        # A_features = torch.cat((audio_inputs, audio_context_inputs), dim=1)  # Shape is [batch_size, 1024*2]
        # V_features = torch.cat((vision_inputs, vision_context_inputs), dim=1)  # Shape is [batch_size, 1024*2]

        # CME layers
        text_inputs, text_attn_mask = self.prepend_cls(raw_output, text_mask, 'text')  # add cls token
        audio_inputs, audio_attn_mask = self.prepend_cls(audio_inputs, audio_mask, 'audio')  # add cls token
        vision_inputs, vision_attn_mask = self.prepend_cls(vision_inputs, audio_mask, 'audio')  # add cls token
        text_context_inputs, text_context_attn_mask = self.prepend_cls(t_context_hidden_states, text_context_mask,
                                                                       'text')  # add cls token
        audio_context_inputs, audio_context_attn_mask = self.prepend_cls(audio_inputs, audio_context_mask,
                                                                         'audio')  # add cls token
        vision_context_inputs, vision_context_attn_mask = self.prepend_cls(vision_context_inputs, audio_context_mask,
                                                                           'audio')  # add cls token

        for layer_module in self.CME_layers:
            text_inputs, audio_inputs, vision_inputs = layer_module(text_inputs, text_attn_mask,
                                                                    audio_inputs, audio_attn_mask,
                                                                    vision_inputs, vision_attn_mask
                                                                    )
        for layer_module in self.CME_layers:
            text_context_inputs, audio_context_inputs, vision_context_inputs = layer_module(text_context_inputs,
                                                                                            text_context_attn_mask,
                                                                                            audio_context_inputs,
                                                                                            audio_context_attn_mask,
                                                                                            vision_context_inputs,
                                                                                            vision_context_attn_mask)

        x_t_simi1 = self.con_projector(text_inputs)
        x_v_simi1 = self.con_projector(vision_inputs)
        x_a_simi1 = self.con_projector(audio_inputs)
        x_t_dissimi1 = self.text_projector(text_context_inputs)
        x_v_dissimi1 = self.viison_projector(vision_context_inputs)
        x_a_dissimi1 = self.audio_projector(audio_context_inputs)

        x1_s = torch.cat((x_t_simi1, x_v_simi1, x_a_simi1), dim=-1)
        x1_ds = torch.cat((x_t_dissimi1, x_v_dissimi1, x_a_dissimi1), dim=-1)
        x1_all = torch.cat((x1_s, x1_ds), dim=-1)
        x1_sds = torch.cat((x_t_simi1, x_v_simi1, x_a_simi1, x_t_dissimi1, x_v_dissimi1, x_a_dissimi1,
                            ), dim=0)
        x_sds = x1_sds
        x2 = None
        x = x1_all
        if sample2 is not None:
            text2 = sample2['raw_text']
            vision2 = sample2['vision'].clone().detach().float()
            audio2 = sample2['audio'].clone().detach().float()
            label2 = sample2['regression_labels'].clone().detach().float()  # .squeeze()
            label_T2 = sample2['regression_labels'].clone().detach().float()  # .squeeze()
            label_V2 = sample2['regression_labels'].clone().detach().float()  # .squeeze()
            label_A2 = sample2['regression_labels'].clone().detach().float()  # .squeeze()
            key_padding_mask_V2, key_padding_mask_A2 = (sample2['vision_padding_mask'].clone().detach(),
                                                        sample2['audio_padding_mask'].clone().detach())

            x_t_embed2 = self.text_encoder(text2).squeeze()
            x_v_embed2 = self.vision_encoder(vision2, key_padding_mask=key_padding_mask_V2).squeeze()
            x_a_embed2 = self.audio_encoder(audio2, key_padding_mask=key_padding_mask_A2).squeeze()

            x_t_simi2 = self.T_simi_proj(x_t_embed2)
            x_v_simi2 = self.V_simi_proj(x_v_embed2)
            x_a_simi2 = self.A_simi_proj(x_a_embed2)
            x_t_dissimi2 = self.T_dissimi_proj(x_t_embed2)
            x_v_dissimi2 = self.V_dissimi_proj(x_v_embed2)
            x_a_dissimi2 = self.A_dissimi_proj(x_a_embed2)

            x2_s = torch.cat((x_t_simi2, x_v_simi2, x_a_simi2), dim=-1)
            x2_ds = torch.cat((x_t_dissimi2, x_v_dissimi2, x_a_dissimi2), dim=-1)
            x2_all = torch.cat((x2_s, x2_ds), dim=-1)
            x2_sds = torch.cat((x_t_simi2, x_v_simi2, x_a_simi2, x_t_dissimi2, x_v_dissimi2, x_a_dissimi2,
                                ), dim=0)

            if return_loss:
                sup_const_loss = 0
                # sds_loss = 0
                if sample2 is not None:
                    # For sequence [Ts,T1s,T2s...T6s, Vs,V1s.....,As,A1s,...], construct corresponding positive and negative embedding pairs, which is use for contrastive learning later.
                    t1, p, t2, n = torch.tensor([0, 0, 7, 7, 14, 14,
                                                 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]), \
                        torch.tensor([1, 2, 8, 9, 15, 16,
                                      7, 14, 8, 15, 9, 16, 10, 17, 11, 18, 12, 19, 13, 20]), \
                        torch.tensor([0, 0, 0, 0, 7, 7, 7, 7, 14, 14, 14, 14,
                                      0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6]), \
                        torch.tensor([3, 4, 5, 6, 10, 11, 12, 13, 17, 18, 19, 20,
                                      21, 28, 35, 22, 29, 36, 23, 30, 37, 24, 31, 38, 25, 32, 39, 26, 33, 40, 27,
                                      34, 41])

                    indices_tuple = (t1, p, t2, n)
                    pre_sample_label = torch.tensor([0, 0, 0, 1, 2, 3, 4, 0, 0, 0, 1, 2, 3, 4, 0, 0, 0, 1, 2, 3, 4,
                                                     5, 5, 5, 6, 7, 8, 9, 5, 5, 5, 6, 7, 8, 9, 5, 5, 5, 6, 7, 8, 9, ])
                    for i in range(len(x1_all)):
                        pre_sample_x = []
                        for fea1, fea2 in zip(
                                [x_t_simi1, x_v_simi1, x_a_simi1, x_t_dissimi1, x_v_dissimi1, x_a_dissimi1, ],
                                [x_t_simi2, x_v_simi2, x_a_simi2, x_t_dissimi2, x_v_dissimi2,
                                 x_a_dissimi2, ]):
                            pre_sample_x.append(torch.cat((fea1[i].unsqueeze(0), fea2[6 * i:6 * (i + 1)]), dim=0))

                        sup_const_loss += self.ntxent_loss(torch.cat(pre_sample_x, dim=0), pre_sample_label,
                                                           # pre_sample_x=list{6},list{i}=tensor(7,384)
                                                           indices_tuple=indices_tuple)

                    sup_const_loss /= len(x1_all)

        uni_output = self.mono_decoder(x_sds)
        fused_output = self.fused_output_layers(x)  # Shape is [batch_size, 1]

        return {
            'uni': uni_output,
            'M': fused_output
        }
        # return None