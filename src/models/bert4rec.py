import torch
from torch import nn as nn
import numpy as np
import pytorch_lightning as pl
from loguru import logger

from .embedding import BERTEmbedding, SimpleEmbedding
from .new_transformer import TransformerBlock


class BERT(pl.LightningModule):
    def __init__(
        self,
        max_len: int = None,
        num_items: int = None,
        n_layer: int = None,
        n_head: int = None,
        n_b: int = None,
        d_model: int = None,
        dropout: float = 0.0,
        battn: bool = None,
        bpff: bool = None,
        brpb: bool = None,
        plm_size: int = 512,
        img_size: int = 768,
        modal_type: str = "img_text",
        item_mm_fusion: str = "dynamic_shared",
    ):
        super().__init__()
        self.d_model = d_model
        self.num_items = num_items
        self.n_b = n_b
        self.battn = battn
        self.bpff = bpff
        self.brpb = brpb

        self.modal_type = modal_type
        self.plm_size = plm_size
        self.img_size = img_size

        plm_embedding_weight = self.load_plm_embedding()
        self.plm_embedding = self.weight2emb(plm_embedding_weight, self.plm_size)
        img_embedding_weight = self.load_img_embedding()
        self.img_embedding = self.weight2emb(img_embedding_weight, self.img_size)
        plm_embedding_weight_aug = self.load_plm_embedding_aug()
        self.plm_embedding_aug = self.weight2emb(
            plm_embedding_weight_aug, self.plm_size
        )

        if "text" in self.modal_type and "img" in self.modal_type:
            if self.item_mm_fusion == "dynamic_shared":
                self.fusion_factor = nn.Parameter(
                    data=torch.tensor(0, dtype=torch.float)
                )
            elif self.item_mm_fusion == "dynamic_instance":
                self.fusion_factor = nn.Parameter(
                    data=torch.zeros(self.num_items, dtype=torch.float)
                )
        if "text" in self.modal_type:
            self.text_adaptor = nn.Linear(self.plm_size, self.d_model)

        if "img" in self.modal_type:
            self.img_adaptor = nn.Linear(self.img_size, self.d_model)

        vocab_size = num_items + 1 + n_b  # add padding and mask
        # if self.brpb:
        if True:
            # simple embedding, adding behavioral relative positional bias in transformer blocks
            self.embedding = SimpleEmbedding(
                vocab_size=vocab_size, embed_size=d_model, dropout=dropout
            )
        else:
            # embedding for BERT, sum of positional, token embeddings
            self.embedding = BERTEmbedding(
                vocab_size=vocab_size,
                embed_size=d_model,
                max_len=max_len,
                dropout=dropout,
            )
        # multi-layers transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model, n_head, d_model * 4, n_b, battn, bpff, brpb, dropout
                )
                for _ in range(n_layer)
            ]
        )

    def get_embedding_empty_mask(self, embedding_table):
        empty_mask_data = ~embedding_table.weight.data.sum(-1).bool()
        return IndexableBuffer(empty_mask_data)

    def load_plm_embedding(self):
        feat_path = "/home/lllrrr/Datasets/yelp/yelp.feat1CLS"
        loaded_feat = np.fromfile(feat_path, dtype=np.float32).reshape(
            -1, self.plm_size
        )
        logger.info(f"plm_emb.shape: {loaded_feat.shape}")
        # mapped_feat = np.zeros((self.num_items, self.plm_size))
        # for i, token in enumerate(self.field2id_token['item_id']):
        #     if token == '[PAD]':
        #         continue
        #     mapped_feat[i] = loaded_feat[int(token)]
        # return mapped_feat
        return loaded_feat

    def load_img_embedding(self):
        feat_path = "/home/lllrrr/Datasets/yelp/yelp.feat3CLS"
        loaded_feat = np.fromfile(feat_path, dtype=np.float32).reshape(
            -1, self.img_size
        )
        logger.info(f"img_emb.shape: {loaded_feat.shape}")
        # Get the embedding matrix by map
        # mapped_feat = np.zeros((self.num_items, self.img_size))
        # for i, token in enumerate(self.field2id_token['item_id']):
        #     if token == '[PAD]':
        #         continue
        #     mapped_feat[i] = loaded_feat[int(token)]
        # return mapped_feat
        return loaded_feat

    def load_plm_embedding_aug(self):
        feat_path = "/home/lllrrr/Datasets/yelp/yelp.feat2CLS"
        loaded_feat = np.fromfile(feat_path, dtype=np.float32).reshape(
            -1, self.img_size
        )
        logger.info(f"plm_emb_aug.shape: {loaded_feat.shape}")
        # Get the embedding matrix by map
        # mapped_feat = np.zeros((self.num_items, self.img_size))
        # for i, token in enumerate(self.field2id_token['item_id']):
        #     if token == '[PAD]':
        #         continue
        #     mapped_feat[i] = loaded_feat[int(token)]
        # return mapped_feat
        return loaded_feat

    def weight2emb(self, weight, emd_size):
        # plm_embedding = nn.Embedding(self.item_num, self.plm_size, padding_idx=0)
        plm_embedding = nn.Embedding(self.num_items, emd_size, padding_idx=0)
        plm_embedding.weight.requires_grad = False
        plm_embedding.weight.data.copy_(torch.from_numpy(weight))
        return plm_embedding

    def forward(self, x, b_seq):
        # get padding masks
        mask = x > 0
        # embedding the indexed sequence to sequence of vectors
        # x = self.embedding(x) # TODO: Update the embedding
        text_emb_seq = self.plm_embedding(x)
        img_emb_seq = self.img_embedding(x)
        # text_emb_empty_mask_seq = self.plm_embedding_empty_mask(x)
        # img_emb_empty_mask_seq = self.img_embedding_empty_mask(x)

        text_emb = self.text_adaptor(text_emb_seq)
        img_emb = self.img_adaptor(img_emb_seq)

        item_emb_list = 0
        item_emb_list = item_emb_list + text_emb
        item_emb_list = item_emb_list + img_emb

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(item_emb_list, b_seq, mask)
        return x

    def _compute_embedding(self, item_seq):
        pass
