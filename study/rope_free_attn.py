"""Atencao "leve": exatamente o Lfm2Attention original (mesmo q/k/v/out_proj,
mesmo QK-Norm, mesmo GQA), so sem aplicar RoPE. O profiling por operador
(study/latency_breakdown.txt, ponto 4) mostrou que RoPE (rotate_half: Neg,
Concat, Mul, Unsqueeze) e o broadcast do GQA (Expand) somam boa parte do
custo de uma camada de atencao que nao e matmul. Isso remove so o RoPE por
enquanto -- estrutura identica ao original, so tira uma chamada.

Warm-start perfeito: como as dimensoes nao mudam (mesmos pesos, mesma forma),
carrega o state_dict do self_attn ORIGINAL da propria camada direto, sem
adaptacao nenhuma -- e o unico modulo desta serie de experimentos que comeca
com fidelidade real ao comportamento antigo (so falta RoPE), nao uma
aproximacao emprestada de outro lugar.
"""

import torch
import torch.nn.functional as F
from transformers.models.lfm2.modeling_lfm2 import Lfm2Attention


class RopeFreeAttention(Lfm2Attention):
    def forward(self, hidden_states, position_embeddings=None, attention_mask=None,
               past_key_values=None, **kwargs):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        q = self.q_layernorm(self.q_proj(hidden_states).view(*hidden_shape)).transpose(1, 2)
        k = self.k_layernorm(self.k_proj(hidden_states).view(*hidden_shape)).transpose(1, 2)
        v = self.v_proj(hidden_states).view(*hidden_shape).transpose(1, 2)
        # sem apply_rotary_pos_emb -- essa e a unica diferenca do original

        n_rep = q.shape[1] // k.shape[1]
        attn = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, scale=self.scaling, enable_gqa=n_rep > 1,
        )
        attn = attn.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        return self.out_proj(attn), None
