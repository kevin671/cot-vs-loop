# https://github.com/kevin671/tmlt
import math

import torch
import torch.nn as nn

from .transformer import MLP, LoopedTF, SelfAttention


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# https://github.com/facebookresearch/DiT/blob/main/models.py#L101
class DiTBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.RMSNorm(config.n_embd, elementwise_affine=False)
        self.attn = SelfAttention(config)
        self.norm2 = nn.RMSNorm(config.n_embd, elementwise_affine=False)
        self.mlp = MLP(config)

        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(config.n_embd, 6 * config.n_embd, bias=True))

    def forward(self, x, c, attn_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), attn_mask=attn_mask)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class TimeModulatedLoopedTF(LoopedTF):
    def __init__(self, config):
        super().__init__(config, DiTBlock)
        self.timestep_embedder = TimestepEmbedder(config.n_embd)
        for block in self.transformer.h:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size

        if self.config.is_causal:
            # causal mask: no attention to future tokens
            attn_mask = None
        else:
            attn_mask = idx != self.pad_token_id  # shape (b, t)
            attn_mask = attn_mask[:, None, None, :].to(device)

        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        tok_emb = self.transformer.wte(idx)
        if not self.config.use_rope:
            pos_emb = self.transformer.wpe(pos)
            x = tok_emb + pos_emb
        else:
            x = tok_emb

        x = self.transformer.drop(x)
        for t in range(self.config.n_loop):
            t_emb = self.timestep_embedder(torch.full((b,), t, dtype=torch.long, device=device))
            for block in self.transformer.h:
                x = block(x, t_emb, attn_mask)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        return logits


# https://github.com/facebookresearch/DiT/blob/main/models.py#L27
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
