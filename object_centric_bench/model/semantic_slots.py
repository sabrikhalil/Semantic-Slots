"""
SemanticSlots: Semantic Slots for Video Object-Centric Learning (BMVC 2026).
https://github.com/sabrikhalil/Semantic-Slots

Everything specific to SemanticSlots is in this file. The rest of the codebase is RandSF.Q
(https://github.com/Genera1Z/RandSF.Q) with a few small changes marked with ``[SemanticSlots]``.
"""
from einops import rearrange
import torch as pt
import torch.nn as nn
import torch.nn.functional as ptnf

from .videosaur import VideoSAUR


class SemanticSlots(VideoSAUR):
    """VideoSAUR pipeline (frozen DINO, Slot Attention, feature reconstruction) with two changes:
    - the decoder is the autoregressive Transformer of DINOSAUR, which reads the features of the
      frame it decodes. Slots then only need to say *what* an object is, not *where* it is;
    - no transitioner and no temporal loss. The model is trained on single frames (t=1).

    At inference, ``mode`` sets how slots follow a video:
    - "refined":  Slot Attention on every frame, initialized with the slots of the previous frame;
    - "frozen":   Slot Attention on frame 0 only, its slots decode all frames;
    - "adaptive": like "frozen", but Slot Attention runs again on frames that show new content.
    """

    def __init__(
        self,
        encode_backbone,
        encode_posit_embed,
        encode_project,
        initializ,
        aggregat,
        decode,
        mode="refined",
        adapt_thresh=0.04,
        adapt_iter=5,
    ):
        super().__init__(
            encode_backbone,
            encode_posit_embed,
            encode_project,
            initializ,
            aggregat,
            decode,
            transit=nn.Identity(),  # slots of frame t-1 are the queries of frame t
        )
        assert mode in ["refined", "frozen", "adaptive"]
        self.mode = mode
        self.adapt_thresh = adapt_thresh  # novelty increase that triggers a slot update
        self.adapt_iter = adapt_iter  # Slot Attention iterations on frame 0 and trigger frames

    def forward(self, input, condit=None):
        """
        - input: video, shape=(b,t,c,h,w)
        - condit: condition, shape=(b,t,n,c); on MOVi, the object boxes that initialize the slots
        """
        b, t, c, h, w = input.shape
        input = input.flatten(0, 1)  # (b*t,c,h,w)

        feature = self.encode_backbone(input).detach()  # (b*t,c,h,w)
        bt, c, h, w = feature.shape
        encode = feature.permute(0, 2, 3, 1)  # (b*t,h,w,c)
        encode = self.encode_posit_embed(encode)
        encode = encode.flatten(1, 2)  # (b*t,h*w,c)
        encode = self.encode_project(encode)

        clue = feature.permute(0, 2, 3, 1).flatten(1, 2)  # decoder context (b*t,h*w,c)
        feature = rearrange(feature, "(b t) c h w -> b t c h w", b=b)
        encode = rearrange(encode, "(b t) hw c -> b t hw c", b=b)

        adaptive = self.mode == "adaptive" and not self.training
        frozen = self.mode == "frozen" and not self.training
        niter0 = self.adapt_iter if adaptive else None  # None: aggregat.num_iter

        query = self.initializ(b if condit is None else condit[:, 0, :, :])  # (b,n,c)
        slotz = []
        attent = []
        for i in range(t):
            if i == 0 or not (frozen or adaptive):
                slotz_i, attent_i = self.aggregat(encode[:, i], query, num_iter=niter0)
                if i == 0 and adaptive:
                    trigger = __class__.novelty_trigger(
                        feature, attent_i, self.adapt_thresh
                    )  # (b,t)
            elif adaptive and trigger[:, i].any():
                slotz_i, attent_i = self.aggregat(
                    encode[:, i], query, num_iter=self.adapt_iter
                )
                update = trigger[:, i, None, None].to(query.device)
                slotz_i = slotz_i.where(update, query)  # only triggered videos update
            else:  # slots are kept, Slot Attention does not run
                slotz_i = query
                attent_i = query.new_zeros(b, query.size(1), h * w)
            query = self.transit(slotz_i)
            slotz.append(slotz_i)  # [(b,n,c),..]
            attent.append(attent_i)  # [(b,n,h*w),..]
        slotz = pt.stack(slotz, 1)  # (b,t,n,c)
        attent = pt.stack(attent, 1)  # (b,t,n,h*w)
        attent = rearrange(attent, "b t n (h w) -> b t n h w", h=h)

        recon, attent2 = self.decode(clue, slotz.flatten(0, 1))  # (b*t,h*w,c)
        recon = rearrange(recon, "(b t) (h w) c -> b t c h w", b=b, h=h)
        attent2 = rearrange(attent2, "(b t) n (h w) -> b t n h w", b=b, h=h)

        # attent2 (decoder cross-attention) gives the segmentation, see the configs
        return feature, slotz, attent, attent2, recon

    @staticmethod
    @pt.no_grad()
    def novelty_trigger(feature, attent0, thresh, top=0.05):
        """Flag frames whose content is not explained by the slots of frame 0 (SemanticSlots-Adaptive).

        Each slot of frame 0 becomes an anchor in DINO space, the average of the features it attends.
        The novelty of a frame is the mean over its ``top`` most distant patches of the cosine distance
        to the closest anchor. A frame triggers a slot update when its novelty exceeds the maximum
        novelty of all previous frames by more than ``thresh``.

        - feature: DINO features, shape=(b,t,c,h,w)
        - attent0: Slot Attention of frame 0, shape=(b,n,h*w)
        - return: shape=(b,t), dtype=bool, on cpu; frame 0 is never flagged
        """
        b, t, c, h, w = feature.shape
        feat0 = feature[:, 0].float().flatten(2).transpose(1, 2)  # (b,h*w,c)
        weight = attent0.float()
        anchor = (weight @ feat0) / weight.sum(-1, keepdim=True).clamp_min(1e-9)
        anchor = ptnf.normalize(anchor, dim=-1)  # (b,n,c)
        feat = feature.float().permute(0, 1, 3, 4, 2).reshape(b, t, h * w, c)
        feat = ptnf.normalize(feat, dim=-1)
        dist = 1 - pt.einsum("btsc,bnc->btsn", feat, anchor).max(-1).values  # (b,t,h*w)
        ktop = max(1, int(h * w * top))
        novelty = dist.topk(ktop, dim=-1).values.mean(-1).cpu()  # (b,t)
        past_max = novelty.cummax(1).values  # max novelty up to each frame
        trigger = pt.zeros(b, t, dtype=pt.bool)
        trigger[:, 1:] = novelty[:, 1:] - past_max[:, :-1] > thresh
        return trigger


class ARTransformerDecoder(nn.Module):
    """Autoregressive Transformer decoder of DINOSAUR (Seitzer et al., ICLR 2023).

    Patch i is predicted from the slots (cross-attention) and from the features of patches 0..i-1
    of the same frame (causal self-attention with teacher forcing). The cross-attention weights of
    the last block, averaged over heads, assign each patch to a slot.
    """

    def __init__(
        self, vfm_dim, slot_dim, n_patches, d_model, n_heads=4, n_blocks=4, dropout=0.0
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            linear(vfm_dim, d_model, bias=False), nn.LayerNorm(d_model)
        )
        self.slot_proj = nn.Sequential(
            linear(slot_dim, d_model, bias=False), nn.LayerNorm(d_model)
        )
        self.bos_token = nn.Parameter(pt.zeros(1, 1, d_model))
        nn.init.normal_(self.bos_token, std=0.02)
        self.pos_emb = nn.Parameter(pt.randn(1, n_patches, d_model) * d_model**-0.5)
        gain = (3 * n_blocks) ** -0.5  # init of the residual branches, as in SPOT
        self.blocks = nn.ModuleList(
            [
                ARTransformerDecoderBlock(d_model, n_heads, dropout, gain)
                for _ in range(n_blocks)
            ]
        )
        self.readout = linear(d_model, vfm_dim, bias=False)

    def forward(self, input, slotz):
        """
        - input: features to reconstruct, shape=(b,h*w,c)
        - slotz: slots, shape=(b,n,c)
        - return: reconstruction, shape=(b,h*w,c); attention, shape=(b,n,h*w)
        """
        b, hw, c = input.shape
        x = self.input_proj(input) + self.pos_emb
        x = pt.cat([self.bos_token.expand(b, -1, -1), x[:, :-1]], 1)  # shift right
        causal = pt.full([hw, hw], -pt.inf, device=x.device).triu(1)
        memory = self.slot_proj(slotz)
        for block in self.blocks:
            x, attent = block(x, memory, causal)
        recon = self.readout(x)
        return recon, attent.transpose(1, 2)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # The checkpoints of the paper were saved from an older version of this decoder with an
        # extra head for temporal similarity, never trained (its loss weight was 0). Drop it.
        old = prefix + "output_head."
        if old + "recon_head.weight" in state_dict:
            state_dict[prefix + "readout.weight"] = state_dict.pop(old + "recon_head.weight")
        for k in [_ for _ in state_dict if _.startswith(old)]:
            state_dict.pop(k)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class ARTransformerDecoderBlock(nn.Module):
    """Pre-norm block: causal self-attention, cross-attention to slots, ReLU feed-forward."""

    def __init__(self, d_model, n_heads, dropout, gain):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            linear(d_model, d_model * 4, weight_init="kaiming"),
            nn.ReLU(),
            nn.Dropout(dropout),
            linear(d_model * 4, d_model, gain=gain),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, memory, causal):
        z = self.norm1(x)
        x = x + self.self_attn(z, z, z, attn_mask=causal, need_weights=False)[0]
        z = self.norm2(x)
        y, attent = self.cross_attn(z, memory, memory, need_weights=True)  # (b,h*w,n)
        x = x + y
        x = x + self.ffn(self.norm3(x))
        return x, attent


def linear(in_dim, out_dim, bias=True, weight_init="xavier", gain=1.0):
    m = nn.Linear(in_dim, out_dim, bias)
    if weight_init == "kaiming":
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
    else:
        nn.init.xavier_uniform_(m.weight, gain)
    if bias:
        nn.init.zeros_(m.bias)
    return m


class DINOViT(nn.Module):
    """DINO (v1) ViT from torch.hub, used on MOVi-C/D.
    Returns the patch tokens of the last block without the final LayerNorm, as in SPOT.
    """

    def __init__(self, model_name="dino_vitb16", in_size=224):
        super().__init__()
        self.model = pt.hub.load("facebookresearch/dino:main", model_name)
        self.out_size = in_size // self.model.patch_embed.patch_size

    def forward(self, input):
        """
        - input: shape=(b,c,h,w)
        - return: shape=(b,c,h/p,w/p)
        """
        x = self.model.prepare_tokens(input)
        for block in self.model.blocks:
            x = block(x)
        feature = x[:, 1:, :]  # remove the class token
        return rearrange(feature, "b (h w) c -> b c h w", h=self.out_size)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # The checkpoints of the paper hold an unused LayerNorm of an older version. Drop it.
        for k in [_ for _ in state_dict if _.startswith(prefix + "norm.")]:
            state_dict.pop(k)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
