import torch
import torch.nn as nn
import math
import timm
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import PatchEmbed, Mlp
import einops
import torch.utils.checkpoint
from dataset.pos import get_2d_sincos_pos_embed

# the xformers lib allows less memory, faster training and inference
try:
    import xformers
    import xformers.ops

    XFORMERS_IS_AVAILBLE = True
    print('xformers enabled')
except:
    XFORMERS_IS_AVAILBLE = False
    print('xformers disabled')


def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def patchify(imgs, patch_size):
    x = einops.rearrange(imgs, 'B C (h p1) (w p2) -> B (h w) (p1 p2 C)', p1=patch_size, p2=patch_size)
    return x


def unpatchify(x, channels=3):
    patch_size = int((x.shape[2] // channels) ** 0.5)
    h = w = int(x.shape[1] ** .5)
    assert h * w == x.shape[1] and patch_size ** 2 * channels == x.shape[2]
    x = einops.rearrange(x, 'B (h w) (p1 p2 C) -> B C (h p1) (w p2)', h=h, p1=patch_size, p2=patch_size)
    return x


class Shift(nn.Module):
    def __init__(self, dim, proj_drop=0.):
        super().__init__()
        self.g = dim // 12

    def forward(self, x):
        g = self.g
        x[:, :-1, :g] = x[:, 1:, :g]
        x[:, 1:, g:2 * g] = x[:, :-1, g:2 * g]
        x[:, :-2, 2 * g:3 * g] = x[:, 2:, 2 * g:3 * g]
        x[:, 2:, 3 * g:4 * g] = x[:, :-2, 3 * g:4 * g]
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, L, C = x.shape

        qkv = self.qkv(x)
        if XFORMERS_IS_AVAILBLE:  # the xformers lib allows less memory, faster training and inference
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B L H D', K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B L H D
            x = xformers.ops.memory_efficient_attention(q, k, v)
            x = einops.rearrange(x, 'B L H D -> B L (H D)', H=self.num_heads)
        else:
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, L, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip=False, use_checkpoint=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None
        self.use_checkpoint = use_checkpoint

    def forward(self, x, skip=None):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, skip)
        else:
            return self._forward(x, skip)

    def _forward(self, x, skip=None):
        if self.skip_linear is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# =========================================================================
# === 新增：用於語意蒸餾的純淨 CrossAttention (無距離遮罩，完全輕量化) ===
# =========================================================================
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        # x: 已知區域特徵 (Key/Value), y: 絕對座標 (Query)
        B, L, C = x.shape
        _, K, _ = y.shape

        q = self.q(y)
        k, v = self.k(x), self.v(x)

        q = einops.rearrange(q, 'B L (H D) -> B H L D', H=self.num_heads)
        k = einops.rearrange(k, 'B L (H D) -> B H L D', H=self.num_heads)
        v = einops.rearrange(v, 'B L (H D) -> B H L D', H=self.num_heads)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = self.proj_drop(self.proj((attn @ v).transpose(1, 2).reshape(B, K, C)))
        return out


class UViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm, mlp_time_embed=False, num_classes=-1,
                 use_checkpoint=False, conv=True, skip=True):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_classes = num_classes
        self.in_chans = in_chans

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans * 2, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        ) if mlp_time_embed else nn.Identity()

        self.masked_embed = nn.Parameter(torch.randn(embed_dim))

        if self.num_classes > 0:
            self.extras = 2
        else:
            self.extras = 1

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)

        # ============================================================
        # === 新增：輕量級語意預測器 (Semantic Predictor) ===
        # ============================================================
        self.semantic_proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.semantic_cross_attn = CrossAttention(dim=embed_dim, num_heads=num_heads)

        self.in_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)])

        self.mid_block = Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            norm_layer=norm_layer, use_checkpoint=use_checkpoint)

        self.out_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, skip=skip, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)])

        self.norm = norm_layer(embed_dim)
        self.patch_dim = patch_size ** 2 * in_chans
        self.decoder_pred = nn.Linear(embed_dim, self.patch_dim, bias=True)
        self.final_layer = nn.Conv2d(self.in_chans, self.in_chans, 3, padding=1) if conv else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(math.sqrt(self.pos_embed.shape[1])),
                                            cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    # ============================================================
    # === 新增：L2 特徵蒸餾 Loss 計算函數 (供 train_ldm.py 呼叫) ===
    # ============================================================
    def forward_feature_loss(self, anchor_view, target_pos, target_view):
        """用 Cosine Similarity + L1 模擬對比學習，並回傳 Batch 中每張圖獨立的 Loss"""
        anchor_emb = self.semantic_proj(anchor_view).flatten(2).transpose(1, 2)
        query_pos = target_pos + self.masked_embed
        target_semantic_pred = self.semantic_cross_attn(anchor_emb, query_pos)
        target_emb_gt = self.semantic_proj(target_view).flatten(2).transpose(1, 2)

        # 1. Cosine Similarity (注意：在 L 維度做平均，保留 B 維度)
        cos_sim = torch.nn.functional.cosine_similarity(target_semantic_pred, target_emb_gt.detach(), dim=-1) # [B, L]
        cos_loss = 1.0 - cos_sim.mean(dim=1) # 形狀: [B]

        # 2. L1 Loss (注意：reduction='none' 才能保留 B 維度)
        l1_loss = torch.nn.functional.l1_loss(target_semantic_pred, target_emb_gt.detach(), reduction='none') # [B, L, D]
        l1_loss = l1_loss.mean(dim=[1, 2]) # 形狀: [B]

        # 回傳每張圖片各自的總 Loss，形狀為 [B]
        loss = 1.0 * cos_loss + 0.1 * l1_loss
        return loss

    def forward(self, x, conditions, timesteps):
        anchor_view, target_pos = conditions

        # ============================================================
        # === 新增：擴散去噪前，先生成語意藍圖 ===
        # ============================================================
        # 提取已知區域的基礎特徵
        anchor_emb = self.semantic_proj(anchor_view).flatten(2).transpose(1, 2)
        # 生成語意藍圖
        query_pos = target_pos + self.masked_embed
        target_semantic_pred = self.semantic_cross_attn(anchor_emb, query_pos)

        x = torch.cat([anchor_view, x], dim=1)  # batch, 3+1, H, W
        x = self.patch_embed(x)
        target_pos = target_pos + self.masked_embed

        # add time embeddings
        time_token = self.time_embed(timestep_embedding(timesteps, self.embed_dim))
        time_token = time_token.unsqueeze(dim=1)
        x = x + self.pos_embed
        B, L, D = x.shape

        # ============================================================
        # === 魔法融合：絕對座標 + 語意藍圖 + 雜訊 ===
        # ============================================================
        x = target_pos + x + target_semantic_pred

        # add conditions
        x = torch.cat((time_token, x), dim=1)

        skips = []
        for blk in self.in_blocks:
            x = blk(x)
            skips.append(x)

        x = self.mid_block(x)

        for blk in self.out_blocks:
            x = blk(x, skips.pop())

        x = self.norm(x)
        x = self.decoder_pred(x)
        x = x[:, -L:, :]
        x = unpatchify(x, self.in_chans)
        x = self.final_layer(x)
        return x