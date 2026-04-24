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


def rotate_half(x):
    """將特徵的最後一個維度切半並旋轉"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_emb(x, freqs):
    """對 Query 和 Key 獨立應用 X 與 Y 的旋轉位置編碼"""
    # 確保切分為 X 子空間與 Y 子空間
    d_2 = x.shape[-1] // 2
    x_x = x[..., :d_2]  # X 軸特徵
    x_y = x[..., d_2:]  # Y 軸特徵

    freqs_x = freqs[..., :d_2]
    freqs_y = freqs[..., d_2:]

    # 各自在獨立的子空間內旋轉 (X 不會污染 Y)
    out_x = x_x * freqs_x.cos() + rotate_half(x_x) * freqs_x.sin()
    out_y = x_y * freqs_y.cos() + rotate_half(x_y) * freqs_y.sin()

    return torch.cat([out_x, out_y], dim=-1)

class RoPE2D(nn.Module):
    def __init__(self, dim, base=10000.0):
        super().__init__()
        # 2D 座標有 X 和 Y，每個佔用一半的 head dimension
        self.half_dim = dim // 2
        # 建立頻率衰減 (維度為 half_dim / 2)
        inv_freq = 1.0 / (base ** (torch.arange(0, self.half_dim, 2).float() / self.half_dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, coords):
        """
        coords: [B, L, 2] 代表每個 Patch 的 (X, Y) 網格絕對/相對座標
        回傳 freqs: [B, L, dim]
        """
        x_coords = coords[..., 0].float()
        y_coords = coords[..., 1].float()

        # 計算 X 和 Y 各自的旋轉頻率 [B, L, half_dim / 2]
        freqs_x = torch.einsum("bl,f->blf", x_coords, self.inv_freq)
        freqs_y = torch.einsum("bl,f->blf", y_coords, self.inv_freq)

        # 為了搭配 rotate_half，直接複製拼接 [f1, f2] -> [f1, f2, f1, f2]
        freqs_x = torch.cat([freqs_x, freqs_x], dim=-1) # [B, L, half_dim]
        freqs_y = torch.cat([freqs_y, freqs_y], dim=-1) # [B, L, half_dim]

        # 拼接 X 和 Y 的特徵形成完整的 head dimension [B, L, dim]
        freqs = torch.cat([freqs_x, freqs_y], dim=-1)
        return freqs

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

    def forward(self, x, freqs=None):  # <-- 新增 freqs 參數
        B, L, C = x.shape
        qkv = self.qkv(x)

        if XFORMERS_IS_AVAILBLE:
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B L H D', K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # [B, L, H, D]

            # --- 套用 RoPE ---
            if freqs is not None:
                freqs_xf = freqs.unsqueeze(2)  # 變成 [B, L, 1, D] 以適應 xformers 形狀
                q = apply_rotary_emb(q, freqs_xf)
                k = apply_rotary_emb(k, freqs_xf)
            # -----------------

            x = xformers.ops.memory_efficient_attention(q, k, v)
            x = einops.rearrange(x, 'B L H D -> B L (H D)', H=self.num_heads)
        else:
            qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads)
            q, k, v = qkv[0], qkv[1], qkv[2]  # [B, H, L, D]

            # --- 套用 RoPE ---
            if freqs is not None:
                freqs_std = freqs.unsqueeze(1)  # 變成 [B, 1, L, D] 以適應標準 attention
                q = apply_rotary_emb(q, freqs_std)
                k = apply_rotary_emb(k, freqs_std)
            # -----------------

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

    def forward(self, x, skip=None, freqs=None):  # <-- 傳遞 freqs
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, skip, freqs)
        else:
            return self._forward(x, skip, freqs)

    def _forward(self, x, skip=None, freqs=None):
        if self.skip_linear is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))

        # 注意：將 freqs 傳給 Attention
        attn_out = self.attn(self.norm1(x), freqs=freqs)
        x = x + attn_out
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
        self.num_features = self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.in_chans = in_chans

        # 注意：你的模型將 Anchor View (3通道) 和 x (3通道) 合併，所以 in_chans 要 * 2
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans * 2, embed_dim=embed_dim)

        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        ) if mlp_time_embed else nn.Identity()

        # 這裡初始化 2D RoPE (計算 head_dim)
        head_dim = embed_dim // num_heads
        self.rope2d = RoPE2D(dim=head_dim)

        # 拔除 self.pos_embed 和 self.cross_attn (或是將 pos_embed 當成純粹的可學習 bias 留下也可以，這裡為乾淨直接拔除)

        self.in_blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  qk_scale=qk_scale, norm_layer=norm_layer, use_checkpoint=use_checkpoint)
            for _ in range(depth // 2)])

        self.mid_block = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale, norm_layer=norm_layer,
                               use_checkpoint=use_checkpoint)

        self.out_blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  qk_scale=qk_scale, norm_layer=norm_layer, skip=skip, use_checkpoint=use_checkpoint)
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

    def forward(self, x, conditions, timesteps):
        # ⚠️ 重點修改：傳入的不再是 Target_pos 的高維 Embedding，而是實體的網格座標 (H, W)
        # pos_coords 的預期形狀為: [B, L, 2] (例如 [Batch, 196, 2])
        anchor_view, pos_coords = conditions

        x = x.float()
        anchor_view = anchor_view.float()
        pos_coords = pos_coords.float()

        # 影像特徵融合
        x = torch.cat([anchor_view, x], dim=1)  # batch, 6, H, W
        x = self.patch_embed(x)

        # 1. 加入時間特徵
        time_token = self.time_embed(timestep_embedding(timesteps, self.embed_dim)).unsqueeze(dim=1)
        x = torch.cat((time_token, x), dim=1)

        # 2. 計算 RoPE 的頻率矩陣
        # 因為前面 concat 了一個 time_token，所以我們幫 time_token 的座標補上 (0,0) 以保持形狀對齊
        B, L_coords, _ = pos_coords.shape
        time_coords = torch.zeros((B, 1, 2), device=x.device, dtype=x.dtype)
        full_coords = torch.cat([time_coords, pos_coords], dim=1)  # [B, 1+L, 2]

        freqs = self.rope2d(full_coords)  # [B, 1+L, head_dim]

        # 3. 進入帶有 RoPE 的 Transformer Blocks
        skips = []
        for blk in self.in_blocks:
            x = blk(x, freqs=freqs)
            skips.append(x)

        x = self.mid_block(x, freqs=freqs)

        for blk in self.out_blocks:
            x = blk(x, skip=skips.pop(), freqs=freqs)

        x = self.norm(x)
        x = self.decoder_pred(x)

        # 捨棄 time_token
        x = x[:, 1:, :]

        x = unpatchify(x, self.in_chans)
        x = self.final_layer(x)
        return x