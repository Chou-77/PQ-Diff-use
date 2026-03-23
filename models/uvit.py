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

def window_partition(x, window_size):
    """將 2D 特徵圖切割成局部視窗"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """將局部視窗還原回 2D 特徵圖"""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
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
        x[:, 1:, g:2*g] = x[:, :-1, g:2*g]
        x[:, :-2, 2*g:3*g] = x[:, 2:, 2*g:3*g]
        x[:, 2:, 3*g:4*g] = x[:, :-2, 3*g:4*g]
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
        # 恢復標準的全域注意力，不切 Window，也不管 H, W
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip=False, use_checkpoint=False,
                 window_size=None):  # 保留 window_size 參數但不使用，以免報錯
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.skip_linear = nn.Linear(2 * dim, dim) if skip else None
        self.use_checkpoint = use_checkpoint

    def forward(self, x, H=None, W=None, skip=None):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, skip)
        else:
            return self._forward(x, skip)

    def _forward(self, x, skip=None):
        if self.skip_linear is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))

        # 恢復原始邏輯：time_token 必須與空間 token 一起進入注意力層運算！
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

# --- 修改 2：加入距離遮罩與稀疏採樣的 CrossAttention (方法一與二) ---
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sparse_ratio=2):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # 用於稀疏全局注意力的空間降維層
        self.sparse_ratio = sparse_ratio
        self.sr = nn.Conv2d(dim, dim, kernel_size=sparse_ratio,
                            stride=sparse_ratio) if sparse_ratio > 1 else nn.Identity()

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # ==========================================
        # 【加入這兩行：零初始化技巧】
        # 強制讓這個全新的注意力機制在剛開始訓練時輸出為 0
        # 避免隨機初始化的雜訊破壞原本 8 萬步的預訓練權重
        # ==========================================
        # nn.init.zeros_(self.proj.weight)
        # nn.init.zeros_(self.proj.bias)

    def forward(self, x, y, H, W, distance_mask=None):
        B, L, C = x.shape
        _, K, _ = y.shape

        q = self.q(y)

        # 稀疏全局注意力：將 Anchor 區塊特徵進行降維提取關鍵 Token
        if getattr(self, 'sr', None) is not None and self.sparse_ratio > 1:
            x_2d = x.transpose(1, 2).view(B, C, H, W)
            x_sparse = self.sr(x_2d).view(B, C, -1).transpose(1, 2)
            k, v = self.k(x_sparse), self.v(x_sparse)
        else:
            k, v = self.k(x), self.v(x)

        q = einops.rearrange(q, 'B L (H D) -> B H L D', H=self.num_heads)
        k = einops.rearrange(k, 'B L (H D) -> B H L D', H=self.num_heads)
        v = einops.rearrange(v, 'B L (H D) -> B H L D', H=self.num_heads)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # 邊界權重強化 (BWCA)：套用距離衰減遮罩
        if distance_mask is not None:
            # 確保 distance_mask 維度為 (B, 1, K, L_sparse) 或可廣播的形狀
            attn = attn + distance_mask

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, K, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out



class UViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm, mlp_time_embed=False, num_classes=-1,
                 use_checkpoint=False, conv=True, skip=True,
                 window_size=6, sparse_ratio=2):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
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

        # 【新增：把剛剛定義的 CrossAttention 實體化裝上來】
        self.cross_attn = CrossAttention(embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            sparse_ratio=sparse_ratio)

        self.in_blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                norm_layer=norm_layer, use_checkpoint=use_checkpoint,
                window_size=window_size)
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

    # 【新增：產生距離遮罩的空殼函數，未來可以換成真正的相對位置演算法】
    def _get_distance_mask(self, H, W, sparse_ratio, device, threshold=4.0, penalty=-10000.0):
        """
        計算並回傳邊界距離遮罩 (Boundary-Weighted Mask)

        參數:
        - H, W: 原始特徵圖的高與寬
        - sparse_ratio: CrossAttention 中 Key/Value 的降維比例
        - threshold: 距離閥值 (歐幾里得距離)，小於此距離的注意力保留，大於的則被遮蔽
        - penalty: 賦予過遠像素的極小懲罰值 (Softmax 後趨近於 0)
        """
        # 步驟 1：產生 Target (目標區 Query) 的 2D 網格座標
        # y_q, x_q 形狀均為 (H, W)
        y_q, x_q = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )
        # 將座標攤平並組合成 (N, 2) 的矩陣，N = H * W
        coords_q = torch.stack([y_q.flatten(), x_q.flatten()], dim=-1).float()

        # 步驟 2：產生 Anchor (已知區 Key) 的 2D 網格座標
        if sparse_ratio > 1:
            H_k, W_k = H // sparse_ratio, W // sparse_ratio
            # 考慮降維卷積 (stride=sparse_ratio) 的感受野中心點偏移
            # 例如 ratio=2 時，index 0 代表原始 0,1 的中心 (0.5)
            y_k = torch.arange(H_k, device=device) * sparse_ratio + (sparse_ratio - 1) / 2.0
            x_k = torch.arange(W_k, device=device) * sparse_ratio + (sparse_ratio - 1) / 2.0

            y_k, x_k = torch.meshgrid(y_k, x_k, indexing='ij')
            coords_k = torch.stack([y_k.flatten(), x_k.flatten()], dim=-1).float()
        else:
            # 如果沒有降維，Anchor 座標就跟 Target 一模一樣
            coords_k = coords_q

        # 步驟 3：計算兩兩之間的「歐幾里得距離」(Euclidean Distance)
        # coords_q: (N, 2), coords_k: (M, 2)
        # dist_matrix 輸出形狀: (N, M)，代表每一個 Query 對應每一個 Key 的實際物理距離
        dist_matrix = torch.cdist(coords_q, coords_k, p=2.0)

        # 步驟 4：設定閥值 (Threshold)，將距離轉換成權重遮罩
        mask = torch.zeros_like(dist_matrix)  # 預設距離近的權重為 0 (不影響原始注意力)

        # 距離大於閥值的地方，填入 -10000
        mask[dist_matrix > threshold] = penalty

        # 步驟 5：輸出成對應形狀的 mask
        # 為了能與 Attention 分數 (B, Heads, Query_Len, Key_Len) 相加
        # 我們將 mask 擴充維度成 (1, 1, N, M)
        mask = mask.unsqueeze(0).unsqueeze(0)

        return mask

    def forward(self, x, conditions, timesteps):
        anchor_view, target_pos = conditions

        x = torch.cat([anchor_view, x], dim=1)  # batch, 3+1, H, W
        x = self.patch_embed(x)

        # 【新增：動態計算特徵圖的高(H)與寬(W)】
        H = W = int(math.sqrt(x.shape[1]))

        target_pos = target_pos + self.masked_embed
        time_token = self.time_embed(timestep_embedding(timesteps, self.embed_dim))
        time_token = time_token.unsqueeze(dim=1)

        x = x + self.pos_embed
        B, L, D = x.shape

        #x = target_pos + x

        # 這裡原本是 x = target_pos + x，這是錯的 (兩者代表不同意義，不能直接相加)
        # 把它移除，改在中間層用 CrossAttention 處理

        # 把時間標籤貼到最前面 (長度變成 L+1)
        x = torch.cat((time_token, x), dim=1)
        # target_pos 不需要加 time_token，因為它只是用來查詢 (Query) 的位置特徵

        skips = []
        for blk in self.in_blocks:
            # 【修改：傳入 H 和 W】
            x = blk(x, H=H, W=W)
            skips.append(x)

        # 【新增：在中間層插入邊界權重強化交叉注意力 (BWCA)】
        # 1. 為了不影響 x 裡面的時間標籤，我們先把 x 的空間特徵分離出來當作參考 (Key, Value)
        x_spatial = x[:, 1:, :]

        # 2. 計算稀疏化後的長度
        sparse_L = (H // self.cross_attn.sparse_ratio) * (W // self.cross_attn.sparse_ratio)

        # 3. 取得距離遮罩
        # 3. 取得距離遮罩 (傳入空間維度與降維比例)
        dist_mask = self._get_distance_mask(
            H=H,
            W=W,
            sparse_ratio=self.cross_attn.sparse_ratio,
            device=x.device,
            threshold=4.0,  # 你可以根據實驗結果調整此參數 (例如 3.0 ~ 8.0)
            penalty=-10000.0
        )

        # 4. 讓 target_pos 去跟 x_spatial 做交叉注意力
        target_features = self.cross_attn(x_spatial, target_pos, H=H, W=W, distance_mask=dist_mask)

        # 5. 將運算結果加回 x 身上 (我們把 target_features 視為對 x_spatial 的特徵補充)
        #    注意：要跟原本的空間特徵維度對齊
        x = x + torch.cat([torch.zeros_like(x[:, :1, :]), target_features], dim=1)

        # 繼續進入下一關
        x = self.mid_block(x, H=H, W=W)

        for blk in self.out_blocks:
            # 【修改：傳入 H、W 和 skip】
            x = blk(x, H=H, W=W, skip=skips.pop())

        x = self.norm(x)
        x = self.decoder_pred(x)
        x = x[:, -L:, :]  # 濾除第 0 個位置的 time_token
        x = unpatchify(x, self.in_chans)
        x = self.final_layer(x)
        return x
