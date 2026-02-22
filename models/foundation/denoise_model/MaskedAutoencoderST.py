import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Block


class MaskedAutoencoderST(nn.Module):
    def __init__(self, img_size=(64, 288), patch_size=(8, 12), in_chans=3,
                embed_dim=1024, depth=24, num_heads=16,
                decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # 1) 使用 timm 的 PatchEmbed, 给定一个 "初始" img_size=(64,288), 
        #    实际 forward 时会根据输入 x.shape 做动态卷积切分
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        # 2) 可学习位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=True)

        # 编码器
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # 解码器
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=True)
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size[0]*patch_size[1]*in_chans, bias=True)

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

        # 额外添加：在 forward_encoder 记录 (H, W), 供 unpatchify 使用
        self.last_hw = None

    def initialize_weights(self):
        # 不再使用正弦-余弦初始化; 改为对可学习参数做随机初始化
        nn.init.normal_(self.pos_embed, std=.02)
        nn.init.normal_(self.decoder_pos_embed, std=.02)
        nn.init.normal_(self.mask_token, std=.02)

        # 初始化 patch_embed.proj
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # ========== patchify / unpatchify ==========
    def patchify(self, imgs):
        """
        (B, 3, H, W) -> (B, N, p_s * p_t * 3),
        其中 p_s=patch_size[0], p_t=patch_size[1]
        """
        p_s = self.patch_embed.patch_size[0]
        p_t = self.patch_embed.patch_size[1]
        B, C, H, W = imgs.shape

        g_s = H // p_s  # 行方向 patch 个数
        g_t = W // p_t  # 列方向 patch 个数
        assert g_s * p_s == H and g_t * p_t == W, "图像尺寸与 patch_size 不整除"

        # 重排
        # (B, C, g_s, p_s, g_t, p_t)
        x = imgs.reshape(B, C, g_s, p_s, g_t, p_t)
        # (B, g_s, g_t, p_s, p_t, C)
        x = x.permute(0, 2, 4, 3, 5, 1)
        # (B, g_s*g_t, p_s*p_t*C)
        x = x.reshape(B, g_s*g_t, p_s*p_t*C)
        return x

    def unpatchify(self, x):
        """
        (B, num_patches, p_s*p_t*3) -> (B, 3, H, W)
        借助 self.last_hw = (H, W) 恢复原图大小
        """
        p_s = self.patch_embed.patch_size[0]
        p_t = self.patch_embed.patch_size[1]
        B, N, C = x.shape
        # C = p_s * p_t * 3
        assert C == p_s * p_t * 3, f"通道数与 patch_size 不匹配: {C} vs {p_s}*{p_t}*3"

        # 取出之前记录的 (H, W)
        if self.last_hw is None:
            raise ValueError("未记录 last_hw, 无法 unpatchify; 请在 forward 过程中先执行 forward_encoder")

        H, W = self.last_hw
        g_s = H // p_s
        g_t = W // p_t
        assert g_s*g_t == N, f"num_patches={N} 与 (H//p_s)*(W//p_t)={g_s*g_t} 不匹配"

        # x: (B, g_s*g_t, p_s*p_t*3) -> (B, g_s, g_t, p_s, p_t, 3)
        x = x.view(B, g_s, g_t, p_s, p_t, 3)
        # -> (B, 3, g_s, p_s, g_t, p_t)
        x = x.permute(0, 5, 1, 3, 2, 4)
        # -> (B, 3, g_s*p_s, g_t*p_t)
        imgs = x.reshape(B, 3, g_s*p_s, g_t*p_t)
        return imgs

    # ========== 两种 Mask 策略: random_masking & right_half_masking ==========

    def random_masking(self, x, mask_ratio):
        """
        x: (B, N, D)
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1,
            index=ids_keep.unsqueeze(-1).repeat(1, 1, D)
        )

        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def right_half_masking(self, x):
        """
        只 Mask 右半边对应的 token (保留左半边 token)
        """
        B, L, D = x.shape
        # 取出占位时计算的网格
        g_s, g_t = self.patch_embed.grid_size  # 例如 (8, 24)
        assert g_s*g_t == L, f"Token数 {L} != grid_size {g_s*g_t}"

        noise = torch.zeros((B, L), device=x.device)
        half_w = g_t // 2
        # 构造 noise: 左半边=0, 右半边=1
        for row in range(g_s):
            for col in range(half_w, g_t):
                idx = row*g_t + col
                noise[:, idx] = 1.0

        ids_shuffle = torch.argsort(noise, dim=1)  # (B, L)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        len_keep = half_w*g_s

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1,
            index=ids_keep.unsqueeze(-1).repeat(1, 1, D)
        )
        mask = torch.ones([B, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    # ========== 编码器、解码器、loss、forward ==========

    def forward_encoder(self, x, mask_ratio, mask_strategy='random'):
        """
        x: (B, 3, H, W)
        这里记录 (H, W) 以支持 unpatchify
        """
        B, C, H, W = x.shape
        self.last_hw = (H, W)  # 记录下原图大小

        # 动态切分: => (B, N_actual, embed_dim)
        x = self.patch_embed(x)
        N_actual = x.shape[1]

        # pos_embed 仅取前 N_actual
        x = x + self.pos_embed[:, :N_actual, :]

        # 根据策略 Mask
        if mask_strategy == 'random':
            x_masked, mask, ids_restore = self.random_masking(x, mask_ratio)
        elif mask_strategy == 'right_half':
            x_masked, mask, ids_restore = self.right_half_masking(x)
        else:
            raise ValueError(f"Unknown mask_strategy={mask_strategy}")

        # 编码器
        for blk in self.blocks:
            x_masked = blk(x_masked)
        x_masked = self.norm(x_masked)
        return x_masked, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)
        B, L, D = x.shape

        # 还原被 mask 掉的 token
        num_patches = ids_restore.shape[1]
        mask_tokens = self.mask_token.repeat(B, num_patches - L, 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        # 重排
        x_ = torch.gather(
            x_, dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1, D)
        )
        # 解码器位置编码
        x_ = x_ + self.decoder_pos_embed[:, :num_patches, :]

        # 解码器
        for blk in self.decoder_blocks:
            x_ = blk(x_)
        x_ = self.decoder_norm(x_)
        x_ = self.decoder_pred(x_)
        return x_

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: (B, 3, H, W)
        pred: (B, N, p_s*p_t*3)
        mask: (B, N)
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss
    
    def forward(self, imgs, mask_ratio=0.75, mask_strategy='random'):
        """
        主前向：返回 (loss, pred, mask)
        - imgs: (B, 3, H, W)
        - mask_ratio: 随机 Mask 比例（若 mask_strategy='random'）
        - mask_strategy: 'random' or 'right_half'
        """
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio, mask_strategy)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask
