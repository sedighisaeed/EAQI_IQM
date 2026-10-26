# -*- coding: utf-8 -*-
"""E-AQI prototype model (Manuscript Sections III-A, III-E, III-F).

Prototype status (stated in Section III-A of the revised manuscript): the
supplied code contains SIMPLIFIED edge and saliency modules and does NOT
certify implementation equivalence to a full production model.  Architecture
choices (ConvNeXt / MobileNetV3 encoders) are candidates to evaluate; no
parameter or latency budget is claimed here — profiling must be measured
(Section VI-E, latency.py / torchinfo).

Design points implemented:
  * common H/4 x W/4 lattice for clipping / edge / context features BEFORE
    the 1x1 projection and stride-4 patch embedding (16-px token stride);
  * cross-attention with queries from backbone features at RGB strides
    4/8/16 and keys/values from the map-token lattice (lengths may differ);
  * FPN-style combination + global pooling + MLP + sigmoid severity output;
  * optional auxiliary heads: family (5), severity class (3), spatial map,
    edge-interaction (binary).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as Fn


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 3, stride=stride, padding=1, groups=cin,
                            bias=False)
        self.bn = nn.BatchNorm2d(cin)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.pw(self.act(self.bn(self.dw(x)))))


class CrossAttentionFusion(nn.Module):
    """Cross-attention: queries = backbone tokens, keys/values = map tokens."""

    def __init__(self, q_dims: tuple[int, ...], kv_dim: int = 48,
                 embed_dim: int = 256, heads: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.heads = heads
        self.kv_proj = nn.Conv2d(kv_dim, embed_dim, kernel_size=1)
        self.q_projs = nn.ModuleList(
            nn.Conv2d(d, embed_dim, kernel_size=1) for d in q_dims)
        self.out_projs = nn.ModuleList(
            nn.Conv2d(embed_dim, d, kernel_size=1) for d in q_dims)

    def forward(self, queries: list[torch.Tensor],
                maps: torch.Tensor) -> list[torch.Tensor]:
        """queries: feature maps [N,C,h,w] at strides 4/8/16;
        maps: [N, kv_dim, H/4, W/4] lattice features.
        Returns residual-added outputs at each query resolution."""
        k = self.kv_proj(maps)                                  # [N,E,Hk,Wk]
        N, E, Hk, Wk = k.shape
        kv = k.flatten(2).transpose(1, 2)                       # [N,Hk*Wk,E]
        outs = []
        for q_in, q_proj, out_proj in zip(queries, self.q_projs, self.out_projs):
            q = q_proj(q_in)                                    # [N,E,h,w]
            _, _, h, w = q.shape
            qt = q.flatten(2).transpose(1, 2)                   # [N, h*w, E]
            attn = torch.nn.functional.scaled_dot_product_attention(
                qt, kv, kv,
                dropout_p=0.0,
                scale=(self.embed_dim // self.heads) ** -0.5)
            attn = attn.transpose(1, 2).reshape(N, E, h, w)
            outs.append(q_in + out_proj(attn))                  # residual add
        return outs


class PrototypeEAQI(nn.Module):
    """Simplified prototype combining corrected operators with a compact
    conv encoder and multi-scale cross-attention fusion."""

    def __init__(self, in_ch: int = 3, base: int = 32,
                 aux_heads: bool = True):
        super().__init__()
        self.aux_heads = aux_heads

        # encoder stages at strides 2 and 4 (queries at stride 4/8/16 come
        # from stage outputs + a stride-8 downsample of stage 2)
        self.stem = DepthwiseSeparableConv(in_ch, base, stride=2)     # /2
        self.stage2 = DepthwiseSeparableConv(base, base * 2, stride=2)  # /4
        self.stage3 = DepthwiseSeparableConv(base * 2, base * 4, stride=2)  # /8
        self.stage4 = nn.Sequential(  # /16
            nn.MaxPool2d(2),
            DepthwiseSeparableConv(base * 4, base * 8),
        )

        # branch projections to a common lattice feature (kv_dim = 48)
        self.clip_proj = nn.Conv2d(3, 12, 1)      # per-channel saturation maps
        self.edge_proj = nn.Conv2d(1, 8, 1)       # edge response (simple Sobel init)
        self.ctx_proj = nn.Conv2d(1, 28, 1)       # entropy + saliency placeholder

        kv_dim = 12 + 8 + 28
        self.fusion = CrossAttentionFusion(
            q_dims=(base * 2, base * 4, base * 8), kv_dim=kv_dim,
            embed_dim=256, heads=4)

        # FPN-style head: lateral 1x1 projections to a common width,
        # then a 3x3 smoothing conv on that width
        width = base * 2
        self.fpn_lateral = nn.ModuleList([
            nn.Conv2d(base * 2, width, 1),
            nn.Conv2d(base * 4, width, 1),
            nn.Conv2d(base * 8, width, 1),
        ])
        self.fpn_smooth = nn.ModuleList([
            nn.Conv2d(width, width, 3, padding=1) for _ in range(3)
        ])
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base * 2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),                          # severity s in [0, 1]
        )

        if aux_heads:
            self.family_head = nn.Linear(base * 2, 5)          # 5 families
            self.severity_head = nn.Linear(base * 2, 3)        # 3 classes
            self.spatial_head = nn.Conv2d(base * 2, 1, 1)      # /4 map
            self.edge_head = nn.Linear(base * 2, 1)            # binary

        # fixed Sobel edge detector (simplified stand-in for PiDiNet)
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        sobel_y = sobel_x.T
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_y.view(1, 1, 3, 3))

    def edge_response(self, gray: torch.Tensor) -> torch.Tensor:
        gx = Fn.conv2d(gray, self.sobel_x, padding=1)
        gy = Fn.conv2d(gray, self.sobel_y, padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-12)

    def forward(self, images: torch.Tensor) -> dict:
        # normalized input assumed in [0, 1]; code values for saturation map
        s2 = self.stem(images)                       # /2
        s4 = self.stage2(s2)                         # /4  (queries)
        s8 = self.stage3(s4)                         # /8  (queries)
        s16 = self.stage4(s8)                        # /16 (queries)

        # ---- branch maps on the common H/4 x W/4 lattice (Section III-E)
        H4, W4 = s4.shape[-2:]
        luma = images.mean(dim=1, keepdim=True)                       # [N,1,H,W]

        # clipping map (corrected two-sided operator, differentiable path):
        # piecewise-soft two-sided response with fixed 8-bit-equivalent
        # cutoffs (adaptive selection is a hard branch; use smooth surrogate
        # from surrogate.py when input-gradient differentiability matters)
        x01 = images.clamp(0, 1)
        lo, hi = 5.0 / 255.0, 250.0 / 255.0
        alpha = 0.5 * 255.0
        a_lo = torch.sigmoid(alpha * (lo - x01))
        a_hi = torch.sigmoid(alpha * (x01 - hi))
        clip_maps = 1.0 - (1.0 - a_lo) * (1.0 - a_hi)                 # [N,3,H,W]
        clip_lat = Fn.adaptive_avg_pool2d(clip_maps, (H4, W4))

        edge = self.edge_response(luma)                               # [N,1,H,W]
        edge_lat = Fn.adaptive_avg_pool2d(edge, (H4, W4)) / edge.max().clamp(min=1e-6)

        # context placeholder: local entropy (numpy path documented in
        # entropy.py; differentiable soft-binned version required for
        # end-to-end E-AQI-Loss — see Section III-H audit note)
        ctx_lat = torch.tanh(Fn.avg_pool2d(luma - luma.mean(), 3, stride=4,
                                           padding=1, count_include_pad=False))
        ctx_lat = Fn.interpolate(ctx_lat, size=(H4, W4), mode="bilinear",
                                 align_corners=False)                 # [N,1,H4,W4]

        maps = torch.cat([self.clip_proj(clip_lat),
                          self.edge_proj(edge_lat),
                          self.ctx_proj(ctx_lat)], dim=1)             # [N,48,H4,W4]

        fused = self.fusion([s4, s8, s16], maps)

        # ---- FPN-style combination (top-down: /16 -> /8 -> /4)
        p = None
        feats = []
        rev_fused = list(reversed(fused))          # [s16, s8, s4]
        rev_lat = list(reversed(self.fpn_lateral))
        for f, lat, smooth in zip(rev_fused, rev_lat, self.fpn_smooth):
            p = lat(f) if p is None else lat(f) + Fn.interpolate(
                p, size=f.shape[-2:], mode="bilinear", align_corners=False)
            p = smooth(p)
            feats.insert(0, p)                      # feats[0] is the /4 map

        severity = self.regressor(p).squeeze(-1)                      # [N]

        out = {"severity": severity}
        if self.aux_heads:
            pooled = p.mean(dim=(2, 3))                               # [N, C]
            out["family_logits"] = self.family_head(pooled)
            out["severity_logits"] = self.severity_head(pooled)
            out["spatial_map"] = self.spatial_head(feats[0])          # /4 map
            out["edge_logits"] = self.edge_head(pooled).squeeze(-1)
        return out


if __name__ == "__main__":
    net = PrototypeEAQI()
    imgs = torch.rand(2, 3, 128, 128)
    o = net(imgs)
    for k, v in o.items():
        print(k, tuple(v.shape))
    print("params (M):",
          sum(p.numel() for p in net.parameters()) / 1e6)
