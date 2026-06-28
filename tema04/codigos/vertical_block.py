import torch
import torch.nn as nn


class VerticalSpatialBlock(nn.Module):
    N_LEVEL_CHANNELS = 6
    L_DIM = 3

    def __init__(
        self,
        in_channels: int = 7,
        out_channels: int = 19,
        kernel_size: tuple = (3, 3, 3),
        compression: str = "attention",  # agora padrão
    ):
        super().__init__()

        assert out_channels > 1
        assert in_channels == 7

        self.compression = compression
        n_lev = self.N_LEVEL_CHANNELS
        L = self.L_DIM
        C_lev = out_channels - 1

        padding = tuple(k // 2 for k in kernel_size)

        self.conv3d = nn.Conv3d(n_lev, C_lev, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(C_lev)
        self.relu = nn.ReLU(inplace=True)

        self.attn = nn.Sequential(
            nn.Conv3d(C_lev, C_lev, kernel_size=1),
            nn.Softmax(dim=2)
        )

        # shortcut mais estável
        self.shortcut = nn.Conv2d(n_lev * L, C_lev, kernel_size=1, bias=False)

        # peso residual aprendível
        self.alpha = nn.Parameter(torch.tensor(0.5))

        # tp branch
        self.tp_conv = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )

        self._init_weights(n_lev, L)

    def _init_weights(self, n_lev, L):
        with torch.no_grad():
            nn.init.xavier_uniform_(self.shortcut.weight)
            nn.init.kaiming_normal_(self.conv3d.weight, nonlinearity="relu")
            nn.init.kaiming_normal_(self.tp_conv[0].weight, nonlinearity="relu")

    def forward(self, x):
        B, C, T, H, W, L = x.shape

        tp = x[:, 0:1, :, :, :, 0]
        x_lev = x[:, 1:, :, :, :, :]

        # tp branch
        tp_bt = tp.permute(0, 2, 1, 3, 4).reshape(B * T, 1, H, W)
        tp_out = self.tp_conv(tp_bt)

        # level vars
        x_lev = x_lev.permute(0, 2, 1, 5, 3, 4)
        x_lev_bt = x_lev.reshape(B * T, 6, L, H, W)

        out = self.relu(self.bn(self.conv3d(x_lev_bt)))

        attn = self.attn(out)
        out = (out * attn).sum(dim=2)

        # shortcut
        sc = x_lev_bt.reshape(B * T, 6 * L, H, W)
        sc = self.shortcut(sc)

        # 🔥 combinação balanceada
        out = self.alpha * out + (1 - self.alpha) * sc

        # reshape
        C_lev = out.shape[1]
        out = out.reshape(B, T, C_lev, H, W).permute(0, 2, 1, 3, 4)
        tp_out = tp_out.reshape(B, T, 1, H, W).permute(0, 2, 1, 3, 4)

        return torch.cat([out, tp_out], dim=1)


class STConvS2S_LV(nn.Module):
    def __init__(
        self,
        stconvs2s_input_shape,
        num_layers=3,
        hidden_dim=32,
        kernel_size=5,
        device="cpu",
        dropout=0.5,
        step=5,
        out_channels_lv=19,
        kernel_lv=(3, 3, 3),
        compression="attention",
    ):
        super().__init__()

        from model.stconvs2s import STConvS2S_R

        self.vblock = VerticalSpatialBlock(
            in_channels=7,
            out_channels=out_channels_lv,
            kernel_size=kernel_lv,
            compression=compression,
        )

        self.stconvs2s = STConvS2S_R(
            stconvs2s_input_shape,
            num_layers,
            hidden_dim,
            kernel_size,
            device,
            dropout,
            step,
        )

    def forward(self, x):
        x = self.vblock(x)
        x = x.unsqueeze(-1)
        return self.stconvs2s(x)