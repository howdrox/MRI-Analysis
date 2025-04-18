# unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=(64,128,256,512)):
        super().__init__()
        self.features = features

        # Downsampling path
        self.downs = nn.ModuleList([
            self._block(
                in_channels if i == 0 else features[i-1],
                features[i]
            )
            for i in range(len(self.features))
        ])

        # Bottleneck
        self.bottleneck = self._block(self.features[-1], self.features[-1]*2)

        # Upsampling convtransposes
        self.ups = nn.ModuleList([
            nn.ConvTranspose2d(
                (self.features[-1]*2 if i == len(self.features)-1 else features[i+1]),
                features[i], kernel_size=2, stride=2
            )
            for i in reversed(range(len(self.features)))
        ])

        # Convs after concat
        self.up_convs = nn.ModuleList([
            self._block(features[i]*2, features[i])
            for i in reversed(range(len(self.features)))
        ])

        # Final 1×1 conv
        self.final = nn.Conv2d(self.features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        # Encoder
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = F.max_pool2d(x, kernel_size=2)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for up, conv, skip in zip(self.ups, self.up_convs, reversed(skips)):
            x = up(x)
            # center‐crop skip to match x
            if x.shape != skip.shape:
                skip = self._center_crop(skip, x.shape)
            x = conv(torch.cat([skip, x], dim=1))

        return torch.sigmoid(self.final(x))

    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def _center_crop(self, layer, target_shape):
        _, _, h, w = layer.size()
        _, _, th, tw = target_shape
        dy = (h - th) // 2
        dx = (w - tw) // 2
        return layer[:, :, dy:dy+th, dx:dx+tw]
