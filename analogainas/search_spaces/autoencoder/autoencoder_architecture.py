import torch
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, depth, padding='same'):
        super().__init__()
        layers = []
        for i in range(depth):
            conv = nn.Conv2d(
                in_channels=(in_channels if i == 0 else out_channels),
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=(stride if i == 0 else 1),
                padding=(kernel_size // 2 if padding == 'same' else 0)
            )
            layers.append(conv)
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class AutoEncoder(nn.Module):
    def __init__(self, config, input_channels=3, input_size=(32,32)):
        super().__init__()
        self.config = config
        self.embedding_dim = config["embedding_dim"]

        H, W = input_size
        current_channels = input_channels

        self.encoder_specs = []
        encoder_blocks = []
        for i in range(1, 4):
            depth = config[f"encoder_convblock{i}_depth"]
            kernel_size = config[f"encoder_convblock{i}_kernel_size"]
            filters = config[f"encoder_convblock{i}_filters"]
            stride = config[f"encoder_convblock{i}_stride"]

            block = ConvBlock(current_channels, filters, kernel_size, stride, depth)
            encoder_blocks.append(block)

            self.encoder_specs.append((current_channels, filters, kernel_size, stride, depth))
            current_channels = filters

        self.encoder = nn.Sequential(*encoder_blocks)

        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, H, W)
            self.encoder_shapes = [(input_channels, H, W)]
            x = dummy_input
            for blk in self.encoder:
                x = blk(x)
                self.encoder_shapes.append((x.shape[1], x.shape[2], x.shape[3]))
            encoded_feat_dim = x.numel()
            self.encoded_shape = self.encoder_shapes[-1]

        self.fc_mu = nn.Linear(encoded_feat_dim, self.embedding_dim)
        self.fc_dec = nn.Linear(self.embedding_dim, encoded_feat_dim)

        decoder_blocks = []
        dec_current_channels = self.encoded_shape[0]

        for i in reversed(range(3)):
            in_c, out_c, kernel_size, stride, depth = self.encoder_specs[i]

            target_c, target_h, target_w = self.encoder_shapes[i]

            in_shape = self.encoder_shapes[i+1]
            in_h, in_w = in_shape[1], in_shape[2]

            pad = kernel_size // 2

            out_pad_h = target_h - ((in_h - 1)*stride - 2*pad + kernel_size)
            out_pad_h = max(out_pad_h, 0)

            out_pad_w = target_w - ((in_w - 1)*stride - 2*pad + kernel_size)
            out_pad_w = max(out_pad_w, 0)

            deconv_block = self._build_deconv_block(dec_current_channels, in_c, kernel_size, stride, depth, pad, (out_pad_h, out_pad_w))
            decoder_blocks.append(deconv_block)
            dec_current_channels = in_c

        self.decoder = nn.Sequential(*decoder_blocks)
        self.final_conv = nn.Conv2d(dec_current_channels, input_channels, kernel_size=3, padding=1)

    def _build_deconv_block(self, in_channels, out_channels, kernel_size, stride, depth, pad, output_padding):
        layers = []
        for i in range(depth):
            current_stride = stride if i == 0 else 1
            current_out_pad = output_padding if i == 0 else (0, 0)
            deconv = nn.ConvTranspose2d(
                in_channels=(in_channels if i == 0 else out_channels),
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=current_stride,
                padding=pad,
                output_padding=current_out_pad
            )
            layers.append(deconv)
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def encode(self, x):
        h = self.encoder(x)
        h_flat = torch.flatten(h, start_dim=1)
        z = self.fc_mu(h_flat)
        return z

    def decode(self, z):
        h_flat = self.fc_dec(z)
        h = h_flat.view(-1, *self.encoded_shape)
        h = self.decoder(h)
        x_recon = self.final_conv(h)
        return x_recon

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon