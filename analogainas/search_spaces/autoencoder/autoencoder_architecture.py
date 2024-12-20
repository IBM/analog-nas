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
                padding=kernel_size // 2 if padding == 'same' else 0
            )
            layers.append(conv)
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, depth, padding='same'):
        super().__init__()
        layers = []
        for i in range(depth):
            opad = (stride - 1 if i == 0 and stride > 1 else 0)
            deconv = nn.ConvTranspose2d(
                in_channels=(in_channels if i == 0 else out_channels),
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=(stride if i == 0 else 1),
                padding=kernel_size // 2 if padding == 'same' else 0,
                output_padding=opad
            )
            layers.append(deconv)
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class AutoEncoder(nn.Module):
    def __init__(self, config, input_channels=3, input_size=(32,32)):
        super().__init__()
        self.config = config
        self.embedding_dim = config["embedding_dim"]
        self.input_channels = input_channels
        self.input_height, self.input_width = input_size

        current_channels = input_channels

        encoder_blocks = []
        for i in range(1, 4):
            depth = config[f"encoder_convblock{i}_depth"]
            kernel_size = config[f"encoder_convblock{i}_kernel_size"]
            filters = config[f"encoder_convblock{i}_filters"]
            stride = config[f"encoder_convblock{i}_stride"]
            encoder_blocks.append(
                ConvBlock(current_channels, filters, kernel_size, stride, depth)
            )
            current_channels = filters
        self.encoder = nn.Sequential(*encoder_blocks)

        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, self.input_height, self.input_width)
            encoded_feat = self.encoder(dummy_input)
            self.encoded_shape = encoded_feat.shape[1:]  # C,H,W
            encoded_feat_dim = encoded_feat.numel()

        self.fc_mu = nn.Linear(encoded_feat_dim, self.embedding_dim)
        self.fc_dec = nn.Linear(self.embedding_dim, encoded_feat_dim)

        decoder_blocks = []
        for i in range(1, 4):
            depth = config[f"encoder_convblock{4-i}_depth"]
            kernel_size = config[f"encoder_convblock{4-i}_kernel_size"]
            filters = config[f"encoder_convblock{4-i}_filters"]
            stride = config[f"encoder_convblock{4-i}_stride"]
            decoder_blocks.append(
                DeconvBlock(
                    current_channels, filters, kernel_size, stride, depth
                )
            )
            current_channels = filters
        self.decoder = nn.Sequential(*decoder_blocks)

        self.pen_conv = nn.Conv2d(current_channels, input_channels, kernel_size=3, stride=1, padding=1)


        decoded_shape = None
        self.need_fc_out = False

        with torch.no_grad():
            dummy_embedding = torch.zeros(1, self.embedding_dim)

            dummy_ret = self.fc_dec(dummy_embedding)
            dummy_ret = dummy_ret.view(-1, *self.encoded_shape)
            dummy_ret = self.decoder(dummy_ret)
            dummy_ret = self.pen_conv(dummy_ret)

            decoded_shape = dummy_ret.shape[1:]

            self.need_fc_out = (decoded_shape != (input_channels, self.input_height, self.input_width))
            print("Need FC Out:", self.need_fc_out)
            print("Decoded Shape:", decoded_shape)
            print("Input Size:", input_size)


        self.fc_out = nn.Linear(input_channels * decoded_shape[1] * decoded_shape[2], input_channels * input_size[0] * input_size[1])



    def encode(self, x):
        h = self.encoder(x)
        h_flat = torch.flatten(h, start_dim=1)
        z = self.fc_mu(h_flat)  # Embedding
        return z

    def decode(self, z):
        h_flat = self.fc_dec(z)
        h = h_flat.view(-1, *self.encoded_shape)
        h = self.decoder(h)
        h = self.pen_conv(h)

        if self.need_fc_out:
            N = h.size(0)
            h = h.view(N, -1)
            h = self.fc_out(h)
            h = h.view(N, self.input_channels, self.input_height, self.input_width)
        return h

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon

