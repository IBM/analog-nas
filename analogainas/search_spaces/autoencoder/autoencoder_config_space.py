"""Hyperparameter Configuration Space."""

from analogainas.search_spaces.config_space.interface.config_space import ConfigSpace

from analogainas.search_spaces.config_space.base_config_space import BaseConfigSpace


class AutoEncoderConfigSpace(BaseConfigSpace):
    def __init__(self, dataset="CIFAR-10"):
        # initialize super
        super().__init__("autoencoder-like", dataset)
        self.dataset = dataset  # VWW, KWS
        self.search_space = "autoencoder-like"
        self.hyperparameters = []  # list of Hyperparameters to search for
        self.set_hyperparameters()

    def sample_arch_uniformly(self, n):
        archs = []
        for i in range(n):
            tmp = self.sample_arch()
            archs.append(tmp)
        return archs

    def set_hyperparameters(self):
        self.add_hyperparameter_range("embedding_dim", "discrete", [16, 32, 64, 128, 256, 512, 1024, 2048])

        # 3 Blocks of variable number of conv layers of varying configs and sizes

        # Block 1
        self.add_hyperparameter("encoder_convblock1_depth", "discrete", min_value=1, max_value=5)
        self.add_hyperparameter_range("encoder_convblock1_kernel_size", "discrete", [3, 5, 7])
        self.add_hyperparameter_range("encoder_convblock1_filters", "discrete", [8, 16, 32, 64, 128, 256, 512])
        self.add_hyperparameter_range("encoder_convblock1_stride", "discrete", [1, 2])

        # Encoder Block 2
        self.add_hyperparameter("encoder_convblock2_depth", "discrete", min_value=1, max_value=5)
        self.add_hyperparameter_range("encoder_convblock2_kernel_size", "discrete", [3, 5, 7])
        self.add_hyperparameter_range("encoder_convblock2_filters", "discrete", [8, 16, 32, 64, 128, 256, 512])
        self.add_hyperparameter_range("encoder_convblock2_stride", "discrete", [1, 2])

        # Encoder Block 3
        self.add_hyperparameter("encoder_convblock3_depth", "discrete", min_value=1, max_value=5)
        self.add_hyperparameter_range("encoder_convblock3_kernel_size", "discrete", [3, 5, 7])
        self.add_hyperparameter_range("encoder_convblock3_filters", "discrete", [8, 16, 32, 64, 128, 256, 512])
        self.add_hyperparameter_range("encoder_convblock3_stride", "discrete", [1, 2])

        # Decoder Block 1
        self.add_hyperparameter("decoder_convblock1_depth", "discrete", min_value=1, max_value=5)
        self.add_hyperparameter_range("decoder_convblock1_kernel_size", "discrete", [3, 5, 7])
        self.add_hyperparameter_range("decoder_convblock1_filters", "discrete", [8, 16, 32, 64, 128, 256, 512])
        self.add_hyperparameter_range("decoder_convblock1_stride", "discrete", [1, 2])

        # Decoder Block 2
        self.add_hyperparameter("decoder_convblock2_depth", "discrete", min_value=1, max_value=5)
        self.add_hyperparameter_range("decoder_convblock2_kernel_size", "discrete", [3, 5, 7])
        self.add_hyperparameter_range("decoder_convblock2_filters", "discrete", [8, 16, 32, 64, 128, 256, 512])
        self.add_hyperparameter_range("decoder_convblock2_stride", "discrete", [1, 2])

        # Decoder Block 3
        self.add_hyperparameter("decoder_convblock3_depth", "discrete", min_value=1, max_value=5)
        self.add_hyperparameter_range("decoder_convblock3_kernel_size", "discrete", [3, 5, 7])
        self.add_hyperparameter_range("decoder_convblock3_filters", "discrete", [8, 16, 32, 64, 128, 256, 512])
        self.add_hyperparameter_range("decoder_convblock3_stride", "discrete", [1, 2])
