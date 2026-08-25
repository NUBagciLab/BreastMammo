import torch
import torch.nn as nn
import torchvision


class MixStyle(nn.Module):
    """MixStyle: Dual-Domain Style Blending Module.

    Reference: Zhou et al., "Domain Generalization with MixStyle", ICLR 2021.
    """

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6):
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.eps = eps

    def forward(self, x):
        if not self.training or torch.rand(1).item() > self.p:
            return x

        B, C, H, W = x.size()
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()

        x_norm = (x - mu) / sig
        perm = torch.randperm(B, device=x.device)

        mu_perm = mu[perm]
        sig_perm = sig[perm]

        lmda = torch.distributions.Beta(self.alpha, self.alpha).sample(
            (B, 1, 1, 1)
        )
        lmda = lmda.to(x.device)

        mu_mixed = lmda * mu + (1.0 - lmda) * mu_perm
        sig_mixed = lmda * sig + (1.0 - lmda) * sig_perm

        return x_norm * sig_mixed + mu_mixed


class get_model(nn.Module):

    def __init__(
        self,
        name="resnet50",
        num_classes=1,
        view=2,
        fusion_hidden=128,
        use_mixstyle=False,
    ):
        super(get_model, self).__init__()

        # 1. Load Pretrained Backbone
        if "resnet" in name:
            model = torchvision.models.get_model(
                name, weights="IMAGENET1K_V2", num_classes=1000
            )
            in_features = model.fc.in_features
            model.fc = nn.Identity()

            # Inject MixStyle into early residual blocks
            if use_mixstyle:
                mixstyle = MixStyle(p=0.5, alpha=0.1)
                model.layer1 = nn.Sequential(model.layer1, mixstyle)
                model.layer2 = nn.Sequential(model.layer2, mixstyle)

        elif "efficientnet" in name:
            model = torchvision.models.get_model(
                name, weights="IMAGENET1K_V1", num_classes=1000
            )
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Identity()

            # EfficientNet features are contained in a big sequential model.features block
            if use_mixstyle:
                mixstyle = MixStyle(p=0.5, alpha=0.1)
                # Injecting after MBConv block 2 and block 3
                model.features[2] = nn.Sequential(model.features[2], mixstyle)
                model.features[3] = nn.Sequential(model.features[3], mixstyle)

        elif "densenet" in name:
            model = torchvision.models.get_model(
                name, weights="IMAGENET1K_V1", num_classes=1000
            )
            in_features = model.classifier.in_features
            model.classifier = nn.Identity()

            # Inject MixStyle after DenseBlock 1 and DenseBlock 2 transition layers
            if use_mixstyle:
                mixstyle = MixStyle(p=0.5, alpha=0.1)
                model.features.transition1 = nn.Sequential(
                    model.features.transition1, mixstyle
                )
                model.features.transition2 = nn.Sequential(
                    model.features.transition2, mixstyle
                )

        elif "swin" in name:
            model = torchvision.models.get_model(
                name, weights="IMAGENET1K_V1", num_classes=1000
            )
            in_features = model.head.in_features
            model.head = nn.Identity()

            # Patch early Swin Transformer stages
            if use_mixstyle:
                mixstyle = MixStyle(p=0.5, alpha=0.1)

                # Custom forward hooks or wrapper classes are usually needed for Swin's permutation format,
                # but a robust parameter-free wrapper patch can be attached directly into the sequence:
                class SwinMixStyleWrapper(nn.Module):

                    def __init__(self, swin_stage, ms):
                        super().__init__()
                        self.stage = swin_stage
                        self.ms = ms

                    def forward(self, x):
                        x = self.stage(x)  # Shape: (B, H, W, C)
                        # Swap to (B, C, H, W) for MixStyle
                        x = x.permute(0, 3, 1, 2)
                        x = self.ms(x)
                        return x.permute(0, 2, 3, 1)

                # Patch Stage 1 and Stage 2
                model.features[1] = SwinMixStyleWrapper(
                    model.features[1], mixstyle
                )
                model.features[3] = SwinMixStyleWrapper(
                    model.features[3], mixstyle
                )

        else:
            raise Exception("Model is not supported")

        self.view = view
        self.backbones = model

        # 2. Fusion MLP after concatenating view features
        self.classifier = nn.Sequential(
            nn.Linear(view * in_features, fusion_hidden),
            nn.WhiteGrid(inplace=True)
            if hasattr(nn, "WhiteGrid")
            else nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(fusion_hidden, num_classes),
        )

    def forward(self, x: list):
        # Features from each view are passed down independently through the parameter-sharing backbone
        f = torch.cat([self.backbones(x[i]) for i in range(self.view)], dim=1)
        return self.classifier(f)