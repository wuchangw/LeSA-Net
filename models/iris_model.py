from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder_3d import Encoder3D


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_channels)
        self.act = nn.LeakyReLU(inplace=True)
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + residual)


class UpBlock3D(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.block = ResidualBlock3D(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class SupportPrototypeEncoder(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(channels // 2, 32)
        self.projection = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.LayerNorm(channels),
        )

    def forward(self, support_features: torch.Tensor, support_masks: torch.Tensor) -> torch.Tensor:
        resized_mask = F.interpolate(
            support_masks,
            size=support_features.shape[-3:],
            mode="trilinear",
            align_corners=False,
        )
        masked = support_features * resized_mask
        denom = resized_mask.sum(dim=(2, 3, 4)).clamp_min(1e-6)
        pooled = masked.sum(dim=(2, 3, 4)) / denom
        return self.projection(pooled)


class BottleneckConditioner(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(channels // 2, 32)
        self.affine = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels * 2),
        )

    def forward(self, bottleneck: torch.Tensor, support_embedding: torch.Tensor) -> torch.Tensor:
        gamma_beta = self.affine(support_embedding)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        gamma = torch.tanh(gamma).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        beta = torch.tanh(beta).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return bottleneck * (1.0 + gamma) + beta


class LesionnessPromptBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(channels // 2, 16)
        self.prompt_head = nn.Sequential(
            nn.Conv3d(channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(hidden),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(hidden, 1, kernel_size=1),
        )
        self.prompt_fuse = nn.Sequential(
            nn.Conv3d(channels + 1, channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(channels),
            nn.LeakyReLU(inplace=True),
            ResidualBlock3D(channels, channels),
        )

    def predict_prompt(self, bottleneck: torch.Tensor) -> torch.Tensor:
        return self.prompt_head(bottleneck)

    def forward(
        self,
        bottleneck: torch.Tensor,
        prompt_logits: Optional[torch.Tensor] = None,
    ):
        if prompt_logits is None:
            prompt_logits = self.predict_prompt(bottleneck)
        prompt_prob = torch.sigmoid(prompt_logits)
        if prompt_prob.shape[0] != bottleneck.shape[0]:
            prompt_prob = prompt_prob.expand(bottleneck.shape[0], -1, -1, -1, -1).contiguous()
            prompt_logits = prompt_logits.expand(bottleneck.shape[0], -1, -1, -1, -1).contiguous()
        fused = self.prompt_fuse(torch.cat([bottleneck, prompt_prob], dim=1))
        return bottleneck + fused, prompt_logits


class CrossInteractionBottleneck(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.interaction = nn.Sequential(
            nn.Conv3d(channels * 2 + 1, channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(channels),
            nn.LeakyReLU(inplace=True),
            ResidualBlock3D(channels, channels),
        )

    def forward(self, bottleneck: torch.Tensor, support_embeddings: torch.Tensor) -> torch.Tensor:
        support_map = support_embeddings.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        support_map = support_map.expand(-1, -1, *bottleneck.shape[-3:]).contiguous()
        query_norm = F.normalize(bottleneck, p=2, dim=1)
        support_norm = F.normalize(support_map, p=2, dim=1)
        similarity_map = (query_norm * support_norm).sum(dim=1, keepdim=True)
        fused = self.interaction(torch.cat([bottleneck, support_map, similarity_map], dim=1))
        return bottleneck + fused


class CoarseToFineDecoder3D(nn.Module):
    def __init__(self, encoder_channels, num_classes: int) -> None:
        super().__init__()
        self.decoder4 = UpBlock3D(encoder_channels[-1], encoder_channels[-2], encoder_channels[-2])
        self.decoder3 = UpBlock3D(encoder_channels[-2], encoder_channels[-3], encoder_channels[-3])
        self.decoder2 = UpBlock3D(encoder_channels[-3], encoder_channels[-4], encoder_channels[-4])
        self.decoder1 = UpBlock3D(encoder_channels[-4], encoder_channels[-5], encoder_channels[-5])
        self.coarse_head = nn.Conv3d(encoder_channels[-2], 1, kernel_size=1)
        self.refine_head = nn.Conv3d(encoder_channels[-3], 1, kernel_size=1)
        self.head = nn.Sequential(
            nn.Conv3d(encoder_channels[-5], encoder_channels[-5], kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(encoder_channels[-5]),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(encoder_channels[-5], num_classes, kernel_size=1),
        )

    def _resize_gate(self, logits: torch.Tensor, target_size) -> torch.Tensor:
        if logits.shape[2:] != target_size:
            logits = F.interpolate(logits, size=target_size, mode="trilinear", align_corners=False)
        return torch.sigmoid(logits)

    def forward(self, features, return_details: bool = False):
        x4 = self.decoder4(features[-1], features[-2])
        coarse_logits = self.coarse_head(x4)
        coarse_gate = torch.sigmoid(coarse_logits)
        skip3_gate = self._resize_gate(coarse_logits, features[-3].shape[2:])
        x3 = self.decoder3(x4 * (1.0 + coarse_gate), features[-3] * (1.0 + skip3_gate))

        refine_logits = self.refine_head(x3)
        refine_gate = torch.sigmoid(refine_logits)
        skip2_gate = self._resize_gate(refine_logits, features[-4].shape[2:])
        x2 = self.decoder2(x3 * (1.0 + refine_gate), features[-4] * (1.0 + skip2_gate))

        x1 = self.decoder1(x2, features[-5])
        logits = self.head(x1)

        if not return_details:
            return logits

        return logits, {
            "coarse_logits": coarse_logits,
            "refine_logits": refine_logits,
        }


class IRISFinalModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 16,
        num_classes: int = 1,
        fusion_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_classes = num_classes
        self.fusion_temperature = fusion_temperature

        self.encoder = Encoder3D(
            in_channels=in_channels,
            base_channels=base_channels,
            num_blocks_per_stage=2,
        )
        encoder_channels = self.encoder.get_feature_channels()
        self.support_encoder = SupportPrototypeEncoder(encoder_channels[-1])
        self.conditioner = BottleneckConditioner(encoder_channels[-1])
        self.cross_interaction = CrossInteractionBottleneck(encoder_channels[-1])
        self.lesion_prompt = LesionnessPromptBlock(encoder_channels[-1])
        self.decoder = CoarseToFineDecoder3D(encoder_channels=encoder_channels, num_classes=num_classes)

        self.config = {
            "model_name": "iris_final",
            "in_channels": in_channels,
            "base_channels": base_channels,
            "num_classes": num_classes,
            "fusion_temperature": fusion_temperature,
            "encoder_channels": encoder_channels,
        }

    def encode_image(self, image: torch.Tensor):
        return self.encoder(image)

    def encode_support(self, support_bottleneck: torch.Tensor, support_masks: torch.Tensor) -> torch.Tensor:
        return self.support_encoder(support_bottleneck, support_masks)

    def _repeat_query_features(self, query_features, batch_size: int):
        repeated = []
        for feature in query_features:
            repeated.append(feature.expand(batch_size, -1, -1, -1, -1).contiguous())
        return repeated

    def _decode_candidates(
        self,
        query_features,
        support_embeddings: torch.Tensor,
    ):
        batch_size = support_embeddings.shape[0]
        repeated_features = self._repeat_query_features(query_features, batch_size=batch_size)
        shared_prompt_logits = self.lesion_prompt.predict_prompt(query_features[-1])
        bottleneck = self.conditioner(repeated_features[-1], support_embeddings)
        bottleneck = self.cross_interaction(bottleneck, support_embeddings)
        bottleneck, lesionness_logits = self.lesion_prompt(bottleneck, shared_prompt_logits)
        repeated_features[-1] = bottleneck
        candidate_logits, decoder_details = self.decoder(repeated_features, return_details=True)
        return candidate_logits, {
            "lesionness_logits": lesionness_logits,
            "coarse_logits": decoder_details["coarse_logits"],
            "refine_logits": decoder_details["refine_logits"],
        }

    def forward_multi_support(
        self,
        query_image: torch.Tensor,
        support_images: torch.Tensor,
        support_masks: torch.Tensor,
        support_scores: Optional[torch.Tensor] = None,
        support_case_ids=None,
        return_details: bool = False,
    ):
        if support_images.dim() != 5 or support_masks.dim() != 5:
            raise ValueError("support_images and support_masks must have shape (K, C, D, H, W) and (K, 1, D, H, W)")
        if support_images.shape[0] == 0:
            raise ValueError("support_images must contain at least one support case")

        query_features = self.encode_image(query_image)
        support_features = self.encode_image(support_images)
        support_embeddings = self.encode_support(support_features[-1], support_masks)
        candidate_logits, auxiliary_outputs = self._decode_candidates(query_features, support_embeddings)

        if support_scores is None:
            weight_logits = torch.zeros(candidate_logits.shape[0], device=candidate_logits.device)
        else:
            weight_logits = support_scores.to(candidate_logits.device).view(-1)

        weights = torch.softmax(weight_logits / max(self.fusion_temperature, 1e-6), dim=0)
        fused_logits = (weights.view(-1, 1, 1, 1, 1) * candidate_logits).sum(dim=0, keepdim=True)
        best_index = int(torch.argmax(weights).item())

        if not return_details:
            return fused_logits

        metadata = []
        for index in range(candidate_logits.shape[0]):
            metadata.append(
                {
                    "support_case_id": support_case_ids[index] if support_case_ids is not None else f"support_{index}",
                    "support_score": float(weight_logits[index].item()),
                }
            )

        return {
            "logits": fused_logits,
            "candidate_logits": candidate_logits,
            "candidate_weights": weights,
            "candidate_metadata": metadata,
            "best_support_case_id": metadata[best_index]["support_case_id"],
            "lesionness_logits": auxiliary_outputs["lesionness_logits"],
            "coarse_logits": auxiliary_outputs["coarse_logits"],
            "refine_logits": auxiliary_outputs["refine_logits"],
        }

    def forward(
        self,
        query_image: torch.Tensor,
        support_images: torch.Tensor,
        support_masks: torch.Tensor,
        support_scores: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.forward_multi_support(
            query_image=query_image,
            support_images=support_images,
            support_masks=support_masks,
            support_scores=support_scores,
            return_details=False,
        )

    def get_model_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        support_params = sum(p.numel() for p in self.support_encoder.parameters())
        conditioner_params = sum(p.numel() for p in self.conditioner.parameters())
        cross_params = sum(p.numel() for p in self.cross_interaction.parameters())
        prompt_params = sum(p.numel() for p in self.lesion_prompt.parameters())
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "encoder_parameters": encoder_params,
            "decoder_parameters": decoder_params,
            "support_encoder_parameters": support_params,
            "conditioner_parameters": conditioner_params,
            "cross_interaction_parameters": cross_params,
            "lesion_prompt_parameters": prompt_params,
            "config": self.config,
        }


IRISModel = IRISFinalModel
