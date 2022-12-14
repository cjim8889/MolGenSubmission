from survae.flows import ConditionalInverseFlow
from survae.distributions import ConditionalNormal
from einops.layers.torch import Rearrange

from .surjectives import ArgmaxSurjection
from .adjacency import AdjacencyBlockFlow
from .conditional import ConditionalAdjacencyBlockFlow
from torch import nn
import torch

class ContextNet(nn.Module):
    def __init__(self, context_size=16, num_classes=4, embedding_dim=7, hidden_dim=32):
        super(ContextNet, self).__init__()
        #Assume input B x 45
        self.net = nn.Sequential(
            nn.Embedding(num_classes, embedding_dim), # B x 45 x embedding_dim
            Rearrange("B N E -> B (N E)"),
            nn.LazyLinear(num_classes * embedding_dim),
        )

    def forward(self, x):
        return self.net(x)

# DO NOT USE THIS
class ArgmaxFlow(nn.Module):
    def __init__(self, context_size=8,
                    num_classes=5,
                    embedding_dim=7,
                    hidden_dim=128,
                    max_nodes=9,
                    t=12,
                    mask_ratio=9.,
                    inverted_mask=False):

        super().__init__()

        self.transforms = nn.ModuleList()


        context_net = ContextNet(context_size=context_size, num_classes=num_classes, embedding_dim=embedding_dim, hidden_dim=hidden_dim)

        encoder_base = ConditionalNormal(nn.Sequential(
            nn.Linear(embedding_dim, num_classes),
            nn.ReLU(),
            nn.Conv2d(context_size, 2, kernel_size=3, stride=1, padding=1)
        ), split_dim=1)

        transform = [ConditionalAdjacencyBlockFlow(max_nodes=max_nodes, embedding_dim=embedding_dim, num_classes=num_classes, context_size=context_size, inverted_mask=inverted_mask, mask_ratio=mask_ratio) for _ in range(t // 2)]
        conditional_flow = ConditionalInverseFlow(base_dist=encoder_base, context_init=context_net, transforms=transform)

        surjection = ArgmaxSurjection(conditional_flow, num_classes)

        self.transforms.append(surjection)

        for i in range(t):
            flow = AdjacencyBlockFlow(hidden_dim=hidden_dim, inverted_mask=inverted_mask, mask_ratio=mask_ratio)

            self.transforms.append(flow)

    def forward(self, x):
        log_prob = torch.zeros(x.shape[0], device=x.device)

        for transform in self.transforms:
            x, ldj = transform(x)
            log_prob += ldj

        return x, log_prob

    def inverse(self, z):

        log_prob = torch.zeros(z.shape[0], device=z.device)
        for idx in range(len(self.transforms) - 1, -1, -1):
            z, ldj = self.transforms[idx].inverse(z)
            log_prob += ldj

        return z, log_prob