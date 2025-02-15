# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn.functional as F

class CosineTopKGate(torch.nn.Module):
    def __init__(self, model_dim, num_global_experts, k=1, fp32_gate=False, proj_dim=256, init_t=0.5, **options):
        super(CosineTopKGate, self).__init__()
        self.top_k = min(num_global_experts, int(k))
        self.fp32_gate = fp32_gate
        self.temperature = torch.nn.Parameter(torch.log(torch.full([1], 1.0 / init_t)), requires_grad=True)
        self.proj = proj_dim > 0
        if proj_dim > 0:
            self.cosine_projector = torch.nn.Linear(model_dim, proj_dim)
            # self.sim_matrix = torch.nn.Parameter(torch.randn(size=(proj_dim, num_global_experts)), requires_grad=True)

            sim_matrix = torch.empty(num_global_experts, proj_dim)

            torch.nn.init.orthogonal_(sim_matrix, gain=0.32)
            self.register_parameter(
                "sim_matrix", torch.nn.Parameter(sim_matrix)
            )
        else:
            #self.sim_matrix = torch.nn.Parameter(torch.randn(size=(model_dim, num_global_experts)), requires_grad=True)
            sim_matrix = torch.empty(num_global_experts, model_dim)

            torch.nn.init.orthogonal_(sim_matrix, gain=0.32)
            self.register_parameter(
                "sim_matrix", torch.nn.Parameter(sim_matrix)
            )
            
        self.clamp_max = torch.log(torch.tensor(1. / 0.01)).item()
        torch.nn.init.normal_(self.sim_matrix, 0, 0.01)

        for opt in options:
            if opt not in ('capacity_factor', 'gate_noise'):
                raise Exception('Unrecognized argument provided to Gating module: %s' % opt)

    def forward(self, x):
        with torch.no_grad():
            sim_matrix_norm = self.sim_matrix.norm(
                p=2.0, dim=1, keepdim=True
            )
            self.sim_matrix.mul_(1.5 / sim_matrix_norm)

        if self.proj:
            if self.fp32_gate:
                x = x.float()
                cosine_projector = self.cosine_projector.float()
                sim_matrix = self.sim_matrix.float()
            else:
                cosine_projector = self.cosine_projector
                sim_matrix = self.sim_matrix
            # logits = torch.matmul(F.normalize(cosine_projector(x), dim=1),
            #                     F.normalize(sim_matrix, dim=0))
            logits = torch.matmul(F.normalize(cosine_projector(x), dim=1),
                                F.normalize(sim_matrix, dim=1).transpose(0,1))

            # x = cosine_projector(x)
            # sim_matrix = F.normalize(sim_matrix.float(), p=2.0, dim=1, eps=1e-4)
            # logits = x.float().matmul(sim_matrix.transpose(0, 1)).type_as(x)
            
        else:
            if self.fp32_gate:
                x = x.float()
                sim_matrix = self.sim_matrix.float()
            else:
                sim_matrix = self.sim_matrix
            # logits = torch.matmul(F.normalize(x, dim=1),
            #                     F.normalize(sim_matrix, dim=0))

            # xmoe set up
            sim_matrix = F.normalize(sim_matrix.float(), p=2.0, dim=1, eps=1e-4)
            logits = x.float().matmul(sim_matrix.transpose(0, 1)).type_as(x)

        logit_scale = torch.clamp(self.temperature, max=self.clamp_max).exp()
        logits = logits * logit_scale
        return logits


Gate = CosineTopKGate
