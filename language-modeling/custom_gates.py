import os, sys
import argparse
import math, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import tree

import pdb
import numpy as np
from fmoe.gates.base_gate import BaseGate
#from sklearn.cluster import KMeans

__all__ = [
    "CustomNaiveGate_Balance_SMoE",
    "CustomNaiveGate_Balance_XMoE",
    "CustomNaiveGate_Balance_StableMoE",
    "CustomNaiveGate_Balance_EllipticalXMoE",
    "CustomNaiveGate_Balance_SparseProjectMoE",
    "SpectralGate_SMoE",
    "Balance_Elliptical2XMoE",
    "KSpectral_Balance_SMoE",
    "COSAGate_Balance",
    "KmeansSpectral",
    "EllipticalSMoE",
    "LDAGate"
]

class CustomNaiveGate_Balance_SparseProjectMoE(BaseGate):
    def __init__(self, d_model, num_expert, world_size, top_k=2, g_balance=False, show_gate_W = False, mean_scale = False):
        super().__init__(num_expert, world_size)
        self.gate = nn.Linear(d_model, self.tot_expert)
        self.top_k = top_k
        self.dense_moe_flag = False
        self.g_balance = g_balance
        self.loss = 0.0
        self.delta = nn.Parameter(torch.tensor(150.))

        #expert_embeddings = torch.empty(num_expert, 8)
        expert_embeddings = torch.empty(num_expert, d_model) # no dimension reduction
        torch.nn.init.orthogonal_(expert_embeddings, gain=0.32)
        self.register_parameter(
            "expert_embeddings", torch.nn.Parameter(expert_embeddings)
        )

        self.inp_reduction = torch.nn.Linear(d_model, 8, bias=False) # for smoe-m, hidden_size = 352, num_heads = 8, therefore head_dim = 44 (double check)
        self.show_gate_W = show_gate_W
        self.mean_scale = mean_scale

    def set_load_balance(self, gate, gate_top_k_idx):
        # gate_top_k_idx (tokens_number, top-k)
        # gate_top_k_val (tokens_number, top-k)

        score = F.softmax(gate / 0.3, dim=-1)
        valid_idx = gate_top_k_idx[gate_top_k_idx > -1]
        fraction_expert = (
            torch.scatter_add(
                torch.zeros(self.tot_expert, device=valid_idx.device),
                0,
                valid_idx,
                torch.ones_like(valid_idx, dtype=torch.float),
            )
            / valid_idx.numel()
        )
        prob_expert = score.sum(dim=0) / valid_idx.numel()

        loss = (fraction_expert * prob_expert).sum() * self.tot_expert
        self.loss = loss

    def forward(self, inp, gate_top_k_idx, return_all_scores=False):

        #reduced_inp = self.inp_reduction(inp)
        reduced_inp = inp # no dimension reduction

        with torch.no_grad():
            expert_embeddings_norm = self.expert_embeddings.norm(
                p=2.0, dim=1, keepdim=True
            )
            self.expert_embeddings.mul_(1.5 / expert_embeddings_norm)
        #breakpoint()
        gate = self._sparse_route(reduced_inp, self.expert_embeddings, gate_top_k_idx[:,0])
        gate = self._make_finite(gate)
        #breakpoint()

        if self.dense_moe_flag:
            gate = torch.ones_like(gate)  # average the importance of all experts
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.tot_expert, dim=-1, largest=True, sorted=False
            )
            gate_top_k_val = gate_top_k_val.view(-1, self.tot_expert)
        else:
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.top_k, dim=-1, largest=True, sorted=False
            )  # [.. x top_k]
            gate_top_k_val = gate_top_k_val.view(-1, self.top_k)  # (BxL) x 1 x top_k

        gate_score = F.softmax(gate_top_k_val, dim=-1)
        if self.g_balance: # should be False
            self.set_load_balance(gate, gate_top_k_idx)

        if return_all_scores:
            return gate_top_k_idx, gate_score, gate
        return gate_top_k_idx, gate_score

    def _cosine(self, mat1, mat2, eps=1e-4):
        assert mat1.dim() == 2
        assert mat2.dim() == 2
        # mat1 = F.normalize(mat1, p=2.0, dim=1, eps=eps)
        mat2 = F.normalize(mat2.float(), p=2.0, dim=1, eps=eps)
        return mat1.float().matmul(mat2.transpose(0, 1)).type_as(mat1)
    
    def _sparse_project(self, mat1, cluster_labels, delta = None, stab = 1e-3):
        with torch.no_grad():
            #breakpoint()
            mat1 = mat1.detach()
            cluster_labels = cluster_labels.detach()

            # downsampling trial
            random_indices = torch.randperm(mat1.shape[0])[:2000] # 2000 arbitrarily chosen
            mat1 = mat1[random_indices]
            cluster_labels = cluster_labels[random_indices]

            n, d = mat1.shape
            assert cluster_labels.numel() == n
            
            n_clusters = torch.unique(cluster_labels).numel()
            ##### original, full batch code #####
            # Compute the first term
            diff_matrix = torch.abs(mat1.unsqueeze(1) - mat1.unsqueeze(0))
            first_term = diff_matrix.sum(dim=(0, 1)) / n

            # Compute the second term
            second_term = torch.zeros(d, device=mat1.device)
            for k in range(n_clusters):
                cluster_mask = (cluster_labels == k)
                n_k = cluster_mask.sum()

                if n_k > 0:
                    X_k = mat1[cluster_mask]
                    diff_matrix_k = torch.abs(X_k.unsqueeze(1) - X_k.unsqueeze(0))
                    second_term += diff_matrix_k.sum(dim=(0, 1)) / n_k

            # Compute the final result
            a = first_term - second_term
            ###### original full batch code ######


            ###### loop over features memory-saving code ######
            # a = torch.zeros(d).to(mat1.device)
            # for feature in range(d):
                
            #     # Calculate the first term
            #     x_d = mat1[:, feature]
            #     first_term = torch.sum(torch.abs(x_d.unsqueeze(0) - x_d.unsqueeze(1))) / n

            #     # Calculate the second term
            #     second_term = torch.tensor(0.).to(mat1.device)
            #     for k in range(n_clusters): 
            #         cluster_indices = (cluster_labels == k).nonzero(as_tuple=True)[0]
            #         n_k = len(cluster_indices)
            #         if n_k > 0:  # Ensure there is at least one element in the cluster
            #             x_d_cluster = mat1[cluster_indices, feature]
            #             if n_k > 1:  # Ensure there are at least two elements to compute differences
            #                 second_term += torch.sum(torch.abs(x_d_cluster.unsqueeze(0) - x_d_cluster.unsqueeze(1))) / n_k
            #     a[feature] = first_term - second_term
            ###### loop over features memory-saving code #######
            #breakpoint()
            # compute the soft threshold
            if delta is None:
                delta = self.delta
           
        num = torch.sign(torch.relu(a)) * torch.relu(torch.relu(a) - delta)
        denom = torch.norm(num, p=1)
        w = num / (denom + stab) # stab for stability

        #w = torch.sqrt(w) # trial square root the weights as per Tibshirani
        #breakpoint()

        # collect summary stats for w
        num_zeros = torch.sum(w == 0.).item()
        mx, max_index =  torch.max(w, dim = 0)
        mn, min_index = torch.min(w, dim = 0)
        mean = torch.mean(w).item()
        std = torch.std(w).item()
            
        self.sparse_w_stats = (delta, d, num_zeros, (mx.item(), max_index.item()), (mn.item(), min_index.item()), mean, std)

        #w = w / torch.max(w) #### get rid of this hardcoding
        return w
    
    def _sparse_route(self, mat1, mat2, cluster_labels, eps = 1e-4):
        assert mat1.dim() == 2
        assert mat2.dim() == 2
        #breakpoint()
        # mat1 = F.normalize(mat1, p=2.0, dim=1, eps=eps) 
        # mat2 = F.normalize(mat2.float(), p=2.0, dim=1, eps=eps) # trial removing expert normalization
        W = self._sparse_project(mat1, cluster_labels)
        mat1W = mat1 * W
        #mat1W = mat1
        return mat1W.float().matmul(mat2.transpose(0, 1)).type_as(mat1)
        
    def _make_finite(self, scores):
        ok = scores.isfinite()
        if not ok.all():
            # NaNs here can break the assignment algorithm
            scores[~ok] = scores[ok].min()
        return scores


class CustomNaiveGate_Balance_EllipticalXMoE(BaseGate):
    def __init__(self, d_model, num_expert, world_size, top_k=2, g_balance=False, show_gate_W = False, mean_scale = False, root_invert = True,
     intra_layer = True, exp_distance = False, reduce_dim = False, use_elliptical = True):
        super().__init__(num_expert, world_size)
        self.gate = nn.Linear(d_model, self.tot_expert)
        self.top_k = top_k
        self.dense_moe_flag = False
        self.g_balance = g_balance
        self.loss = 0.0
        self.reduce_dim = reduce_dim

        #expert_embeddings = torch.empty(num_expert, 8)
        if reduce_dim: # LDA inspired dim reduction where clustering dim < clusters -1
            expert_embeddings = torch.empty(num_expert, num_expert-1) 
        else:
            expert_embeddings = torch.empty(num_expert, d_model) # no dimension reduction
        torch.nn.init.orthogonal_(expert_embeddings, gain=0.32)
        self.register_parameter(
            "expert_embeddings", torch.nn.Parameter(expert_embeddings)
        )

        self.inp_reduction = torch.nn.Linear(d_model, num_expert-1, bias=False) #
        self.show_gate_W = show_gate_W
        self.mean_scale = mean_scale
        self.root_invert = root_invert
        self.intra_layer = intra_layer
        self.exp_distance = exp_distance
        self.use_elliptical = use_elliptical

        self.gate_W = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.gate_W_scaled = (0.0, 0.0, 0.0, 0.0, 0.0)

    def set_load_balance(self, gate, gate_top_k_idx):
        # gate_top_k_idx (tokens_number, top-k)
        # gate_top_k_val (tokens_number, top-k)

        score = F.softmax(gate / 0.3, dim=-1)
        valid_idx = gate_top_k_idx[gate_top_k_idx > -1]
        fraction_expert = (
            torch.scatter_add(
                torch.zeros(self.tot_expert, device=valid_idx.device),
                0,
                valid_idx,
                torch.ones_like(valid_idx, dtype=torch.float),
            )
            / valid_idx.numel()
        )
        prob_expert = score.sum(dim=0) / valid_idx.numel()

        loss = (fraction_expert * prob_expert).sum() * self.tot_expert
        self.loss = loss

    def forward(self, inp, fwds =  None, return_all_scores=False):

        if self.reduce_dim:
            reduced_inp = self.inp_reduction(inp)
            print('not here')
        else:
            reduced_inp = inp # no dimension reduction

        with torch.no_grad():
            expert_embeddings_norm = self.expert_embeddings.norm(
                p=2.0, dim=1, keepdim=True
            )
            self.expert_embeddings.mul_(1.5 / expert_embeddings_norm)
        breakpoint()
        if self.use_elliptical:
            
            if not self.intra_layer:
                if type(fwds) is not tuple: # generic xmoe
                    gate = self._cosine(reduced_inp, self.expert_embeddings)
                else: # use elliptical once fwds is fully populated with (fwd_last, fwd_last2)
                    gate = self._elliptical_cosine(reduced_inp, self.expert_embeddings, fwds)
            else: # for intra-layer we run with return_fwd on so fwds should always be populated
                gate = self._elliptical_cosine(reduced_inp, self.expert_embeddings, fwds)
        else:
            print('not using elliptical')
            gate = self._cosine(reduced_inp, self.expert_embeddings)
        gate = self._make_finite(gate)
        #breakpoint()

        if self.dense_moe_flag:
            gate = torch.ones_like(gate)  # average the importance of all experts
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.tot_expert, dim=-1, largest=True, sorted=False
            )
            gate_top_k_val = gate_top_k_val.view(-1, self.tot_expert)
        else:
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.top_k, dim=-1, largest=True, sorted=False
            )  # [.. x top_k]
            gate_top_k_val = gate_top_k_val.view(-1, self.top_k)  # (BxL) x 1 x top_k

        gate_score = F.softmax(gate_top_k_val, dim=-1)
        if self.g_balance: # should be False
            self.set_load_balance(gate, gate_top_k_idx)

        if return_all_scores:
            return gate_top_k_idx, gate_score, gate
        return gate_top_k_idx, gate_score

    def _cosine(self, mat1, mat2, eps=1e-4):
        assert mat1.dim() == 2
        assert mat2.dim() == 2
        # mat1 = F.normalize(mat1, p=2.0, dim=1, eps=eps)
        mat2 = F.normalize(mat2.float(), p=2.0, dim=1, eps=eps)
        return mat1.float().matmul(mat2.transpose(0, 1)).type_as(mat1)
     
    
    def _elliptical_cosine(self, mat1, mat2, fwds, eps=1e-4):
        assert mat1.dim() == 2
        assert mat2.dim() == 2
        #breakpoint()
        if self.intra_layer:
            W = self.compute_W(v_last = fwds)
            
        else:
            if type(fwds) is tuple:
                v_last, v_last2 = fwds
            else: # when intra  layer is on, we take fwds when its not a tuple as well
                v_last, v_last2 = fwds, None
            W = self.compute_W(v_last, v_last2)
        # mat1 = F.normalize(mat1, p=2.0, dim=1, eps=eps)
        
        mat2 = F.normalize(mat2.float(), p=2.0, dim=1, eps=eps)
        mat1W = mat1 @ W
        #mat1W = mat1

        return mat1W.float().matmul(mat2.transpose(0, 1)).type_as(mat1)
    
    def compute_W(self, v_last, v_last2 = None, k_last = None, k_last2 = None, delta = 1, lamb = None):
        with torch.no_grad():
            # v shape: [(bsize x seqlen) x dim]
            v_last = v_last.detach()
            if v_last2 is not None:
                v_last2 = v_last2.detach()
            # k_last = k_last.detach()
            # k_last2 = k_last2.detach()
            
            if self.intra_layer:
                
                w =  torch.var(v_last, dim = 0) # shape: [dim]
                dim = w.shape[0]

                if self.exp_distance:
                    if lamb is None:
                        #lamb  = torch.sqrt(torch.tensor(dim))   
                        lamb = torch.tensor(1.)
                    w = F.softmax(-w / lamb)
                    

                elif self.root_invert:
                    w = 1 / (w + 1e-4)
                    print('inverting')
                
                # store weights  
                W_mean = torch.mean(w)
                W_std = torch.std(w)
                W_max, max_idx = torch.max(w, dim = 0)
                W_min, min_idx = torch.min(w, dim = 0)
                
                self.gate_W = (w, W_std, W_max, max_idx, W_min, min_idx, W_mean)
                # if self.mean_scale:
                #     weights_scaled = w / torch.mean(w)
                # else:
                #     weights_scaled = w / torch.max(w)
                # scaled_mean = torch.mean(weights_scaled)
                # scaled_std = torch.std(weights_scaled)
                # scaled_max = torch.max(weights_scaled)
                # scaled_min = torch.min(weights_scaled)
                # self.gate_W_scaled = (weights_scaled, scaled_std, scaled_max, scaled_min, scaled_mean)

                if not self.exp_distance:
                    #w = w / torch.max(w)
                    print(f'scaling by {w.mean()}')
                    w = w / torch.mean(w) # try meanscale

        return torch.diag_embed(w)


        #return v_last
        #     seqlen = v_last.size(0)
        #     if delta is None:
        #         deltas = torch.abs(k_last - k_last2) #include small term for stability and gradient attenuation
        #         difference_quotients = (v_last - v_last2) /deltas # entrywise (v' - v)_nd / (q' - q)_nd

        #     else:
        #         #delta = torch.mean(torch.abs(k_last - k_last2))
        #         difference_quotients = (v_last-v_last2) / delta
            
        #     W = torch.norm(difference_quotients, p = 1, dim = 0) /seqlen #columnwise average l1 norms
        #     # W dim: [h]

        #     if self.root_invert:
        #         #W = 1/ (torch.sqrt(W) + 1e-3) # trial D^{-1/2} as from sphering the data in LDA
        #         W  = W**2 # trial squaring and inverting
        #         W = 1 / (W + 1e-4) # trial inversion but no sqrt

        #     # store weights  
        #     W_mean = torch.mean(W)
        #     W_std = torch.std(W)
        #     W_max, max_idx = torch.max(W, dim = 0)
        #     W_min, min_idx = torch.min(W, dim = 0)
            
        #     self.gate_W = (W, W_std, W_max, max_idx, W_min, min_idx, W_mean)
        #     if self.mean_scale:
        #         weights_scaled = W / torch.mean(W)
        #     else:
        #         weights_scaled = W / torch.max(W)
        #     scaled_mean = torch.mean(weights_scaled)
        #     scaled_std = torch.std(weights_scaled)
        #     scaled_max = torch.max(weights_scaled)
        #     scaled_min = torch.min(weights_scaled)
        #     self.gate_W_scaled = (weights_scaled, scaled_std, scaled_max, scaled_min, scaled_mean)

            
        #     if self.mean_scale:
        #         W = W / torch.mean(W)
        #     else: # default max scale
        #         #W = W / torch.max(W, dim = -1, keepdim=True)[0] # maxscale for multidim W
        #         W = W / torch.max(W) # W is 0-dim here just [h] vector.
        #         #print('here')
        #     W = torch.diag_embed(W)
            
        # return W
      

    def _make_finite(self, scores):
        ok = scores.isfinite()
        if not ok.all():
            # NaNs here can break the assignment algorithm
            scores[~ok] = scores[ok].min()
        return scores

class LDAGate(BaseGate):
    def __init__(self, d_model, num_expert, world_size, top_k=2, g_balance=False, show_gate_W = False, mean_scale = False, root_invert = False, intra_layer  = False):
        super().__init__(num_expert, world_size)
        self.gate = nn.Linear(d_model, self.tot_expert)
        self.top_k = top_k
        self.dense_moe_flag = False
        self.g_balance = g_balance
        self.loss = 0.0

        expert_embeddings = torch.empty(num_expert, num_expert)
        torch.nn.init.orthogonal_(expert_embeddings, gain=0.32)
        self.register_parameter(
            "expert_embeddings", torch.nn.Parameter(expert_embeddings)
        )

        self.inp_reduction = torch.nn.Linear(d_model, num_expert, bias=False) # ESL states LDA in k-1 dimensions 
        self.show_gate_W = show_gate_W
        self.mean_scale = mean_scale
        self.root_invert = root_invert

    def set_load_balance(self, gate, gate_top_k_idx):
        # gate_top_k_idx (tokens_number, top-k)
        # gate_top_k_val (tokens_number, top-k)

        score = F.softmax(gate / 0.3, dim=-1)
        valid_idx = gate_top_k_idx[gate_top_k_idx > -1]
        fraction_expert = (
            torch.scatter_add(
                torch.zeros(self.tot_expert, device=valid_idx.device),
                0,
                valid_idx,
                torch.ones_like(valid_idx, dtype=torch.float),
            )
            / valid_idx.numel()
        )
        prob_expert = score.sum(dim=0) / valid_idx.numel()

        loss = (fraction_expert * prob_expert).sum() * self.tot_expert
        self.loss = loss

    def forward(self, inp, eigs, return_all_scores=False):

        #inp = self.inp_reduction(inp)

        with torch.no_grad():
            expert_embeddings_norm = self.expert_embeddings.norm(
                p=2.0, dim=1, keepdim=True
            )
            self.expert_embeddings.mul_(1.5 / expert_embeddings_norm)

        gate = self._cosine(inp, self.expert_embeddings, eigs)
        gate = self._make_finite(gate)
        #breakpoint()

        if self.dense_moe_flag:
            gate = torch.ones_like(gate)  # average the importance of all experts
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.tot_expert, dim=-1, largest=True, sorted=False
            )
            gate_top_k_val = gate_top_k_val.view(-1, self.tot_expert)
        else:
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.top_k, dim=-1, largest=True, sorted=False
            )  # [.. x top_k]
            gate_top_k_val = gate_top_k_val.view(-1, self.top_k)  # (BxL) x 1 x top_k

        gate_score = F.softmax(gate_top_k_val, dim=-1)
        if self.g_balance: # should be False
            self.set_load_balance(gate, gate_top_k_idx)

        if return_all_scores:
            return gate_top_k_idx, gate_score, gate
        return gate_top_k_idx, gate_score

    def _cosine(self, mat1, mat2, eigs, eps=1e-4):
        assert mat1.dim() == 2
        assert mat2.dim() == 2
        # mat1 = F.normalize(mat1, p=2.0, dim=1, eps=eps)
        mat1 = self.sphere(mat1, eigs) # LDA sphering
        mat2 = F.normalize(mat2.float(), p=2.0, dim=1, eps=eps)
        return mat1.float().matmul(mat2.transpose(0, 1)).type_as(mat1)
     
    
    def sphere(self, inp, eigs):
        with torch.no_grad():
            # inp shape: [(BxM) x d] ordinary dim projected down to num_exp for dim-reduction LDA style and because current code setup picks top num_exp largest eigenvectors from attention layer
            # need to think about this, it's possible we're losing information here and we shouldn't be cutting down the dimension like this. Otherwise we reweigh all dims by the largest variance eigenvectors
            # as opposed to each feature by its corresponding eigenvector / value
            inp = inp.detach()
            # eigvectors shape: B x d x M
            # eigvals shape: B x d - check
            eigvectors, eigvals = eigs[0].detach(), eigs[1].detach()
            breakpoint()
            n, d = eigvectors.shape
            B, d = eigvals.shape
            
            # sphering computed as X <- D^{-1/2}XU. U: [d,d] eigvectors of the covariance matrix. 
            
            # eigvectors of covariance matrix obtained via projection from eigvectors of gram matrix. u = X'v / eigval : [d,d]
            U = inp.transpose(-2,-1) @ eigvectors.permute(0,2,1).reshape(B*M, -1) # d x d
            #U = U / torch.sqrt(eigvals) # colwise division
            D = 1 / (eigvals + 1e-4) # divide by square root twice, once in U reweight and once in formula of sphering
            
            inp = (inp @ U) * D # B*M x d 
            
        return inp
      

    def _make_finite(self, scores):
        ok = scores.isfinite()
        if not ok.all():
            # NaNs here can break the assignment algorithm
            scores[~ok] = scores[ok].min()
        return scores



class Balance_Elliptical2XMoE(BaseGate): # perform over-layers averaging but of inp(layer) and inp(layer-1) rather than of fwds
    def __init__(self, d_model, num_expert, world_size, top_k=2, g_balance=False, show_gate_W = False, mean_scale = False, root_invert = False, intra_layer  = False):
        super().__init__(num_expert, world_size)
        self.gate = nn.Linear(d_model, self.tot_expert)
        self.top_k = top_k
        self.dense_moe_flag = False
        self.g_balance = g_balance
        self.loss = 0.0

        #expert_embeddings = torch.empty(num_expert, 8)
        expert_embeddings = torch.empty(num_expert, d_model) # no dimension reduction
        torch.nn.init.orthogonal_(expert_embeddings, gain=0.32)
        self.register_parameter(
            "expert_embeddings", torch.nn.Parameter(expert_embeddings)
        )

        self.inp_reduction = torch.nn.Linear(d_model, 8, bias=False) # for smoe-m, hidden_size = 352, num_heads = 8, therefore head_dim = 44 (double check)
        self.show_gate_W = show_gate_W
        self.mean_scale = mean_scale
        self.root_invert = root_invert

    def set_load_balance(self, gate, gate_top_k_idx):
        # gate_top_k_idx (tokens_number, top-k)
        # gate_top_k_val (tokens_number, top-k)

        score = F.softmax(gate / 0.3, dim=-1)
        valid_idx = gate_top_k_idx[gate_top_k_idx > -1]
        fraction_expert = (
            torch.scatter_add(
                torch.zeros(self.tot_expert, device=valid_idx.device),
                0,
                valid_idx,
                torch.ones_like(valid_idx, dtype=torch.float),
            )
            / valid_idx.numel()
        )
        prob_expert = score.sum(dim=0) / valid_idx.numel()

        loss = (fraction_expert * prob_expert).sum() * self.tot_expert
        self.loss = loss

    def forward(self, inp, inp_last =  None, return_all_scores=False):

        #reduced_inp = self.inp_reduction(inp)
        reduced_inp = inp # no dimension reduction

        with torch.no_grad():
            expert_embeddings_norm = self.expert_embeddings.norm(
                p=2.0, dim=1, keepdim=True
            )
            self.expert_embeddings.mul_(1.5 / expert_embeddings_norm)

        gate = self._elliptical_cosine(inp, self.expert_embeddings, inp_last)
        #gate = self._cosine(reduced_inp, self.expert_embeddings)
        gate = self._make_finite(gate)
        #breakpoint()

        if self.dense_moe_flag:
            gate = torch.ones_like(gate)  # average the importance of all experts
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.tot_expert, dim=-1, largest=True, sorted=False
            )
            gate_top_k_val = gate_top_k_val.view(-1, self.tot_expert)
        else:
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.top_k, dim=-1, largest=True, sorted=False
            )  # [.. x top_k]
            gate_top_k_val = gate_top_k_val.view(-1, self.top_k)  # (BxL) x 1 x top_k

        gate_score = F.softmax(gate_top_k_val, dim=-1)
        if self.g_balance: # should be False
            self.set_load_balance(gate, gate_top_k_idx)

        if return_all_scores:
            return gate_top_k_idx, gate_score, gate
        return gate_top_k_idx, gate_score

    def _cosine(self, mat1, mat2, eps=1e-4):
        assert mat1.dim() == 2
        assert mat2.dim() == 2
        # mat1 = F.normalize(mat1, p=2.0, dim=1, eps=eps)
        mat2 = F.normalize(mat2.float(), p=2.0, dim=1, eps=eps)
        return mat1.float().matmul(mat2.transpose(0, 1)).type_as(mat1)
     
    
    def _elliptical_cosine(self, inp, expert_embeddings, inp_last =  None, eps = 1e-4):
        assert inp.dim() == 2
        assert expert_embeddings.dim() == 2
        #breakpoint()
        if inp_last is not None:
            W = self.compute_W(inp, inp_last)
            inp = F.normalize(inp, p=2.0, dim=1, eps=eps)
            expert_embeddings = F.normalize(expert_embeddings.float(), p=2.0, dim=1, eps=eps)
            inp = inp * W
            #breakpoint()

        else:
            inp = F.normalize(inp, p=2.0, dim=1, eps=eps)
            expert_embeddings = F.normalize(expert_embeddings.float(), p=2.0, dim=1, eps=eps)
           
        return inp.float().matmul(expert_embeddings.transpose(0, 1)).type_as(inp)
    
    def compute_W(self, inp, inp_last):
        with torch.no_grad():
            # v shape: [(bsize x seqlen) x dim]
            inp = inp.detach()
            inp_last = inp_last.detach()
            # k_last = k_last.detach()
            # k_last2 = k_last2.detach()

        #return v_last
            seqlen = inp.size(0)
            # if delta is None:
            #     deltas = torch.abs(k_last - k_last2) #include small term for stability and gradient attenuation
            #     difference_quotients = (v_last - v_last2) /deltas # entrywise (v' - v)_nd / (q' - q)_nd

            #delta = torch.mean(torch.abs(k_last - k_last2))
            difference_quotients = inp-inp_last
            
            W = torch.norm(difference_quotients, p = 1, dim = 0) /seqlen #columnwise average l1 norms
            # W dim: [h]
            
            if self.root_invert:
                W = 1/ (torch.sqrt(W) + 1e-3) # trial D^{-1/2} as from sphering the data in LDA

            # store weights  
            W_mean = torch.mean(W)
            W_std = torch.std(W)
            W_max, max_idx = torch.max(W, dim = 0)
            W_min, min_idx = torch.min(W, dim = 0)
            self.gate_W = (W, W_std, W_max, max_idx, W_min, min_idx, W_mean)
            if self.mean_scale:
                weights_scaled = W / torch.mean(W)
            else:
                weights_scaled = W / torch.max(W)
            scaled_mean = torch.mean(weights_scaled)
            scaled_std = torch.std(weights_scaled)
            scaled_max = torch.max(weights_scaled)
            scaled_min = torch.min(weights_scaled)
            self.gate_W_scaled = (weights_scaled, scaled_std, scaled_max, scaled_min, scaled_mean)

            #self.gate_deltas =  torch.max(torch.abs(k_last-k_last2)), torch.min(torch.abs(k_last-k_last2)), torch.mean(torch.abs(k_last - k_last2)), torch.std(torch.abs(k_last-k_last2))

            
            if self.mean_scale:
                W = W / torch.mean(W)
            else: # default max scale
                #W = W / torch.max(W, dim = -1, keepdim=True)[0] # maxscale for multidim W

                W = W / torch.max(W) # W is 0-dim here just [h] vector.
                #print('here')
            #W = torch.diag_embed(W)
            
        return W
      

    def _make_finite(self, scores):
        ok = scores.isfinite()
        if not ok.all():
            # NaNs here can break the assignment algorithm
            scores[~ok] = scores[ok].min()
        return scores




class SpectralGate_SMoE(BaseGate):
    def __init__(self, num_expert, world_size, top_k=2, g_blance=False):
        super().__init__(num_expert, world_size)
        #self.gate = nn.Linear(d_model, self.tot_expert)
        self.top_k = top_k
        self.dense_moe_flag = False
        self.g_blance = g_blance
        self.loss = None
        self.num_expert = num_expert

    def forward(self, attn_logit, return_all_scores=False):

        gate_top_k_val, gate_top_k_idx =  self.spectral_cluster(attn_logit)
        
        gate_score = F.softmax(gate_top_k_val, dim=-1)

        # if self.dense_moe_flag:
        #     gate = torch.ones_like(gate)  # average the importance of all experts
        #     gate_top_k_val, gate_top_k_idx = torch.topk(
        #         gate, k=self.tot_expert, dim=-1, largest=True, sorted=False
        #     )
        #     gate_top_k_val = gate_top_k_val.view(-1, self.tot_expert)
        # else:
        #     gate_top_k_val, gate_top_k_idx = torch.topk(
        #         gate, k=self.top_k, dim=-1, largest=True, sorted=False
        #     )  # [.. x top_k]
        #     gate_top_k_val = gate_top_k_val.view(-1, self.top_k)  # (BxL) x 1 x top_k

        # gate_score = F.softmax(gate_top_k_val, dim=-1)
        # if self.g_blance:
        #     self.set_load_balance(gate, gate_top_k_idx)

        # if return_all_scores:
        #     return gate_top_k_idx, gate_score, gate

        return gate_top_k_idx, gate_score
    
    def spectral_cluster(self, attn_logit, threshold  = None):
        with torch.no_grad():
            attn_logit =  attn_logit.detach()
            bsize, seqlen, _ =  attn_logit.shape
            if threshold is not None:
                # apply thresholding to sparsify
                attn_logit[attn_logit <= threshold] = 0.
            
            D = torch.eye(seqlen, device = attn_logit.device)*seqlen # degree matrix
            L = D - attn_logit # laplacian
            eigvals, eigvecs = torch.linalg.eigh(L) 
            #breakpoint()
            eigvecs = eigvecs[:, :, :self.num_expert] # arranged in ascending order
            eigvecs =  eigvecs.reshape(bsize*seqlen, -1)
            gate_top_k_val, gate_top_k_idx = torch.topk(eigvecs, k = self.top_k, largest=True, sorted = False)
        
        return gate_top_k_val, gate_top_k_idx


class CustomNaiveGate_Balance_SMoE(BaseGate):
    def __init__(self, d_model, num_expert, world_size, top_k=2, g_blance=False):
        super().__init__(num_expert, world_size)
        self.gate = nn.Linear(d_model, self.tot_expert)
        self.top_k = top_k
        self.dense_moe_flag = False
        self.g_blance = g_blance
        self.loss = None

    def set_load_balance(self, gate, gate_top_k_idx):

        score = F.softmax(gate, dim=-1)
        valid_idx = gate_top_k_idx[gate_top_k_idx > -1]
        fraction_expert = (
            torch.scatter_add(
                torch.zeros(self.tot_expert, device=valid_idx.device),
                0,
                valid_idx,
                torch.ones_like(valid_idx, dtype=torch.float),
            )
            / valid_idx.numel()
        )
        prob_expert = score.sum(dim=0) / valid_idx.numel()

        loss = (fraction_expert * prob_expert).sum() * self.tot_expert
        self.loss = loss

    def forward(self, inp, return_all_scores=False):

        gate = self.gate(inp)

        if self.dense_moe_flag:
            gate = torch.ones_like(gate)  # average the importance of all experts
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.tot_expert, dim=-1, largest=True, sorted=False
            )
            gate_top_k_val = gate_top_k_val.view(-1, self.tot_expert)
        else:
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.top_k, dim=-1, largest=True, sorted=False
            )  # [.. x top_k]
            gate_top_k_val = gate_top_k_val.view(-1, self.top_k)  # (BxL) x 1 x top_k

        gate_score = F.softmax(gate_top_k_val, dim=-1)
        if self.g_blance:
            self.set_load_balance(gate, gate_top_k_idx)

        if return_all_scores:
            return gate_top_k_idx, gate_score, gate
        return gate_top_k_idx, gate_score

class EllipticalSMoE(BaseGate):
    def __init__(self, d_model, num_expert, world_size, top_k=2, g_blance=False):
        super().__init__(num_expert, world_size)
        self.gate = nn.Linear(d_model, self.tot_expert)
        self.top_k = top_k
        self.dense_moe_flag = False
        self.g_blance = g_blance
        self.loss = None

        self.gate_W = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.gate_W_scaled = (0.0, 0.0, 0.0, 0.0, 0.0)

    def set_load_balance(self, gate, gate_top_k_idx):

        score = F.softmax(gate, dim=-1)
        valid_idx = gate_top_k_idx[gate_top_k_idx > -1]
        fraction_expert = (
            torch.scatter_add(
                torch.zeros(self.tot_expert, device=valid_idx.device),
                0,
                valid_idx,
                torch.ones_like(valid_idx, dtype=torch.float),
            )
            / valid_idx.numel()
        )
        prob_expert = score.sum(dim=0) / valid_idx.numel()

        loss = (fraction_expert * prob_expert).sum() * self.tot_expert
        self.loss = loss

    def forward(self, inp, fwds, return_all_scores=False):

        W = self.compute_W(fwds)
        inp = inp @ W

        gate = self.gate(inp)

        if self.dense_moe_flag:
            gate = torch.ones_like(gate)  # average the importance of all experts
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.tot_expert, dim=-1, largest=True, sorted=False
            )
            gate_top_k_val = gate_top_k_val.view(-1, self.tot_expert)
        else:
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.top_k, dim=-1, largest=True, sorted=False
            )  # [.. x top_k]
            gate_top_k_val = gate_top_k_val.view(-1, self.top_k)  # (BxL) x 1 x top_k

        gate_score = F.softmax(gate_top_k_val, dim=-1)
        if self.g_blance:
            self.set_load_balance(gate, gate_top_k_idx)

        if return_all_scores:
            return gate_top_k_idx, gate_score, gate
        return gate_top_k_idx, gate_score
    
    def compute_W(self, fwds):
        with torch.no_grad():
            # fwds shape: [(bsize x seqlen) x dim]
            fwds = fwds.detach()
            
            w =  torch.var(fwds, dim = 0) # shape: [dim]
            dim = w.shape[0]

            # if self.exp_distance:
            #     if lamb is None:
            #         #lamb  = torch.sqrt(torch.tensor(dim))   
            #         lamb = torch.tensor(1.)
            #     w = F.softmax(-w / lamb)
            
            ## invert
            w = 1 / (w + 1e-4)

            # featscale
            w = w / w.mean()
            
            # store weights  
            W_mean = torch.mean(w)
            W_std = torch.std(w)
            W_max, max_idx = torch.max(w, dim = 0)
            W_min, min_idx = torch.min(w, dim = 0)
            
            self.gate_W = (w, W_std, W_max, max_idx, W_min, min_idx, W_mean)

        return torch.diag_embed(w)

class CustomNaiveGate_Balance_XMoE(BaseGate):
    def __init__(self, d_model, num_expert, world_size, top_k=2, g_balance=False):
        super().__init__(num_expert, world_size)
        self.gate = nn.Linear(d_model, self.tot_expert)
        self.top_k = top_k
        self.dense_moe_flag = False
        self.g_balance = g_balance
        self.loss = 0.0

        expert_embeddings = torch.empty(num_expert, 8) # rather than 8 as default
        torch.nn.init.orthogonal_(expert_embeddings, gain=0.32)
        self.register_parameter(
            "expert_embeddings", torch.nn.Parameter(expert_embeddings)
        )

        self.inp_reduction = torch.nn.Linear(d_model, 8, bias=False)

    def set_load_balance(self, gate, gate_top_k_idx):
        # gate_top_k_idx (tokens_number, top-k)
        # gate_top_k_val (tokens_number, top-k)

        score = F.softmax(gate / 0.3, dim=-1)
        valid_idx = gate_top_k_idx[gate_top_k_idx > -1]
        fraction_expert = (
            torch.scatter_add(
                torch.zeros(self.tot_expert, device=valid_idx.device),
                0,
                valid_idx,
                torch.ones_like(valid_idx, dtype=torch.float),
            )
            / valid_idx.numel()
        )
        prob_expert = score.sum(dim=0) / valid_idx.numel()

        loss = (fraction_expert * prob_expert).sum() * self.tot_expert
        self.loss = loss

    def forward(self, inp, return_all_scores=False):
        
        reduced_inp = self.inp_reduction(inp)
        #reduced_inp = inp # no input reduction
        
        with torch.no_grad():
            expert_embeddings_norm = self.expert_embeddings.norm(
                p=2.0, dim=1, keepdim=True
            )
            self.expert_embeddings.mul_(1.5 / expert_embeddings_norm)

        gate = self._cosine(reduced_inp, self.expert_embeddings)
        gate = self._make_finite(gate)

        if self.dense_moe_flag:
            gate = torch.ones_like(gate)  # average the importance of all experts
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.tot_expert, dim=-1, largest=True, sorted=False
            )
            gate_top_k_val = gate_top_k_val.view(-1, self.tot_expert)
        else:
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.top_k, dim=-1, largest=True, sorted=False
            )  # [.. x top_k]
            gate_top_k_val = gate_top_k_val.view(-1, self.top_k)  # (BxL) x 1 x top_k

        gate_score = F.softmax(gate_top_k_val, dim=-1)
        if self.g_balance:
            self.set_load_balance(gate, gate_top_k_idx)

        if return_all_scores:
            return gate_top_k_idx, gate_score, gate
        return gate_top_k_idx, gate_score

    def _cosine(self, mat1, mat2, eps=1e-4):
        assert mat1.dim() == 2
        assert mat2.dim() == 2
        # mat1 = F.normalize(mat1, p=2.0, dim=1, eps=eps)
        mat2 = mat2.float()
        mat2 = F.normalize(mat2, p=2.0, dim=1, eps=eps)
        return mat1.float().matmul(mat2.transpose(0, 1)).type_as(mat1)

    def _make_finite(self, scores):
        ok = scores.isfinite()
        if not ok.all():
            # NaNs here can break the assignment algorithm
            scores[~ok] = scores[ok].min()
        return scores

class COSAGate_Balance(BaseGate):
    def __init__(self, d_model, num_expert, world_size, top_k=2, g_balance=False, use_var = False, smoe_base = False, mad = False, mix_weights = False, skip_connect = False, temp_disp = False):
        super().__init__(num_expert, world_size)
        self.gate = nn.Linear(d_model, self.tot_expert)
        self.top_k = top_k
        self.dense_moe_flag = False
        self.g_balance = g_balance
        self.loss = 0.0
        self.use_var = use_var
        self.smoe_base = smoe_base
        self.mad = mad
        self.mix_weights = mix_weights
        self.skip_connect = skip_connect
        self.temp_disp = temp_disp

        # if skip_connect:
        #     self.norm1 = nn.LayerNorm(d_model)

        expert_embeddings = torch.empty(num_expert, d_model)
        torch.nn.init.orthogonal_(expert_embeddings, gain=0.32)
        self.register_parameter(
            "expert_embeddings", torch.nn.Parameter(expert_embeddings)
        )

        #self.inp_reduction = torch.nn.Linear(d_model, 8, bias=False)
        self.avg_distances_per_cluster = 0.0, 0.0, 0.0, 0.0, 0.0
        self.W = 0.0, 0.0, 0.0, 0.0, 0.0

    def set_load_balance(self, gate, gate_top_k_idx):
        # gate_top_k_idx (tokens_number, top-k)
        # gate_top_k_val (tokens_number, top-k)

        score = F.softmax(gate / 0.3, dim=-1)
        valid_idx = gate_top_k_idx[gate_top_k_idx > -1]
        fraction_expert = (
            torch.scatter_add(
                torch.zeros(self.tot_expert, device=valid_idx.device),
                0,
                valid_idx,
                torch.ones_like(valid_idx, dtype=torch.float),
            )
            / valid_idx.numel()
        )
        prob_expert = score.sum(dim=0) / valid_idx.numel()

        loss = (fraction_expert * prob_expert).sum() * self.tot_expert
        self.loss = loss

    def forward(self, inp, gate_top_k_idx, gate_score = None, return_all_scores=False, moe_inp_last = None, fwds = None):
        
        #reduced_inp = self.inp_reduction(inp)
        if self.smoe_base:
            if self.use_var:
                W = self.compute_W_var(inp, gate_top_k_idx, gate_score)
                #print('using smoe')

            else:
                W = self.compute_W(inp, gate_top_k_idx, gate_score)
            
            inp = inp* W

            gate = self.gate(inp)

            if self.dense_moe_flag:
                gate = torch.ones_like(gate)  # average the importance of all experts
                gate_top_k_val, gate_top_k_idx = torch.topk(
                    gate, k=self.tot_expert, dim=-1, largest=True, sorted=False
                )
                gate_top_k_val = gate_top_k_val.view(-1, self.tot_expert)
            else:
                gate_top_k_val, gate_top_k_idx = torch.topk(
                    gate, k=self.top_k, dim=-1, largest=True, sorted=False
                )  # [.. x top_k]
                gate_top_k_val = gate_top_k_val.view(-1, self.top_k)  # (BxL) x 1 x top_k

                gate_score = F.softmax(gate_top_k_val, dim=-1)
                if self.g_balance:
                    self.set_load_balance(gate, gate_top_k_idx)

                if return_all_scores:
                    return gate_top_k_idx, gate_score, gate
                return gate_top_k_idx, gate_score

        else: # xmoe base
            #reduced_inp = self.inp_reduction(inp)
            reduced_inp = inp
            with torch.no_grad():
                expert_embeddings_norm = self.expert_embeddings.norm(
                    p=2.0, dim=1, keepdim=True
                )
                self.expert_embeddings.mul_(1.5 / expert_embeddings_norm)

            gate = self._cosa_dot(reduced_inp, self.expert_embeddings, cluster_labels= gate_top_k_idx, mixing_weights= gate_score, moe_inp_last=moe_inp_last, fwds = fwds)
            gate = self._make_finite(gate)

            if self.dense_moe_flag:
                gate = torch.ones_like(gate)  # average the importance of all experts
                gate_top_k_val, gate_top_k_idx = torch.topk(
                    gate, k=self.tot_expert, dim=-1, largest=True, sorted=False
                )
                gate_top_k_val = gate_top_k_val.view(-1, self.tot_expert)
            else:
                gate_top_k_val, gate_top_k_idx = torch.topk(
                    gate, k=self.top_k, dim=-1, largest=True, sorted=False
                )  # [.. x top_k]
                gate_top_k_val = gate_top_k_val.view(-1, self.top_k)  # (BxL) x 1 x top_k

            gate_score = F.softmax(gate_top_k_val, dim=-1)
            if self.g_balance:
                self.set_load_balance(gate, gate_top_k_idx)

            if return_all_scores:
                return gate_top_k_idx, gate_score, gate
            return gate_top_k_idx, gate_score
    
    def _cosa_dot(self, mat1, mat2, cluster_labels, mixing_weights = None, moe_inp_last = None, fwds = None, eps = 1e-4):
        if self.use_var:
            W = self.compute_W_var(mat1, cluster_labels, mixing_weights= mixing_weights, moe_inp_last= moe_inp_last, fwds = fwds)
            #print('using var xmoe')
        else:
            W = self.compute_W(mat1, cluster_labels, mixing_weights)
            
        assert W.requires_grad == False
        
        mat2 = F.normalize(mat2.float(), p=2.0, dim=1, eps=eps)
        mat1 = mat1 * W
        
        return mat1.float().matmul(mat2.transpose(0, 1)).type_as(mat1)

    def compute_W(self, tokens, cluster_assignments, mixing_weights = None, lamb = None):
        with torch.no_grad():
            tokens = tokens.detach()
            cluster_assignments = cluster_assignments.detach()
            
            top_cluster_assignments = cluster_assignments[:,0] # take only top cluster assignment as label

            # downsampling trial
            # random_indices = torch.randperm(tokens.shape[0])[:1000] # 1000 arbitrarily chosen
            # tokens = tokens[random_indices]
            # cluster_assignments_original = top_cluster_assignments
            # top_cluster_assignments = top_cluster_assignments[random_indices]
            #if not self.use_var:
            # get [n,n,d] distances tensor
            n, d = tokens.shape
            
            # use absolute distance
            distances = torch.abs(tokens[:, None, :] - tokens[None, :, :])

            # use scaled squared distance
            # tokens = tokens / (torch.var(tokens, dim = 0) +1e-4)
            # distances = (tokens[:, None, :] - tokens[None, :, :])**2
            
            # [n,n,d] -> [n*n, d]
            distances = distances.flatten(start_dim=0, end_dim=1)

            d = distances.shape[1]
            k = cluster_assignments.max().item() + 1
            # [n,n] matrix representing whether two points are in the same cluster
            cluster_pair_assignments = top_cluster_assignments.unsqueeze(1) == top_cluster_assignments.unsqueeze(0)
            # [n,n] matrix representing the cluster assignments of each pair of points, = 0 if not in the same cluster
            cluster_pair_assignments = cluster_pair_assignments * (top_cluster_assignments + 1).unsqueeze(1)
            # [n,n] -> [d,n*n] and fill diagonal with 0 to avoid calculate [i,i] pairs
            cluster_pair_assignments = cluster_pair_assignments.fill_diagonal_(0).flatten()
            #breakpoint()
            # [d,n*n] -> [k+1,d] using cluster_pair_assignments as index to reduce
            # if self.use_var:
            #     avg_distances_per_cluster = torch.zeros((k + 1,d), device = cluster_assignments.device).index_reduce(0, cluster_pair_assignments, tokens, 'var', include_self=False)
            # else:
            avg_distances_per_cluster = torch.zeros((k + 1,d), device = top_cluster_assignments.device).index_reduce(0, cluster_pair_assignments, distances, 'mean', include_self=False)
            #avg_distances_per_cluster = torch.ones((k + 1,d), device = top_cluster_assignments.device).index_reduce(0, cluster_pair_assignments, distances, 'mean', include_self=False) # fill ones when downsampling
            #breakpoint()
            # eliminate the first row since it is the average of pairs not in the same cluster
            avg_distances_per_cluster = avg_distances_per_cluster[1:]
                        
            
            # get exponential distances
            if lamb is None:
                lamb = torch.tensor(0.5)
            #breakpoint()
            #W = torch.softmax(-avg_distances_per_cluster / lamb, dim = -1)
            W = 1 / (avg_distances_per_cluster + 1e-4) # [k,d] cluster weights
            self.W = W[:2], W.min(), W.max(), W.mean(), W.std()

            if self.mix_weights:
                assert mixing_weights is not None
                # Create an index tensor of shape (n, 2) based on the labels
                index = cluster_assignments.view(n, self.top_k, 1).expand(n, 2, d)
                # Add a new dimension to the weights tensor to match the index tensor
                W = W.unsqueeze(0).expand(n, k, d)
                # Gather the corresponding weights for each label. 
                W = torch.gather(W, 1, index) # [n, top_k, d]

                # mix weights according to the gate top k val
                mixing_weights = mixing_weights.view(n, self.top_k, 1)
                W = (W*mixing_weights).sum(dim = 1) # n x d
            
            else:
                # broadcast up to shape [n, d] which rows arranged by corresponding cluster assignments of n
                W = W[top_cluster_assignments, :]


        return W

    def compute_W_var(self, tokens, cluster_assignments, mixing_weights = None, moe_inp_last = None, fwds = None, lamb = None):
        with torch.no_grad():
            tokens = tokens.detach()
            cluster_assignments = cluster_assignments.detach()
            
            top_cluster_assignments = cluster_assignments[:,0] # take only top cluster assignment as label
            n, d = tokens.shape
            k = cluster_assignments.max().item() + 1

            if self.temp_disp:
                fwd, fwd_last = fwds[0].detach(), fwds[1].detach()
                diff_mat = torch.abs(fwd - fwd_last)[::self.top_k] # compute dimwise differences and take alternating elements (corresponding to each tokens top 1 assignment)
                
                assert diff_mat.shape == tokens.shape
                mean_dimwise_clusterwise = torch.zeros((k,d), device = tokens.device).index_reduce(0, top_cluster_assignments, diff_mat, 'mean', include_self=False)
                
                # invert
                W = 1 / (mean_dimwise_clusterwise + 1e-4)

                # clip thresholding
                W_mean = W.mean()
                clip_threshold = 10 * W_mean
                W = torch.clamp(W, max = clip_threshold)
                
                # weight scale
                #print(f'scaling by {W.mean(dim=0).shape} features')
                W = W/ W.mean(dim = 0) # dimwise meanscaling
                #print('using temp disp')
            if self.skip_connect:
                assert moe_inp_last is not None
                assert moe_inp_last.shape == tokens.shape

                moe_inp_last = moe_inp_last.detach()
                mix = 0.75
                tokens = mix*tokens + (1-mix)*moe_inp_last
                #print('using the skip connection')


            # global_abs_dev = torch.mean(torch.abs(tokens - torch.mean(tokens, dim=0)), dim=0) # mean absolute dev per dim
            # tokens = tokens / (global_abs_dev + 1e-4) # scale dimwise stdev
            # #print(global_abs_dev)
    
            if self.mad: # mean absolute deviation
                mean_dimwise_clusterwise = torch.zeros((k,d), device = tokens.device).index_reduce(0, top_cluster_assignments, tokens, 'mean', include_self=False)
                centered_tokens = torch.abs(tokens - mean_dimwise_clusterwise[top_cluster_assignments, :]) # abs of each token minus its cluster mean
                W_mad = torch.zeros((k,d), device = tokens.device).index_reduce(0, top_cluster_assignments, centered_tokens, 'mean', include_self = False)
                
                # add noise
                # if self.training:
                #     W_mad = W_mad + torch.randn_like(W_mad)*0.01 # 0.05 original stdev
                #     W_mad = torch.clamp(W_mad, min = 0) # no negative mads
                
                if False: # softmax
                    lamb = 0.5
                    W = W_mad + 1e-4 # add small term for 0 var stability
                    W_mean = W.mean()
                    clip_threshold =  W_mean / 10
                    W = torch.clamp(W, min = clip_threshold)

                    W = torch.softmax(-W / lamb, dim = 1)
                    
                    W = W/ W.mean(dim = 0) # dimwise meanscaling
                    #W = W/W.mean(dim=1).unsqueeze(1) # clusterwise meanscale
                    # print(f'Min: {W.min()}')
                    # print(f'Max" {W.max()}')
                else:

                    # invert
                    W = 1 / (W_mad + 1e-4) # previous eps=0.1

                    # clip thresholding
                    W_mean = W.mean()
                    clip_threshold = 10 * W_mean # trial eval only 1.5 * mean clamp roof
                    W = torch.clamp(W, max = clip_threshold)
                    
                    # weight scale
                    #print(f'scaling by {W.mean(dim=0).shape} features')
                    W = W/ W.mean(dim = 0) # dimwise meanscaling
                    #print('here mad')
                    
                    #W = W / W.mean(dim = 1).unsqueeze(1) # clusterwise meanscaling
                    # W = W / W_mean # global mean feature scaling

                    # no cosa
                    # W = torch.ones_like(W_mad)
            
                
            else:
                mean_dimwise_clusterwise = torch.zeros((k,d), device = tokens.device).index_reduce(0, top_cluster_assignments, tokens, 'mean', include_self=False)
                mean_square_dimwise_clusterwise = torch.zeros((k,d), device = tokens.device).index_reduce(0, top_cluster_assignments, torch.square(tokens), 'mean', include_self=False)
                variance_dimwise_clusterwise = mean_square_dimwise_clusterwise - mean_dimwise_clusterwise**2
                W = 1 / ((variance_dimwise_clusterwise) + 0.1) # [k,d] cluster-feature weights # adjust this eps=0.1 as an attentuation on the weights. Trial 2x by equivalence MeanPairwiseDistance = 2Var
                #print('here var')
            #self.W = W[:2], W.min(), W.max(), W.mean(), W.std()
            #print('here cosa')

            if self.mix_weights:
                assert mixing_weights is not None
                
                # Create an index tensor of shape (n, 2) based on the labels
                index = cluster_assignments.view(n, self.top_k, 1).expand(n, 2, d)
                # Add a new dimension to the weights tensor to match the index tensor
                W = W.unsqueeze(0).expand(n, k, d)
                # Gather the corresponding weights for each label. 
                W = torch.gather(W, 1, index) # [n, top_k, d]
                #breakpoint()
                # mix weights according to the gate top k val
                mixing_weights = mixing_weights.view(n, self.top_k, 1)
                W = (W*mixing_weights).sum(dim = 1) # n x d
                
            else:
                # broadcast up to shape [n, d] which rows arranged by corresponding cluster assignments of n
                W = W[top_cluster_assignments, :] # rescale via MAPD - MAD relationship
                

        return W

    def _make_finite(self, scores):
        ok = scores.isfinite()
        if not ok.all():
            # NaNs here can break the assignment algorithm
            scores[~ok] = scores[ok].min()
        return scores



class CustomNaiveGate_Balance_StableMoE(BaseGate):
    r"""
    Naive Gate StableMoE
    """

    def __init__(self, d_model, num_expert, world_size, top_k=2, g_balance=False):
        super().__init__(num_expert, world_size)
        self.top_k = top_k
        self.dense_moe_flag = False
        self.g_balance = g_balance
        self.loss = 0.0

        expert_embeddings = torch.empty(num_expert, d_model)
        torch.nn.init.orthogonal_(expert_embeddings, gain=0.32)
        self.register_parameter(
            "expert_embeddings", torch.nn.Parameter(expert_embeddings)
        )

    def set_load_balance(self, gate, gate_top_k_idx):

        score = F.softmax(gate / 0.3, dim=-1)
        valid_idx = gate_top_k_idx[gate_top_k_idx > -1]
        fraction_expert = (
            torch.scatter_add(
                torch.zeros(self.tot_expert, device=valid_idx.device),
                0,
                valid_idx,
                torch.ones_like(valid_idx, dtype=torch.float),
            )
            / valid_idx.numel()
        )
        prob_expert = score.sum(dim=0) / valid_idx.numel()

        loss = (fraction_expert * prob_expert).sum() * self.tot_expert
        self.loss = loss

    def forward(self, inp, return_all_scores=False):

        gate = self._cosine(inp, self.expert_embeddings)
        gate = self._make_finite(gate)

        if self.dense_moe_flag:
            gate = torch.ones_like(gate)  # average the importance of all experts
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.tot_expert, dim=-1, largest=True, sorted=False
            )
            gate_top_k_val = gate_top_k_val.view(-1, self.tot_expert)
        else:
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.top_k, dim=-1, largest=True, sorted=False
            )  # [.. x top_k]
            gate_top_k_val = gate_top_k_val.view(-1, self.top_k)
        # (BxL) x 1 x top_k

        gate_score = F.softmax(gate_top_k_val, dim=-1)
        if self.g_balance:
            self.set_load_balance(gate, gate_top_k_idx)

        if return_all_scores:
            return gate_top_k_idx, gate_score, gate
        return gate_top_k_idx, gate_score

    def _cosine(self, mat1, mat2, eps=1e-4):
        assert mat1.dim() == 2
        assert mat2.dim() == 2
        # mat1 = F.normalize(mat1, p=2.0, dim=1, eps=eps)
        mat2 = F.normalize(mat2.float(), p=2.0, dim=1, eps=eps)
        return mat1.float().matmul(mat2.transpose(0, 1)).type_as(mat1)

    def _make_finite(self, scores):
        ok = scores.isfinite()
        if not ok.all():
            # NaNs here can break the assignment algorithm
            scores[~ok] = scores[ok].min()
        return scores

class KSpectral_Balance_SMoE(BaseGate):
    def __init__(self, d_model, num_expert, world_size, top_k=2, g_blance=False, gate_then_mix = False, gate_with_eigenvectors = True):
        super().__init__(num_expert, world_size)
        self.gate = nn.Linear(d_model, self.tot_expert)
        self.eigengate = nn.Linear(self.tot_expert, self.tot_expert)
        self.top_k = top_k
        self.dense_moe_flag = False
        self.g_blance = g_blance
        self.loss = None

        self.gate_then_mix = gate_then_mix
        self.gate_with_eigenvectors = gate_with_eigenvectors

        expert_embeddings = torch.empty(num_expert, d_model)
        torch.nn.init.orthogonal_(expert_embeddings, gain=0.32)
        self.register_parameter(
            "expert_embeddings", torch.nn.Parameter(expert_embeddings)
        )

        self.W = 0.0, 0.0, 0.0, 0.0, 0.0
        self.eigen_var_weight = False
        self.xmoe_base = False

    def set_load_balance(self, gate, gate_top_k_idx):

        score = F.softmax(gate, dim=-1)
        valid_idx = gate_top_k_idx[gate_top_k_idx > -1]
        fraction_expert = (
            torch.scatter_add(
                torch.zeros(self.tot_expert, device=valid_idx.device),
                0,
                valid_idx,
                torch.ones_like(valid_idx, dtype=torch.float),
            )
            / valid_idx.numel()
        )
        prob_expert = score.sum(dim=0) / valid_idx.numel()

        loss = (fraction_expert * prob_expert).sum() * self.tot_expert
        self.loss = loss

    def forward(self, inp, eigenvectors, return_all_scores=False):
        if self.eigen_var_weight:
            
            W = self.compute_W(eigenvectors)
            inp = inp * W
        
        if self.xmoe_base:
            with torch.no_grad():
                expert_embeddings_norm = self.expert_embeddings.norm(
                    p=2.0, dim=1, keepdim=True
                )
                self.expert_embeddings.mul_(1.5 / expert_embeddings_norm)

            gate = self._cosine(inp, self.expert_embeddings)
            gate = self._make_finite(gate)

            if self.dense_moe_flag:
                gate = torch.ones_like(gate)  # average the importance of all experts
                gate_top_k_val, gate_top_k_idx = torch.topk(
                    gate, k=self.tot_expert, dim=-1, largest=True, sorted=False
                )
                gate_top_k_val = gate_top_k_val.view(-1, self.tot_expert)
            else:
                gate_top_k_val, gate_top_k_idx = torch.topk(
                    gate, k=self.top_k, dim=-1, largest=True, sorted=False
                )  # [.. x top_k]
                gate_top_k_val = gate_top_k_val.view(-1, self.top_k)  # (BxL) x 1 x top_k

            gate_score = F.softmax(gate_top_k_val, dim=-1)
            if self.g_blance:
                self.set_load_balance(gate, gate_top_k_idx)

            if return_all_scores:
                return gate_top_k_idx, gate_score, gate
            return gate_top_k_idx, gate_score

        assert (self.gate_then_mix != self.gate_with_eigenvectors) # exactly one has to be true
        if self.gate_then_mix:
            gate = self.gate(inp) # d_model -> num_exp
            gate = self.mix_embedding(gate, eigenvectors)
        
        elif self.gate_with_eigenvectors:
            eigenvectors = eigenvectors.detach()
            assert eigenvectors.requires_grad == False
            B, _, M = eigenvectors.shape
            eigenvectors = eigenvectors.reshape(B*M, -1)
            #gate = self.eigengate(eigenvectors)
            gate = eigenvectors # try with no gate at all
            #print('here')


        if self.dense_moe_flag:
            gate = torch.ones_like(gate)  # average the importance of all experts
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.tot_expert, dim=-1, largest=True, sorted=False
            )
            gate_top_k_val = gate_top_k_val.view(-1, self.tot_expert)
        else:
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.top_k, dim=-1, largest=True, sorted=False
            )  # [.. x top_k]
            gate_top_k_val = gate_top_k_val.view(-1, self.top_k)  # (BxL) x 1 x top_k

        gate_score = F.softmax(gate_top_k_val, dim=-1)
        if self.g_blance:
            self.set_load_balance(gate, gate_top_k_idx)

        if return_all_scores:
            return gate_top_k_idx, gate_score, gate
        return gate_top_k_idx, gate_score
    
    def mix_embedding(self, gate, eigenvectors):
        # mix gate output with its own spectral embedding
        # eigvectors:  B x num_exp x M
        # gate: B_M x num_exp
        with torch.no_grad():
            eigenvectors = eigenvectors.detach().clone()
            gate = gate.detach().clone()
            B, _, M = eigenvectors.shape

            eigenvectors = eigenvectors.reshape(B*M, -1)
            # add and normlalize
            out = gate + eigenvectors 
            out = torch.nn.functional.normalize(out, p = 2)
        return out

    def _cosine(self, mat1, mat2, eps=1e-4):
        assert mat1.dim() == 2
        assert mat2.dim() == 2
        # mat1 = F.normalize(mat1, p=2.0, dim=1, eps=eps)
        mat2 = F.normalize(mat2.float(), p=2.0, dim=1, eps=eps)
        return mat1.float().matmul(mat2.transpose(0, 1)).type_as(mat1)

    def _make_finite(self, scores):
        ok = scores.isfinite()
        if not ok.all():
            # NaNs here can break the assignment algorithm
            scores[~ok] = scores[ok].min()
        return scores
    
    def compute_W(self, eigenvectors):
        with torch.no_grad():
            
            eigenvectors = eigenvectors.detach()
            B, M, D = eigenvectors.shape # integrate this further up
            eigenvectors = eigenvectors.reshape(B*M, D)
            W = torch.var(eigenvectors, dim = 0)
            W = 1 / (W + 1e-4)
            self.W = W, W.min(), W.max(), W.mean(), W.std()
            #print('kspec here')
        return W

    

class KmeansSpectral(BaseGate):
    def __init__(self, d_model, num_expert, world_size, top_k=2, g_blance=False, gate_then_mix=True, gate_with_eigenvectors=False):
        super().__init__(num_expert, world_size)
        self.kmeans = KMeans(n_clusters=self.tot_expert)
        self.eigengate = nn.Linear(self.tot_expert, self.tot_expert)
        self.top_k = top_k
        self.g_blance = g_blance
        self.loss = None

    def set_load_balance(self, gate, gate_top_k_idx):

        score = F.softmax(gate, dim=-1)
        valid_idx = gate_top_k_idx[gate_top_k_idx > -1]
        fraction_expert = (
            torch.scatter_add(
                torch.zeros(self.tot_expert, device=valid_idx.device),
                0,
                valid_idx,
                torch.ones_like(valid_idx, dtype=torch.float),
            )
            / valid_idx.numel()
        )
        prob_expert = score.sum(dim=0) / valid_idx.numel()

        loss = (fraction_expert * prob_expert).sum() * self.tot_expert
        self.loss = loss

    def forward(self, inp, eigenvectors, return_all_scores=False):
        # eigvectors:  B x num_exp x M
        B, num_exp, M = eigenvectors.shape
        with torch.no_grad():
            #breakpoint()
            eigenvectors = eigenvectors.detach()
            eigenvectors_cpu = eigenvectors.reshape(B*M, -1).cpu() # sklearn not gpu compatible
            labels = self.kmeans.fit_predict(eigenvectors_cpu)
            labels = torch.tensor(labels, device = eigenvectors.device, dtype = torch.int64)

        gate = self.eigengate(eigenvectors.transpose(-1,-2))
        gate_top_k_val, gate_top_k_idx = torch.topk(
            gate, k=1, dim=-1, largest=True, sorted=False
        )  # [.. x top_k]
        gate_top_k_idx = labels
        #print(torch.sum(labels>15))
        # gate_top_k_idx = torch.cat((labels, gate_top_k_idx), dim=-1)
        # eigen_val = eigenvectors.transpose(-1,-2).gather(-1,labels)
        # # gate_top_k_val = torch.cat((eigen_val, gate_top_k_val), dim=-1)
        # gate_top_k_val = eigen_val
        # gate_top_k_val = gate_top_k_val.view(-1, self.top_k)  # (BxL) x 1 x top_k

        # gate_score = F.softmax(gate_top_k_val, dim=-1)
        # if self.g_blance:
        #     self.set_load_balance(gate, gate_top_k_idx)

        gate_score = torch.ones(B*M, device = gate_top_k_idx.device)
        if return_all_scores:
            return gate_top_k_idx, gate_score, gate
        return gate_top_k_idx, gate_score
