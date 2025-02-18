from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath
from typing import Any, Callable, Dict, Optional, Set, Tuple, Type, Union, List
from timm.layers.helpers import to_2tuple


def softmax(x: torch.Tensor, dim: int | tuple[int, ...])  -> torch.Tensor:
# def softmax(x: torch.Tensor, dim: Union[int, Tuple[int, ...]])  -> torch.Tensor: # for python <3.10
    """
    Compute the softmax along the specified dimensions.
    This function adds the option to specify multiple dimensions

    Args:
        x (torch.Tensor): Input tensor.
        dims (int or tuple[int]): The dimension or list of dimensions along which the softmax probabilities are computed.

    Returns:
        torch.Tensor: Output tensor containing softmax probabilities along the specified dimensions.
    """
    max_vals = torch.amax(x, dim=dim, keepdim=True)
    e_x = torch.exp(x - max_vals)
    sum_exp = e_x.sum(dim=dim, keepdim=True)
    return e_x / sum_exp


class SoftMoELayerWrapper(nn.Module):
    """
    A wrapper class to create a Soft Mixture of Experts layer.

    From "From Sparse to Soft Mixtures of Experts"
    https://arxiv.org/pdf/2308.00951.pdf
    """

    def __init__(
        self,
        dim: int,
        num_experts: int,
        slots_per_expert: int,
        layer: Callable,
        normalize: bool = True,
        acmoe = False,
        return_topk = False,
        mad = True,
        mix_weights = False,
        mix_k = 8,
        **layer_kwargs,
    ) -> None:
        """
        Args:
            dim (int): Dimensionality of input features.
            num_experts (int): Number of experts.
            slots_per_expert (int): Number of token slots per expert.
            layer (Callable): Network layer of the experts.
            normalize (bool): Normalize input and phi (sec. 2.3 from paper)
            **layer_kwargs: Additional keyword arguments for the layer class.
        """
        super().__init__()

        self.dim = dim
        self.num_experts = num_experts
        self.slots_per_expert = slots_per_expert
        self.normalize = normalize

        # Initialize phi and normalization scaling factor
        self.phi = nn.Parameter(torch.zeros(dim, num_experts, slots_per_expert))
        if self.normalize:
            self.scale = nn.Parameter(torch.ones(1))

        # Initialize phi using LeCun normal initialization
        # https://github.com/google-research/vmoe/blob/662341d007650d5bbb7c6a2bef7f3c759a20cc7e/vmoe/projects/soft_moe/router.py#L49C1-L49C1
        nn.init.normal_(self.phi, mean=0, std=1 / dim**0.5)

        # Create a list of expert networks
        self.experts = nn.ModuleList(
            [layer(**layer_kwargs) for _ in range(num_experts)]
        )

        self.acmoe = acmoe
        self.return_topk = return_topk
        self.mad = mad
        self.mix_weights = mix_weights

        self.is_moe = True
        self.mix_k = mix_k

        self.router_stability = 0.0

    def forward(self, x: torch.Tensor, cluster_assignments = None, mixing_weights = None) -> torch.Tensor:
        """
        Forward pass through the Soft-MoE layer (algorithm 1 from paper).

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, input_dim].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, input_dim].
        """
        assert (
            x.shape[-1] == self.dim
        ), f"Input feature dim of {x.shape[-1]} does not match layer dim of {self.dim}"
        assert (
            len(x.shape) == 3
        ), f"Input expected to have 3 dimensions but has {len(x.shape)}"
        
        b, m, d = x.shape
        
        phi = self.phi

        # Normalize input and phi
        if self.normalize:
            x = F.normalize(x, dim=2)  # [b, m, d]
            phi = self.scale * F.normalize(phi, dim=0)  # [d, n, p]

        # Compute dispatch and combine weights
        #breakpoint()
        if self.acmoe:
            M = self.compute_acmoe(x, cluster_assignments = cluster_assignments, mixing_weights=mixing_weights)
            x = x.view(-1, d) * M
            x = x.view(b,m,d)
        
        # elif self.expert_acmoe:
        #     M = self.expert_M # [n,d]
        #     phi = M.transpose(-2,-1).unsqueeze() * phi # [d,n,p]
            
        
        logits = torch.einsum("bmd,dnp->bmnp", x, phi)

        #### get top cluster assignments ###
        b, m, n, p = logits.shape
        expert_affinities = logits.view(b, m, n*p)

        #breakpoint()
        ### compute router stability ###
        
        topk_previous = cluster_assignments # store previous assignments
        ### compute router stability ###

        mixing_weights, slot_assignments = torch.topk(expert_affinities, dim = 2, k = self.mix_k) # top k over 'np' dimension
        cluster_assignments = slot_assignments // p # each expert takes p slots, recover the ith expert corresponding to the slot idx
        ###

        ### start compute router stability ###
        
        if topk_previous is not None:
            topk_previous_expanded = topk_previous[:,0].unsqueeze(0)  # Shape: (1, n)
            topk_previous_t_expanded = topk_previous[:,0].unsqueeze(1)  # Shape: (n, 1)

            sim_previous = (topk_previous_expanded == topk_previous_t_expanded).int()  # Shape: (n, n), with 1 where labels match and 0 otherwise

            topk_expanded = cluster_assignments[:,0].unsqueeze(0)  # Shape: (1, n)
            topk_t_expanded = cluster_assignments[:,0].unsqueeze(1)  # Shape: (n, 1)

            sim_here = (topk_expanded == topk_t_expanded).int()  # Shape: (n, n), with 1 where labels match and 0 otherwise


            self.router_stability = 0.8*self.router_stability + 0.2*torch.mean((sim_previous != sim_here).float())
        ### end compute router stability ###

        d = softmax(logits, dim=1)
        c = softmax(logits, dim=(2, 3))

        # if self.save_expert_acmoe:
        #     self.compute_exp_acmoe(x, d)

        # Compute input slots as weighted average of input tokens using dispatch weights
        xs = torch.einsum("bmd,bmnp->bnpd", x, d)

        # Apply expert to corresponding slots
        ys = torch.stack(
            [f_i(xs[:, i, :, :]) for i, f_i in enumerate(self.experts)], dim=1
        )

        # Compute output tokens as weighted average of output slots using combine weights
        y = torch.einsum("bnpd,bmnp->bmd", ys, c)

        return y, (mixing_weights, cluster_assignments)
        
        #return y

    def compute_acmoe(self, input, cluster_assignments, mixing_weights = None, reserve_dims=1):
        # input : [b,m,d]
        # cluster_assignments : [b,m,topk]
        #breakpoint()
        original_shape, original_dtype  = input.shape, input.dtype

        tokens = input.reshape(-1, original_shape[-reserve_dims:].numel())
        top_k = cluster_assignments.shape[2]

        cluster_assignments = cluster_assignments.reshape(-1, top_k).squeeze()

        with torch.no_grad():
            tokens = tokens.detach()
            cluster_assignments = cluster_assignments.detach()
            
            if top_k > 1:
                top_cluster_assignments = cluster_assignments[:,0] # take only top cluster assignment as label
            else:
                top_cluster_assignments = cluster_assignments
            
            n, d = tokens.shape
            
            k = cluster_assignments.max().item() + 1

            #tokens = tokens / (torch.std(tokens, dim = 0) + 1e-4) # divide by featurewise stds
            
            if self.mad: # mean absolute deviation
                mean_dimwise_clusterwise = torch.zeros((k,d), device = tokens.device).index_reduce(0, top_cluster_assignments, tokens, 'mean', include_self=False)
                centered_tokens = torch.abs(tokens - mean_dimwise_clusterwise[top_cluster_assignments, :]) # abs of each token minus its cluster mean
                W_mad = torch.zeros((k,d), device = tokens.device).index_reduce(0, top_cluster_assignments, centered_tokens, 'mean', include_self = False)
                
                # invert
                W = 1 / (W_mad + 0.35) # previous eps=0.35

                # clip thresholding
                W_mean = W.mean()
                top_clamp = 5* W_mean
                #bottom_clamp = 0 * W_mean
                
                # top_clamp = 1.25 * W_mean
                # bottom_clamp = (1/1.25) * W_mean
                
                W = torch.clamp(W,  max = top_clamp)

                # alpha = 0.1
                # W = alpha*torch.ones_like(W) + (1-alpha)*W

                #alpha = 0 # remove this once running a uniform smoothing
                # if self.epoch >= 45:
                #     # begin smoothing to uniform distribution
                #     alpha = self.smooth_dist[self.alpha_counter]
                #     W = alpha*torch.ones_like(W) + (1-alpha)*W
                #     #print(alpha)
                
                # weight scale
                #W = W/ W.mean(dim = 0) # dimwise meanscaling
                #print('cosa here')
                # W = W / W.mean(dim = 1).unsqueeze(1) # clusterwise meanscaling
                W = W / W.mean() # global mean feature scaling
                #W = W / W.max(dim=0)[0]
                
            else:
                mean_dimwise_clusterwise = torch.zeros((k,d), device = tokens.device).index_reduce(0, top_cluster_assignments, tokens, 'mean', include_self=False)
                mean_square_dimwise_clusterwise = torch.zeros((k,d), device = tokens.device).index_reduce(0, top_cluster_assignments, torch.square(tokens), 'mean', include_self=False)
                variance_dimwise_clusterwise = mean_square_dimwise_clusterwise - mean_dimwise_clusterwise**2
                W = 1 / ((variance_dimwise_clusterwise) + 1e-4) # [k,d] cluster-feature weights 
                 # clip thresholding
                W_mean = W.mean()
                clip_threshold = 10 * W_mean
                W = torch.clamp(W, max = clip_threshold)
                
                # weight scale
                #W = W/ W.mean(dim = 0) # dimwise meanscaling
                #print('cosa here')
                # W = W / W.mean(dim = 1).unsqueeze(1) # clusterwise meanscaling
                W = W / W_mean # global mean feature scaling

                #print('here var')
            self.W = W.min(), W.max(), W.mean(), W.std(), top_clamp
            # print('here cosa')

            if self.mix_weights:
                assert mixing_weights is not None
                assert top_k == self.mix_k
                #breakpoint()
                # mixing_weights: [b,m,top_k]
                mixing_weights = mixing_weights.view(-1, top_k) #[b*m, top_k]
                mixing_weights = torch.softmax(mixing_weights, dim = 1)
                assert torch.mean(torch.sum(mixing_weights, dim = 1)) == 1
                
                # Create an index tensor of shape (n, top_k) based on the labels
                index = cluster_assignments.view(n, top_k, 1).expand(n, top_k, d)
                # Add a new dimension to the weights tensor to match the index tensor
                W = W.unsqueeze(0).expand(n, k, d)
                # Gather the corresponding weights for each label. 
                W = torch.gather(W, 1, index) # [n, top_k, d]

                # mix weights according to the gate top k val
                mixing_weights = mixing_weights.view(n, top_k, 1)
                W = (W*mixing_weights).sum(dim = 1) # n x d
                
            else:
                # broadcast up to shape [n, d] which rows arranged by corresponding cluster assignments of n
                W = W[top_cluster_assignments, :] # rescale via MAPD - MAD relationship
                
        return W

    def compute_exp_acmoe(self, input, dispatch_weights, mixing_weights = None, reserve_dims=1):
        # input : [b,m,d]
        # dispatch_weights : [b,m,n]
        #breakpoint()
        b,m,d = input.shape
        n = dispatch_weights.shape[2]

        input = input.unsqueeze(2) # [b,m,1,d]
        input = input.expand(-1,-1,n,-1) # [b,m,n,d]

        with torch.no_grad():
            input = input.detach()
            dispatch_weights = dispatch_weights.detach()

            input = input * dispatch_weights.unsqueeze() # each token weighted by its dispatch weight

            input = input.permute(2,0,1,3) # [n, b, m, d]
            input = input.view(n, b*m, d) # b*m d-dimensional inputs per expert

            expert_means = torch.mean(input, dim =1, keepdim = True)
            input_demean = torch.abs(input - expert_means)
            W_mad = torch.mean(input_demean, dim = 1) # [n,d]

            W = 1 / (W_mad + 1e-4)

            # clip thresholding
            W_mean = W.mean()
            top_clamp = 5* W_mean
            #bottom_clamp = 0 * W_mean
            
            W = torch.clamp(W,  max = top_clamp)

            # weight scale
            #W = W/ W.mean(dim = 0) # dimwise meanscaling
            #print('cosa here')
            # W = W / W.mean(dim = 1).unsqueeze(1) # clusterwise meanscaling
            W = W / W.mean() # global mean feature scaling

            self.W = W.min(), W.max(), W.mean(), W.std(), top_clamp

            self.expert_M = W
        
        return 

          

class LayerScale(nn.Module):
    def __init__(
            self,
            dim: int,
            init_values: float = 1e-5,
            inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_norm=False, attn_drop=0., proj_drop=0., norm_layer=nn.LayerNorm):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # sqrt (D)
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn) 

        # @ is a matrix multiplication
        x = (attn @ v)
        x = x.transpose(1, 2).reshape(B,N,C)
       
        x = self.proj(x)
        x = self.proj_drop(x)
       
        return x

class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
            is_moe = False,
            use_acmoe = False
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.is_moe = is_moe
        self.use_acmoe = use_acmoe

    def forward(self, x):
        if type(x) is tuple:
            x, (mixing_weights, cluster_assignments) = x
        
        x_attn_out = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        #breakpoint()
        if self.is_moe: # acmoe
            if self.use_acmoe:
                assert cluster_assignments is not None
                x_moe_out, (mixing_weights, cluster_assignments) = self.mlp(self.norm2(x_attn_out), 
                                                                            cluster_assignments = cluster_assignments, mixing_weights = mixing_weights)
            else:
                ### temporary patch over for computing router stability in baseline model, take out for final build
                try:
                    x_moe_out, (mixing_weights, cluster_assignments) = self.mlp(self.norm2(x_attn_out), 
                                                                            cluster_assignments = cluster_assignments, mixing_weights = mixing_weights)
                except:
                    x_moe_out, (mixing_weights, cluster_assignments) = self.mlp(self.norm2(x_attn_out)) # should just be this line
                #### 
        
            x_out = x_attn_out + self.drop_path2(self.ls2(x_moe_out))

            return (x_out, (mixing_weights, cluster_assignments))
        
        else: # regular MLP
            x_out = x_attn_out + self.drop_path2(self.ls2(self.mlp(self.norm2(x_attn_out))))
        
            return x_out