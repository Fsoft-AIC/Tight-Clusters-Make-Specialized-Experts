import os, sys
import argparse
import math, random
import torch
import tqdm

PARAMS_CONFIG = {
    # env-specific
    "env_params": {
        "--distributed": {
            "action": "store_true",
            "default": False,
            "help": "enable distributed training."
            "(otherwise will use all available GPUs with dataparallel)",
            "dest": "distributed",
        },
        "--local_rank": {
            "type": int,
            "default": int(os.environ['LOCAL_RANK']),
            "help": "used in distributed training",
            "dest": "local_rank",
        },
    },
    # data-specific
    "data_params": {
        "--data": {
            "type": str,
            "default": "data/text8",
            "help": "data location " "(must contain train.txt, valid.txt and test.txt)",
            "dest": "data_path",
        },
        "--data_name": {
            "type": str,
            "default": "text8",
            "help": "The name of dataset",
            "dest": "data_name",
        },
        "--wt103_attack": {
            "action": "store_true",
            "default": False,
            "help": "Flag to run wt103 text attack",
            "dest": "wt103_attack",
        },
    },
    # model-specific
    "model_params": {
        "--hid-sz": {
            "type": int,
            "default": 256,
            "help": "hidden size (i.e. model size)",
            "dest": "hidden_size",
        },
        "--inner-hid-sz": {
            "type": int,
            "default": 1024,
            "help": "inner hidden size of FF layer",
            "dest": "inner_hidden_size",
        },
        "--nlayers": {
            "type": int,
            "default": 8,
            "help": "number of layers",
            "dest": "nb_layers",
        },
        "--block-sz": {
            "type": int,
            "default": 64,
            "help": "block size " "(the length of sequence to process in parallel)",
            "dest": "block_size",
        },
        "--nheads": {
            "type": int,
            "default": 2,
            "help": "number of self-attention heads",
            "dest": "nb_heads",
        },
        "--attn-span": {
            "type": int,
            "default": 32,
            "help": "length of the attention span",
            "dest": "attn_span",
        },
        "--dropout": {
            "type": float,
            "default": 0.2,
            "help": "dropout rate of ReLU and attention",
            "dest": "dropout",
        },
        "--architecture": {
            "type": str,
            "default": None,
            "help": "arch",
            "dest": "architecture",
        },
        "--base_arch": {
            "type": str,
            "default": None,
            "help": "arch",
            "dest": "base_arch",
        },
        "--smoe_dropout": {
            "action": "store_true",
            "default": False,
            "help": "enable SMoE-drop - Freeze gate",
            "dest": "smoe_dropout",
        },
        "--optimal_policy": {
            "action": "store_true",
            "default": False,
            "help": "Searching the best routing policy",
            "dest": "optimal_policy",
        },
        "--load_balance": {
            "type": float,
            "default": 1.0,
            "help": "Ratio of blance loss",
            "dest": "load_balance",
        },
        "--moe_top_k": {
            "type": int,
            "default": 2,
            "help": "Number of activate experts",
            "dest": "moe_top_k",
        },
        "--freq": {
            "type": float,
            "default": 0.03,
            "help": "Frequent for searching optimal policy",
            "dest": "freq",
        },
        "--freq_type": {
            "type": str,
            "default": "fix",
            "help": "Type of frequent for searching optimal policy. Choice: fix or function",
            "dest": "freq_type",
        },
        "--alpha": {
            "type": float,
            "default": 1.0,
            "help": "Impact of optimal loss",
            "dest": "alpha",
        },
        "--gate_name": {
            "type": str,
            "default": "smoe",
            "help": "Names of gates: smoe, smoe-dropout, xmoe, stablemoe",
            "dest": "gate_name",
        },
        "--act_experts": {
            "type": str,
            "default": "shuffle",
            "help": "Type to activate all experts: shuffle OR linear",
            "dest": "act_experts",
        },
        "--g_blance": {
            "action": "store_true",
            "default": False,
            "help": "Activate balance loss for router",
            "dest": "g_blance",
        },
        "--opt_blance": {
            "action": "store_true",
            "default": False,
            "help": "Activate blancing for optimal router",
            "dest": "opt_blance",
        },
        "--combine_gate": {
            "action": "store_true",
            "default": False,
            "help": "Utilize previous information for better consistancy",
            "dest": "combine_gate",
        },
        "--opt_loss": {
            "type": str,
            "default": "mse",
            "help": "Type of loss for optimal policy searching",
            "dest": "opt_loss",
        },
        "--gamma": {
            "type": float,
            "default": 1.0,
            "help": "Learning rate",
            "dest": "gamma",
        },
        "--mu": {
            "type": float,
            "default": 0.9,
            "help": "Momentum parameter",
            "dest": "mu",
        },
        "--layer-n": {
            "type": float,
            "default": 0.0,
            "help": "layer to get expert count",
            "dest": "layer_n",
        },
        "--ssm": {
            "action": "store_true",
            "default": False,
            "help": "ssm momentum flag",
            "dest": "ssm",
        },
        "--compute_load_balance": {
            "action": "store_true",
            "default": False,
            "help": "compute load balance over experts",
            "dest": "compute_load_balance",
        },
        "--compute-rep-collapse": {
            "action": "store_true",
            "default": False,
            "help": "compute rep collapse for smoe out",
            "dest": "compute_rep_collapse",
        },
        "--show-gate-W": {
            "action": "store_true",
            "default": False,
            "help": "store the M gate weights in an attribute show_gate_W",
            "dest": "show_gate_W",
        },    
        "--mean-scale": {
        "action": "store_true",
        "default": False,
        "help": "mean scale, as opposed to max scale, gate weights",
        "dest": "mean_scale",
        },
        "--root-invert": {
        "action": "store_true",
        "default": False,
        "help": "square root and invert variation weights",
        "dest": "root_invert",
        },
        "--intra-layer": {
        "action": "store_true",
        "default": False,
        "help": "compute feature weights using a single fwds tensor from the previous layer as opposed to averaging two fwds from two layers",
        "dest": "intra_layer",
        },
        "--exp-distance": {
        "action": "store_true",
        "default": False,
        "help": "use a negative softmax over feature weights",
        "dest": "exp_distance",
        },
        "--reduce-dim": {
        "action": "store_true",
        "default": False,
        "help": "LDA style dimension rerduction to num_experts-1",
        "dest": "reduce_dim",
        },
        "--return-fwd": {
        "action": "store_true",
        "default": False,
        "help": "g specified gate will return fwd, designed for use with ellipticalxmoe",
        "dest": "return_fwd",
        },
        "--return-2fwds": {
        "action": "store_true",
        "default": False,
        "help": "return two layers of fwds",
        "dest": "return_2fwds",
        },
        "--use-var": {
        "action": "store_true",
        "default": False,
        "help": "use variance in cosa rather than average pairwise disptances",
        "dest": "use_var",
        },
        "--smoe-base": {
        "action": "store_true",
        "default": False,
        "help": "use smoe as the base in cosa",
        "dest": "smoe_base",
        },
        "--mad": {
        "action": "store_true",
        "default": False,
        "help": "use mean absolute deviation within the use_var cosa-var",
        "dest": "mad",
        },
        "--mix-weights": {
        "action": "store_true",
        "default": False,
        "help": "mix clusterwise weights together according to cluster assignment in cosa",
        "dest": "mix_weights",
        },
        "--skip-connect": {
        "action": "store_true",
        "default": False,
        "help": "use skip connection on tokens in cosa",
        "dest": "skip_connect",
        },
        "--temp-disp": {
        "action": "store_true",
        "default": False,
        "help": "use temporal dispersion in cosa",
        "dest": "temp_disp",
        },
    },
    # optimization-specific
    "optim_params": {
        "--lr": {"type": float, "default": 0.03, "help": "learning rate", "dest": "lr"},
        "--momentum": {
            "type": float,
            "default": 0.9,
            "help": "SGD momentum",
            "dest": "momentum",
        },
        "--optim": {
            "type": str,
            "default": "sgd",
            "help": "optimization method: sgd | adagrad",
            "dest": "optim",
        },
        "--lr-warmup": {
            "type": int,
            "default": 0,
            "help": "linearly increase LR from 0 " "during first lr_warmup updates",
            "dest": "lr_warmup",
        },
        "--grad-clip": {
            "type": float,
            "default": 0,
            "help": "[only works with adagrad!] "
            "clip gradient of each module parameters by a given "
            "value",
            "dest": "grad_clip",
        },
    },
    # trainer-specific
    "trainer_params": {
        "--batch-sz": {
            "type": int,
            "default": 64,
            "help": "batch size",
            "dest": "batch_size",
        },
        "--batch-split": {
            "type": int,
            "default": 1,
            "help": "split a batch into smaller parts to fit in GPU memory",
            "dest": "batch_split",
        },
        "--nbatches": {
            "type": int,
            "default": 1000,
            "help": "number of batches in each iteration",
            "dest": "nb_batches_per_iter",
        },
        "--niter": {
            "type": int,
            "default": 1000,
            "help": "number of iterations to train",
            "dest": "nb_iter",
        },
        "--checkpoint": {
            "type": str,
            "default": "",
            "help": "path to save/load model",
            "dest": "checkpoint_path",
        },
        "--resume": {
            "action": "store_true",
            "default": False,
            "help": "resume training",
            "dest": "resume",
        },
        "--pretrained_weight": {
            "type": str,
            "default": "",
            "help": "path to save/load model",
            "dest": "pretrained_weight",
        },
        "--full-eval-mode": {
            "action": "store_true",
            "default": False,
            "help": "do evaluation on the whole validation and the test data",
            "dest": "full_eval_mode",
        },
        "--debug": {
            "action": "store_true",
            "default": False,
            "help": "run in debug mode",
            "dest": "debug",
        },
        "--show-sparse-w-stats": {
            "action": "store_true",
            "default": False,
            "help": "print and log sparse w stats",
            "dest": "show_sparse_w_stats",
        },
        "--show-gate-w-stats": {
            "action": "store_true",
            "default": False,
            "help": "print and log gate w stats",
            "dest": "show_gate_w_stats",
        },
    },
    # adaptive attention span specific params
    "adapt_span_params": {
        "--adapt-span": {
            "action": "store_true",
            "default": False,
            "help": "enable adaptive attention span",
            "dest": "adapt_span_enabled",
        },
        "--adapt-span-loss": {
            "type": float,
            "default": 0,
            "help": "the loss coefficient for span lengths",
            "dest": "adapt_span_loss",
        },
        "--adapt-span-ramp": {
            "type": int,
            "default": 32,
            "help": "ramp length of the soft masking function",
            "dest": "adapt_span_ramp",
        },
        "--adapt-span-init": {
            "type": float,
            "default": 0,
            "help": "initial attention span ratio",
            "dest": "adapt_span_init",
        },
        "--adapt-span-cache": {
            "action": "store_true",
            "default": False,
            "help": "adapt cache size as well to reduce memory usage",
            "dest": "adapt_span_cache",
        },
    },
    "wandb_params": {
        "--project-name": {
            "type": str,
            "default": "project_name",
            "help": "wandb project name",
            "dest": "project_name",
        },
        "--job-name": {
            "type": str,
            "default": "job_name",
            "help": "wandb job name",
            "dest": "job_name",
        },
        "--wandb-flag": {
            "action": "store_true",
            "default": False,
            "help": "use wandb",
            "dest": "wandb_flag",
        },
    },
}
