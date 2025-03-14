import os, sys
import warnings

warnings.filterwarnings("ignore")

import argparse
import math, random
import torch
import time

from config import PARAMS_CONFIG
from data import get_train_val_test_data, get_val_test_data
from models import TransformerSeq
from trainer import train_iteration, full_eval
import datetime
import wandb
import numpy as np
from utils_smoe import (
    get_params,
    set_up_env,
    get_optimizer_and_scheduler,
    load_checkpoint,
    save_checkpoint,
    create_exp_dir,
    freeze_gate_weight,
    Logger,
    set_freq_optimal_search,
)


def launch(
    env_params,
    model_params,
    adapt_span_params,
    optim_params,
    data_params,
    trainer_params,
    wandb_params,
):
    if trainer_params["debug"]:
        print('Running Debug Mode: No model saving or checkpoint writing')
        trainer_params["batch_size"] = 12
        wandb_params["wandb_flag"] = False

    wandb_flag = wandb_params["wandb_flag"]
    if wandb_flag:
        wandb.init(project=wandb_params["project_name"])
        wandb_params["job_name"] = trainer_params["checkpoint_path"].split("/")[-1].split(".")[0]
        wandb.run.name = wandb_params["job_name"]
        wandb.config.update(model_params)
    # global val
    best_val_loss = None
    # ENVIRONMENT (device, distributed, etc.)
    set_up_env(env_params)
    device = env_params["device"]
    distributed = env_params["distributed"]
    resume = trainer_params["resume"]
    compute_load_balance = model_params["compute_load_balance"]
    compute_rep_collapse = model_params["compute_rep_collapse"]
    show_sparse_w_stats = trainer_params["show_sparse_w_stats"]
    show_gate_w_stats = trainer_params["show_gate_w_stats"]

    if distributed == False or env_params["rank"] == 0:
        print("data_params:\t", data_params)
        print("model_params:\t", model_params)
        print("optim_params:\t", optim_params)
        print("trainer_params:\t", trainer_params)
        print("adapt_span_params:\t", adapt_span_params)

    # DATA
    if data_params["wt103_attack"]:
       val_data, test_data = get_val_test_data(
        data_params=data_params,
        env_params=env_params,
        batch_size=trainer_params["batch_size"],
        device=device,
        attack = True
    ) 
    else:
        train_data, val_data, test_data = get_train_val_test_data(
            data_params=data_params,
            env_params=env_params,
            batch_size=trainer_params["batch_size"],
            device=device,
        )



    # MODEL
    model = TransformerSeq(
        vocab_size=data_params["vocab_size"],
        **model_params,
        adapt_span_params=adapt_span_params,
    )
    #print(model)
    if distributed:
        local_rank = env_params["local_rank"]
        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
    else:
        model = torch.nn.DataParallel(model)
        model = model.to(device)

    # OPTIMIZER AND SCHEDULER
    optimizer, scheduler = get_optimizer_and_scheduler(
        model=model, optim_params=optim_params
    )

    # create logger
    logger = Logger()
    fold_name = trainer_params["checkpoint_path"].split("/")[-1].split(".")[0]
    folder_path = "/".join(trainer_params["checkpoint_path"].split("/")[:-1])
    if trainer_params["debug"]:
        fold_name = 'debug_log'
    logging = create_exp_dir(f"{folder_path}/experiments/{fold_name}")
    # log paramters
    logging(f"Training Parameters:\n {trainer_params}")
    logging(f"Models Parameters:\n {model_params}")
    # logging time
    current_time = datetime.datetime.now()
    logging(str(current_time))
    if show_gate_w_stats:
        logging_gate_w_stats = create_exp_dir(f"{folder_path}/experiments/{fold_name}/gate_w_stats")
    
    # log model
    # logging(str(model))
    # logging(f"Total of Parameters: {sum(p.numel() for p in model.parameters())}")
    # logging(
    #     f"Total of Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    # )
    # resume training from last checkpoint if exists
    iter_init = load_checkpoint(
        trainer_params["checkpoint_path"],
        model,
        optimizer,
        scheduler,
        logger,
        distributed,
        resume,
    )
    # fix gate
    if model_params["smoe_dropout"]:
        freeze_gate_weight(model)
    # calculate time
    start_time = time.time()
    # eval model
    if trainer_params["full_eval_mode"]:
        # evaluate the model on test data
        with torch.no_grad():
            loss_val, expert_count_all_val = full_eval(
                model,
                optimizer,
                scheduler,
                val_data,
                model_params["block_size"],
                model_params["hidden_size"],
            )
            loss_test, expert_count_all_test = full_eval(
                model,
                optimizer,
                scheduler,
                test_data,
                model_params["block_size"],
                model_params["hidden_size"],
            )
            if distributed:
                # collect results into rank0
                stats = torch.tensor([loss_val, loss_test]).to(device)
                expert_count_all_val = expert_count_all_val.to(device)
                expert_count_all_test = expert_count_all_test.to(device)
                torch.distributed.reduce(stats, 0)
                if env_params["rank"] == 0:
                    loss_val = stats[0] / env_params["world_size"]
                    loss_test = stats[1] / env_params["world_size"]
                    if compute_load_balance:
                        logging_load_balance = create_exp_dir(f"{folder_path}/experiments/{fold_name}/expertcount")
                        # expert_count_all_val = expert_count_all_val / env_params["world_size"]
                        # expert_count_all_test = expert_count_all_test / env_params["world_size"]
                        # expert_count_val_std = np.std(expert_count_all_val.cpu().numpy())
                        # expert_count_test_std = np.std(expert_count_all_test.cpu().numpy())
                        # layer_n = model_params["layer_n"]
                        # # rewrite this saving using the logging() function

                        # logging_load_balance(f" ----- validation expert count, layer {layer_n} ----- ")
                        # logging_load_balance(f"val counts: {expert_count_all_val.cpu().numpy()} | val load std: {expert_count_val_std:.3f}")

                        # logging_load_balance(f" ------ test expert count, layer {layer_n} ------- ")
                        # logging_load_balance(f"test counts: {expert_count_all_test.cpu().numpy()} | val load std: {expert_count_test_std:.3f}")
                        
                        print('computing load balance over test data')
                        mod = model.module if distributed else model
                        vars = []
                        for idx, layer in enumerate(mod.layers):
                            if layer.use_smoe:
                                load = layer.load_counter.float()
                                load_dist = (load / torch.sum(load))*100
                                var = torch.std(load_dist)
                                print(f'layer {idx} | std over expert distribution: {var}')
                                vars.append(var)
        
                        mean_over_layers = torch.mean(torch.stack(vars))
                        std_over_layers = torch.std(torch.stack(vars))
                        print(f'Load balance over all layers: {mean_over_layers} +/- {std_over_layers}')
                        assert 1==2

                        # f = open(f"{folder_path}/experiments/{fold_name}/expert_count_val.txt", "a")
                        # f.write(f"validation expert count, layer {layer_n}: \n")
                        # f.close()
                        # f = open(f"{folder_path}/experiments/{fold_name}/expert_count_test.txt", "a")
                        # f.write(f"test expert count, layer {layer_n}: \n")
                        # f.close()
                        # np.savetxt(f"{folder_path}/experiments/{fold_name}/expert_count_val.txt", expert_count_all_val.cpu().numpy())
                        # np.savetxt(f"{folder_path}/experiments/{fold_name}/expert_count_test.txt", expert_count_all_test.cpu().numpy())
                        # print(expert_count_all_val)
                        # print(f"val expert load std: {expert_count_val_std:.3f}")
                        # print(expert_count_all_test)
                        # print(f"test expert load std: {expert_count_test_std:.3f}")
                    if compute_rep_collapse:
                        logging_rep_collapse = create_exp_dir(f"{folder_path}/experiments/{fold_name}/repcollapse")
                        mod = model.module
                        for idx, layer in enumerate(mod.layers):
                            #ith_average_sim =  layer.cossim / env_params["world_size"]
                            ith_average_sim = layer.cossim
                            print(f'Layer: {idx} | average cosine sim: {ith_average_sim:.3f}')
                            logging_rep_collapse(f'Layer: {idx} | average cosine sim: {ith_average_sim:.3f}')

                else:
                    return

            print('Test BPC: {:.4f}'.format(loss_test / math.log(2)))
            if ("enwik8" in data_params["data_path"]) or (
                "text8" in data_params["data_path"]
            ):
                logging("Val: {:.3f} BPC".format(loss_val / math.log(2)))
                logging("Test: {:.3f} BPC".format(loss_test / math.log(2)))
            else:
                logging("Val: {:.3f} PPL".format(math.exp(loss_val)))
                logging("Test: {:.3f} PPL".format(math.exp(loss_test)))
        return

    # position of current batch
    data_pos = [0] * 2
    # initialize caches for train and valid
      
    hid_cache = [
        [
            torch.zeros(
                train_data.size(0),
                model.module.layers[layer_i].attn.attn.get_cache_size(),
                model_params["hidden_size"],
            ).to(device)
            for layer_i in range(model.module.attn_layer_count)
        ]
        for _ in range(2)
    ]

    nb_batches_per_iter = trainer_params["nb_batches_per_iter"]
    for iter_no in range(iter_init, trainer_params["nb_iter"]):
        # freq type
        if model_params["freq_type"] == "function":
            _threshold = 2.0 / (2.0 + math.sqrt((iter_no + 1)))
            set_freq_optimal_search(model, _threshold)

        # time storing
        t_sta = time.time()
        loss_train, data_pos[0], hid_cache[0], expert_count_all_train = train_iteration(
            model,
            model_params["load_balance"],
            optimizer,
            scheduler,
            train_data,
            nb_batches_per_iter,
            model_params["block_size"],
            False,
            data_pos[0],
            hid_cache[0],
            trainer_params["batch_split"],
            trainer_params["checkpoint_path"],
            debug = False
        )
        elapsed = 1000 * (time.time() - t_sta) / nb_batches_per_iter
        with torch.no_grad():
            loss_val, data_pos[1], hid_cache[1], expert_count_all_train = train_iteration(
                model,
                model_params["load_balance"],
                optimizer,
                scheduler,
                val_data,
                nb_batches_per_iter,
                model_params["block_size"],
                True,
                data_pos[1],
                hid_cache[1],
                trainer_params["batch_split"],
                trainer_params["checkpoint_path"],
                debug = False
            )

        if distributed:
            # collect results into rank0
            stats = torch.tensor([loss_train, loss_val]).to(device)
            torch.distributed.reduce(stats, 0)
            if env_params["rank"] == 0:
                loss_train = stats[0] / env_params["world_size"]
                loss_val = stats[1] / env_params["world_size"]
            else:
                continue

        if show_sparse_w_stats:
            logging_sparse_w_stats = create_exp_dir(f"{folder_path}/experiments/{fold_name}/sparse_w_stats")
            mod = model.module if distributed else model
            logging_sparse_w_stats(f' ------ EPOCH: {iter_no} -----')
            for idx, layer in enumerate(mod.layers):
                #ith_average_sim =  layer.cossim / env_params["world_size"]
                #breakpoint()
                if not hasattr(layer.smoe.gate, 'sparse_w_stats'):
                    continue
                delta, d, num_zeros, (mx, max_index), (mn, min_index), mean, std = layer.smoe.gate.sparse_w_stats
                # print(f'\n')
                # print(f'Layer {idx}:')
                # print(f'Number of dimensions: {d} | Delta: {delta}')
                # print(f'Number of 0s: {num_zeros}')
                # print(f'Max weight: {mx:.4f} | index: {max_index}')
                # print(f'Min weight: {mn:.4f} | index: {min_index}')
                # print(f'Mean weight: {mean:.4f}')
                # print(f'std over weights: {std:.4f}')
                # print(f'\n')
                logging_sparse_w_stats(f'\n')
                logging_sparse_w_stats(f'Layer {idx}:')
                logging_sparse_w_stats(f'Number of dimensions: {d} | Delta: {delta}')
                logging_sparse_w_stats(f'Number of 0s: {num_zeros}')
                logging_sparse_w_stats(f'Max weight: {mx:.4f} | index: {max_index}')
                logging_sparse_w_stats(f'Min weight: {mn:.4f} | index: {min_index}')
                logging_sparse_w_stats(f'Mean weight: {mean:.4f}')
                logging_sparse_w_stats(f'std over weights: {std:.4f}')
                logging_sparse_w_stats(f'\n')
        
        if show_gate_w_stats:
            mod = model.module if distributed else model
            logging_gate_w_stats(f' ------ EPOCH: {iter_no} -----')
            
            for idx, layer in enumerate(mod.layers):
                #ith_average_sim =  layer.cossim / env_params["world_size"]
                
                
                if layer.smoe is not None: # check if layer is an smoe
                    if hasattr(layer.smoe.gate, 'W'):
                        w, mn, mx, mean, std  = layer.smoe.gate.W
                        logging_gate_w_stats(f'\n')
                        logging_gate_w_stats(f'Layer {idx}:')
                        logging_gate_w_stats(f'First 2 rows of W')
                        logging_gate_w_stats(f'{w}')
                        logging_gate_w_stats(f'min: {mn}')
                        logging_gate_w_stats(f'max: {mx}')
                        logging_gate_w_stats(f'mean: {mean}')
                        logging_gate_w_stats(f'std: {std}')
                        continue

                    if not hasattr(layer.smoe.gate, 'gate_W'):
                        continue
                    
                    _, W_std, W_max, max_idx, W_min, min_idx, W_mean = layer.smoe.gate.gate_W

                    # print(f'\n')
                    # print(f'Layer {idx}:')
                    # print(f'Weight std: {W_std:.4f}')
                    # print(f'Max weight: {W_max:.4f} | index: {max_idx}')
                    # print(f'Min weight: {W_min:.4f} | index: {min_idx}')
                    # print(f'Mean weight: {W_mean:.4f}')
                    # print(f'\n')

                    logging_gate_w_stats(f'\n')
                    logging_gate_w_stats(f'Layer {idx}:')
                    logging_gate_w_stats(f'Weights: {_}')
                    logging_gate_w_stats(f'Weight std: {W_std:.4f}')
                    logging_gate_w_stats(f'Max weight: {W_max:.4f} | index: {max_idx}')
                    logging_gate_w_stats(f'Min weight: {W_min:.4f} | index: {min_idx}')
                    logging_gate_w_stats(f'Mean weight: {W_mean:.4f}')
                    logging_gate_w_stats(f'\n')

                    if hasattr(layer.smoe.gate, 'gate_W_scaled'):
                        logging_gate_w_stats(f'-------Scaled Stats-----')
                        logging_gate_w_stats(f'\n')
                        
                        _, W_std, W_max, W_min, W_mean = layer.smoe.gate.gate_W_scaled
                        logging_gate_w_stats(f'Layer {idx}:')
                        logging_gate_w_stats(f'Weights: {_}')
                        logging_gate_w_stats(f'Weight std: {W_std:.4f}')
                        logging_gate_w_stats(f'Max weight: {W_max:.4f} | index: {max_idx}')
                        logging_gate_w_stats(f'Min weight: {W_min:.4f} | index: {min_idx}')
                        logging_gate_w_stats(f'Mean weight: {W_mean:.4f}')




        logging(f"=================== EPOCHS {iter_no} ======================")
        if ("enwik8" in data_params["data_path"]) or (
            "text8" in data_params["data_path"]
        ):
            msg_result = "Epochs: {} | loss_train: {:.3f} ~ {:.3f} BPC | loss_val: {:.3f} ~ {:.3f} BPC | elapsed: {:.1f}".format(
                iter_no,
                loss_train,
                float(loss_train / math.log(2)),
                loss_val,
                float(loss_val / math.log(2)),
                elapsed,
            )
        else:
            msg_result = "Epochs: {} | loss_train: {:.3f} ~ {:.3f} PPL | loss_val: {:.3f} ~ {:.3f} PPL | elapsed: {:.1f}".format(
                iter_no,
                loss_train,
                float(math.exp(loss_train)),
                loss_val,
                float(math.exp(loss_val)),
                elapsed,
            )
        logging(msg_result)
        if wandb_flag:
            wandb.log({'train_ppl':float(math.exp(loss_train)),'Epoch':iter_no,'valid_ppl':float(math.exp(loss_val))})
        # logger.log_iter(iter_no, nb_batches_per_iter, loss_train, loss_val, elapsed, model)
        # Save the model if the validation loss is the best we've seen so far.
        if (best_val_loss is None) or loss_val < best_val_loss:
            best_val_loss = loss_val
            if not trainer_params["debug"]: # no model saving in debug mode
                save_checkpoint(
                    trainer_params["checkpoint_path"],
                    iter_no,
                    model,
                    optimizer,
                    scheduler,
                    logger,
                )
        # save_checkpoint(trainer_params['checkpoint_path'], nb_batches_per_iter, model, optimizer, scheduler, logger)
    end_time = time.time()
    logging(f"Training time total: {(end_time - start_time)/3600} h")

    logging(f'------------Running Test Eval---------')
    logging(f'Loading Best Model...')
    iter_init = load_checkpoint(
        trainer_params["checkpoint_path"],
        model,
        optimizer,
        scheduler,
        logger,
        distributed,
        resume = True,
    )
    logging(f'Evaluating...')
    with torch.no_grad():
        loss_val, expert_count_all_val = full_eval(
            model,
            optimizer,
            scheduler,
            val_data,
            model_params["block_size"],
            model_params["hidden_size"],
        )
        loss_test, expert_count_all_test = full_eval(
            model,
            optimizer,
            scheduler,
            test_data,
            model_params["block_size"],
            model_params["hidden_size"],
        )
        if distributed:
            # collect results into rank0
            stats = torch.tensor([loss_val, loss_test]).to(device)
            expert_count_all_val = expert_count_all_val.to(device)
            expert_count_all_test = expert_count_all_test.to(device)
            torch.distributed.reduce(stats, 0)
            if env_params["rank"] == 0:
                loss_val = stats[0] / env_params["world_size"]
                loss_test = stats[1] / env_params["world_size"]
                if compute_load_balance:
                    logging_load_balance = create_exp_dir(f"{folder_path}/experiments/{fold_name}/expertcount")
                    expert_count_all_val = expert_count_all_val / env_params["world_size"]
                    expert_count_all_test = expert_count_all_test / env_params["world_size"]
                    expert_count_val_std = np.std(expert_count_all_val.cpu().numpy())
                    expert_count_test_std = np.std(expert_count_all_test.cpu().numpy())
                    layer_n = model_params["layer_n"]
                    # rewrite this saving using the logging() function

                    logging_load_balance(f" ----- validation expert count, layer {layer_n} ----- ")
                    logging_load_balance(f"val counts: {expert_count_all_val.cpu().numpy()} | val load std: {expert_count_val_std:.3f}")

                    logging_load_balance(f" ------ test expert count, layer {layer_n} ------- ")
                    logging_load_balance(f"test counts: {expert_count_all_test.cpu().numpy()} | val load std: {expert_count_test_std:.3f}")
                    
                    # f = open(f"{folder_path}/experiments/{fold_name}/expert_count_val.txt", "a")
                    # f.write(f"validation expert count, layer {layer_n}: \n")
                    # f.close()
                    # f = open(f"{folder_path}/experiments/{fold_name}/expert_count_test.txt", "a")
                    # f.write(f"test expert count, layer {layer_n}: \n")
                    # f.close()
                    # np.savetxt(f"{folder_path}/experiments/{fold_name}/expert_count_val.txt", expert_count_all_val.cpu().numpy())
                    # np.savetxt(f"{folder_path}/experiments/{fold_name}/expert_count_test.txt", expert_count_all_test.cpu().numpy())
                    # print(expert_count_all_val)
                    # print(f"val expert load std: {expert_count_val_std:.3f}")
                    # print(expert_count_all_test)
                    # print(f"test expert load std: {expert_count_test_std:.3f}")
                if compute_rep_collapse:
                    logging_rep_collapse = create_exp_dir(f"{folder_path}/experiments/{fold_name}/repcollapse")
                    mod = model.module
                    for idx, layer in enumerate(mod.layers):
                        #ith_average_sim =  layer.cossim / env_params["world_size"]
                        ith_average_sim = layer.cossim
                        print(f'Layer: {idx} | average cosine sim: {ith_average_sim:.3f}')
                        logging_rep_collapse(f'Layer: {idx} | average cosine sim: {ith_average_sim:.3f}')

            else:
                return

        print('Test BPC: {:.4f}'.format(loss_test / math.log(2)))
        if ("enwik8" in data_params["data_path"]) or (
            "text8" in data_params["data_path"]
        ):
            logging("Val: {:.3f} BPC".format(loss_val / math.log(2)))
            logging("Test: {:.3f} BPC".format(loss_test / math.log(2)))
        else:
            logging("Val: {:.3f} PPL".format(math.exp(loss_val)))
            logging("Test: {:.3f} PPL".format(math.exp(loss_test)))
    return



if __name__ == "__main__":
    os.environ["WANDB_API_KEY"]="f9b91afe90c0f06aa89d2a428bd46dac42640bff" # rachel key
    #os.environ["WANDB_API_KEY"]= "971b65e110dc2de0cf18976cd29bd2037496ed0d" # my key
    launch(**get_params(params_config=PARAMS_CONFIG))
