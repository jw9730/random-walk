import os
import copy
import random
import numpy as np

import torch
import torch.optim as optim
import torch_geometric.seed
from torch_geometric.loader import DataLoader

import wandb

from definitions import ROOT_DIR
from synthexp.data.data_loading import load_dataset
from synthexp.parsers.load_parsers import load_config, load_yaml, get_parser
from synthexp.utils.train import train, evaluate, Evaluator
from synthexp.nn.utils.model_lookup import model_lookup


def main(config):
    """The common training and evaluation script used by all the experiments."""

    # set device
    device = torch.device("cuda:" + str(config.exp.device)) if torch.cuda.is_available() else torch.device("cpu")

    if config.exp.seed is None:
        config.exp.seed = config.exp.start_seed

    if config.exp.wandb:
        wandb.init(project="barbell-clique",
                   group=config.data.dataset,
                   name=config.model.model_name + "-" + str(config.model.walk_len) + "-" + config.data.dataset)
        wandb.config.update(config.get_dic())
        print(f"Running with wandb account: {config.exp.entity}")

    print("==========================================================")
    print("Using device", str(device))
    print(f"Seed: {config.exp.seed}")
    print("======================== Args ===========================")
    print(config.get_dic())
    print("===================================================")

    # Set the seed for everything
    torch.manual_seed(config.exp.seed)
    torch.cuda.manual_seed(config.exp.seed)
    torch.cuda.manual_seed_all(config.exp.seed)
    np.random.seed(config.exp.seed)
    random.seed(config.exp.seed)
    torch_geometric.seed.seed_everything(config.exp.seed)

    # load data
    root = os.path.join(ROOT_DIR, '', 'datasets')
    train_data, val_data, test_data = load_dataset(config.data.dataset, root=root, config=config)

    train_loader = DataLoader(train_data, batch_size=config.optim.batch_size, shuffle=True)
    valid_loader = DataLoader(val_data, batch_size=config.optim.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=config.optim.batch_size, shuffle=False)

    # Automatic evaluator, takes dataset name as input
    evaluator = Evaluator(config.data.eval_metric)

    # Instantiate model
    model = model_lookup(config)
    # model.reset_parameters()
    model.to(device)

    print("============= Model Parameters =================")
    trainable_params = 0
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.size())
            trainable_params += param.numel()
        total_params += param.numel()
    print("============= Params stats ==================")
    print(f"Trainable params: {trainable_params}")
    print(f"Total params    : {total_params}")

    # instantiate optimiser
    if config.model.model_name == "BuNNNode":
        w_params, bunn_params, tau_params = model.grouped_parameters()
        optimizer = optim.Adam(
                        [{'params': w_params, 'lr': config.optim.lr},
                         {'params': bunn_params, 'lr': config.optim.lr_bunn},
                         {'params': bunn_params, 'lr': config.optim.lr.tau}]
        )
    elif config.optim.optimizer == "ADAMW":
        optimizer = optim.AdamW(model.parameters(), lr=config.optim.lr, weight_decay=config.optim.weight_decay)
    else:
        assert config.optim.optimizer == "ADAM"
        optimizer = optim.Adam(model.parameters(), lr=config.optim.lr, weight_decay=config.optim.weight_decay)

    # instantiate learning rate decay
    if config.optim.lr_scheduler == 'ReduceLROnPlateau':
        mode = 'min' if config.data.minimize else 'max'
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode,
                                                               factor=config.optim.lr_scheduler_decay_rate,
                                                               patience=config.optim.lr_scheduler_patience,
                                                               verbose=True)
    elif config.optim.lr_scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.optim.lr_scheduler_decay_steps,
                                                    gamma=config.optim.lr_scheduler_decay_rate)
    elif config.optim.lr_scheduler == 'None':
        scheduler = None
    else:
        raise NotImplementedError(f'Scheduler {config.optim.lr_scheduler} is not currently supported.')

    # (!) start training/evaluation
    best_val_epoch = 0
    valid_curve = []
    test_curve = []
    train_curve = []
    train_loss_curve = []
    params = []

    if config.data.unbalanced:
        # compute weights for unbalanced case
        weight = torch.zeros(train_data[0].y.shape)
        for data in train_data:
            weight += data.y
        weight = weight/weight.sum()
    else:
        weight = None

    if config.data.dataset in ['chameleon', 'squirrel', 'cornell', 'texas',
                               'wisconsin', 'film', 'cora', 'pubmed', 'citeseer',
                               'roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions']:  # TODO: make this better
        train_mask = train_data.train_mask
        val_mask = train_data.val_mask
        test_mask = train_data.test_mask
    else:
        train_mask = val_mask = test_mask = None

    # training
    for epoch in range(1, config.optim.epochs + 1):
        # perform one epoch
        print("=====Epoch {}".format(epoch))
        print('Training...')
        epoch_train_curve = train(model, device, train_loader, optimizer,
                                  config.data.task_type, weight=weight, mask=train_mask, cfg=config)
        train_loss_curve += epoch_train_curve
        epoch_train_loss = float(np.mean(epoch_train_curve))

        # evaluate model
        print('Evaluating...')
        if epoch == 1 or epoch % config.exp.train_eval_period == 0:
            train_perf, _ = evaluate(model, device, train_loader, evaluator,
                                 config.data.task_type, weight=weight,
                                 mask=train_mask, cfg=config)
        train_curve.append(train_perf)
        valid_perf, epoch_val_loss = evaluate(model, device,
                                          valid_loader, evaluator, config.data.task_type,
                                          mask=val_mask, cfg=config)
        valid_curve.append(valid_perf)

        if test_loader is not None:
            test_perf, epoch_test_loss = evaluate(model, device, test_loader, evaluator,
                                              config.data.task_type,
                                              mask=test_mask, cfg=config)
        else:
            test_perf = np.nan
            epoch_test_loss = np.nan
        test_curve.append(test_perf)

        print(f'Train: {train_perf:.3f} | Validation: {valid_perf:.3f} | Test: {test_perf:.3f}'
              f' | Train Loss {epoch_train_loss:.3f} | Val Loss {epoch_val_loss:.3f}'
              f' | Test Loss {epoch_test_loss:.3f}')

        if config.exp.wandb:
            wandb.log({"train_loss": epoch_train_loss,
                       "val_loss": epoch_val_loss,
                       "test_loss": epoch_test_loss,
                       "train_score": train_perf,
                       "val_score": valid_perf,
                       "test_score": test_perf,
                       "lr": optimizer.param_groups[0]['lr']
                       })

        # decay learning rate
        if scheduler is not None:
            if config.optim.lr_scheduler == 'ReduceLROnPlateau':
                scheduler.step(valid_perf)
                # We use a strict inequality here like in the benchmarking GNNs paper code
                # https://github.com/graphdeeplearning/benchmarking-gnns/blob/master/main_molecules_graph_regression.py#L217
                if config.optim.early_stop and optimizer.param_groups[0]['lr'] < config.optim.lr_scheduler_min:
                    print("\n!! The minimum learning rate has been reached.")
                    break
            else:
                scheduler.step()

        i = 0
        new_params = []
        if epoch % config.exp.train_eval_period == 0:
            print("====== Slowly changing params ======= ")
        for name, param in model.named_parameters():
            new_params.append(param.data.detach().mean().item())
            if len(params) > 0 and epoch % config.exp.train_eval_period == 0:
                if abs(params[i] - new_params[i]) < 1e-6:
                    print(f"Param {name}: {params[i] - new_params[i]}")
            i += 1
        params = copy.copy(new_params)

    if not config.data.minimize:
        best_val_epoch = np.argmax(np.array(valid_curve))
    else:
        best_val_epoch = np.argmin(np.array(valid_curve))

    print('Final Evaluation...')
    final_train_perf = np.nan
    final_val_perf = np.nan
    final_test_perf = np.nan
    final_train_perf, _ = evaluate(model, device, train_loader, evaluator, config.data.task_type, mask=train_mask, cfg=config)
    final_val_perf, _ = evaluate(model, device, valid_loader, evaluator, config.data.task_type, mask=val_mask, cfg=config)
    if test_loader is not None:
        final_test_perf, _ = evaluate(model, device, test_loader, evaluator, config.data.task_type, mask=test_mask, cfg=config, is_test=True)

    # save results
    curves = {
        'train_loss': train_loss_curve,
        'train': train_curve,
        'val': valid_curve,
        'test': test_curve,
        'last_val': final_val_perf,
        'last_test': final_test_perf,
        'last_train': final_train_perf,
        'best': best_val_epoch}

    if config.exp.wandb:
        wandb.log({
            'last_val': final_val_perf,
            'last_test': final_test_perf,
            'last_train': final_train_perf,
            'best': best_val_epoch,
            'best_train': train_curve[best_val_epoch],
            'best_val': valid_curve[best_val_epoch],
            'best_test': test_curve[best_val_epoch],
            'trainable_params': trainable_params,
            'total_params': total_params
        })

    msg = (
       f'========== Result ============\n'
       f'Dataset:        {config.data.dataset}\n'
       f'------------ Best epoch -----------\n'
       f'Train:          {train_curve[best_val_epoch]}\n'
       f'Validation:     {valid_curve[best_val_epoch]}\n'
       f'Test:           {test_curve[best_val_epoch]}\n'
       f'Best epoch:     {best_val_epoch}\n'
       '------------ Last epoch -----------\n'
       f'Train:          {final_train_perf}\n'
       f'Validation:     {final_val_perf}\n'
       f'Test:           {final_test_perf}\n'
       '-------------------------------\n\n')
    print(msg)

    msg += str(config)
    if config.exp.wandb:
        wandb.finish()
    return curves


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    yaml_name = args.yaml_name
    cfg = load_yaml(yaml_name)
    config = load_config(cfg)
    # validate_args(args)

    print(f'Starting {config.model.model_name} on {config.data.dataset} | SHA: {config.exp.sha}')
    curves = main(config)
    print(f'Finished {config.model.model_name} on {config.data.dataset} | SHA: {config.exp.sha}')
