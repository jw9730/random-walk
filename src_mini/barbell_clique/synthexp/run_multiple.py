import itertools
import numpy as np
import wandb

from synthexp.run import main
from synthexp.parsers.load_parsers import load_config, load_yaml, get_parser


def exp_main(config):
    results = []
    for seed in range(config.exp.start_seed, config.exp.start_seed + config.exp.num_seeds):
        config.exp.seed = seed
        curves = main(config)
        results.append(curves)

    # Extract results
    train_curves = [curves['train'] for curves in results]
    val_curves = [curves['val'] for curves in results]
    test_curves = [curves['test'] for curves in results]
    best_idx = [curves['best'] for curves in results]
    last_train = [curves['last_train'] for curves in results]
    last_val = [curves['last_val'] for curves in results]
    last_test = [curves['last_test'] for curves in results]

    # Extract results at the best validation epoch.
    best_epoch_train_results = [train_curves[i][best] for i, best in enumerate(best_idx)]
    best_epoch_train_results = np.array(best_epoch_train_results, dtype=np.float64)
    best_epoch_val_results = [val_curves[i][best] for i, best in enumerate(best_idx)]
    best_epoch_val_results = np.array(best_epoch_val_results, dtype=np.float64)
    best_epoch_test_results = [test_curves[i][best] for i, best in enumerate(best_idx)]
    best_epoch_test_results = np.array(best_epoch_test_results, dtype=np.float64)

    # Compute stats for the best validation epoch
    mean_train_perf = np.mean(best_epoch_train_results)
    std_train_perf = np.std(best_epoch_train_results, ddof=1)  #  makes the estimator unbiased  # TODO
    mean_val_perf = np.mean(best_epoch_val_results)
    std_val_perf = np.std(best_epoch_val_results, ddof=1)  # makes the estimator unbiased
    mean_test_perf = np.mean(best_epoch_test_results)
    std_test_perf = np.std(best_epoch_test_results, ddof=1)  # ddof=1 makes the estimator unbiased
    min_perf = np.min(best_epoch_test_results)
    max_perf = np.max(best_epoch_test_results)

    # Compute stats for the last epoch
    mean_final_train_perf = np.mean(last_train)
    std_final_train_perf = np.std(last_train)
    mean_final_val_perf = np.mean(last_val)
    std_final_val_perf = np.std(last_val)
    mean_final_test_perf = np.mean(last_test)
    std_final_test_perf = np.std(last_test)
    final_test_min = np.min(last_test)
    final_test_max = np.max(last_test)

    if config.exp.wandb:
        wandb.init(project="barbell-clique",
                   group="avg_res-"+config.data.dataset,
                   name="final-eval-" + config.model.model_name + "-" + str(config.model.walk_len) + "-" + config.data.dataset)
        wandb.config.update(config.get_dic())
        print(f"Running with wandb account: {config.exp.entity}")

        print("======================== Args ===========================")
        print(config.get_dic())
        print("===================================================")

        wandb.log({
            'final_val_mean': mean_final_val_perf,
            'final_val_std': std_final_val_perf,
            'final_test_mean': mean_final_test_perf,
            'final_test_std': std_final_test_perf,
            'final_train_mean': mean_final_train_perf,
            'final_train_std': std_final_train_perf,
            'final_test_min': final_test_min,
            'final_test_max': final_test_max,

            'best_train_mean': mean_train_perf,
            'best_train_std': std_train_perf,
            'best_val_mean': mean_val_perf,
            'best_val_std': std_val_perf,
            'best_test_mean': mean_test_perf,
            'best_test_std': std_test_perf,
            'best_min_perf': min_perf,
            'best_max_perf': max_perf
        })

        wandb.finish()

    msg = (
        f"========= Final result ==========\n"
        f'Dataset:                {config.data.dataset}\n'
        f'SHA:                    {config.exp.sha}\n'
        f'----------- Best epoch ----------\n'
        f'Train:                  {mean_train_perf} ± {std_train_perf}\n'
        f'Valid:                  {mean_val_perf} ± {std_val_perf}\n'
        f'Test:                   {mean_test_perf} ± {std_test_perf}\n'
        f'Test Min:               {min_perf}\n'
        f'Test Max:               {max_perf}\n'
        f'----------- Last epoch ----------\n'
        f'Train:                  {mean_final_train_perf} ± {std_final_train_perf}\n'
        f'Valid:                  {mean_final_val_perf} ± {std_final_val_perf}\n'
        f'Test:                   {mean_final_test_perf} ± {std_final_test_perf}\n'
        f'Test Min:               {final_test_min}\n'
        f'Test Max:               {final_test_max}\n'
        f'---------------------------------\n\n')
    print(msg)

    return {
        'mean_train': mean_train_perf,
        'std_train': std_train_perf,
        'mean_val': mean_val_perf,
        'std_val': std_val_perf
    }


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    yaml_name = args.yaml_name
    cfg = load_yaml(yaml_name)

    new_config = {}

    for meta_key in ["exp", "data", "model", "optim"]:
        keys, values = zip(*cfg[meta_key].items())
        new_config[meta_key] = [dict(zip(keys, v)) for v in itertools.product(*values)]  # all combinations
    for (exp_cfg, data_cfg, model_cfg, optim_cfg) in itertools.product(*new_config.values()):
        config_dic = {"exp": exp_cfg, "data": data_cfg,
                      "model": model_cfg, "optim": optim_cfg}

        config = load_config(config_dic)
        exp_main(config)
