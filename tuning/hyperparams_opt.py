from typing import Any, Dict
import numpy as np
import optuna


# based on https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/rl_zoo3/hyperparams_opt.py
def sample_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    # gamma = trial.suggest_categorical("gamma", [0.95, 0.99])
    clip_param = trial.suggest_categorical("clip_param", [0.025, 0.05, 0.1, 0.15, 0.2])
    # use_recurrent_policy = trial.suggest_categorical("use_recurrent_policy", [True, False])
    entropy_coef = trial.suggest_categorical("entropy_coef", [0, 0.01, 0.001])
    use_linear_lr_decay = trial.suggest_categorical("use_linear_lr_decay", [True, False])
    n_epoch = trial.suggest_categorical("n_epoch", [5, 10, 15])
    
    ppo_epoch = n_epoch
    critic_epoch = n_epoch
    
    return {
        "ppo_epoch": ppo_epoch,
        "critic_epoch" : critic_epoch,
        "use_linear_lr_decay": use_linear_lr_decay,
        "entropy_coef": entropy_coef,
        # "use_recurrent_policy": use_recurrent_policy,
        "clip_param": clip_param,
        # "gamma": gamma
    }

def sample_trpo_params(trial: optuna.Trial) -> Dict[str, Any]:
    # gamma = trial.suggest_categorical("gamma", [0.95, 0.99])
    accept_ratio = trial.suggest_categorical("accept_ratio", [0.5])
    # use_recurrent_policy = trial.suggest_categorical("use_recurrent_policy", [True, False])
    use_linear_lr_decay = trial.suggest_categorical("use_linear_lr_decay", [True, False])
    kl_threshold = trial.suggest_categorical("kl_threshold", [0.001, 0.005, 0.01])
    backtrack_coeff = trial.suggest_categorical("backtrack_coeff", [0.8])

    return {
        "accept_ratio": accept_ratio,
        "use_linear_lr_decay": use_linear_lr_decay,
        "kl_threshold": kl_threshold,
        "backtrack_coeff": backtrack_coeff,
    }
