import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history, plot_param_importances
from .hyperparams_opt import sample_ppo_params, sample_trpo_params

import argparse
import json
from harl.utils.configs_tools import get_defaults_yaml_args, update_args


N_TRIALS = 20
N_JOBS = 1
N_STARTUP_TRIALS = 1
N_EVALUATIONS = 10
N_TIMESTEPS = int(1e6)
EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)
N_EVAL_EPISODES = 40
N_EVAL_ENVS = 4
TIMEOUT = int(60 * 30)  # 30 minutes



def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="happo",
        choices=[
            "happo",
            "hatrpo",
            "haa2c",
            "haddpg",
            "hatd3",
            "hasac",
            "had3qn",
            "maddpg",
            "mappo",
            "clip_happo",
            "clip_hatrpo",
        ],
        help="Algorithm name. Choose from: happo, hatrpo, haa2c, haddpg, hatd3, hasac, had3qn, maddpg, mappo.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="pettingzoo_mpe",
        choices=[
            "smac",
            "mamujoco",
            "pettingzoo_mpe",
            "gym",
            "football",
            "dexhands",
            "smacv2",
            "lag",
        ],
        help="Environment name. Choose from: smac, mamujoco, pettingzoo_mpe, gym, football, dexhands, smacv2, lag.",
    )
    parser.add_argument(
        "--exp_name", type=str, default="installtest", help="Experiment name."
    )
    parser.add_argument(
        "--n_step_each_trial", type=int, default=2e6, help="max step in each trial."
    )
    parser.add_argument(
        "--load_config",
        type=str,
        default="",
        help="If set, load existing experiment config file instead of reading from yaml config file.",
    )
    args, unparsed_args = parser.parse_known_args()

    def process(arg):
        try:
            return eval(arg)
        except:
            return arg

    keys = [k[2:] for k in unparsed_args[0::2]]  # remove -- from argument
    values = [process(v) for v in unparsed_args[1::2]]
    unparsed_dict = {k: v for k, v in zip(keys, values)}
    args = vars(args)  # convert to dict
    if args["load_config"] != "":  # load config from existing config file
        with open(args["load_config"], encoding="utf-8") as file:
            all_config = json.load(file)
        args["algo"] = all_config["main_args"]["algo"]
        args["env"] = all_config["main_args"]["env"]
        args["exp_name"] = all_config["main_args"]["exp_name"]
        algo_args = all_config["algo_args"]
        env_args = all_config["env_args"]
    else:  # load config from corresponding yaml file
        algo_args, env_args = get_defaults_yaml_args(args["algo"], args["env"])

    args["exp_name"] = args["exp_name"] + "_tuning"
    def objective(trial: optuna.Trial):
        tuning_params = None
        if "ppo" in args["algo"]:
            tuning_params = sample_ppo_params(trial)
        elif "trpo" in args["algo"]:
            tuning_params = sample_trpo_params(trial)

        unparsed_dict.update(tuning_params)
        # only run for 2M steps at most
        if "num_env_steps" in unparsed_dict:
            unparsed_dict["num_env_steps"] = min(unparsed_dict["num_env_steps"], args["n_step_each_trial"])
        else:
            unparsed_dict["num_env_steps"] = args["n_step_each_trial"]
        update_args(unparsed_dict, algo_args, env_args)  # update args from command line

        if args["env"] == "dexhands":
            import isaacgym  # isaacgym has to be imported before PyTorch

        # note: isaac gym does not support multiple instances, thus cannot eval separately
        if args["env"] == "dexhands":
            algo_args["eval"]["use_eval"] = False
            algo_args["train"]["episode_length"] = env_args["hands_episode_length"]

        # start training
        from harl.runners import RUNNER_REGISTRY

        runner = RUNNER_REGISTRY[args["algo"]](args, algo_args, env_args, trial)
        ret = runner.run()
        runner.close()
        return ret


    sampler = TPESampler()
    # Do not prune before 1/3 of the max budget is used
    pruner = MedianPruner()

    study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")
    
    try:
        study.optimize(objective, n_trials=N_TRIALS, n_jobs=N_JOBS)
    except KeyboardInterrupt:
        pass
        
    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")

    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print(f"    {key}: {value}")


    from datetime import datetime
    current_datetime = datetime.now().strftime("%Y-%m-%d %H-%M-%S")

    # Write report
    study.trials_dataframe().to_csv(f"tuned_{args['algo']}_{env_args['scenario']}_{env_args['agent_conf']}_{current_datetime}.csv")

    with open("study.pkl", "wb+") as f:
        pkl.dump(study, f)

    fig1 = plot_optimization_history(study)
    fig2 = plot_param_importances(study)

    fig1.show()
    fig2.show()


if __name__ == "__main__":
    main()
