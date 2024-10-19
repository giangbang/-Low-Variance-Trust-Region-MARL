import numpy as np
import torch
import torch.nn as nn
from harl.utils.envs_tools import check
from harl.utils.trans_tools import _t2n

def share_param_train(self,
                      actor_buffer: "List[OnPolicyActorBuffer]",
                      advantages: torch.Tensor,
                      num_agents: int,
                      state_type: str,
                      agent_order: "List[int]"):
    train_info = {}
    train_info["policy_loss"] = 0
    train_info["dist_entropy"] = 0
    train_info["actor_grad_norm"] = 0
    train_info["ratio"] = 0
    train_info["adv_variance"] = 0

    train_infos = [train_info.copy() for _ in range(num_agents)]
    
    # `episode_length` include the last timestep, which should be -1 later
    episode_length, n_rollout_threads = actor_buffer[0].obs.shape[:2]
    episode_length -= 1

    advantages_list = []

    if state_type == "EP":
        advantages_ori_list = []
        advantages_copy_list = []
        for agent_id in range(num_agents):
            advantages_ori = advantages.copy()
            advantages_ori_list.append(advantages_ori)
            advantages_copy = advantages.copy()
            advantages_copy[actor_buffer[agent_id].active_masks[:-1] == 0.0] = np.nan
            advantages_copy_list.append(advantages_copy)
        advantages_ori_tensor = np.array(advantages_ori_list)
        advantages_copy_tensor = np.array(advantages_copy_list)
        mean_advantages = np.nanmean(advantages_copy_tensor)
        std_advantages = np.nanstd(advantages_copy_tensor)
        normalized_advantages = (advantages_ori_tensor - mean_advantages) / (
                std_advantages + 1e-5
        )

        for agent_id in range(num_agents):
            advantages_list.append(normalized_advantages[agent_id])

    elif state_type == "FP":
        for agent_id in range(num_agents):
            advantages_list.append(advantages[:, :, agent_id])

    # the following reshaping combines the first two dimensions (i.e. episode_length and n_rollout_threads) to form a batch
    # only used to compute the factor
    available_actions_list = []
    for agent_id in range(num_agents):
        available_actions = (
            None
            if actor_buffer[agent_id].available_actions is None
            else actor_buffer[agent_id]
                     .available_actions[:-1]
                     .reshape(-1, *actor_buffer[agent_id].available_actions.shape[2:])
        )
        available_actions_list.append(available_actions)

    for _ in range(self.ppo_epoch):
        # factor is used for considering updates made by previous agents
        factor = np.ones(
            (
                episode_length,
                n_rollout_threads,
                1,
            ),
            dtype=np.float32,
        )

        data_generators = []
        for agent_id in range(num_agents):
            if self.use_recurrent_policy:
                data_generator = actor_buffer[agent_id].recurrent_generator_actor(
                    advantages_list[agent_id],
                    self.actor_num_mini_batch,
                    self.data_chunk_length,
                )
            elif self.use_naive_recurrent_policy:
                data_generator = actor_buffer[agent_id].naive_recurrent_generator_actor(
                    advantages_list[agent_id], self.actor_num_mini_batch
                )
            else:
                data_generator = actor_buffer[agent_id].feed_forward_generator_actor(
                    advantages_list[agent_id], self.actor_num_mini_batch
                )
            data_generators.append(data_generator)

        for _ in range(self.actor_num_mini_batch):
            for agent_id in agent_order:
                actor_buffer[agent_id].update_factor(
                    factor
                )

                sample = next(data_generators[agent_id])
                policy_loss, dist_entropy, actor_grad_norm, imp_weights, info = self.update(
                    sample
                )

                # compute action log probs for updated agent
                new_actions_logprob, _, _ = self.evaluate_actions(
                    actor_buffer[agent_id]
                        .obs[:-1]
                        .reshape(-1, *actor_buffer[agent_id].obs.shape[2:]),
                    actor_buffer[agent_id]
                        .rnn_states[0:1]
                        .reshape(-1, *actor_buffer[agent_id].rnn_states.shape[2:]),
                    actor_buffer[agent_id].actions.reshape(
                        -1, *actor_buffer[agent_id].actions.shape[2:]
                    ),
                    actor_buffer[agent_id]
                        .masks[:-1]
                        .reshape(-1, *actor_buffer[agent_id].masks.shape[2:]),
                    available_actions_list[agent_id],
                    actor_buffer[agent_id]
                        .active_masks[:-1]
                        .reshape(-1, *actor_buffer[agent_id].active_masks.shape[2:]),
                )

                old_actions_logprob = check(actor_buffer[agent_id]
                                            .action_log_probs
                                            .reshape(-1, actor_buffer[agent_id]
                                                     .action_log_probs.shape[-1])).to(self.device)

                # update factor for next agent
                # assert new_actions_logprob.shape == old_actions_logprob.shape
                factor = factor * _t2n(
                    getattr(torch, self.action_aggregation)(
                        torch.exp(new_actions_logprob - old_actions_logprob), dim=-1
                    ).reshape(
                        episode_length,
                        n_rollout_threads,
                        1,
                    )
                )

                train_info = train_infos[agent_id]
                train_info["policy_loss"] += policy_loss.item()
                train_info["dist_entropy"] += dist_entropy.item()
                train_info["actor_grad_norm"] += actor_grad_norm
                train_info["ratio"] += imp_weights.mean()
                train_info["adv_variance"] += info["adv_variance"]
                

    num_updates = self.ppo_epoch * self.actor_num_mini_batch
    for train_info in train_infos:
        for k in train_info.keys():
            train_info[k] /= num_updates

    return train_infos