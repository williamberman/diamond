from agent import Agent
from envs import make_atari_env
from hydra.utils import instantiate
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import random
from PIL import Image
import imageio
from models.diffusion import DiffusionSampler
import torch.distributed as dist
import os
import time
import gc
import matplotlib.pyplot as plt
import itertools
import tqdm
import numpy as np

OmegaConf.register_new_resolver("eval", eval)

torch.set_grad_enabled(False)
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(device)

@hydra.main(config_path="../config", config_name="trainer", version_base="1.3")
def main(cfg_: DictConfig):
    global cfg
    cfg = cfg_
    cfg.common.device = f"cuda:{device}"

    dist.init_process_group(backend="nccl")
    init_global_state()

    skip_count = 0

    if device == 0:
        env_loop_ = env_loop()

    while True:
        t0 = time.perf_counter()

        if device == 0:
            action = env_loop_.step1()

            if action is None and skip_count <= 0:
                obs_send, actions_send = env_loop_.obs, env_loop_.actions

                obs_send = torch.stack(obs_send, dim=1)
                actions_send = torch.tensor(actions_send, device=device, dtype=torch.long).unsqueeze(0)

                scatter_data = [obs_send.shape, actions_send.shape, True, env_loop_.ctr]

                obs_send = [obs_send for _ in range(dist.get_world_size())]
                actions_send = [actions_send for _ in range(dist.get_world_size())]
            else:
                if skip_count > 0:
                    action = 0
                    skip_count -= 1

                scatter_data = [None, None, False, env_loop_.ctr]
                obs_send, actions_send = None, None
        else:
            scatter_data = None
            obs_send, actions_send = None, None

        scatter_data = [scatter_data for _ in range(dist.get_world_size())]
        scatter_data_output = [None]
        dist.scatter_object_list(scatter_data_output, scatter_data, src=0)
        obs_shape, actions_shape, do_scatter, ctr = scatter_data_output[0]

        if do_scatter:
            obs = torch.empty(obs_shape, device=device, dtype=torch.float32)
            actions = torch.empty(actions_shape, device=device, dtype=torch.long)

            dist.scatter(obs, obs_send, 0)
            dist.scatter(actions, actions_send, 0)

            action = choose_action(obs, actions, ctr=ctr)
            skip_count = 0

        if device == 0:
            end = env_loop_.step2(action)
            end = torch.tensor([end], device=device)
            scatter_data = [end for _ in range(dist.get_world_size())]
        else:
            end = torch.tensor([False], device=device)
            scatter_data = None

        dist.scatter(end, scatter_data, 0)

        if do_scatter:
            gc.collect()
            torch.cuda.empty_cache()

        if device == 0:
            print(f"time taken: {time.perf_counter() - t0}")

        if end:
            break

    if device == 0:
        save(env_loop_.obs, env_loop_.ctr)
        
def init_global_state():
    global sampler, rew_end_model, num_steps_conditioning, num_actions

    env = make_atari_env(num_envs=cfg.collection.test.num_envs, device=device, **cfg.env.test)
    num_actions = int(env.num_actions)

    # ckpt = "/mnt/raid/diamond/better4/Breakout_100k_labeled_1000_actor_critic_cont/checkpoints/agent_versions/agent_epoch_01000.pt"
    ckpt = "/workspace/Breakout.pt"
    # ckpt = "/workspace/my_wm.pt"
    agent = Agent(instantiate(cfg.agent, num_actions=num_actions)).to(device, dtype=torch.bfloat16)
    agent.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")))
    agent.eval()

    num_steps_conditioning = agent.denoiser.inner_model.cfg.num_steps_conditioning

    denoiser = torch.compile(agent.denoiser, mode="reduce-overhead")
    rew_end_model = torch.compile(agent.rew_end_model, mode="reduce-overhead")
    # denoiser = agent.denoiser
    # rew_end_model = agent.rew_end_model

    sampler = DiffusionSampler(denoiser, cfg.world_model_env.diffusion_sampler)

class env_loop:
    def __init__(self):
        self.env = make_atari_env(num_envs=cfg.collection.test.num_envs, device=device, **cfg.env.test)
        state, info = self.env.reset()

        self.info = info

        self.obs = [state]
        self.actions = []

        self.lives_1_started = False
        self.lives_1_ctr = 0

        self.ctr = 0
        self.first_did_pred = None

    def step1(self):
        print(self.ctr)

        self.did_pred = False

        if self.ctr == 0 or self.info['lives'] != 1 or not self.lives_1_started:
            action = 1

            if self.info['lives'] == 1 and not self.lives_1_started:
                print("started playing with 1 life")
                self.lives_1_started = True
        else:
            if self.lives_1_ctr < 10:
                action = 0
                self.lives_1_ctr += 1
            else:
                action = None
                self.did_pred = True
                if self.first_did_pred is None:
                    self.first_did_pred = self.ctr

        return action

    def step2(self, action):
        self.actions.append(action)
        state, _, end, trunc, info = self.env.step(torch.tensor([action]))
        self.info = info
        self.obs.append(state)

        if self.did_pred:
            save(self.obs[self.first_did_pred:], self.ctr)

        self.ctr += 1

        return end or trunc

# save_prefix = "/workspace/run_see_if_access_to_fire_helps/"
# save_prefix = "/workspace/running_with_theirs_2/"
# save_prefix = "/workspace/running_with_theirs_reward_averaging/"
# save_prefix = "/workspace/running_with_theirs_test_avg_no_mc/"
# save_prefix = "/workspace/running_with_theirs_test_avg_no_zero/"
save_prefix = "/workspace/test/"
if device == 0:
    os.makedirs(save_prefix, exist_ok=True)

def save(obs, idx):
    if isinstance(obs, list):
        obs = torch.cat(obs, dim=0)
    obs = obs.permute(0, 2, 3, 1).mul(0.5).add(0.5).mul(255).clamp(0, 255).byte().cpu().numpy()
    obs = [Image.fromarray(x).resize((256, 256)) for x in obs]
    imageio.mimsave(os.path.join(save_prefix, f"obs_{idx}.mp4"), obs, fps=1)
    os.makedirs(os.path.join(save_prefix, f"obs_{idx}"), exist_ok=True)
    # for i, x in enumerate(obs):
    #     x.save(os.path.join(save_prefix, f"obs_{idx}/{i}.png"))
    if device == 0:
        print("saved")

available_sample_actions = [0, 1, 2, 3]

avg_rews_dbg = {k: [] for k in available_sample_actions}
end_dbg = {k: [] for k in available_sample_actions}

def choose_action(prefix_obs, prefix_actions, ctr, depth=9, max_batch_size=2048):
    assert prefix_obs.ndim == 5
    assert prefix_actions.ndim == 2
    assert prefix_obs.shape[1] == prefix_actions.shape[1] + 1

    if prefix_obs.shape[1] < num_steps_conditioning:
        return random.randint(0, num_actions - 1)

    assert prefix_obs.shape[1] >= num_steps_conditioning

    prefix_obs = prefix_obs.squeeze(0)
    prefix_actions = prefix_actions.squeeze(0)

    paths = {}
    for idx in range(prefix_obs.shape[0]):
        key = tuple(prefix_actions[:idx].cpu().tolist()) # index action up to but not including current idx
        paths[key] = {'obs': prefix_obs[idx], 'rew': None, 'end': None}

    scattered = False

    for depth_idx in range(depth):
        gc.collect()
        torch.cuda.empty_cache()

        if device == 0:
            print(f"rollout_to_depth {depth_idx+1}/{depth}")

        if device == 0:
            if scattered:
                should_scatter = False
            else:
                max_path_len = max([len(x) for x in paths.keys()])
                num_at_max_path_len = len([x for x in paths.keys() if len(x) == max_path_len])
                should_scatter = num_at_max_path_len >= dist.get_world_size()

            should_scatter = torch.tensor([should_scatter], device=device)
        else:
            should_scatter = torch.tensor([False], device=device)

        dist.broadcast(should_scatter, 0)

        if should_scatter.item():
            if device == 0:
                print("scattering")

            scattered = True

            # each device should get each path less than the max path length. 
            # paths at max path length will be split across devices
            paths_ = [paths] if device == 0 else [None]
            dist.broadcast_object_list(paths_, 0)
            paths = paths_[0]

            max_path_len = max([len(x) for x in paths.keys()])

            paths_not_at_max_len = [x for x in paths.keys() if len(x) < max_path_len]

            paths_at_max_len = sorted([x for x in paths.keys() if len(x) == max_path_len])
            paths_at_max_len = [x for i, x in enumerate(paths_at_max_len) if i % dist.get_world_size() == device]

            paths = {k: paths[k] for k in itertools.chain(paths_not_at_max_len, paths_at_max_len)}

            for k in paths.keys():
                paths[k]['obs'] = paths[k]['obs'].to(device)
                if paths[k]['rew'] is not None:
                    paths[k]['rew'] = paths[k]['rew'].to(device)
                if paths[k]['end'] is not None:
                    paths[k]['end'] = paths[k]['end'].to(device)

        if device == 0 or scattered:
            max_path_len = max([len(x) for x in paths.keys()])

            paths_to_step = [] 
            for act in available_sample_actions:
                for path in paths.keys():
                    if len(path) == max_path_len:
                        paths_to_step.append(list(path) + [act])

            for start_idx in tqdm.tqdm(range(0, len(paths_to_step), max_batch_size), disable=device != 0):
                end_idx = min(start_idx + max_batch_size, len(paths_to_step))
                paths_to_step_batch = paths_to_step[start_idx:end_idx]

                obs_batch = []
                actions_batch = []

                for path in paths_to_step_batch:
                    obs_batch.append([])
                    actions_batch.append(torch.tensor(path[-num_steps_conditioning:], device=device, dtype=torch.long))

                    for idx in range(num_steps_conditioning):
                        path_ = tuple(path[:-num_steps_conditioning+idx])
                        obs_batch[-1].append(paths[path_]['obs'])

                actions_batch = torch.stack(actions_batch).contiguous()
                obs_batch = torch.stack([torch.stack(x) for x in obs_batch]).contiguous()

                assert obs_batch.shape[:2] == (end_idx-start_idx, num_steps_conditioning)
                assert actions_batch.shape[:2] == (end_idx-start_idx, num_steps_conditioning)

                next_obs_batch = sampler.sample_next_obs(obs_batch, actions_batch)[0]
                next_obs_batch = torch.cat([obs_batch[:, 1:], next_obs_batch.unsqueeze(1)], dim=1)
                rew_batch, end_batch, _ = rew_end_model(obs_batch, actions_batch, next_obs_batch)

                # probs = rew_batch[:, -1, :].softmax(dim=-1)
                # pos_probs = probs[:, 2]
                # neg_probs = probs[:, 0]
                # rew_batch = pos_probs - neg_probs
                rew_batch = -(end_batch[:, -1, :].softmax(dim=-1))[:, 1]

                end_batch = end_batch[:, -1, :].argmax(dim=-1)

                for path, obs, rew, end in zip(paths_to_step_batch, next_obs_batch, rew_batch, end_batch):
                    paths[tuple(path)] = {'obs': obs[-1], 'rew': rew, 'end': end}

    max_path_len = max([len(x) for x in paths.keys()])
    final_paths = {k: paths[k] for k in paths.keys() if len(k) == max_path_len}

    results_rew = {x: [] for x in available_sample_actions}
    results_end = {x: [] for x in available_sample_actions}

    # for each final path, compute the reward and end
    for path in final_paths.keys():
        path = list(path)
        first_action = path[prefix_obs.shape[0]-1]

        rew = None
        end = False

        while True:
            rew_ = paths[tuple(path)]['rew']
            end_ = paths[tuple(path)]['end']

            if rew_ is None or end_ is None:
                break

            if rew is None:
                rew = rew_.item()
            else:
                rew = min(rew, rew_.item())
            end = end or end_.item()

            path.pop()


        results_rew[first_action].append(rew)
        results_end[first_action].append(end)

    if device == 0:
        all_results_rew = [None for _ in range(dist.get_world_size())]
    else:
        all_results_rew = None

    dist.gather_object(results_rew, all_results_rew)

    if device == 0:
        all_results_end = [None for _ in range(dist.get_world_size())]
    else:
        all_results_end = None

    dist.gather_object(results_end, all_results_end)

    if device == 0:
        results_rew = {x: [] for x in available_sample_actions}
        for x in all_results_rew:
            for k, v in x.items():
                results_rew[k].extend(v)
        results_rew = {k: np.median(x).item() for k, x in results_rew.items()}

        results_end = {x: [] for x in available_sample_actions}
        for x in all_results_end:
            for k, v in x.items():
                results_end[k].extend(v)

        for act in available_sample_actions:
            avg_rews_dbg[act].append(results_rew[act])

        for act in available_sample_actions:
            end_dbg[act].append(sum(results_end[act])/len(results_end[act]))

        action_to_str = {0: "NOOP", 1: "FIRE", 2: "RIGHT", 3: "LEFT"}

        plt.figure()
        for act, rew in avg_rews_dbg.items():
            plt.plot(rew, label=action_to_str[act])
        plt.legend()
        plt.savefig(os.path.join(save_prefix, f"obs_{ctr}.png"))
        plt.close()

        plt.figure()
        for act, end in end_dbg.items():
            plt.plot(end, label=action_to_str[act])
        plt.legend()
        plt.savefig(os.path.join(save_prefix, f"end_{ctr}.png"))
        plt.close()

        best_rew = max(results_rew.values())
        best_action = [x for x in results_rew.keys() if results_rew[x] == best_rew][0]

        return best_action

if __name__ == "__main__":
    main()