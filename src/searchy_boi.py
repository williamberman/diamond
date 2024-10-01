from agent import Agent
from envs import make_atari_env, WorldModelEnv
from hydra.utils import instantiate
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import random
from PIL import Image
import imageio
from models.diffusion import Denoiser, DiffusionSampler, DiffusionSamplerConfig
from models.rew_end_model import RewEndModel
import tqdm
from collections import defaultdict
import torch.distributed as dist
import os
import time
import gc
import math
import matplotlib.pyplot as plt

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

            choose_action_out = choose_action(obs, actions, ctr)
            if device == 0:
                action, skip_count = choose_action_out

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
save_prefix = "/workspace/running_with_theirs_test_avg/"
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

# available_sample_actions = [0, 2, 3]
# available_sample_actions = [0, 1, 2, 3]
available_sample_actions = [1, 2, 3]
# available_sample_actions = [2, 3]

def init_search_actions(n):
    acts = [[x] for x in available_sample_actions]

    if n == 1:
        return acts

    n_1 = init_search_actions(n-1)

    rv = []

    for act in acts:
        for n_1_act in n_1:
            rv.append(act + n_1_act)

    return rv

def choose_action(obs, actions, ctr):
    assert obs.ndim == 5
    assert actions.ndim == 2
    assert obs.shape[1] == actions.shape[1] + 1

    if obs.shape[1] < num_steps_conditioning:
        return random.randint(0, num_actions - 1)

    input_seq_len = obs.shape[1]

    all_rew = defaultdict(list)
    all_end = defaultdict(list)
    all_obs = defaultdict(list)
    all_actions = defaultdict(list)

    # init_search_actions_ = init_search_actions(10) # when 2 options
    # init_search_actions_ = init_search_actions(7)
    # init_search_actions_ = init_search_actions(8) # when 3 options
    # init_search_actions_ = init_search_actions(5)
    # init_search_actions_ = init_search_actions(15)
    # init_search_actions_ = init_search_actions(5)
    init_search_actions_ = init_search_actions(10)

    init_search_actions_ = [x for i, x in enumerate(init_search_actions_) if i % dist.get_world_size() == device]

    n_samples = 1
    # n_samples = 5

    mc_traj_len = 20

    outter_loop = [x for x in range(0, len(init_search_actions_), 1024)]

    pbar = tqdm.tqdm(total=len(outter_loop)*(len(init_search_actions_[0])+(n_samples*mc_traj_len)), desc="initial rollout", disable=device != 0)

    for start_idx in outter_loop:
        end_idx = min(start_idx + 1024, len(init_search_actions_))
        init_search_actions_batch = init_search_actions_[start_idx:end_idx]

        # if device == 0:
        #     print("doing initial rollout")
        obs_, actions_, rew, end = rollout_trajectory(obs, actions, init_search_actions_batch, pbar)
        # if device == 0:
        #     print("done initial rollout")

        for _ in range(n_samples):
            # obs__, actions__, rew_, end_ = sample_trajectory(obs_, actions_, 15) # when initial search depth 10
            obs__, actions__, rew_, end_ = sample_trajectory(obs_, actions_, mc_traj_len, pbar)
            # obs__, actions__, rew_, end_ = sample_trajectory(obs_, actions_, 10)

            rew_ = torch.cat([rew, rew_], dim=1)
            end_ = torch.cat([end, end_], dim=1)

            # rew_ = rew_.sum(dim=-1)
            rew_ = rew_.max(dim=-1).values
            end_ = end_.any(dim=-1)

            rew_ = torch.where(end_, torch.tensor([-1.0], device=device), rew_)

            for init_act, obs___, actions___, rew__, end__ in zip(init_search_actions_batch, obs__, actions__, rew_, end_):
                all_rew[tuple(init_act)].append(rew__)
                # all_end[tuple(init_act)].append(end__)
                # all_obs[tuple(init_act)].append(obs___)
                # all_actions[tuple(init_act)].append(actions___)

        gc.collect()
        torch.cuda.empty_cache()

    pbar.close()

    # return choose_action_orig(all_obs, all_rew, all_end, input_seq_len, ctr)
    return choose_action_take_avg(all_rew, ctr)

avg_rews_dbg = {0: [], 1: [], 2: [], 3: []}

def choose_action_take_avg(all_rew, ctr):
    sum_rews = torch.zeros((max(available_sample_actions)+1,), device=device, dtype=torch.float32)
    counts = torch.zeros((max(available_sample_actions)+1,), device=device, dtype=torch.float32)

    for idx in range(len(available_sample_actions)):
        if idx not in available_sample_actions:
            sum_rews[idx] = float("-inf")

    for actions, rews in all_rew.items():
        for rew in rews:
            sum_rews[actions[0]] += rew
            counts[actions[0]] += 1

    # gather sum_rews and counts to device 0
    if device == 0:
        sum_rews_target = [torch.zeros_like(sum_rews) for _ in range(dist.get_world_size())]
        counts_target = [torch.zeros_like(counts) for _ in range(dist.get_world_size())]
    else:
        sum_rews_target = None
        counts_target = None

    dist.gather(sum_rews, sum_rews_target, 0)
    dist.gather(counts, counts_target, 0)

    if device == 0:
        sum_rews = torch.stack(sum_rews_target, dim=1).sum(dim=1)
        counts = torch.stack(counts_target, dim=1).sum(dim=1)
        avg_rews = sum_rews / counts
        action = avg_rews.argmax().item()

        print_str = ", ".join(f"{count} {rew:.4f} {avg_rew:.4f}" for rew, count, avg_rew in zip(sum_rews, counts, avg_rews))
        print(print_str)
        print(f"device {device}: chose best action {action} with avg_rew {avg_rews[action].item()}")

        for act in available_sample_actions:
            it = avg_rews[act].item()
            if math.isinf(it):
                it = 0
            avg_rews_dbg[act].append(it)

        # write out line graph of each avg_rews_dbg[i] to file
        plt.figure()
        for act, rew in avg_rews_dbg.items():
            plt.plot(rew, label=f"action {act}")
        plt.legend()
        plt.savefig(os.path.join(save_prefix, f"obs_{ctr}.png"))
        plt.close()

        skip_count = 10
        for act in available_sample_actions:
            if avg_rews[act].item() != 1.0:
                skip_count = 0

        if skip_count > 0:
            print(f"All scores are 1.0! skipping for {skip_count} steps")

        return action, skip_count


def choose_action_orig(all_obs, all_rew, all_end, input_seq_len, ctr):
    n_samples = len(next(iter(all_obs.values())))

    # avg_rews = {act: sum(rew)/len(rew) for act, rew in all_rew.items()}
    avg_rews = {act: max(rew) for act, rew in all_rew.items()}
    ends = {act: any(end) for act, end in all_end.items()}

    best_rew = 0
    best_action = None

    for action in avg_rews.keys():
        avg_rew = avg_rews[action]
        end = ends[action]

        if avg_rew > best_rew and not end:
            best_rew = avg_rew
            best_action = action

    suffix = f"_best"
    if best_action is None:
        best_action = random.choice(list(avg_rews.keys()))
        suffix = f"_random"

    # if device == 0:
        # print(f"avg_rews: {avg_rews}, ends: {ends}, best_action: {best_action}, best_rew: {best_rew}")
        # print(f"best_action: {best_action}, best_rew: {best_rew}")

    print(f"device {device}: best_action: {best_action}, best_rew: {best_rew}")

    for i in range(n_samples):
        save(all_obs[best_action][i][max(input_seq_len-3, 0):], f"pred_{ctr}_{device}_{i}_{best_rew:.4f}_{best_action}{suffix}")

    action = best_action[0]

    best_rew = best_rew.item()

    # gather best_rew
    best_rew = torch.tensor([best_rew], device=device)
    if device == 0:
        best_rew_target = [torch.zeros_like(best_rew) for _ in range(dist.get_world_size())]
    else:
        best_rew_target = None
    dist.gather(best_rew, best_rew_target, 0)

    # gather action
    action = torch.tensor([action], device=device)
    if device == 0:
        action_target = [torch.zeros_like(action) for _ in range(dist.get_world_size())]
    else:
        action_target = None
    dist.gather(action, action_target, 0)

    if device == 0:
        best_rew = torch.concat(best_rew_target, dim=0).tolist()
        action = torch.concat(action_target, dim=0).tolist()
        action_rew = zip(action, best_rew)
        action_rew = sorted(action_rew, key=lambda x: x[1], reverse=True)
        print(f"device {device}: chose best action {action_rew[0][0]} with rew {action_rew[0][1]}")
        action = action_rew[0][0]
        return action

def rollout_trajectory(obs, actions, rollout, pbar):
    assert obs.ndim == 5
    assert actions.ndim == 2
    assert obs.shape[1] == actions.shape[1] + 1

    obs = obs.repeat((len(rollout), 1, 1, 1, 1))
    actions = actions.repeat(len(rollout), 1) 
    rew = None
    end = None

    for rollout_idx in range(len(rollout[0])):
        new_actions = torch.tensor([x[rollout_idx] for x in rollout], dtype=torch.long, device=device).reshape(len(rollout), 1)
        actions = torch.cat([actions, new_actions], dim=1)
        assert actions.shape[:2] == obs.shape[:2]

        obs_ = obs[:, -num_steps_conditioning:]
        act_ = actions[:, -num_steps_conditioning:]
        next_obs = sampler.sample_next_obs(obs_, act_)[0]

        next_obs_ = torch.cat([obs_[:, 1:], next_obs.unsqueeze(1)], dim=1)
        obs_ = obs_.contiguous()
        act_ = act_.contiguous()
        next_obs_ = next_obs_.contiguous()
        rew_, end_, _ = rew_end_model(obs_, act_, next_obs_)
        probs = rew_[:, -1, :].softmax(dim=-1)
        pos_probs = probs[:, 2]
        neg_probs = probs[:, 0]
        # pos_probs = torch.where(pos_probs <= 0.02, torch.tensor([0.0], device=device), pos_probs)
        rew_ = pos_probs - neg_probs
        end_ = end_[:, -1, :].argmax(dim=-1)
        if rew is None:
            rew = rew_.unsqueeze(1)
        else:
            rew = torch.cat([rew, rew_.unsqueeze(1)], dim=1)
        if end is None:
            end = end_.unsqueeze(1)
        else:
            end = torch.cat([end, end_.unsqueeze(1)], dim=1)

        obs = torch.cat([obs, next_obs.unsqueeze(1)], dim=1)

        pbar.update(1)

    return obs, actions, rew, end

def rand_action_tensor(shape):
    n = 1
    for x in shape:
        n *= x
    return torch.tensor([random.choice(available_sample_actions) for _ in range(n)], dtype=torch.long, device=device).reshape(shape)

def sample_trajectory(obs, actions, trajectory_length, pbar):
    assert obs.ndim == 5
    assert actions.ndim == 2
    assert obs.shape[0] == actions.shape[0]
    assert obs.shape[1] == actions.shape[1]+1

    rew = None
    end = None

    for _ in range(trajectory_length):
        actions = torch.cat([actions, rand_action_tensor((actions.shape[0], 1))], dim=1)

        obs_ = obs[:, -num_steps_conditioning:]
        act_ = actions[:, -num_steps_conditioning:]

        next_obs = sampler.sample_next_obs(obs_, act_)[0]
        next_obs_ = torch.cat([obs_[:, 1:], next_obs.unsqueeze(1)], dim=1)

        assert obs_.shape[:2] == next_obs_.shape[:2]
        assert obs_.shape[:2] == act_.shape[:2]

        obs_ = obs_.contiguous()
        act_ = act_.contiguous()
        next_obs_ = next_obs_.contiguous()
        rew_, end_, _ = rew_end_model(obs_, act_, next_obs_)

        probs = rew_[:, -1, :].softmax(dim=-1)
        pos_probs = probs[:, 2]
        neg_probs = probs[:, 0]
        # pos_probs = torch.where(pos_probs <= 0.02, torch.tensor([0.0], device=device), pos_probs)
        rew_ = pos_probs - neg_probs
        rew_ = rew_.unsqueeze(1)

        end_ = end_[:, -1, :].argmax(dim=-1, keepdim=True)

        if rew is None:
            rew = rew_
        else:
            rew = torch.cat([rew, rew_], dim=1)
        if end is None:
            end = end_
        else:
            end = torch.cat([end, end_], dim=1)

        obs = torch.cat([obs, next_obs.unsqueeze(1)], dim=1)

        pbar.update(1)

    assert rew.shape[0] == obs.shape[0]
    assert rew.shape[1] == trajectory_length
    assert end.shape[1] == trajectory_length

    return obs, actions, rew, end


if __name__ == "__main__":
    main()