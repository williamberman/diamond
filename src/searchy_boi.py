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


OmegaConf.register_new_resolver("eval", eval)

torch.set_grad_enabled(False)
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device = 2

@hydra.main(config_path="../config", config_name="trainer", version_base="1.3")
def main(cfg: DictConfig):
    global num_actions, num_steps_conditioning, ctr

    # ckpt = "/mnt/raid/diamond/better4/Breakout_100k_labeled_1000_actor_critic_cont/checkpoints/agent_versions/agent_epoch_01000.pt"
    ckpt = "/workspace/Breakout.pt"

    test_env = make_atari_env(num_envs=cfg.collection.test.num_envs, device=device, **cfg.env.test)
    num_actions = int(test_env.num_actions)

    agent = Agent(instantiate(cfg.agent, num_actions=num_actions)).to(device, dtype=torch.bfloat16)
    agent.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")))
    agent.eval()

    num_steps_conditioning = agent.denoiser.inner_model.cfg.num_steps_conditioning

    denoiser = torch.compile(agent.denoiser, mode="reduce-overhead")
    rew_end_model = torch.compile(agent.rew_end_model, mode="reduce-overhead")

    sampler = DiffusionSampler(denoiser, cfg.world_model_env.diffusion_sampler)

    state, info = test_env.reset()
    obs = [state]
    actions = []

    ctr = 0
    lives_1_started = False
    lives_1_ctr = 0

    while True:
        print(ctr)

        did_pred = False

        if ctr == 0 or info['lives'] != 1 or not lives_1_started:
            action = 1

            if info['lives'] == 1 and not lives_1_started:
                print("started playing with 1 life")
                lives_1_started = True
        else:
            if lives_1_ctr < 10:
                action = 0
                lives_1_ctr += 1
            else:
                action = choose_action(sampler, rew_end_model, obs, actions)
                did_pred = True

        actions.append(action)
        state, _, end, trunc, info = test_env.step(torch.tensor([action]))
        obs.append(state)

        if end or trunc:
            break

        if did_pred:
            save(obs, ctr)

        ctr += 1

    save(obs, ctr)

def save(obs, idx):
    if isinstance(obs, list):
        obs = torch.cat(obs, dim=0)
    obs = obs.permute(0, 2, 3, 1).mul(0.5).add(0.5).mul(255).clamp(0, 255).byte().cpu().numpy()
    obs = [Image.fromarray(x).resize((256, 256)) for x in obs]
    imageio.mimsave(f"/workspace/obs_{idx}.mp4", obs, fps=5)
    print("saved")

# available_sample_actions = [0, 2, 3]
available_sample_actions = [2, 3]

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

def choose_action(sampler, rew_end_model, obs, actions):
    assert len(obs) == len(actions) + 1

    if len(obs) < num_steps_conditioning:
        return random.randint(0, num_actions - 1)

    all_rew = defaultdict(list)
    all_end = defaultdict(list)
    all_obs = defaultdict(list)
    all_actions = defaultdict(list)

    init_search_actions_ = init_search_actions(10)
    n_samples = 2

    print("doing initial rollout")
    obs_, actions_, rew, end = rollout_trajectory(sampler, rew_end_model, obs, actions, init_search_actions_)
    print("done initial rollout")

    for _ in range(n_samples):
        obs__, actions__, rew_, end_ = sample_trajectory(sampler, rew_end_model, obs_, actions_, 15)
        rew_ = torch.cat([rew, rew_], dim=1)
        end_ = torch.cat([end, end_], dim=1)
        rew_ = rew_.sum(dim=-1)
        end_ = end_.any(dim=-1)
        for init_act, obs___, actions___, rew__, end__ in zip(init_search_actions_, obs__, actions__, rew_, end_):
            all_rew[tuple(init_act)].append(rew__)
            all_end[tuple(init_act)].append(end__)
            all_obs[tuple(init_act)].append(obs___)
            all_actions[tuple(init_act)].append(actions___)

    avg_rews = {act: sum(rew)/len(rew) for act, rew in all_rew.items()}
    ends = {act: any(end) for act, end in all_end.items()}

    best_rew = 0
    best_action = None

    for action in avg_rews.keys():
        avg_rew = avg_rews[action]
        end = ends[action]

        if avg_rew > best_rew and not end:
            best_rew = avg_rew
            best_action = action

    if best_action is None:
        best_action = random.choice(list(avg_rews.keys()))

    # print(f"avg_rews: {avg_rews}, ends: {ends}, best_action: {best_action}, best_rew: {best_rew}")
    print(f"best_action: {best_action}, best_rew: {best_rew}")

    save(all_obs[best_action][0], f"pred_{ctr}_0")
    save(all_obs[best_action][1], f"pred_{ctr}_1")

    return best_action[0]

def rollout_trajectory(sampler, rew_end_model, obs, actions, rollout):
    obs = [x for x in obs]
    actions = [x for x in actions]

    assert len(obs) == len(actions) + 1

    obs = torch.cat(obs, dim=0).unsqueeze(0).repeat((len(rollout), 1, 1, 1, 1))
    actions = torch.tensor(actions, device=device, dtype=torch.long).unsqueeze(0).repeat(len(rollout), 1) 
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
        rew_ = rew_[:, -1, :].argmax(dim=-1) - 1
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

    return obs, actions, rew, end

def rand_action_tensor(shape):
    n = 1
    for x in shape:
        n *= x
    return torch.tensor([random.choice(available_sample_actions) for _ in range(n)], dtype=torch.long, device=device).reshape(shape)

def sample_trajectory(sampler, rew_end_model, obs, actions, trajectory_length):
    assert obs.shape[0] == actions.shape[0]
    assert obs.shape[1] == actions.shape[1]+1

    rew = None
    end = None

    for _ in tqdm.tqdm(range(trajectory_length)):
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

        rew_ = rew_[:, -1, :].argmax(dim=-1, keepdim=True) - 1
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

    assert rew.shape[0] == obs.shape[0]
    assert rew.shape[1] == trajectory_length
    assert end.shape[1] == trajectory_length

    return obs, actions, rew, end


if __name__ == "__main__":
    main()