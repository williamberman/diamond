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

device = 2

@hydra.main(config_path="../config", config_name="trainer", version_base="1.3")
def main(cfg: DictConfig):
    global num_actions, num_steps_conditioning

    ckpt = "/mnt/raid/diamond/better4/Breakout_100k_labeled_1000_actor_critic_cont/checkpoints/agent_versions/agent_epoch_01000.pt"

    test_env = make_atari_env(num_envs=cfg.collection.test.num_envs, device=device, **cfg.env.test)
    num_actions = int(test_env.num_actions)

    agent = Agent(instantiate(cfg.agent, num_actions=num_actions)).to(device)
    agent.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")))
    agent.eval()

    sampler = DiffusionSampler(agent.denoiser, cfg.world_model_env.diffusion_sampler)

    num_steps_conditioning = sampler.denoiser.inner_model.cfg.num_steps_conditioning
    rew_end_model = agent.rew_end_model

    state, info = test_env.reset()
    obs = [state]
    actions = []

    ctr = 0
    lives_1_started = False

    while True:
        print(ctr)

        if ctr == 0 or info['lives'] != 1 or not lives_1_started:
            action = 1

            if info['lives'] == 1 and not lives_1_started:
                print("started playing with 1 life")
                lives_1_started = True
        else:
            action = choose_action(sampler, rew_end_model, obs, actions)

        actions.append(action)
        state, _, end, trunc, info = test_env.step(torch.tensor([action]))
        obs.append(state)

        if end or trunc:
            break

        ctr += 1

        if (ctr+1) % 10 == 0:
            save(obs, ctr+1)

    save(obs, ctr)

def save(obs, idx):
    obs = torch.cat(obs, dim=0).permute(0, 2, 3, 1).mul(0.5).add(0.5).mul(255).clamp(0, 255).byte().cpu().numpy()
    obs = [Image.fromarray(x).resize((256, 256)) for x in obs]
    imageio.mimsave(f"/workspace/obs_{idx}.mp4", obs, fps=5)
    print("saved")

def init_search_actions(n):
    # acts = [[0], [2], [3]]
    acts = [[2], [3]]

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

    init_search_actions_ = init_search_actions(5)
    n_samples = 2

    pbar = tqdm.tqdm(total=len(init_search_actions_)*n_samples)

    for init_act in init_search_actions_:
        obs_, actions_ = rollout_trajectory(sampler, rew_end_model, obs, actions, init_act)
        for _ in range(n_samples):
            rew, end = sample_trajectory(sampler, rew_end_model, obs_, actions_, 20)
            all_rew[tuple(init_act)].append(rew)
            all_end[tuple(init_act)].append(end)
            pbar.update(1)

    avg_rews = {act: sum(rew)/len(rew) for act, rew in all_rew.items()}
    ends = {act: any(end) for act, end in all_end.items()}

    pbar.close()

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

    print(f"avg_rews: {avg_rews}, ends: {ends}, best_action: {best_action}, best_rew: {best_rew}")

    return best_action[0]

def rollout_trajectory(sampler, rew_end_model, obs, actions, rollout):
    obs = [x for x in obs]
    actions = [x for x in actions]

    assert len(obs) == len(actions) + 1

    for action in rollout:
        actions.append(action)
        obs_ = torch.cat(obs[-num_steps_conditioning:], dim=0).unsqueeze(0)
        act_ = torch.tensor(actions[-num_steps_conditioning:], device=device, dtype=torch.long).unsqueeze(0)

        next_obs = sampler.sample_next_obs(obs_, act_)[0]
        obs.append(next_obs)

    return obs, actions

def sample_trajectory(sampler, rew_end_model, obs, actions, trajectory_length):
    obs = [x for x in obs]
    actions = [x for x in actions]

    assert len(obs) == len(actions)+1

    rew = []
    end = []

    for _ in range(trajectory_length):
        actions.append(random.randint(0, num_actions - 1))

        obs_ = torch.cat(obs[-num_steps_conditioning:], dim=0).unsqueeze(0)
        act_ = torch.tensor(actions[-num_steps_conditioning:], device=device, dtype=torch.long).unsqueeze(0)

        next_obs = sampler.sample_next_obs(obs_, act_)[0]
        next_obs_ = torch.cat([obs_, next_obs.unsqueeze(1)], dim=1)
        next_obs_ = next_obs_[:, 1:]

        assert obs_.shape[:2] == next_obs_.shape[:2]
        assert obs_.shape[:2] == act_.shape[:2]

        rew_, end_, _ = rew_end_model(obs_, act_, next_obs_)

        rew_ = rew_[:, -1, :].argmax(dim=-1).item() - 1
        end_ = end_[:, -1, :].argmax(dim=-1).item()

        rew.append(rew_)
        end.append(end_)
        obs.append(next_obs)

    assert len(rew) == trajectory_length
    assert len(end) == trajectory_length

    rew_ = sum(rew)
    end_ = any(end)

    # if rew_ > 0:
    #     import ipdb; ipdb.set_trace()
    #     save(obs[-trajectory_length-5:], "foo")

    return rew_, end_


if __name__ == "__main__":
    main()