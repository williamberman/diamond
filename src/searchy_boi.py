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

    state, _ = test_env.reset()
    obs = [state]
    actions = []

    ctr = 0

    while True:
        print(ctr)
        action = choose_action(sampler, rew_end_model, obs, actions)
        actions.append(action)
        state, _, end, trunc, _ = test_env.step(torch.tensor([action]))
        obs.append(state)

        if end or trunc:
            break

        ctr += 1

    obs = torch.cat(obs, dim=0).permute(0, 2, 3, 1).mul(0.5).add(0.5).mul(255).clamp(0, 255).byte().cpu().numpy()
    obs = [Image.fromarray(x).resize((256, 256)) for x in obs]
    imageio.mimsave("/workspace/obs.mp4", obs, fps=30)
    print("saved")

    # print(agent.world_model.encoder.forward(torch.randn(1, 4, 84, 84)))

def choose_action(sampler, rew_end_model, obs, actions):
    assert len(obs) == len(actions) + 1

    if len(obs) < num_steps_conditioning:
        return random.randint(0, num_actions - 1)

    all_rew = []
    all_end = []

    pbar = tqdm.tqdm(total=num_actions*5)

    for act in range(num_actions):
        all_rew.append([])
        all_end.append([])

        for _ in range(5):
            rew, end = sample_trajectory(sampler, rew_end_model, obs, actions + [act], 10)
            all_rew[-1].append(rew)
            all_end[-1].append(end)
            pbar.update(1)

    avg_rews = [sum(rew)/len(rew) for rew in all_rew]
    ends = [any(end) for end in all_end]

    pbar.close()

    best_rew = -float("inf")
    best_action = None

    for action, avg_rew, end in zip(range(num_actions), avg_rews, ends):
        if avg_rew > best_rew and not end:
            best_rew = avg_rew
            best_action = action

    if best_action is None:
        best_action = random.randint(0, num_actions - 1)

    print(f"search_results: {[x for x in zip(avg_rews, ends)]}, best_action: {best_action}, best_rew: {best_rew}")

    return best_action

def sample_trajectory(sampler, rew_end_model, obs, actions, trajectory_length):
    obs = [x for x in obs]
    actions = [x for x in actions]

    assert len(obs) == len(actions)

    rew = []
    end = []

    for _ in range(trajectory_length):
        if len(obs) == len(actions) + 1:
            actions.append(random.randint(0, num_actions - 1))
        elif len(obs) == len(actions):
            pass
        else:
            assert False

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

    rew = sum(rew)
    end = any(end)

    return rew, end


if __name__ == "__main__":
    main()