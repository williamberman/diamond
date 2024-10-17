import argparse
from pathlib import Path
from typing import Tuple
import concurrent.futures
import json

from huggingface_hub import hf_hub_download
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader

from agent import Agent
from coroutines.collector import make_collector, NumToCollect
from data import BatchSampler, collate_segments_to_batch, Dataset
from envs import make_atari_env, WorldModelEnv
from game import ActionNames, DatasetEnv, Game, get_keymap_and_action_names, Keymap, NamedEnv, PlayEnv
from utils import ATARI_100K_GAMES, get_path_agent_ckpt, prompt_atari_game
import tqdm
import math
import os
import random

# python src/play.py --pretrained --record --recording-dir ./test_recording --default-env test --game 7 --headless-collect-n-episodes 100
# cd src
"""
from data import Dataset, SegmentId

dataset = Dataset("../test_recording/0/", name="name", save_on_disk=True, use_manager=False)
dataset.load_from_default_path()
dataset[SegmentId(episode_id=0, start=0, stop=10)]
"""

OmegaConf.register_new_resolver("eval", eval)


def download(filename: str) -> Path:
    path = hf_hub_download(repo_id="eloialonso/diamond", filename=filename)
    return Path(path)


def parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pretrained", action="store_true", help="Download pretrained world model and agent.")
    parser.add_argument("-d", "--dataset-mode", action="store_true", help="Dataset visualization mode.")
    parser.add_argument("-r", "--record", action="store_true", help="Record episodes in PlayEnv.")
    parser.add_argument("-n", "--num-steps-initial-collect", type=int, default=1000, help="Num steps initial collect.")
    parser.add_argument("--store-denoising-trajectory", action="store_true", help="Save denoising steps in info.")
    parser.add_argument("--store-original-obs", action="store_true", help="Save original obs (pre resizing) in info.")
    parser.add_argument("--store-final-obs", action="store_true", help="Save final obs in info.")
    parser.add_argument("--fps", type=int, default=15, help="Frame rate.")
    parser.add_argument("--size", type=int, default=640, help="Window size.")
    parser.add_argument("--no-header", action="store_true")
    parser.add_argument("--recording-dir", type=str, default=None, help="Directory to store recordings.")
    parser.add_argument("--default-env", choices=["wm", "test", "train"], default="wm", help="Default environment.")
    parser.add_argument("--game", type=str, default=None, help="Game to play.")
    parser.add_argument("--headless-collect-n-episodes", type=int, default=None, help="Number of episodes to collect in headless mode.")
    parser.add_argument("--headless-collect-n-steps", type=int, default=None, help="Number of stepsto collect in headless mode.")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to use.")
    parser.add_argument("--path-ckpt", type=str, default=None, help="Path to the checkpoint.")
    parser.add_argument("--horizon", type=int, default=50, help="Horizon for the world model environment.")
    parser.add_argument("--write-rewards", type=str, default=None, help="Path to write rewards.")
    parser.add_argument("--headless-collect-n-threads", type=int, default=20, help="Number of threads to collect in headless mode.")
    parser.add_argument("--eps", type=str, default=None, help="Epsilon for random action selection.")
    parser.add_argument("--eps-min", type=float, default=None, help="Minimum epsilon for dynamic eps.")
    parser.add_argument("--eps-max", type=float, default=None, help="Maximum epsilon for dynamic eps.")
    parser.add_argument("--eps-length", type=int, default=None, help="Length of epsilon decay.")
    parser.add_argument("--only-reset-after-n-deaths", type=int, default=None, help="Only reset after n deaths.")
    parser.add_argument("--has-negative-rewards", type=int, required=True)
    parser.add_argument("--headless-collect-until-min-things", action="store_true", help="Collect until minimum actions and rewards in headless mode.")
    parser.add_argument("--headless-collect-until-min-actions", type=int, default=None, help="Minimum number of actions to collect in headless mode.")
    parser.add_argument("--headless-collect-until-min-rewards", type=int, default=None, help="Minimum number of rewards to collect in headless mode.")
    return parser


def check_args(args: argparse.Namespace) -> None:
    if args.dataset_mode:
        if not Path("dataset").is_dir():
            print(f"Error: {str(Path('dataset').absolute())} not found, cannot use dataset mode.")
            return False
        if Path(".git").is_dir():
            print("Error: cannot run dataset mode the root of the repository.")
            return False
        if args.pretrained or args.record:
            print("Warning: dataset mode, ignoring --pretrained and --record")
    else:
        if not args.record and (args.store_denoising_trajectory or args.store_original_obs):
            print("Warning: not in recording mode, ignoring --store* options")
    
    if args.headless_collect_n_episodes is not None or args.headless_collect_n_steps is not None:
        args.default_env = "train"

    if args.eps is not None:
        if args.eps == "dynamic":
            assert args.eps_min is not None and args.eps_max is not None and args.eps_length is not None
        else:
            args.eps = float(args.eps)

    assert args.has_negative_rewards in [0, 1], "Invalid value for --has-negative-rewards. Must be 0 or 1."
    args.has_negative_rewards = bool(args.has_negative_rewards)
    return True


def prepare_dataset_mode(cfg: DictConfig) -> Tuple[DatasetEnv, Keymap, ActionNames]:
    datasets = []
    for p in Path("dataset").iterdir():
        if p.is_dir():
            d = Dataset(p, p.stem)
            d.load_from_default_path()
            datasets.append(d)
    _, env_action_names = get_keymap_and_action_names(cfg.env.keymap)
    dataset_env = DatasetEnv(datasets, env_action_names)
    keymap, _ = get_keymap_and_action_names("dataset_mode")
    return dataset_env, keymap

def get_game_name(game):
    try:
        name = ATARI_100K_GAMES[int(game)]
    except ValueError:
        assert game in ATARI_100K_GAMES, f"Game {game} not found in Atari 100K games."
        name = game
    return name

def prepare_play_mode(cfg: DictConfig, args: argparse.Namespace, thread_id=None) -> Tuple[PlayEnv, Keymap, ActionNames]:
    # Checkpoint
    if args.pretrained:
        if args.game is None:
            name = prompt_atari_game()
        else:
            name = get_game_name(args.game)

        path_ckpt = download(f"atari_100k/models/{name}.pt")
        # Override config
        cfg.agent = agent_cfg
        cfg.env = env_cfg
        cfg.env.train.id = cfg.env.test.id = f"{name}NoFrameskip-v4"
        cfg.world_model_env.horizon = args.horizon
    elif args.path_ckpt is not None:
        path_ckpt = Path(args.path_ckpt)

        assert args.game is not None, "Must provide game name when using --path-ckpt"
        name = get_game_name(args.game)
        cfg.env.train.id = cfg.env.test.id = f"{name}NoFrameskip-v4"
        cfg.world_model_env.horizon = args.horizon
    else:
        path_ckpt = get_path_agent_ckpt("checkpoints", epoch=-1)

    device = torch.device(args.device)

    # Real envs
    train_env = make_atari_env(num_envs=1, device=device, **cfg.env.train)
    test_env = make_atari_env(num_envs=1, device=device, **cfg.env.test)

    # Models
    agent = Agent(instantiate(cfg.agent, num_actions=test_env.num_actions)).to(device).eval()
    agent.load(path_ckpt)

    # Collect for imagination's initialization
    n = args.num_steps_initial_collect
    if thread_id is not None:
        dataset_dir = Path(f"dataset/{path_ckpt.stem}_{n}_{thread_id}")
    else:
        dataset_dir = Path(f"dataset/{path_ckpt.stem}_{n}")
    dataset = Dataset(dataset_dir)
    dataset.load_from_default_path()
    if len(dataset) == 0:
        print(f"Collecting {n} steps in real environment for world model initialization.")
        collector = make_collector(test_env, agent.actor_critic, dataset, epsilon=0)
        collector.send(NumToCollect(steps=n))
        dataset.save_to_default_path()

    # World model environment
    bs = BatchSampler(dataset, 1, cfg.agent.denoiser.inner_model.num_steps_conditioning, None, False)
    dl = DataLoader(dataset, batch_sampler=bs, collate_fn=collate_segments_to_batch)
    wm_env_cfg = instantiate(cfg.world_model_env, num_batches_to_preload=1)
    wm_env = WorldModelEnv(agent.denoiser, agent.rew_end_model, dl, wm_env_cfg, return_denoising_trajectory=True)

    envs_ = {
        "wm": NamedEnv("wm", wm_env),
        "test": NamedEnv("test", test_env),
        "train": NamedEnv("train", train_env),
    }
    envs = [envs_.pop(args.default_env)]
    envs.extend(envs_.values())

    env_keymap, env_action_names = get_keymap_and_action_names(cfg.env.keymap)

    if args.recording_dir is not None:
        if thread_id is not None:
            recording_dir = os.path.join(args.recording_dir, str(thread_id))
        else:
            recording_dir = args.recording_dir
    else:
        recording_dir = None

    play_env = PlayEnv(
        agent,
        envs,
        env_action_names,
        env_keymap,
        args.record,
        args.store_denoising_trajectory,
        args.store_original_obs,
        args.store_final_obs,
        recording_dir,
    )

    return play_env, env_keymap

def stupid_preproc(cfg: DictConfig, args: argparse.Namespace):
    if args.pretrained:
        if args.game is None:
            name = prompt_atari_game()
        else:
            name = get_game_name(args.game)

        path_ckpt = download(f"atari_100k/models/{name}.pt")
        # Override config
        cfg.agent = agent_cfg
        cfg.env = env_cfg
        cfg.env.train.id = cfg.env.test.id = f"{name}NoFrameskip-v4"
        cfg.world_model_env.horizon = args.horizon
    elif args.path_ckpt is not None:
        path_ckpt = Path(args.path_ckpt)

        assert args.game is not None, "Must provide game name when using --path-ckpt"
        name = get_game_name(args.game)
        cfg.env.train.id = cfg.env.test.id = f"{name}NoFrameskip-v4"
        cfg.world_model_env.horizon = args.horizon
    else:
        path_ckpt = get_path_agent_ckpt("checkpoints", epoch=-1)

    return cfg

def get_num_actions(cfg: DictConfig):
    stupid_preproc(cfg, args)
    device = torch.device(args.device)
    test_env = make_atari_env(num_envs=1, device=device, **cfg.env.test)
    num_actions = test_env.num_actions
    return num_actions

@torch.no_grad()
def main(args):
    ok = check_args(args)
    if not ok:
        return

    global agent_cfg, env_cfg

    agent_cfg = OmegaConf.load(download("atari_100k/config/agent/default.yaml"))
    env_cfg = OmegaConf.load(download("atari_100k/config/env/atari.yaml"))

    with initialize(version_base="1.3", config_path="../config"):
        cfg = compose(config_name="trainer")

    if args.headless_collect_n_episodes is not None or args.headless_collect_n_steps is not None or args.headless_collect_until_min_things is not None:
        if args.headless_collect_n_episodes is not None:
            print(f"Collecting {args.headless_collect_n_episodes} episodes in headless mode.")
        elif args.headless_collect_n_steps is not None:
            print(f"Collecting {args.headless_collect_n_steps} steps in headless mode.")
        elif args.headless_collect_until_min_things:
            print(f"Collecting until minimum actions: {args.headless_collect_until_min_actions} and rewards: {args.headless_collect_until_min_rewards} in headless mode.")
        else:
            assert False

        assert not args.dataset_mode

        if args.headless_collect_n_episodes is not None:
            pbar = tqdm.tqdm(total=args.headless_collect_n_threads * math.ceil(args.headless_collect_n_episodes / args.headless_collect_n_threads))
        elif args.headless_collect_n_steps is not None:
            pbar = tqdm.tqdm(total=args.headless_collect_n_threads * math.ceil(args.headless_collect_n_steps / args.headless_collect_n_threads))
        elif args.headless_collect_until_min_things is not None:
            pass
        else:
            assert False

        action_counts = {i: 0 for i in range(get_num_actions(cfg))}
        reward_counts = {i: 0 for i in ([-1, 0, 1] if args.has_negative_rewards else [0, 1])}

        def do(thread_id):
            env, _ = prepare_play_mode(cfg, args, thread_id)

            assert not env.is_human_player

            rewards = []
            episode_ctr = 0
            step_ctr = 0

            while True:
                env.reset()
                rewards.append(0)
                info = None
                    
                doing_no_eps = args.eps == "dynamic" and (random.random() < 0.1 or step_ctr == 0)
                if doing_no_eps:
                    eps = 0.0

                while True:
                    if args.eps == "dynamic" and step_ctr % args.eps_length == 0 and not doing_no_eps:
                        eps = random.uniform(args.eps_min, args.eps_max)

                    on_last_life = info is not None and info["lives"].item() == 1

                    _, rew, end, trunc, info = env.step(0, eps=eps, on_last_life=on_last_life)
                    rewards[-1] += rew.item()
                    step_ctr += 1

                    if rew.item() < 0:
                        rew_item = -1
                    elif rew.item() > 0:
                        rew_item = 1
                    else:
                        rew_item = 0

                    reward_counts[rew_item] += 1
                    action_counts[env.last_action] += 1

                    if args.headless_collect_n_steps is not None:
                        pbar.update(1)

                    if on_last_life and (end or trunc):
                        break

                    if args.headless_collect_n_steps is not None:
                        if step_ctr >= math.ceil(args.headless_collect_n_steps / args.headless_collect_n_threads):
                            env.add_cur_episode_to_dataset()
                            break

                    if args.headless_collect_until_min_things is not None:
                        if thread_id == 0 and step_ctr % 100 == 0:
                            print(f"{step_ctr}: {action_counts} {reward_counts}")

                        if all([x > args.headless_collect_until_min_actions for x in action_counts.values()]) and all([x > args.headless_collect_until_min_rewards for x in reward_counts.values()]):
                            env.add_cur_episode_to_dataset()
                            break

                print(f"{thread_id}: Episode {episode_ctr} step {step_ctr} reward: {rewards[-1]} average reward: {sum(rewards) / len(rewards):.2f}")

                if args.headless_collect_n_episodes is not None:
                    pbar.update(1)

                episode_ctr += 1
                
                if args.headless_collect_n_episodes is not None:
                    if episode_ctr >= math.ceil(args.headless_collect_n_episodes / args.headless_collect_n_threads):
                        break
                elif args.headless_collect_n_steps is not None:
                    print(f"{thread_id}: step_ctr: {step_ctr} >= {math.ceil(args.headless_collect_n_steps / args.headless_collect_n_threads)}")
                    if step_ctr >= math.ceil(args.headless_collect_n_steps / args.headless_collect_n_threads):
                        break
                elif args.headless_collect_until_min_things is not None:
                    if all([x > args.headless_collect_until_min_actions for x in action_counts.values()]) and all([x > args.headless_collect_until_min_rewards for x in reward_counts.values()]):
                        break
                else:
                    assert False

            return rewards

        all_rewards = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.headless_collect_n_threads) as executor:
            futures = [executor.submit(do, i) for i in range(args.headless_collect_n_threads)]
            for future in concurrent.futures.as_completed(futures):
                all_rewards.extend(future.result())

        print(f"Collected {len(all_rewards)} episodes with average reward: {sum(all_rewards) / len(all_rewards):.2f}")

        if args.write_rewards is not None:
            os.makedirs(os.path.dirname(args.write_rewards), exist_ok=True)
            with open(args.write_rewards, "w") as f:
                json.dump({args.game: all_rewards}, f, indent=4)
    else:
        env, keymap = prepare_dataset_mode(cfg) if args.dataset_mode else prepare_play_mode(cfg, args)
        size = (args.size // cfg.env.train.size) * cfg.env.train.size  # window size
        game = Game(env, keymap, (size, size), fps=args.fps, verbose=not args.no_header, only_reset_after_n_deaths=args.only_reset_after_n_deaths)
        game.run()


if __name__ == "__main__":
    parser = parser()
    args = parser.parse_args()
    main(args)
