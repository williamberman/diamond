from datetime import datetime
from functools import partial
from pathlib import Path
import shutil
import time
from typing import Tuple

import hydra
from hydra.utils import instantiate
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import wandb

from agent import Agent
from coroutines.collector import make_collector, NumToCollect
from data import BatchSampler, collate_segments_to_batch, Dataset, DatasetTraverser
from envs import make_atari_env, WorldModelEnv
from utils import (
    CommonTools,
    configure_opt,
    count_parameters,
    get_lr_sched,
    keep_agent_copies_every,
    Logs,
    process_confusion_matrices_if_any_and_compute_classification_metrics,
    save_info_for_import_script,
    save_with_backup,
    set_seed,
    StateDictMixin,
    try_until_no_except,
    wandb_log,
)
from compute_atari_100k import RANDOM_SCORES, HUMAN_SCORES


class Trainer(StateDictMixin):
    def __init__(self, cfg: DictConfig) -> None:
        torch.backends.cuda.matmul.allow_tf32 = True
        OmegaConf.resolve(cfg)
        self._cfg = cfg

        # Seed
        if cfg.common.seed is None:
            cfg.common.seed = int(datetime.now().timestamp()) % 10**5
        set_seed(cfg.common.seed)

        # Init wandb
        wandb.init(config=OmegaConf.to_container(cfg, resolve=True), reinit=True, resume=True, **cfg.wandb)

        # Flags
        self._is_static_dataset = cfg.collection.path_to_static_dataset is not None
        self._is_model_free = cfg.training.model_free
        self._use_cuda = "cuda" in cfg.common.device

        # Device
        self._device = torch.device(cfg.common.device)
        if self._use_cuda:
            torch.cuda.set_device(self._device)  # fix compilation error on multi-gpu nodes

        # Checkpointing
        self._path_ckpt_dir = Path("checkpoints")
        self._path_state_ckpt = self._path_ckpt_dir / "state.pt"
        self._keep_agent_copies = partial(
            keep_agent_copies_every,
            every=cfg.checkpointing.save_agent_every,
            path_ckpt_dir=self._path_ckpt_dir,
            num_to_keep=cfg.checkpointing.num_to_keep,
        )
        self._save_info_for_import_script = partial(
            save_info_for_import_script, run_name=cfg.wandb.name, path_ckpt_dir=self._path_ckpt_dir
        )

        # First time, init files hierarchy
        if not cfg.common.resume:
            self._path_ckpt_dir.mkdir(exist_ok=False, parents=False)
            path_config = Path("config") / "trainer.yaml"
            path_config.parent.mkdir(exist_ok=False, parents=False)
            shutil.move(".hydra/config.yaml", path_config)
            wandb.save(str(path_config))
            shutil.copytree(src=(Path(hydra.utils.get_original_cwd()) / "src"), dst="./src")
            shutil.copytree(src=(Path(hydra.utils.get_original_cwd()) / "scripts"), dst="./scripts")

        # Datasets
        num_workers = cfg.training.num_workers_data_loaders
        use_manager = cfg.training.cache_in_ram and (num_workers > 0)

        if self._is_static_dataset:
            self.train_dataset = Dataset(Path(cfg.collection.path_to_static_dataset), "train_dataset", cfg.training.cache_in_ram, use_manager)
        else:
            self.train_dataset = Dataset(Path("dataset/test"), "train_dataset", cfg.training.cache_in_ram, use_manager)

        self.test_dataset = Dataset(Path("dataset/test"), "test_dataset", cache_in_ram=True)
        self.train_dataset.load_from_default_path()
        self.test_dataset.load_from_default_path()

        # Envs
        train_env = make_atari_env(num_envs=cfg.collection.train.num_envs, device=self._device, **cfg.env.train)
        test_env = make_atari_env(num_envs=cfg.collection.test.num_envs, device=self._device, **cfg.env.test)

        # Create models
        num_actions = int(test_env.num_actions)
        self.agent = Agent(instantiate(cfg.agent, num_actions=num_actions)).to(self._device)
        self.agent.denoiser.compute_loss = torch.compile(self.agent.denoiser.compute_loss, mode="reduce-overhead")

        if cfg.initialization.path_to_ckpt is not None:
            self.agent.load(**cfg.initialization)

        if cfg.collection.train.action_labeler is not None:
            cfg.collection.train.action_labeler.num_classes = num_actions
            action_labeler = instantiate(cfg.collection.train.action_labeler).eval().requires_grad_(False).to(self._device)
            action_labeler.load_state_dict(torch.load(cfg.collection.train.action_labeler_checkpoint))
            print(f"Action labeler loaded from {cfg.collection.train.action_labeler_checkpoint}")
        else:
            print("No action labeler, using real action labels")
            action_labeler = None

        # Collectors
        self._train_collector = make_collector(
            train_env, self.agent.actor_critic, self.train_dataset, cfg.collection.train.epsilon, action_labeler=action_labeler
        )
        self._test_collector = make_collector(
            test_env, self.agent.actor_critic, self.test_dataset, cfg.collection.test.epsilon, reset_every_collect=True
        )

        ######################################################

        # Optimizers and LR schedulers

        def build_opt(name: str) -> torch.optim.AdamW:
            return configure_opt(getattr(self.agent, name), **getattr(cfg, name).optimizer)

        def build_lr_sched(name: str) -> torch.optim.lr_scheduler.LambdaLR:
            return get_lr_sched(self.opt.get(name), getattr(cfg, name).training.lr_warmup_steps)

        self._model_names = ["denoiser", "rew_end_model", "actor_critic"]
        self.opt = CommonTools(*map(build_opt, self._model_names))
        self.lr_sched = CommonTools(*map(build_lr_sched, self._model_names))

        # Data loaders

        make_data_loader = partial(
            DataLoader,
            dataset=self.train_dataset,
            collate_fn=collate_segments_to_batch,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
            pin_memory=self._use_cuda,
            pin_memory_device=str(self._device) if self._use_cuda else "",
        )

        c = cfg.denoiser.training
        seq_length = cfg.agent.denoiser.inner_model.num_steps_conditioning + 1 + c.num_autoregressive_steps
        bs = BatchSampler(self.train_dataset, c.batch_size, seq_length, c.sample_weights)
        dl_denoiser_train = make_data_loader(batch_sampler=bs)
        dl_denoiser_test = DatasetTraverser(self.test_dataset, c.batch_size, seq_length)

        c = cfg.rew_end_model.training
        bs = BatchSampler(self.train_dataset, c.batch_size, c.seq_length, c.sample_weights, can_sample_beyond_end=True)
        dl_rew_end_model_train = make_data_loader(batch_sampler=bs)
        dl_rew_end_model_test = DatasetTraverser(self.test_dataset, c.batch_size, c.seq_length)

        self._data_loader_train = CommonTools(dl_denoiser_train, dl_rew_end_model_train, None)
        self._data_loader_test = CommonTools(dl_denoiser_test, dl_rew_end_model_test, None)

        # RL env

        if self._is_model_free:
            rl_env = make_atari_env(num_envs=cfg.actor_critic.training.batch_size, device=self._device, **cfg.env.train)

        else:
            c = cfg.actor_critic.training
            sl = cfg.agent.denoiser.inner_model.num_steps_conditioning
            bs = BatchSampler(self.train_dataset, c.batch_size, sl, c.sample_weights)
            dl_actor_critic = make_data_loader(batch_sampler=bs)
            wm_env_cfg = instantiate(cfg.world_model_env)
            rl_env = WorldModelEnv(self.agent.denoiser, self.agent.rew_end_model, dl_actor_critic, wm_env_cfg)

            if cfg.training.compile_wm:
                rl_env.predict_next_obs = torch.compile(rl_env.predict_next_obs, mode="reduce-overhead")
                rl_env.predict_rew_end = torch.compile(rl_env.predict_rew_end, mode="reduce-overhead")

        # Setup training
        sigma_distribution_cfg = instantiate(cfg.denoiser.sigma_distribution)
        actor_critic_loss_cfg = instantiate(cfg.actor_critic.actor_critic_loss)
        self.agent.setup_training(sigma_distribution_cfg, actor_critic_loss_cfg, rl_env)

        # Training state (things to be saved/restored)
        self.epoch = 0
        self.num_epochs_collect = None
        self.num_episodes_test = 0
        self.num_batch_train = CommonTools(0, 0, 0)
        self.num_batch_test = CommonTools(0, 0, 0)

        if cfg.common.resume:
            self.load_state_checkpoint()
        else:
            self.save_checkpoint()

        for name in self._model_names:
            print(f"{count_parameters(getattr(self.agent, name))} parameters in {name}")
        print(self.train_dataset)
        print(self.test_dataset)

    def run(self) -> None:
        to_log = []

        if hasattr(self._cfg, "only_run_validation") and self._cfg.only_run_validation:
            to_log += self.collect_test()
            if not self._is_model_free:
                to_log += self.test_agent()
            wandb_log(to_log, 0)
            return

        if self.epoch == 0:
            if self._is_model_free or self._is_static_dataset:
                self.num_epochs_collect = 0
            else:
                self.num_epochs_collect, to_log_ = self.collect_initial_dataset()
                to_log += to_log_

        num_epochs = self.num_epochs_collect + self._cfg.training.num_final_epochs

        while self.epoch < num_epochs:
            self.epoch += 1

            print(f"\nEpoch {self.epoch} / {num_epochs}\n")
            start_time = time.time()

            # Training
            should_collect_train = (
                not self._is_model_free and not self._is_static_dataset and self.epoch <= self.num_epochs_collect
            )

            if should_collect_train:
                c = self._cfg.collection.train
                to_log += self._train_collector.send(NumToCollect(steps=c.steps_per_epoch))

            if self._cfg.training.should:
                to_log += self.train_agent()

            # Evaluation
            should_test = self._cfg.evaluation.should and (self.epoch % self._cfg.evaluation.every == 0)

            if should_test:
                to_log += self.collect_test()

            if should_test and not self._is_model_free:
                to_log += self.test_agent()

            # Logging
            to_log.append({"duration": (time.time() - start_time) / 3600})
            wandb_log(to_log, self.epoch)
            to_log = []

            # Checkpointing
            self.save_checkpoint()

        # Last collect
        if not self._is_static_dataset:
            wandb_log(self.collect_test(final=True), self.epoch)

    def collect_initial_dataset(self) -> Tuple[int, Logs]:
        print("\nInitial collect\n")
        to_log = []
        c = self._cfg.collection.train
        min_steps = c.first_epoch.min
        steps_per_epoch = c.steps_per_epoch
        max_steps = c.first_epoch.max
        threshold_rew = c.first_epoch.threshold_rew
        assert min_steps % steps_per_epoch == 0

        steps = min_steps
        while True:
            to_log += self._train_collector.send(NumToCollect(steps=steps))
            num_steps = self.train_dataset.num_steps
            total_minority_rew = sum(sorted(self.train_dataset.counts_rew)[:-1])
            if total_minority_rew >= threshold_rew:
                break
            if (max_steps is not None) and num_steps >= max_steps:
                print("Reached the specified maximum for initial collect")
                break
            print(f"Minority reward: {total_minority_rew}/{threshold_rew} -> Keep collecting\n")
            steps = steps_per_epoch

        print("\nSummary of initial collect:")
        print(f"Num steps: {num_steps} / {c.num_steps_total}")
        print(f"Reward counts: {dict(self.train_dataset.counter_rew)}")

        remaining_steps = c.num_steps_total - num_steps
        assert remaining_steps % c.steps_per_epoch == 0
        num_epochs_collect = remaining_steps // c.steps_per_epoch

        return num_epochs_collect, to_log

    def collect_test(self, final: bool = False) -> Logs:
        c = self._cfg.collection.test
        episodes = c.num_final_episodes if final else c.num_episodes
        td = self.test_dataset
        td.clear()
        to_log = self._test_collector.send(NumToCollect(episodes=episodes))
        key_ep_id = f"{td.name}/episode_id"
        to_log = [{k: v + self.num_episodes_test if k == key_ep_id else v for k, v in x.items()} for x in to_log]

        print(f"\nSummary of {'final' if final else 'test'} collect: {td.num_episodes} episodes ({td.num_steps} steps)")
        keys = [key_ep_id, "return", "length"]
        to_log_episodes = [x for x in to_log if set(x.keys()) == set(keys)]
        episode_ids, returns, lengths = [[d[k] for d in to_log_episodes] for k in keys]
        for i, (ep_id, ret, length) in enumerate(zip(episode_ids, returns, lengths)):
            print(f"  Episode {ep_id}: return = {ret} length = {length}\n", end="\n" if i == episodes - 1 else "")

        game = self._cfg.env.test.id.replace("NoFrameskip-v4", "")
        mean_return = np.array(returns).mean()
        min_return = np.array(returns).min()
        max_return = np.array(returns).max()
        atari_100k_score_hns = (mean_return - RANDOM_SCORES[game]) / (HUMAN_SCORES[game] - RANDOM_SCORES[game])
        atari_100k_score_hns = atari_100k_score_hns.item()
        print(f"Atari 100k hns score: {atari_100k_score_hns:.2f} mean_return: {mean_return:.2f} min_return: {min_return:.2f} max_return: {max_return:.2f}")
        to_log.append({"atari_100k_score_hns": atari_100k_score_hns, "atari_100k_mean_return": mean_return, "atari_100k_min_return": min_return, "atari_100k_max_return": max_return})

        self.num_episodes_test += episodes

        if final:
            to_log.append({"final_return_mean": np.mean(returns), "final_return_std": np.std(returns)})
            print(to_log[-1])

        return to_log

    def train_agent(self) -> Logs:
        self.agent.train()
        self.agent.zero_grad()
        to_log = []

        model_names = []

        if self._cfg.denoiser.train:
            model_names.append("denoiser")

        if self._cfg.rew_end_model.train:
            model_names.append("rew_end_model")

        if self._cfg.actor_critic.train:
            model_names.append("actor_critic")

        for name in model_names:
            cfg = getattr(self._cfg, name).training
            if self.epoch > cfg.start_after_epochs:
                steps = cfg.steps_first_epoch if self.epoch == 1 else cfg.steps_per_epoch
                to_log += self.train_component(name, steps)
            self.save_checkpoint()
        return to_log

    @torch.no_grad()
    def test_agent(self) -> Logs:
        self.agent.eval()
        to_log = []
        model_names = []

        if self._cfg.denoiser.train:
            model_names.append("denoiser")

        if self._cfg.rew_end_model.train:
            model_names.append("rew_end_model")

        # we do not eval the actor critic here

        for name in model_names:
            cfg = getattr(self._cfg, name).training
            if self.epoch > cfg.start_after_epochs:
                to_log += self.test_component(name)
        return to_log

    def train_component(self, name: str, steps: int) -> Logs:
        cfg = getattr(self._cfg, name).training
        model = getattr(self.agent, name)
        opt = self.opt.get(name)
        lr_sched = self.lr_sched.get(name)
        data_loader = self._data_loader_train.get(name)

        model.train()
        opt.zero_grad()
        data_iterator = iter(data_loader) if data_loader is not None else None
        to_log = []

        num_steps = cfg.grad_acc_steps * steps

        for i in trange(num_steps, desc=f"Training {name}"):
            batch = next(data_iterator).to(self._device) if data_iterator is not None else None
            loss, metrics = model.compute_loss(batch) if batch is not None else model.compute_loss()
            loss.backward()

            num_batch = self.num_batch_train.get(name)
            metrics[f"num_batch_train_{name}"] = num_batch
            self.num_batch_train.set(name, num_batch + 1)

            if (i + 1) % cfg.grad_acc_steps == 0:
                if cfg.max_grad_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    metrics["grad_norm_before_clip"] = grad_norm

                opt.step()
                opt.zero_grad()

                if lr_sched is not None:
                    metrics["lr"] = lr_sched.get_last_lr()[0]
                    lr_sched.step()

            to_log.append(metrics)

        process_confusion_matrices_if_any_and_compute_classification_metrics(to_log)
        to_log = [{f"{name}/train/{k}": v for k, v in d.items()} for d in to_log]
        return to_log

    @torch.no_grad()
    def test_component(self, name: str) -> Logs:
        model = getattr(self.agent, name)
        data_loader = self._data_loader_test.get(name)
        model.eval()
        to_log = []
        for batch in tqdm(data_loader, desc=f"Evaluating {name}"):
            batch = batch.to(self._device)
            _, metrics = model.compute_loss(batch)
            num_batch = self.num_batch_test.get(name)
            metrics[f"num_batch_test_{name}"] = num_batch
            self.num_batch_test.set(name, num_batch + 1)
            to_log.append(metrics)

        process_confusion_matrices_if_any_and_compute_classification_metrics(to_log)
        to_log = [{f"{name}/test/{k}": v for k, v in d.items()} for d in to_log]
        return to_log

    def load_state_checkpoint(self) -> None:
        self.load_state_dict(torch.load(self._path_state_ckpt, map_location=self._device))

    def save_checkpoint(self) -> None:
        save_with_backup(self.state_dict(), self._path_state_ckpt)
        self.train_dataset.save_to_default_path()
        self.test_dataset.save_to_default_path()
        self._keep_agent_copies(self.agent.state_dict(), self.epoch)
        self._save_info_for_import_script(self.epoch)
