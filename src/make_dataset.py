import torch
from pathlib import Path
from omegaconf import DictConfig

from data import Dataset
from envs import make_atari_env
from agent import Agent
from coroutines.collector import make_collector, NumToCollect
from hydra.utils import instantiate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_train_dataset(cfg: DictConfig, output_dir: Path, name) -> Dataset:
    # Set up the environment
    env = make_atari_env(num_envs=cfg.collection.train.num_envs, device=device, **cfg.env.train)

    agent = Agent(instantiate(cfg.agent, num_actions=env.num_actions)).to(device)
    
    # Create the dataset
    dataset = Dataset(output_dir, name=name, save_on_disk=True, use_manager=False)

    # Create the collector
    collector = make_collector(
        env=env,
        model=agent.actor_critic,
        dataset=dataset,
        epsilon=cfg.collection.train.epsilon,
        reset_every_collect=True,
        verbose=True
    )
    
    collector.send(NumToCollect(steps=100))
    num_steps = dataset.num_steps
    print(f"Collected {num_steps} steps")

    dataset.is_static = True

    # Save the dataset
    dataset.save_to_default_path()
    
    return dataset

# Usage example (you would typically call this from your main training script)
if __name__ == "__main__":
    from hydra import initialize, compose
    
    with initialize(config_path="../config"):
        cfg = compose(config_name="trainer")
    
    output_dir = Path("make_dataset_output") / "train"
    dataset = create_train_dataset(cfg, output_dir=output_dir, name="train_dataset")
    
    print(f"Created dataset with {len(dataset)} steps")

    # do a test dataset
    # Create test dataset
    test_output_dir = Path("make_dataset_output") / "test"
    test_dataset = create_train_dataset(cfg, output_dir=test_output_dir, name="test_dataset")
    
    print(f"Created test dataset with {len(test_dataset)} steps")

    output_dir = Path("make_dataset_output") / "train"
    foo = Dataset(output_dir, "train_dataset", cfg.training.cache_in_ram, use_manager=False)
    foo.load_from_default_path()
    import ipdb; ipdb.set_trace()

