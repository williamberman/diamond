import modal
import train_action_labeler
import play
import os

image = (
    modal.Image.debian_slim()
    .apt_install(["ffmpeg", "libsm6", "libxext6", "git", "build-essential"])
    .pip_install([
        "gymnasium[atari,accept-rom-license]==0.29.1",
        "huggingface-hub",
        "hydra-core==1.3",
        "numpy==1.26.0",
        "opencv-python==4.8.0.76",
        "pillow==10.3.0",
        "pygame==2.5.2",
        "torcheval",
        "tqdm==4.66.4",
        "wandb==0.17.0",
        "ipdb",
        "pandas",
        "scikit-learn",
        "git+https://github.com/google-research/rliable",
        "transformers==4.44.2",
        "einops>=0.8.0",
        "einx>=0.3.0",
        "torch",
    ])
    .pip_install(["torchvision"])
    .pip_install(["awscli"])
)

app = modal.App("diamond-runner", image=image)

with image.imports():
    ...

@app.cls(
    gpu=modal.gpu.A10G(),
    volumes={},
    mounts=[
        modal.Mount.from_local_dir("/home/ec2-user/diamond/config", remote_path="/config"),
        modal.Mount.from_local_dir("/home/ec2-user/diamond/config", remote_path="/root/config"),
    ],
    timeout=60*60*24,
    secrets=[
        modal.Secret.from_name("aws-diamond", required_keys=["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]),
        modal.Secret.from_name("wandb")
    ]
)
class Model:
    @modal.method()
    def check(self):
        print('***************')
        os.system("cat /proc/meminfo")
        print('***************')
        os.system("ls -la /mnt/raid/diamond/")
        print('***************')
        os.system("nvidia-smi")
        print('***************')
        os.system("pwd")
        print('***************')
        os.system("ls -la")
        print('***************')
        os.system("mkdir -p /mnt/raid/diamond/foo")
        os.system("echo hey > /mnt/raid/diamond/foo/bar.txt")
        os.system("cat /mnt/raid/diamond/foo/bar.txt")
        print('***************')

    @modal.method()
    def collect_recordings(self, game):
        os.makedirs("/mnt/raid/diamond/tiny", exist_ok=True)
        parser = play.parser()
        args = parser.parse_args([
            "--pretrained",
            "--record",
            "--recording-dir", f"/mnt/raid/diamond/tiny/{game}_recordings_100k",
            "--game", game,
            "--headless-collect-n-steps", "100000",
            "--eps", "0.1",
            "--store-final-obs",
            "--device", "cuda:0",
        ])
        play.main(args)
        os.system("aws s3 sync /mnt/raid/diamond/tiny/ s3://shvaibackups/diamond/tiny/")

    @modal.method()
    def train_action_labeler(self, game):
        os.makedirs("/mnt/raid/diamond/tiny", exist_ok=True)
        os.system("aws s3 sync s3://shvaibackups/diamond/tiny/ /mnt/raid/diamond/tiny/")
        parser = train_action_labeler.parser_()
        args = parser.parse_args([
            "--epochs", "600",
            "--data_dir", f"/mnt/raid/diamond/tiny/{game}_recordings_100k/",
            "--checkpoint_dir", f"/mnt/raid/diamond/tiny/{game}_action_labelers_100k_1000",
            "--train_size", "0.01",
            "--eval_every_n_epochs", "100",
            "--lr", "1e-6",
            "--gpu", "0",
            "--batch_size", "64",
            "--game", game,
            "--write_new_dataset_dir", f"/mnt/raid/diamond/tiny/{game}_recordings_100k_labeled_1000",
        ])
        train_action_labeler.main(args)
        os.system("aws s3 sync /mnt/raid/diamond/tiny/ s3://shvaibackups/diamond/tiny/")

    @modal.method()
    def train_denoiser(self, game):
        os.makedirs("/mnt/raid/diamond/tiny", exist_ok=True)
        os.system("aws s3 sync s3://shvaibackups/diamond/tiny/ /mnt/raid/diamond/tiny/")
        os.system(f"python src/main.py \
            hydra.run.dir=/mnt/raid/diamond/tiny/{game}_100k_labeled_1000_denoiser \
            env.train.id={game}NoFrameskip-v4 \
            wandb.mode=online \
            wandb.name={game}_100k_labeled_1000_denoiser \
            collection.path_to_static_dataset=/mnt/raid/diamond/tiny/{game}_recordings_100k_labeled_1000/ \
            common.device=cuda:0 \
            denoiser.train=True \
            rew_end_model.train=True \
            actor_critic.train=False")
        os.system("aws s3 sync /mnt/raid/diamond/tiny/ s3://shvaibackups/diamond/tiny/")

    @modal.method()
    def train_actor_critic(self, game):
        os.makedirs("/mnt/raid/diamond/tiny", exist_ok=True)
        os.system("aws s3 sync s3://shvaibackups/diamond/tiny/ /mnt/raid/diamond/tiny/")
        os.system(f"python src/main.py \
            hydra.run.dir=/mnt/raid/diamond/tiny/{game}_100k_labeled_1000_actor_critic \
            env.train.id={game}NoFrameskip-v4 \
            wandb.mode=online \
            wandb.name={game}_100k_labeled_1000_actor_critic \
            collection.path_to_static_dataset=/mnt/raid/diamond/tiny/{game}_recordings_100k_labeled_1000/ \
            common.device=cuda:0 \
            denoiser.train=False \
            rew_end_model.train=False \
            actor_critic.train=True \
            initialization.path_to_ckpt=/mnt/raid/diamond/tiny/{game}_100k_labeled_1000_denoiser/checkpoints/agent_versions/agent_epoch_01000.pt")
        os.system("aws s3 sync /mnt/raid/diamond/tiny/ s3://shvaibackups/diamond/tiny/")

@app.local_entrypoint()
def main():
    # Model().check.remote()

    game = "Amidar"

    Model().collect_recordings.remote(game)
    Model().train_action_labeler.remote(game)
    # Model().train_denoiser.remote(game)
    # Model().train_actor_critic.remote(game)
