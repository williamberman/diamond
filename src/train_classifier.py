import os
from typing import Callable, List, Optional, Type, Union
from torchvision.models.resnet import conv1x1, conv3x3, _log_api_usage_once
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from argparse import ArgumentParser
from torch.utils.data import DataLoader, IterableDataset
import random
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch import Tensor
import torch.optim as optim
import wandb
import tqdm
import numpy as np

class dummy_wandb:
    def log(*args, **kwargs): pass
    def init(*args, **kwargs): pass

device = int(os.environ['LOCAL_RANK'])

def main():
    global args, model, training_set_idx, validation_set_idx, wandb

    dist.init_process_group(backend='nccl')

    args = ArgumentParser()
    args.add_argument('--num_classes', type=int, required=False)
    args.add_argument('--num_input_images', type=int, default=30)
    args.add_argument("--dataset_path", type=str, required=True)
    args.add_argument("--batch_size", type=int, default=512)
    args.add_argument("--validation_batch_size", type=int, default=1024)
    args.add_argument("--validation_steps", type=int, default=1000)
    args.add_argument("--save_steps", type=int, default=1000)
    args.add_argument("--save_dir", type=str, required=True)
    args.add_argument("--train_n_examples", type=int, required=True)
    args.add_argument("--no_wandb", action="store_true")
    args.add_argument("--validation_subset_n", type=int, default=None)
    args.add_argument("--wandb_name", type=str, default=None)
    args.add_argument("--model", type=str, required=True, choices=resnet_configs.keys())
    args = args.parse_args()

    if args.no_wandb:
        wandb = dummy_wandb()

    training_set_idx, validation_set_idx = get_data_splits()

    model = ResNet(
        **resnet_configs[args.model],
        num_classes=args.num_classes,
        input_channels=3*args.num_input_images,
        inplanes=256,
        norm_layer=lambda x: AdaLN(x, args.num_input_images*3-1),
    ).train().requires_grad_(True).to(device)
    model = DDP(model, device_ids=[device])

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    dataloader = iter(DataLoader(TrainingDataset(), batch_size=args.batch_size, num_workers=8))

    if device == 0:
        wandb.init(project="diamond_classifier", name=args.wandb_name)

    step = 0

    if device == 0:
        print("starting training loop")

    while True:
        batch = next(dataloader)
        batch = {k: v.to(device) for k, v in batch.items()}
        pred = model(batch['obs'], batch['cond'])
        loss = F.cross_entropy(pred, batch['target'])
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        dist.barrier()

        if device == 0 and (step+1) % args.save_steps == 0:
            print(f"saving model at step {step}")
            state_dict = dict(model=model.state_dict(), optimizer=optimizer.state_dict())
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save(state_dict, os.path.join(args.save_dir, f"{step+1}.pt"))
            print(f"done saving model at step {step}")

        dist.barrier()

        validation_data = {}

        if (step+1) % args.validation_steps == 0:
            validation_data = validation(subset_n=args.validation_subset_n)
            done = [False]
            if device == 0 and ('validation_loss/avg_prev' in validation_data or 'validation_loss/avg_cur' in validation_data):
                # stop training if we have not improved validation loss by at least .01
                done = [validation_data['validation_loss/avg_prev'] - 0.01 <= validation_data['validation_loss/avg_cur']]

        if device == 0:
            log_args = dict(loss=loss.item(), grad_norm=grad_norm.item(), **validation_data)
            wandb.log(log_args, step=step)
            print(f"{step}: {log_args}")

        step += 1

        dist.broadcast_object_list(done, src=0)
        if done[0]:
            break

    validation_data = validation(subset_n=None)
    if device == 0:
        print(f"final validation data: {validation_data}")
        wandb.log(validation_data, step=step)

def get_data_splits():
    n_shards = 0
    for x in os.listdir(args.dataset_path):
        item_n = int(x.split('_')[0])
        n_shards = max(n_shards, item_n)

    print(f"total n_shards: {n_shards}")

    training_set_idx = {}
    validation_set_idx = {}

    for shard_n in tqdm.tqdm(range(n_shards), desc="making train split"):
        shard_path = os.path.join(args.dataset_path, f"{shard_n}_target.npy")
        n_obs = np.load(shard_path).shape[0]

        total_seen = sum([x['end_idx'] - x['start_idx'] for x in training_set_idx.values()])

        if total_seen + n_obs < args.train_n_examples:
            training_set_idx[shard_n] = dict(start_idx=0, end_idx=n_obs)
        else:
            n_to_add_to_training_set = max(args.train_n_examples - total_seen, 0)
            if n_to_add_to_training_set > 0:
                training_set_idx[shard_n] = dict(start_idx=0, end_idx=n_to_add_to_training_set)
            if n_obs - n_to_add_to_training_set > 0:
                validation_set_idx[shard_n] = dict(start_idx=n_to_add_to_training_set, end_idx=n_obs)

    n_examples_in_training_set = sum([x['end_idx'] - x['start_idx'] for x in training_set_idx.values()])
    n_examples_in_validation_set = sum([x['end_idx'] - x['start_idx'] for x in validation_set_idx.values()])

    print(f"n_examples_in_training_set: {n_examples_in_training_set} n_examples_in_validation_set: {n_examples_in_validation_set}")

    assert n_examples_in_training_set == args.train_n_examples

    return training_set_idx, validation_set_idx

class TrainingDataset(IterableDataset):
    def __iter__(self):
        while True:
            shard_idx = random.choice(list(training_set_idx.keys()))

            obs = torch.from_numpy(np.load(os.path.join(args.dataset_path, f"{shard_idx}_obs.npy")))
            cond = torch.from_numpy(np.load(os.path.join(args.dataset_path, f"{shard_idx}_cond.npy")))
            target = torch.from_numpy(np.load(os.path.join(args.dataset_path, f"{shard_idx}_target.npy")))

            perm = [x for x in range(training_set_idx[shard_idx]['start_idx'], training_set_idx[shard_idx]['end_idx'])]
            random.shuffle(perm)

            for i in perm:
                yield dict(obs=obs[i], cond=cond[i], target=target[i])
    
all_validation_data = []

@torch.no_grad()
def validation(subset_n=None):
    dist.barrier()

    validation_data = None

    pbar = tqdm.tqdm(total=subset_n)

    if device == 0:
        print("running validation")

        validation_loss = []
        correct = 0
        total = 0

        correct_classes = {i: 0 for i in range(args.num_classes)}
        total_classes = {i: 0 for i in range(args.num_classes)}

        validation_model = model.module
        validation_model.eval().requires_grad_(False)

        for shard_idx in validation_set_idx:
            shard_obs = torch.from_numpy(np.load(os.path.join(args.dataset_path, f"{shard_idx}_obs.npy")))
            shard_cond = torch.from_numpy(np.load(os.path.join(args.dataset_path, f"{shard_idx}_cond.npy")))
            shard_target = torch.from_numpy(np.load(os.path.join(args.dataset_path, f"{shard_idx}_target.npy")))

            for start_idx in range(validation_set_idx[shard_idx]['start_idx'], validation_set_idx[shard_idx]['end_idx'], args.validation_batch_size):
                end_idx = min(start_idx + args.validation_batch_size, validation_set_idx[shard_idx]['end_idx'])
                obs = shard_obs[start_idx:end_idx].to(device)
                cond = shard_cond[start_idx:end_idx].to(device)
                target = shard_target[start_idx:end_idx].to(device)

                pred = validation_model(obs, cond)
                loss = F.cross_entropy(pred, target)
                validation_loss.append(loss.item())

                total += target.shape[0]
                correct += (pred.argmax(dim=1) == target).sum().item()

                for i in range(args.num_classes):
                    mask = target == i
                    pred_ = pred[mask]
                    target_ = target[mask]

                    correct_classes[i] += (pred_.argmax(dim=1) == target_).sum().item()
                    total_classes[i] += target_.shape[0]

                pbar.update(target.shape[0])

                if subset_n is not None and total >= subset_n:
                    break

            if subset_n is not None and total >= subset_n:
                break

        validation_loss = sum(validation_loss) / len(validation_loss)
        accuracy = correct / total

        validation_data = {
            "validation_loss/cur": validation_loss,
            "accuracy/total": accuracy,
            "total/total": total,
            "correct/total": correct,
        }

        for i in range(args.num_classes):
            validation_data[f"accuracy/{i}"] = correct_classes[i] / total_classes[i]
            validation_data[f"total/{i}"] = total_classes[i]
            validation_data[f"correct/{i}"] = correct_classes[i]

        all_validation_data.append(validation_data)

        if len(all_validation_data) >= 10:
            prev = [x['validation_loss/cur'] for x in all_validation_data[-10:-5]]
            cur = [x['validation_loss/cur'] for x in all_validation_data[-5:]]
            assert len(prev) == 5 and len(cur) == 5
            avg_prev = sum(prev) / len(prev)
            avg_cur = sum(cur) / len(cur)
            validation_data["validation_loss/avg_prev"] = avg_prev
            validation_data["validation_loss/avg_cur"] = avg_cur

        validation_model.train().requires_grad_(True)

    dist.barrier()

    return validation_data

class AdaLN(nn.Module):
    def __init__(self, num_features, cond_features):
        super().__init__()
        self.num_features = num_features
        self.lin = nn.Linear(cond_features, num_features*2)

    def forward(self, x, cond):
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, (self.num_features,))
        weight, bias = self.lin(cond).chunk(2, dim=1)
        x = x * weight[:, None, None, :] + bias[:, None, None, :]
        x = x.permute(0, 3, 1, 2)
        return x

class BatchNorm2d(nn.BatchNorm2d):
    def forward(self, x, cond):
        return super().forward(x)

class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union['BasicBlock', 'Bottleneck']],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        inplanes: int = 64,
        input_channels: int = 3,
        dims: List[int] = [64, 128, 256, 512],
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        assert len(dims) == len(layers)
        if norm_layer is None:
            norm_layer = BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = inplanes
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(input_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layers = nn.ModuleList()

        for dim, n_layer in zip(dims, layers):
            self.layers.append(self._make_layer(block, dim, n_layer))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

        self.fc = nn.Linear(dim * block.expansion*2, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union['BasicBlock', 'Bottleneck']],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor, cond) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x, cond)
        x = self.relu(x)
        x = self.maxpool(x)

        for layer in self.layers:
            for block in layer:
                x = block(x, cond)

        x = torch.cat([self.avgpool(x).flatten(1), self.maxpool(x).flatten(1)], dim=1)
        x = F.layer_norm(x, (x.shape[1],))
        x = self.fc(x)

        return x

    def forward(self, x: Tensor, cond) -> Tensor:
        return self._forward_impl(x, cond)

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor, cond) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out, cond)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, cond)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out, cond)

        if self.downsample is not None:
            assert len(self.downsample) == 2
            identity = self.downsample[0](x)
            identity = self.downsample[1](identity, cond)

        out += identity
        out = self.relu(out)

        return out

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor, cond) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out, cond)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, cond)

        if self.downsample is not None:
            assert len(self.downsample) == 2
            identity = self.downsample[0](x)
            identity = self.downsample[1](identity, cond)

        out += identity
        out = self.relu(out)

        return out

resnet_configs = dict(
    resnet_smol=dict(block=BasicBlock, layers=[1, 1], dims=[64, 128]),
    resnet18=dict(block=BasicBlock, layers=[2, 2, 2, 2]),
    resnet34=dict(block=BasicBlock, layers=[3, 4, 6, 3]),
    resnet50=dict(block=Bottleneck, layers=[3, 4, 6, 3]),
    resnet101=dict(block=Bottleneck, layers=[3, 4, 23, 3]),
    resnet152=dict(block=Bottleneck, layers=[3, 8, 36, 3])
)

if __name__ == "__main__":
    main()
