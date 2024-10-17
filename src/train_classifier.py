import os
from functools import partial
from typing import Any, Callable, List, Optional, Type, Union
from torchvision.models.resnet import BasicBlock, conv1x1, conv3x3, _log_api_usage_once
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from argparse import ArgumentParser
from data import Dataset as DiamondDataset, Episode
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.utils.data.dataloader import default_collate
import random
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch import Tensor
import torch.optim as optim
import wandb
import tqdm

device = int(os.environ['LOCAL_RANK'])
classifying = 'actions'

def main():
    global args, model

    dist.init_process_group(backend='nccl')

    args = ArgumentParser()
    args.add_argument('--num_classes', type=int, required=False)
    args.add_argument('--num_input_images', type=int, default=30)
    args.add_argument("--has_negative_rewards", type=int, required=True)
    args.add_argument("--classifying", type=str, required=True, choices=["actions", "rewards", "end"])
    args.add_argument("--training_dataset_path", type=str, required=True)
    args.add_argument("--holdout_dataset_path", type=str, required=True)
    args.add_argument("--batch_size", type=int, default=512)
    args.add_argument("--validation_batch_size", type=int, default=1024)
    args.add_argument("--validation_steps", type=int, default=1000)
    args.add_argument("--save_steps", type=int, default=1000)
    args.add_argument("--save_dir", type=str, required=True)
    args = args.parse_args()

    assert args.has_negative_rewards in [0, 1]
    args.has_negative_rewards = bool(args.has_negative_rewards)

    if args.classifying == "actions":
        assert args.num_classes
    elif args.classifying == "rewards":
        if args.has_negative_rewards:
            args.num_classes = 3
        else:
            args.num_classes = 2
    elif args.classifying == "end":
        args.num_classes = 2
    else:
        assert False

    model = ResNet(
        Bottleneck,
        [3, 8, 36, 3],
        num_classes=args.num_classes,
        input_channels=3*args.num_input_images,
        inplanes=256,
        norm_layer=lambda x: AdaLN(x, args.num_input_images*3-1),
    ).train().requires_grad_(True).to(device)
    model = DDP(model, device_ids=[device])

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    data = data_iter()

    if device == 0:
        wandb.init(project="diamond_classifier")

    step = 0

    while True:
        batch = next(data)
        batch = {k: v.to(device) for k, v in batch.items()}
        obs = batch['obs']
        cond = torch.cat([batch['actions'], batch['rewards'], batch['ends']], dim=1)

        pred = model(obs, cond)
        loss = F.cross_entropy(pred, batch['target'])
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        dist.barrier()

        if device == 0 and (step+1) % args.save_steps == 0:
            state_dict = dict(model=model.state_dict(), optimizer=optimizer.state_dict())
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save(state_dict, os.path.join(args.save_dir, f"{step}.pt"))

        dist.barrier()

        validation_data = {}

        if (step+1) % args.validation_steps == 0:
            validation_data = validation()

        if device == 0:
            log_args = dict(loss=loss.item(), grad_norm=grad_norm.item(), **validation_data)
            wandb.log(log_args, step=step)
            print(f"{step}: {log_args}")

        step += 1

@torch.no_grad()
def validation():
    dist.barrier()

    validation_data = None

    pbar = tqdm.tqdm()

    if device == 0:
        print("running validation")

        validation_loss = []
        correct = 0
        total = 0

        correct_classes = {i: 0 for i in range(args.num_classes)}
        total_classes = {i: 0 for i in range(args.num_classes)}

        validation_model = model.module
        validation_model.eval().requires_grad_(False)

        validation_iter_ = validation_iter()

        for batch in validation_iter_:
            batch = {k: v.to(device) for k, v in batch.items()}
            obs = batch['obs']
            cond = torch.cat([batch['actions'], batch['rewards'], batch['ends']], dim=1)

            pred = validation_model(obs, cond)
            loss = F.cross_entropy(pred, batch['target'])
            validation_loss.append(loss.item())

            total += batch['target'].shape[0]
            correct += (pred.argmax(dim=1) == batch['target']).sum().item()

            for i in range(args.num_classes):
                mask = batch['target'] == i
                pred_ = pred[mask]
                target_ = batch['target'][mask]

                correct_classes[i] += (pred_.argmax(dim=1) == target_).sum().item()
                total_classes[i] += target_.shape[0]

            pbar.update(1)

        validation_loss = sum(validation_loss) / len(validation_loss)
        accuracy = correct / total

        validation_data = dict(validation_loss=validation_loss, accuracy=accuracy, total=total, correct=correct)

        for i in range(args.num_classes):
            validation_data[f"accuracy_{i}"] = correct_classes[i] / total_classes[i]
            validation_data[f"total_{i}"] = total_classes[i]
            validation_data[f"correct_{i}"] = correct_classes[i]

        validation_model.train().requires_grad_(True)

    dist.barrier()

    return validation_data

def validation_iter():
    for dataset_path in os.listdir(args.holdout_dataset_path):
        iterable = iter(iterate_diamond_dataset(os.path.join(args.holdout_dataset_path, dataset_path), None))
        done = False

        while True:
            batch = []

            while len(batch) < args.validation_batch_size:
                try:
                    batch.append(next(iterable))
                except StopIteration:
                    done = True
                    break

            yield default_collate(batch)

            if done:
                break

def data_iter():
    data_iters = [
        iter(DataLoader(TrainingDataset(args.training_dataset_path, action_idx), num_workers=1, batch_size=1, collate_fn=lambda x: x))
        for action_idx in range(args.num_classes)
    ]

    data_iter_idx = 0

    while True:
        batch = []

        while len(batch) < args.batch_size:
            batch.append(next(data_iters[data_iter_idx])[0])
            data_iter_idx = (data_iter_idx + 1) % len(data_iters)

        yield default_collate(batch)

class TrainingDataset(IterableDataset):
    def __init__(self, dataset_path, action_idx):
        self.dataset_path = dataset_path
        self.action_idx = action_idx

    def __iter__(self):
        while True:
            dataset_path = os.path.join(self.dataset_path, random.choice(os.listdir(self.dataset_path)))
            for x in iterate_diamond_dataset(dataset_path, self.action_idx):
                yield x

def iterate_diamond_dataset(dataset_path, action_idx):
    dataset = DiamondDataset(dataset_path)
    dataset.load_from_default_path()

    for episode_id in range(dataset.num_episodes):
        episode = dataset.load_episode(episode_id)

        for i in range(len(episode) - args.num_input_images + 1):
            obs = episode.obs[i:i+args.num_input_images].flatten(0, 1)
            actions = episode.act[i:i+args.num_input_images]
            rewards = episode.rew[i:i+args.num_input_images]
            ends = episode.end[i:i+args.num_input_images]

            # target is in the middle
            target_idx = args.num_input_images // 2

            if args.classifying == "actions":
                target = actions[target_idx]
                if action_idx is not None and target != action_idx:
                    continue
                actions = torch.cat([actions[:target_idx], actions[target_idx+1:]])
            elif args.classifying == "rewards":
                assert False, "TODO what is actually in the dataset"
                target_reward = rewards[target_idx]

                if action_idx is not None and target_reward != action_idx:
                    continue
            elif args.classifying == "ends":
                target = ends[target_idx]
                if action_idx is not None and target != action_idx:
                    continue
                ends = torch.cat([ends[:target_idx], ends[target_idx+1:]])
            else:
                assert False

            yield dict(
                obs=obs,
                actions=actions,
                rewards=rewards,
                ends=ends,
                target=target,
            )

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

class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, 'Bottleneck']],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        inplanes: int = 64,
        input_channels: int = 3,
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
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
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
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
        block: Type[Union[BasicBlock, 'Bottleneck']],
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

        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
                x = block(x, cond)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
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
            norm_layer = nn.BatchNorm2d
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

if __name__ == "__main__":
    main()
