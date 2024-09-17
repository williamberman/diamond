class Dataset(torch.utils.data.IterableDataset):
    @torch.no_grad()
    def __iter__(self):
        env = make_env()
        agent = load_random_agent()
        trajectory_ctr = 0

        trajectory_buffer = []

        if use_labeled_actions:
            action_labeler = ImprovedCNN(n_actions, 3).to(device)
            action_labeler.load_state_dict(torch.load("smol_conv_classifier_final.pt", map_location="cpu", weights_only=True))
            action_labeler.eval()

            labeler_mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
            labeler_std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]

        while True:
            t0 = time.perf_counter()

            if (trajectory_ctr+1) % 200 == 0:
                agent = load_random_agent()

            obs = [] 
            act = [] 

            state, _ = env.reset()

            obs.append(state)
            act.append(get_action(agent, obs))

            while True:
                next_obs, reward, terminated, truncated, info = env.step(act[-1])

                obs.append(next_obs)
                act.append(get_action(agent, obs))

                if terminated or truncated:
                    break
            
            if use_labeled_actions:
                obs_ = []
                next_obs_ = []

                for i in range(len(obs)-1):
                    obs_.append(torch.tensor(obs[i]))
                    next_obs_.append(torch.tensor(obs[i+1]))

                obs_ = torch.stack(obs_).to(device).permute(0, 3, 1, 2).float().div(255).sub(labeler_mean).div(labeler_std)
                next_obs_ = torch.stack(next_obs_).to(device).permute(0, 3, 1, 2).float().div(255).sub(labeler_mean).div(labeler_std)
                deltas = next_obs_ - obs_

                labeled_act = action_labeler(deltas).argmax(dim=1).to('cpu')
                labeled_act = torch.concat([labeled_act, act[-1].unsqueeze(0)])
                assert len(labeled_act) == len(obs)
                act = labeled_act
            else:
                act = torch.stack(act)

            obs = torch.stack([torch.tensor(o).div(255).mul(2).sub(1).permute(2, 0, 1) for o in obs])

            trajectory_buffer.append((obs, act))

            if len(trajectory_buffer) == 10:
                yield trajectory_buffer
                trajectory_buffer.clear()

            trajectory_ctr += 1
            # print(f"Collected {trajectory_ctr} trajectories. trajectory_buffer: {len(trajectory_buffer)} time: {time.perf_counter() - t0} steps in trajectory: {obs.shape[0]}")

