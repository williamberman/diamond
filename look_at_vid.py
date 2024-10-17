from PIL import Image
import imageio
import torch
import os
import glob

dir = "/mnt/raid/diamond_v2/Breakout/classifier_holdout_set/"

ctr = 0

for x in glob.glob(os.path.join(dir, "**/*.pt"), recursive=True):
    it = torch.load(x)
    if "obs" not in it:
        continue
    ims = [Image.fromarray(x.permute(1,2,0).numpy()).resize((256,256)) for x in it['obs']]
    imageio.mimsave(f"{ctr}.mp4", ims)
    ctr += 1
