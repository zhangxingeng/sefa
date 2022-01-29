import os
import re
import subprocess
import argparse
import torch
from torchvision import utils # assumes you use torchvision 0.8.2; if you use the latest version, see comments below
import legacy
import dnnlib
from typing import List
import numpy as np
import random

"""
Use closed_form_factorization.py first to create your factor.pt

Usage:

python apply_factor.py -i 1-3 --seeds 10,20 --ckpt models/ffhq.pkl factor.pt --video
Create images and interpolation videos for image-seeds 10 and 20 for eigenvalues one, two and three.

python apply_factor.py -i 10,20 --seeds 100-200 --ckpt models/ffhq.pkl factor.pt --no-video
Create images for each image-seed between 100 and 200 and for eigenvalues 10 and 20.

python apply_factor.py --seeds r3 --ckpt models/ffhq.pkl factor.pt --no-video
Create images for three random seeds and all eigenvalues (this can take a lot of time, especially for videos).

Apply different truncation values by using --truncation.
Apply different increment degree for interpolation video by using --vid_increment.
Apply different scalar factors for moving latent vectors along eigenvector by using --degree.
Change output directory by using --output.
"""

#############################################################################################

def generate_images(args, G, z, label, truncation_psi, noise_mode, direction, file_name):
    if(args.space == 'w'):
        ws = zs_to_ws(G,torch.device('cuda'),label,truncation_psi,[z,z + direction,z - direction])
        img1 = G.synthesis(ws[0], noise_mode=noise_mode, force_fp32=True)
        img2 = G.synthesis(ws[1], noise_mode=noise_mode, force_fp32=True)
        img3 = G.synthesis(ws[2], noise_mode=noise_mode, force_fp32=True)
    else:
        img1 = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        img2 = G(z + direction, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        img3 = G(z - direction, label, truncation_psi=truncation_psi, noise_mode=noise_mode)

    X = torch.cat([img3, img1, img2], 0)
    #print("generate_images: X=",X.size())
    return X

def generate_image(G, z, label, truncation_psi, noise_mode, space):
    #print("generate_image: z:",z.size(), " truncation_psi:",truncation_psi, " noise_mode:",noise_mode, " space:",space)
    if(space == 'w'):
        #img = G.synthesis(z, noise_mode=noise_mode, force_fp32=True)
        #z = z.unsqueeze(0)  #uncomment for stylegan/stylegan2
        img = G.synthesis(z)
    else:
        img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
    #print("generate_image: z:",z.size(), " img:",img.size())
    return img

def line_interpolate(zs, steps):
    out = []
    for i in range(len(zs)-1):
        for index in range(steps):
            t = index/float(steps)
            out.append(zs[i+1]*t + zs[i]*(1-t))
    return out

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c', a range 'a-c' and return as a list of ints or a string with "r{number}".'''
    if "r" in s:
        index = s.index("r")
        return int(s[index+1:])
    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

def zs_to_ws(G,device,label,truncation_psi,zs):
    ws = []
    for z_idx, z in enumerate(zs):
        # z = torch.from_numpy(z).to(device)
        #w = G.mapping(z, label, truncation_psi, truncation_cutoff=8)
        #w = G.mapping(z, label, truncation_psi, 8)
        w = G.mapping(z)['w']
        ws.append(w)
    return ws

def zss_to_ws(G,device,label,truncation_psi,zss):
    ws = None 
    for z_idx, z in enumerate(zss):
        w = G.mapping(z, label, truncation_psi=truncation_psi, truncation_cutoff=8)
        print("Z:",z.size(), " W:",w.size())
        if ws == None:
            ws = w
        else:
            ws = torch.cat([ws, w],dim=0)
    return ws
#############################################################################################
