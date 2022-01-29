"""SeFa."""

import os
import subprocess 
import argparse
from tqdm import tqdm
import numpy as np

import torch

from models import parse_gan_type
from utils import to_tensor
from utils import postprocess
from utils import load_generator
from utils import factorize_weight
from utils import get_layers 
from utils import HtmlPageVisualizer
import torchvision
#from torchvision import utils # assumes you use torchvision 0.8.2; if you use the latest version, see comments below
from common import zs_to_ws , zss_to_ws, generate_image, line_interpolate


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        description='Discover semantics from the pre-trained weight.')
    parser.add_argument('model_name', type=str,
                        help='Name to the pre-trained model.')
    parser.add_argument('--save_dir', type=str, default='results',
                        help='Directory to save the visualization pages. '
                             '(default: %(default)s)')
    parser.add_argument('-L', '--layer_idx', type=str, default='all',
                        help='Indices of layers to interpret. '
                             '(default: %(default)s)')
    parser.add_argument('-N', '--num_samples', type=int, default=5,
                        help='Number of samples used for visualization. '
                             '(default: %(default)s)')
    parser.add_argument('-K', '--num_semantics', type=int, default=5,
                        help='Number of semantic boundaries corresponding to '
                             'the top-k eigen values. (default: %(default)s)')
    parser.add_argument('--start_distance', type=float, default=-3.0,
                        help='Start point for manipulation on each semantic. '
                             '(default: %(default)s)')
    parser.add_argument('--end_distance', type=float, default=3.0,
                        help='Ending point for manipulation on each semantic. '
                             '(default: %(default)s)')
    parser.add_argument('--step', type=int, default=11,
                        help='Manipulation step on each semantic. '
                             '(default: %(default)s)')
    parser.add_argument('--viz_size', type=int, default=256,
                        help='Size of images to visualize on the HTML page. '
                             '(default: %(default)s)')
    parser.add_argument('--trunc_psi', type=float, default=0.7,
                        help='Psi factor used for truncation. This is '
                             'particularly applicable to StyleGAN (v1/v2). '
                             '(default: %(default)s)')
    parser.add_argument('--trunc_layers', type=int, default=8,
                        help='Number of layers to perform truncation. This is '
                             'particularly applicable to StyleGAN (v1/v2). '
                             '(default: %(default)s)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for sampling. (default: %(default)s)')
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='GPU(s) to use. (default: %(default)s)')
    parser.add_argument('--video', type=str, default='no', help='Generate video or not')
    parser.add_argument('--degree', type=float, default=5, help='scalar factors for moving latent vectors along eigenvector')
    parser.add_argument("--vid_increment", type=float, default=0.1, help="increment degree for interpolation video")
    return parser.parse_args()


def main():
    """Main function."""
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    args = parse_args()
    print("ARGS:",args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    os.makedirs(args.save_dir, exist_ok=True)

    # Factorize weights.
    generator = load_generator(args.model_name)
    if 'ada' in args.model_name:
        modulate = {
            k[0]: k[1]
            for k in generator.named_parameters()
            if "affine" in k[0] and "torgb" not in k[0] and "weight" in k[0] or ("torgb" in k[0] and "b4" in k[0] and "weight" in k[0] and "affine" in k[0])
        }

        weight_mat = []
        for k, v in modulate.items():
         weight_mat.append(v)

        W = torch.cat(weight_mat, 0)
        W = W.detach().cpu().numpy()
        W = W / np.linalg.norm(W, axis=0, keepdims=True)
        #W = torch.tensor(W)
        #boundaries = torch.svd(W).V.to("cpu")
        #boundaries = boundaries.detach().cpu().numpy()
        #values = torch.svd(W).S.to("cpu")
        values, eigen_vectors = np.linalg.eig(W.T.dot(W))
        boundaries = eigen_vectors
        print("values:",values.shape)
        print("boundaries:",boundaries.shape)

        layers = get_layers(args.layer_idx, 10)
        gan_type = 'stylegan2_ada'
    else:
        gan_type = parse_gan_type(generator)
        layers, boundaries, values = factorize_weight(generator, args.layer_idx)
        print("LAYERS:",np.asarray(layers).shape)
        print("LAYERS:",layers)
        print("BOUNDARIES:",np.asarray(boundaries).shape)
        print("VALUES:",np.asarray(values).shape)

    print("Done loading")
    # Set random seed.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float32)
    # Prepare codes.
    latents = torch.randint(0,10000,(args.num_samples,))
    if 'ada' in args.model_name:
        z_dim = generator.z_dim
    else:
        z_dim = generator.z_space_dim
    codes = None
    for latent in latents:
        print("LATENT:",latent)
        z = torch.from_numpy(np.random.RandomState(latent).randn(1, z_dim)).cuda()
        if codes == None:
            codes = z
        else:
            codes = torch.cat([codes, z],dim=0)

    print("CODES:",codes.size())
    device = torch.device('cuda')
    codes = torch.tensor(codes).float()
    label = torch.zeros([1, 0], device=device) # assume no class label
    if gan_type == 'pggan':
        codes = generator.layer0.pixel_norm(codes)
    elif gan_type in ['stylegan', 'stylegan2']:
        codes = generator.mapping(codes)['w']
        codes = generator.truncation(codes,
                                     trunc_psi=args.trunc_psi,
                                     trunc_layers=args.trunc_layers)
        codes = codes.detach().cpu().numpy()
        print("AFter truncation CODES SHAPE:", codes.shape)
    elif gan_type in ['stylegan2_ada']:
        lst_codes = []
        for idx in range(codes.size()[0]):
            lst_codes.append(codes[idx].unsqueeze(0))
        codes = zss_to_ws(generator,device,None,args.trunc_psi,lst_codes)
        #codes = np.asarray(codes)
        codes = codes.detach().cpu().numpy()
        print("AFter conversion CODES SHAPE:", codes.shape)

    # Generate visualization pages.
    distances = np.linspace(args.start_distance,args.end_distance, args.step)
    num_sam = args.num_samples
    num_sem = args.num_semantics
    index_list_of_eigenvalues=[i for i in range(num_sem)]
    vizer_1 = HtmlPageVisualizer(num_rows=num_sem * (num_sam + 1),
                                 num_cols=args.step + 1,
                                 viz_size=args.viz_size)
    vizer_2 = HtmlPageVisualizer(num_rows=num_sam * (num_sem + 1),
                                 num_cols=args.step + 1,
                                 viz_size=args.viz_size)

    headers = [''] + [f'Distance {d:.2f}' for d in distances]
    vizer_1.set_headers(headers)
    vizer_2.set_headers(headers)
    for sem_id in range(num_sem):
        value = values[sem_id]
        vizer_1.set_cell(sem_id * (num_sam + 1), 0,
                         text=f'Semantic {sem_id:03d}<br>({value:.3f})',
                         highlight=True)
        for sam_id in range(num_sam):
            vizer_1.set_cell(sem_id * (num_sam + 1) + sam_id + 1, 0,
                             text=f'Sample {sam_id:03d}')
    for sam_id in range(num_sam):
        vizer_2.set_cell(sam_id * (num_sem + 1), 0,
                         text=f'Sample {sam_id:03d}',
                         highlight=True)
        for sem_id in range(num_sem):
            value = values[sem_id]
            vizer_2.set_cell(sam_id * (num_sem + 1) + sem_id + 1, 0,
                             text=f'Semantic {sem_id:03d}<br>({value:.3f})')

    for sam_id in tqdm(range(num_sam), desc='Sample ', leave=False):
        code = codes[sam_id:sam_id + 1]
        #if gan_type == 'stylegan2_ada':
            #images = generate_images(args, generator, temp_code, label, 0.7, 'const', boundaries, 'w')

        for sem_id in tqdm(range(num_sem), desc='Semantic ', leave=False):
            boundary = boundaries[sem_id:sem_id + 1]
            for col_id, d in enumerate(distances, start=1):
                temp_code = code.copy()
                if gan_type == 'pggan':
                    temp_code += boundary * d
                    image = generator(to_tensor(temp_code))['image']
                elif gan_type in ['stylegan', 'stylegan2']:
                    temp_code[:, layers, :] += boundary * d
                    image = generator.synthesis(to_tensor(temp_code))['image']
                elif gan_type == 'stylegan2_ada':
                    #print("BOUNDARY:",boundary.shape," temp:",temp_code.shape)
                    temp_code[:, layers, :] += boundary * d
                    print("temp SHAPE:",temp_code.shape)
                    image= generate_image(generator, torch.from_numpy(temp_code).cuda(), label, 0.7, 'const', 'w')
                image = postprocess(image)[0]
                vizer_1.set_cell(sem_id * (num_sam + 1) + sam_id + 1, col_id,
                                 image=image)
                vizer_2.set_cell(sam_id * (num_sem + 1) + sem_id + 1, col_id,
                                 image=image)

    prefix = (f'{args.model_name}_'
              f'N{num_sam}_K{num_sem}_L{args.layer_idx}_seed{args.seed}')
    vizer_1.save(os.path.join(args.save_dir, f'{prefix}_sample_first.html'))
    vizer_2.save(os.path.join(args.save_dir, f'{prefix}_semantic_first.html'))


    if(args.video == 'yes'):
        print('Processing videos; this may take a while...')
        args.output="./results"
        args.space='w'

        str_seed_list = '-'.join(str(x) for x in latents.cpu().detach().numpy())
        str_index_list = '-'.join(str(x) for x in index_list_of_eigenvalues)

        folder_name = f"seed-{str_seed_list}_index-{str_index_list}_degree-{args.degree}"
        folder_path = os.path.join(args.output, folder_name)

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        lIndex = 0
        for l in latents:
            seed_folder_name = f"seed-{l}"
            seed_folder_path = os.path.join(folder_path, seed_folder_name)

            if not os.path.exists(seed_folder_path):
                os.makedirs(seed_folder_path)

            z = torch.from_numpy(codes[lIndex]).cuda()
            lIndex += 1
            #z = torch.from_numpy(np.random.RandomState(l).randn(1, z_dim)).to(device)

            #print("VID Z:",z.size())


            for j in index_list_of_eigenvalues:
                current_eigvec = torch.from_numpy(boundaries[:, j]).unsqueeze(0).cuda()
                direction = args.degree * current_eigvec
                print("DIR:",direction.size(), " Z:",z.size())

                index_folder_name = f"index-{j}/frames"
                index_folder_path = os.path.join(seed_folder_path, index_folder_name)

                if not os.path.exists(index_folder_path):
                    os.makedirs(index_folder_path)

                negative = z.clone()
                negative[layers,:] -= direction
                positive = z.clone()
                positive[layers,:] += direction
                zs = [negative, positive]
                if(args.space=='w'):
                    truncation_psi=args.trunc_psi
                    #ws = zss_to_ws(generator,device,label,truncation_psi,zs)
                    #pts = line_interpolate(ws, int((args.degree*2)/args.vid_increment))
                    pts = line_interpolate(zs, 100)
                    print("LENGTHS:", len(pts), ":", len(zs))
                else:
                    pts = line_interpolate(zs, int((args.degree*2)/args.vid_increment))
                
                fcount = 0
                for pt in pts:
                    qt = pt.unsqueeze(0)
                    noise_mode = 'const'
                    img = generate_image(generator, qt, label, truncation_psi, noise_mode, args.space)
                    #image= generate_image(generator, pt, label, 0.7, 'const', 'w')
                    grid = torchvision.utils.save_image(
                        img, #['image'],
                        f"{index_folder_path}/{fcount:04}.png",
                        normalize=True,
                        value_range=(-1, 1), # change range to value_range for latest torchvision
                        nrow=1,
                    )
                    fcount+=1
                #cmd=f"ffmpeg -y -r 24 -i {index_folder_path}/%04d.png -vcodec libx264 -pix_fmt yuv420p {seed_folder_path}/seed-{str_seed_list}_index-{j}_degree-{args.degree}.mp4"
                cmd=f"ffmpeg -y -r 72 -i {index_folder_path}/%04d.png -vcodec libx264 -pix_fmt yuv420p {seed_folder_path}/seed-{str_seed_list}_index-{j}_degree-{args.degree}.mp4"
                #print("CMD:",cmd)
                subprocess.call(cmd, shell=True)
                #subprocess.call(cmd)
if __name__ == '__main__':
    main()
