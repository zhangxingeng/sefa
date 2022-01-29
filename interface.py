# python 3.7
"""Demo."""

import numpy as np
import torch
import streamlit as st
#import SessionState

from models import parse_gan_type
from utils import to_tensor
from utils import postprocess
from utils import load_generator
from utils import factorize_weight
from utils import get_layers 
from common import zss_to_ws 
from common import generate_image 


@st.cache(allow_output_mutation=True, show_spinner=False)
def get_model(model_name):
    """Gets model by name."""
    print("MODEL_NAME:",model_name)
    return load_generator(model_name)


@st.cache(allow_output_mutation=True, show_spinner=False)
def factorize_model(model, layer_idx):
    """Factorizes semantics from target layers of the given model."""
    return factorize_weight(model, layer_idx)


def sample(model, gan_type, num=1):
    """Samples latent codes."""
    if gan_type != 'stylegan2_ada':
        z_dim = model.z_space_dim
    else:
        z_dim = model.z_dim
    codes = torch.randn(num, z_dim).cuda()
    if gan_type == 'pggan':
        codes = model.layer0.pixel_norm(codes)
    elif gan_type == 'stylegan':
        codes = model.mapping(codes)['w']
        codes = model.truncation(codes,
                                 trunc_psi=0.7,
                                 trunc_layers=8)
    elif gan_type == 'stylegan2':
        codes = model.mapping(codes)['w']
        codes = model.truncation(codes,
                                 trunc_psi=0.5,
                                 trunc_layers=18)
    elif gan_type == 'stylegan2_ada':
        lst_codes = []
        device = torch.device('cuda')
        for idx in range(codes.size()[0]):
            lst_codes.append(codes[idx].unsqueeze(0))
        trunc_psi = 0.7
        codes = zss_to_ws(model,device,None,trunc_psi,lst_codes)
        #codes = np.asarray(codes)
    
    codes = codes.detach().cpu().numpy()
    #print("CODES SIZE:",codes.size())

    return codes


@st.cache(allow_output_mutation=True, show_spinner=False)
def synthesize(model, gan_type, code):
    """Synthesizes an image with the give code."""
    if gan_type == 'pggan':
        image = model(to_tensor(code))['image']
    elif gan_type in ['stylegan', 'stylegan2']:
        image = model.synthesis(to_tensor(code))['image']
    image = postprocess(image)[0]
    return image


def main():
    """Main function (loop for StreamLi	t)."""
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    st.title('Closed-Form Factorization of Latent Semantics in GANs')
    st.sidebar.title('Options')
    reset = st.sidebar.button('Reset')

    model_name = st.sidebar.selectbox(
        'Model to Interpret',
        ['stylegan_pizza256','stylegan2_ada_pizza256','stylegan_animeface512', 'stylegan_car512', 'stylegan_cat256',
         'pggan_celebahq1024'])

    model = get_model(model_name)
    if 'ada' in model_name:
        gan_type = 'stylegan2_ada'
    else:
        gan_type = parse_gan_type(model)
    layer_idx = st.sidebar.selectbox(
        'Layers to Interpret',
        ['all', '0-1', '2-5', '6-13'])
    if gan_type != 'stylegan2_ada':
        layers, boundaries, eigen_values = factorize_model(model, layer_idx)
    else:
        modulate = {
            k[0]: k[1]
            for k in model.named_parameters()
            if "affine" in k[0] and "torgb" not in k[0] and "weight" in k[0] or ("torgb" in k[0] and "b4" in k[0] and "weight" in k[0] and "affine" in k[0])
        }

        weight_mat = []
        for k, v in modulate.items():
         weight_mat.append(v)

        W = torch.cat(weight_mat, 0)
        W = W.detach().cpu().numpy()
        W = W / np.linalg.norm(W, axis=0, keepdims=True)
        #boundaries = torch.svd(W).V.to("cpu")
        #boundaries = boundaries.detach().cpu().numpy()
        #values = torch.svd(W).S.to("cpu")
        #W = W / np.linalg.norm(W, axis=0, keepdims=True)
        eigen_values, eigen_vectors = np.linalg.eig(W.T.dot(W))
        boundaries = eigen_vectors
        layers = get_layers(layer_idx, 10)
        gan_type = 'stylegan2_ada'

    num_semantics = st.sidebar.number_input(
        'Number of semantics', value=10, min_value=0, max_value=None, step=1)
    steps = {sem_idx: 0 for sem_idx in range(num_semantics)}
    if gan_type == 'pggan':
        max_step = 5.0
    elif gan_type == 'stylegan':
        max_step = 2.0
    elif gan_type == 'stylegan2':
        max_step = 15.0
    elif gan_type == 'stylegan2_ada':
        max_step = 15.0
    for sem_idx in steps:
        eigen_value = eigen_values[sem_idx]
        steps[sem_idx] = st.sidebar.slider(
            f'Semantic {sem_idx:03d} (eigen value: {eigen_value:.3f})',
            value=0.0,
            min_value=-max_step,
            max_value=max_step,
            step=0.04 * max_step if not reset else 0.0)

    image_placeholder = st.empty()
    button_placeholder = st.empty()

    try:
        base_codes = np.load(f'latent_codes/{model_name}_latents.npy')
    except FileNotFoundError:
        base_codes = sample(model, gan_type)

    cur_model_name = st.session_state.get('model_name','')
                           
    if cur_model_name != model_name:
        cur_model_name = model_name
        cur_code_idx = 0
        cur_codes = base_codes[0:1]
    else:
        cur_code_idx = st.session_state.get('code_idx')
        cur_codes = st.session_state.get('codes')


    if button_placeholder.button('Random', key=0):
        cur_code_idx += 1
        if cur_code_idx < base_codes.shape[0]:
            cur_codes = base_codes[cur_code_idx][np.newaxis]
        else:
            cur_codes = sample(model, gan_type)
    
    st.session_state.model_name = cur_model_name
    st.session_state.code_idx = cur_code_idx
    st.session_state.codes = cur_codes

    code = cur_codes.copy()
    for sem_idx, step in steps.items():
        if gan_type == 'pggan':
            code += boundaries[sem_idx:sem_idx + 1] * step
        elif gan_type in ['stylegan', 'stylegan2','stylegan2_ada']:
            code[:, layers, :] += boundaries[sem_idx:sem_idx + 1] * step
    if gan_type != 'stylegan2_ada':
        image = synthesize(model, gan_type, code)
        print("IMAGE:",image.shape)
    else:
        device = torch.device('cuda')
        label = torch.zeros([1, 0], device=device) # assume no class label
        image= generate_image(model, torch.from_numpy(code).cuda(), label, 0.7, 'const', 'w')
        image = postprocess(image)
    image_placeholder.image(image / 255.0)


if __name__ == '__main__':
    main()
