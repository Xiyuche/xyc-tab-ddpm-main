import sys

import torch
import numpy as np
import zero
import os
from tab_ddpm.gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion
from tab_ddpm.utils import FoundNANsError
from utils_train import get_model, make_dataset
from lib import round_columns
import lib

def to_good_ohe(ohe, X):
    indices = np.cumsum([0] + ohe._n_features_outs)
    Xres = []
    for i in range(1, len(indices)):
        x_ = np.max(X[:, indices[i - 1]:indices[i]], axis=1)
        t = X[:, indices[i - 1]:indices[i]] - x_.reshape(-1, 1)
        Xres.append(np.where(t >= 0, 1, 0))
    return np.hstack(Xres)

def sample(
    parent_dir,
    real_data_path = 'data/higgs-small',
    batch_size = 2000,
    num_samples = 0,
    model_type = 'mlp',
    model_params = None,
    model_path = None,
    num_timesteps = 1000,
    gaussian_loss_type = 'mse',
    scheduler = 'cosine',
    T_dict = None,
    num_numerical_features = 0,
    disbalance = None,
    device = torch.device('cpu'),
    seed = 0,
    change_val = False,
    file_path = ''
):
    device = torch.device('cpu')
    zero.improve_reproducibility(seed)

    T = lib.Transformations(**T_dict)
    D = make_dataset(
        real_data_path,
        T,
        num_classes=model_params['num_classes'],
        is_y_cond=model_params['is_y_cond'],
        change_val=change_val
    )

    K = np.array(D.get_category_sizes('train'))
    if len(K) == 0 or T_dict['cat_encoding'] == 'one-hot':
        K = np.array([0])

    num_numerical_features_ = D.X_num['train'].shape[1] if D.X_num is not None else 0
    d_in = np.sum(K) + num_numerical_features_
    model_params['d_in'] = int(d_in)
    model = get_model(
        model_type,
        model_params,
        num_numerical_features_,
        category_sizes=D.get_category_sizes('train')
    )

    model.load_state_dict(
        torch.load(model_path, map_location="cpu")
    )

    diffusion = GaussianMultinomialDiffusion(
        K,
        num_numerical_features=num_numerical_features_,
        denoise_fn=model, num_timesteps=num_timesteps, 
        gaussian_loss_type=gaussian_loss_type, scheduler=scheduler, device=device
    )


    diffusion.to(device)
    diffusion.eval()
    
    _, empirical_class_dist = torch.unique(torch.from_numpy(D.y['train']), return_counts=True)
    # # empirical_class_dist = empirical_class_dist.float() + torch.tensor([-5000., 10000.]).float()

    # x_gen, y_gen = diffusion.sample_all(num_samples, batch_size, empirical_class_dist.float(), ddim=False)


    # try:
    # except FoundNANsError as ex:
    #     print("Found NaNs during sampling!")
    #     loader = lib.prepare_fast_dataloader(D, 'train', 8)
    #     x_gen = next(loader)[0]
    #     y_gen = torch.multinomial(
    #         empirical_class_dist.float(),
    #         num_samples=8,
    #         replacement=True
    #     )

    # Here load X_gen and Y_gen here by modifying codes below

    # X_gen, y_gen = x_gen.numpy(), y_gen.numpy()
    # X_num_train = np.load('ExperimentLocalData/X_num_train.npy')
    # X_cat_train = np.load('ExperimentLocalData/X_cat_train.npy')
    # X_gen = np.concatenate((X_num_train, X_cat_train), axis=1)
    # file_path = '' # file_path is passed from argument
    X_gen = np.load(file_path)  # Resample size = (6400,11)
    if model_params['is_y_cond']:   # only load y_gen if y is conditioned in MLP, X_gen has no y info
        exp_dir = os.path.dirname(file_path)
        sample_name = os.path.splitext(os.path.basename(file_path))[0]
        y_gen = np.load(f'{exp_dir}/{sample_name}_y.npy', allow_pickle=True)


    ###
    # X_num_unnorm = X_gen[:, :num_numerical_features]
    # lo = np.percentile(X_num_unnorm, 2.5, axis=0)
    # hi = np.percentile(X_num_unnorm, 97.5, axis=0)
    # idx = (lo < X_num_unnorm) & (hi > X_num_unnorm)
    # X_gen = X_gen[np.all(idx, axis=1)]
    # y_gen = y_gen[np.all(idx, axis=1)]
    ###

    num_numerical_features = num_numerical_features + int(D.is_regression and not model_params["is_y_cond"])

    X_num_ = X_gen
    if num_numerical_features < X_gen.shape[1]:
        np.save(os.path.join(parent_dir, 'X_cat_unnorm'), X_gen[:, num_numerical_features:])
        # _, _, cat_encoder = lib.cat_encode({'train': X_cat_real}, T_dict['cat_encoding'], y_real, T_dict['seed'], True)
        if T_dict['cat_encoding'] == 'one-hot':
            X_gen[:, num_numerical_features:] = to_good_ohe(D.cat_transform.steps[0][1], X_num_[:, num_numerical_features:])
        X_cat = D.cat_transform.inverse_transform(X_gen[:, num_numerical_features:])

    if num_numerical_features_ != 0:
        # _, normalize = lib.normalize({'train' : X_num_real}, T_dict['normalization'], T_dict['seed'], True)
        np.save(os.path.join(parent_dir, 'X_num_unnorm'), X_gen[:, :num_numerical_features])
        X_num_ = D.num_transform.inverse_transform(X_gen[:, :num_numerical_features])
        X_num = X_num_[:, :num_numerical_features]

        X_num_real = np.load(os.path.join(real_data_path, "X_num_train.npy"), allow_pickle=True)
        disc_cols = []
        for col in range(X_num_real.shape[1]):
            uniq_vals = np.unique(X_num_real[:, col])
            if len(uniq_vals) <= 32 and ((uniq_vals - np.round(uniq_vals)) == 0).all():
                disc_cols.append(col)
        print("Discrete cols:", disc_cols)
        if model_params['num_classes'] == 0:
            y_gen = X_num[:, 0]
            X_num = X_num[:, 1:]
        if len(disc_cols):
            X_num = round_columns(X_num_real, X_num, disc_cols)

    parent_dir = os.path.dirname(file_path)

    # Base filename without the extension
    base_filename = os.path.splitext(os.path.basename(file_path))[0]

    # Save each component in the same parent directory with appended type
    X_num_path = f'{parent_dir}/{base_filename}_num.npy'
    np.save(X_num_path, X_num)
    print("Saved X_num at path:", X_num_path)

    if num_numerical_features < X_gen.shape[1]:     #save cat file only has_cat
        X_cat_path = f'{parent_dir}/{base_filename}_cat.npy'
        np.save(X_cat_path, X_cat)
        print("Saved X_cat at path:", X_cat_path)

    y_gen_path = f'{parent_dir}/{base_filename}_y_gen.npy'
    np.save(y_gen_path, y_gen)
    print("Saved y_gen at path:", y_gen_path)

    sys.exit('Transformation done.')

# Here save the X_num or X_cat or Y_train
#     if num_numerical_features != 0:
#         print("Num shape: ", X_num.shape)
#         np.save(os.path.join(parent_dir, 'X_num_train'), X_num)
#     if num_numerical_features < X_gen.shape[1]:
#         np.save(os.path.join(parent_dir, 'X_cat_train'), X_cat)
#     np.save(os.path.join(parent_dir, 'y_train'), y_gen)