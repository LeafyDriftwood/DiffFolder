import umap
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm
import argparse
import sys

'''
Usage
interact -m 64g -t 24:00:00
conda activate /users/anair27/data/DiffFolder/DiffFolder/env/difffolder_conda
python umap_visual.py -e omegafold -t umap
'''

def esm_embeddings():
    # load embeddings
    path_to_embeddings = '/users/anair27/data/DiffFolder/DiffFolder/esm_embeddings/esm_256_small'

    # load embeddings
    embeddings = []
    for file in tqdm(os.listdir(path_to_embeddings)):
        if file.endswith('.npz'):
            try:
                embedding = np.load(os.path.join(path_to_embeddings, file))
            except Exception as e:
                print(f"Error loading {file}")
                continue
            # keys = embedding.keys()
            embedding = embedding['arr_0']
            print(embedding.shape)
            # collapse by taking average across residues
            embedding = np.mean(embedding, axis=0)
            print(embedding.shape)
            embeddings.append(embedding.reshape(1, -1))
    print("Loaded embeddings")
    # concatenate embeddings
    embeddings = np.concatenate(embeddings, axis=0)
    print(embeddings.shape)
    print("Concatenated embeddings.")
    return embeddings


def omegafold_embeddings(path = None):
    # load embeddings
    if path is None:
        path_to_embeddings = '/users/anair27/data/DiffFolder/DiffFolder/embeddings/all256_embeddings'
    else:
        path_to_embeddings = path
    # load embeddings
    embeddings = []
    for folder in tqdm(os.listdir(path_to_embeddings)[:20]):
        for file in os.listdir(os.path.join(path_to_embeddings, folder)):
            if file.endswith('.npz'):
                try:
                    embedding = np.load(os.path.join(path_to_embeddings, folder, file))
                except Exception as e:
                    print(f"Error loading {file}")
                    continue
                # keys = embedding.keys()
                embedding = embedding['node_repr']
                # print(embedding.shape)
                # collapse by taking average across residues
                embedding = np.mean(embedding, axis=0)
                # print(embedding.shape)
                embeddings.append(embedding.reshape(1, -1))
    print("Loaded embeddings")
    # concatenate embeddings
    embeddings = np.concatenate(embeddings, axis=0)
    # print(embeddings.shape)
    print(f"Concatenated embeddings, shape {embeddings.shape}")
    return embeddings

def parse_args(args):
    parser = argparse.ArgumentParser(description="UMAP visualization of embeddings")
    parser.add_argument('-e', '--embeddings', type=str, default='omegafold', help='omegafold or esm')
    parser.add_argument('-t', '--technique', type=str, default='umap', help='umap or kmeans')
    return parser.parse_args(args)

def main(args):
    args = parse_args(args)

    if args.embeddings == 'omegafold':
        embeddings1 = omegafold_embeddings('/users/anair27/data/DiffFolder/DiffFolder/embeddings/omega_apo_embeddings')
        print("Loaded apo embeddings.")
        embeddings2 = omegafold_embeddings('/users/anair27/data/DiffFolder/DiffFolder/embeddings/omega_cameo_embeddings')
        print("Loaded cameo embeddings.")
        embeddings3 = omegafold_embeddings('/users/anair27/data/DiffFolder/DiffFolder/embeddings/omega_codnas_embeddings')
        print("Loaded codnas embeddings.")
        embeddings = np.concatenate([embeddings1, embeddings2, embeddings3], axis=0)
        # create labels
        labels = []
        for i in range(embeddings1.shape[0]):
            labels.append(1)
        for i in range(embeddings2.shape[0]):
            labels.append(2)
        for i in range(embeddings3.shape[0]):
            labels.append(3)
    elif args.embeddings == 'esm':
        embeddings = esm_embeddings()
    else:
        raise ValueError("Invalid embedding type. Choose omegafold or esm.")
    # normalize
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    embeddings = scaler.fit_transform(embeddings)
    print("Normalized data.")
    # if args.technique == 'umap':
    # reduce dimensionality
    reducer = umap.UMAP(n_components=3)
    embedding = reducer.fit_transform(embeddings)
    print("Reduced by UMAP.")
    # plot
    # sns.scatterplot(x=embedding[:,0], y=embedding[:,1], hue=labels)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(embedding[:,0], embedding[:,1], embedding[:,2], c=labels)
    # save plot
    plt.savefig(f'umap_embeddings_{args.embeddings}.png')

if __name__ == '__main__':
    main(sys.argv[1:])