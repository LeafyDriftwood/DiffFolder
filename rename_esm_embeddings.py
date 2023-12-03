# for each npz file in the directory, load the npz file 
# and get the embedding array
# replace the npz file with a new npz file that has the key
# 'node repr' and the value of the embedding array

import os
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

directory = '/users/anair27/data/DiffFolder/DiffFolder/esm_embeddings/esm_256_all'
def rename_embedding(filename):
    path_to_file = os.path.join(directory, filename)
    # if filename is a npz file
    if not filename.endswith('.npz'):
        return
    embedding = np.load(path_to_file)
    if 'node_repr' in embedding.keys():
        embedding_array = embedding['node_repr']
    else:
        embedding_array = embedding['arr_0']
    # save as npz with key 'node repr'
    # print(f"Saving {filename}.")
    np.savez(path_to_file, **{'node_repr': embedding_array, 'edge_repr': np.zeros((embedding_array.shape[0], embedding_array.shape[0], 128))})

def main():
    # parallelize across filenames
    for file in tqdm(os.listdir(directory)):
        rename_embedding(file)
if __name__ == '__main__':
    main()