import os
import torch
import pickle
import glob
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import seaborn as sns
from pyPrune.utils import plot_loss_accuracy_sparsity, set_seed, CustomLambdaLR


from collections import defaultdict
import numpy as np
import numpy as np
from collections import defaultdict

import numpy as np
from collections import defaultdict

#############################################
# Utility Functions for Loading Metrics
#############################################

def load_metric(pkl_file):
    """Load metrics from a pickle file."""
    with open(pkl_file, 'rb') as file:
        return pickle.load(file)

def load_json(json_file):
    """Load metrics from a JSON checkpoint file."""
    print(f"Processing: {json_file}")
    with open(json_file, 'r') as file:
        return json.load(file)


from collections import defaultdict
import numpy as np
import pprint
from collections import defaultdict
import numpy as np

from collections import defaultdict
import numpy as np


#############################################
# Neuron Similarity Analysis Functions
#############################################

def process_neuron_similarity(neuron_similarity_dir, model_name):

    files_found = glob.glob(os.path.join(neuron_similarity_dir, "*.pkl"))
    print(f"Files found for neuron similarity: {files_found}")
 
    # Extract checkpoint name from the parent folder of neuron_similarity_dir
    checkpoint_dir = os.path.dirname(neuron_similarity_dir)
    checkpoint_name = os.path.basename(os.path.normpath(checkpoint_dir))

    # --- Process the first file for per-file plots (as before) ---
    with open(files_found[0], 'rb') as f:
        pruner = pickle.load(f)

    # Layers activations_step.keys()
    # dict_keys(['conv1', 'conv2', 'fc1', 'fc2'])
    # Activations
    activations_step = pruner.activations_step

    # 1. Compute cosine similarity between consecutive activation steps for each layer.
    maximum_similarities = defaultdict(list)
    for layer, data in activations_step.items():
        data_sorted = sorted(data, key=lambda x: x[0])
        for i in range(len(data_sorted)):
            step = data_sorted[i][0]
            activations = data_sorted[i][1].to('cpu',dtype=torch.float64).detach()+1e-10
            activations = activations.view(activations.size(1),-1)
            norm_activations = torch.norm(activations, dim=1, keepdim=True)
            normalized_activations = activations / norm_activations 
            similarity_matrix = normalized_activations @ normalized_activations.T
            similarity_matrix = torch.abs(similarity_matrix)
            maximum_similarity, _ = torch.max(similarity_matrix - torch.diag(torch.ones(similarity_matrix.shape[0])),dim=1)
            maximum_similarities[layer].append((step, maximum_similarity))
    ######
    save_dir_prev = f"./plots/{model_name}/{checkpoint_name}/activation_similarity/"
    os.makedirs(save_dir_prev, exist_ok=True)
    fig,axs = plt.subplots(1,3,figsize = (9,3))
    steps_to_plot = [0,3,14] ## no pruning, 50% sparsity, and 95% sparsity
    sparsities = ['0', '50', '95']
    plot_data = {}
    for i in steps_to_plot:
        plot_data[i] = []
    for layer, sim_data in maximum_similarities.items():
        sim_data_sorted = sorted(sim_data, key=lambda x: x[0])
        steps_layer = [x[0] for x in sim_data_sorted]
        sims = [x[1] for x in sim_data_sorted]
        for i in steps_to_plot:
            plot_data[i].append(sims[i])
    j=0
    for i in steps_to_plot:
        data = torch.cat(plot_data[i])
        axs[j].hist(data,bins = len(data)//10)
        j+=1
    plt.savefig(os.path.join(save_dir_prev, "maximum_cosine_similarity.png"))
    plt.close()

    print(f"Neuron similarity post-processing complete for model: {model_name}, checkpoint: {checkpoint_name}")

def main():
    """Main function to execute the entire post-processing pipeline."""
    model_names = ["LeNet", "ResNet20", "Vgg16"]
    # The glob pattern is used to select the checkpoint directories.    
    for model_name in model_names:
        checkpoint_glob = f"/projects/modularai/jgafur/results/*{model_name}*"
        for output_dir in glob.glob(checkpoint_glob):
            clear_memory()
            print(output_dir)
            neuron_similarity_dir = os.path.join(output_dir, "neuron_similarity")
            process_neuron_similarity(neuron_similarity_dir, '')
            
def clear_memory():
    import gc
    gc.collect()
    torch.cuda.empty_cache()
           
if __name__ == "__main__":
    main()