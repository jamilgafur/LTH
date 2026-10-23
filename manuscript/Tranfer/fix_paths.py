with open('adversarial_analysis.py', 'r') as f:
    lines = f.readlines()

with open('adversarial_analysis.py', 'w') as f:
    skip = False
    for line in lines:
        if line.startswith('def get_checkpoint_path'):
            skip = True
            f.write('''def get_checkpoint_path(model: str, dataset: str, kind: str):
    import glob
    import os
    # Find directories matching this model and dataset
    pattern = f"{model}_{dataset}_*"
    dirs = glob.glob(pattern)
    if not dirs:
        return None
        
    # Prioritize epochs100_pretrain300 if it exists
    target_dir = dirs[0]
    for d in dirs:
        if "epochs100_pretrain300" in d:
            target_dir = d
            break
            
    base_dir = target_dir
    filename = "final_JF_Control.pt" if kind == "Original" else "final_JF_Dynamic_Region_All_Combined.pt"
    full_path = os.path.abspath(os.path.join(base_dir, "checkpoints", filename))
    
    if os.path.exists(full_path):
        return full_path
    return None

def discover_checkpoints():
    entries = []
    for model in MODEL_ORDER:
        for dataset in DATASET_ORDER:
            for kind in ("Finetuned", "Original"):
                path = get_checkpoint_path(model, dataset, kind)
                if path:
                    entries.append((model, dataset, kind, path))
    return entries

''')
        elif line.startswith('def load_model('):
            skip = False
        
        if not skip:
            f.write(line)

print("Successfully updated the path logic in adversarial_analysis.py!")
