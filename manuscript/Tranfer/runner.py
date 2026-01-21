import glob



# go into every folder in baseline_models
# get the cka_scores.json file 
# parse the model name, dataset name, epochs from the folder name
# call qsub submit_job.pbs with the appropriate arguments

if __name__ == "__main__":
    folders = glob.glob("baseline*/*")
    print(folders)
    for folder in folders:
        model_name = folder.split("/")[1].split("_")[0]
        dataset_name = folder.split("/")[1].split("_")[1]
        epochs = folder[folder.index("break")+5:folder.rindex("_")]
        print(f"Submitting job for Model: {model_name}, Dataset: {dataset_name}, Epochs: {epochs}")
        # load in the cka_scores.json file from the folder
        cka_file = glob.glob(f"{folder}/cka_scores.json")[0]
        print(f"Using CKA file: {cka_file}")
        # read in the collapse start and end layers from the cka_scores.json file
        import json
        with open(cka_file, "r") as f:
            cka_data = json.load(f)
        # for all layer 
        for layer_info in cka_data["layer_names"]:
            collapse_start = layer_info[1]
            collapse_end = layer_info[2]
            print(f"Collapse from {collapse_start} to {collapse_end}")

            # submit job
            import os
            command = f"qsub -v MODEL={model_name},DATASET={dataset_name},EPOCHS={epochs},COLLAPSE_START={collapse_start},COLLAPSE_END={collapse_end} submit_job.pbs"
            print(f"Executing command: {command}")
            print(f"python main_2.py --model {model_name} --dataset {dataset_name} --epochs {epochs} --collapse_start {collapse_start} --collapse_end {collapse_end}")
            os.system(command)