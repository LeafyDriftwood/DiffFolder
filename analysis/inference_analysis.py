# import libaries
import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

# create df for a split
def create_df(filename):

    # Lists to store extracted data
    structure_names = []
    metrics_data = {'rmsd': [], 'tm': [], 'gdt_ts': [], 'gdt_ha': [], 'lddt': []}

    # line matching expression
    line_pattern = re.compile(r'(\S+\.pdb).*?rmsd\': (\d+\.\d+).*?tm\': (\d+\.\d+).*?gdt_ts\': (\d+\.\d+).*?gdt_ha\': (\d+\.\d+).*?lddt\': (\d+\.\d+)')

    # read file line by line and fill out df
    with open(filename, 'r') as file:
        for line in file:
            # Check if the line contains structure name and metrics
            match = line_pattern.search(line)
            if match:
                # Extract data from the matched groups
                structure_name, rmsd, tm, gdt_ts, gdt_ha, lddt = match.groups()
                
                # Append data to the lists
                structure_names.append(structure_name)
                metrics_data['rmsd'].append(float(rmsd))
                metrics_data['tm'].append(float(tm))
                metrics_data['gdt_ts'].append(float(gdt_ts))
                metrics_data['gdt_ha'].append(float(gdt_ha))
                metrics_data['lddt'].append(float(lddt))
    
    # create dataframe with metrics
    df = pd.DataFrame(metrics_data, index=structure_names)

    return df

# loop through file names and calculate averages
def calculate_metrics(model):

    # define out files
    BASE = "../bash_scripts/run_inference__"
    splits = ["apo", "cameo", "codnas"]

    # store metrics per split
    split_metrics = {}

    # loop through all splits
    for split in splits:
        # create df for each split
        filename = f"{BASE}{model}_{split}.out"
        df = create_df(filename)

        # Calculate the average of each metric
        average_metrics = df.median()

        # store split dfs
        split_metrics[split] = average_metrics
    
    # return metrics
    return split_metrics

# loop through all models and print metrics
def summarize_models(models):
    
    # loop through each model
    model_dict = {}
    for model in models:

        # calculate metrics
        metrics = calculate_metrics(model=model)
        model_dict[model] = metrics
    
    # # create separate dfs for each file
    file_dfs = {}

    # loop through models and files
    for model, file_metrics in model_dict.items():
        for file_name, metrics in file_metrics.items():
            if file_name not in file_dfs:
                file_dfs[file_name] = pd.DataFrame(columns=list(metrics.keys()))

            # create df for current model and add to file
            model_df = pd.DataFrame({**metrics}, index=[model])
            file_dfs[file_name] = pd.concat([file_dfs[file_name], model_df])

    # print all metrics for each file
    for key in file_dfs:
        print(f"==========={key}==========")
        print(file_dfs[key])


# define main function
def main():

    # define model types
    models  = ["omega", "prot", "ohe", "esm"]

    # calculate averages
    summarize_models(models)

if __name__ == "__main__":
    main()