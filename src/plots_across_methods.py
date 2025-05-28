#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import argparse
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# custom functions #
import utils


# In[2]:


# If running this interactively, can specify jupyter_args here for argparser to use
if utils.is_interactive():
    methods = "pretrained_subj01_40sess_hypatia_vd2,method2,method3"
    data_path = "/weka/proj-medarc/shared/mindeyev2_dataset"
    criteria = "all"
    print("Methods:", methods)

    jupyter_args = f"--methods={methods} --data_path={data_path} --criteria={criteria}"
    print(jupyter_args)
    jupyter_args = jupyter_args.split()
    
    from IPython.display import clear_output  # Function to clear print outputs in cell
    get_ipython().run_line_magic('load_ext', 'autoreload')
    # This allows you to change functions in models.py or utils.py and have this notebook automatically update with your revisions
    get_ipython().run_line_magic('autoreload', '2')


# In[3]:


parser = argparse.ArgumentParser(description="Compare methods based on metrics")
parser.add_argument(
    "--methods", type=str, required=True,
    help="Comma-separated list of method names to compare",
)
parser.add_argument(
    "--data_path", type=str, default="../dataset",
    help="Path to where metrics CSV files are stored",
)
parser.add_argument(
    "--columns_to_normalize", type=str, default='PixCorr,SSIM,AlexNet(2),AlexNet(5),InceptionV3,CLIP,EffNet-B,SwAV,Brain Corr. nsd_general,Brain Corr. EVC,Brain Corr. higher_vis',
    help="Comma-separated list of metric columns to normalize",
)
parser.add_argument(
    "--criteria", type=str, default="all",
    help="Criteria to use for averaging metrics. 'all' or comma-separated list of metrics",
)
parser.add_argument(
    "--output_path", type=str, default="../figs",
    help="Path to save the output scatter plot",
)
parser.add_argument(
    "--output_file", type=str, default="method_scatter_plot",
    help="Filename to save the output scatter plot",
)
parser.add_argument(
    "--gradient",action=argparse.BooleanOptionalAction,default=False,
)
parser.add_argument(
    "--stimtype", type=str, default='all', choices=['all', 'simple', 'complex'],
    help="Type of stimulus to plot across",
)
parser.add_argument(
    "--subjs", type=str, default='-1',
    help="Comma-separated list of subject indices to average over (e.g., '1,2,3'). Use '-1' for default behavior.",
)
if utils.is_interactive():
    args = parser.parse_args(jupyter_args)
else:
    args = parser.parse_args()

# Create global variables without the args prefix
for attribute_name in vars(args).keys():
    globals()[attribute_name] = getattr(args, attribute_name)

criteria = criteria.replace("*", " ")
methods = [method.strip() for method in methods.split(",")]
columns_to_normalize = [col.strip() for col in columns_to_normalize.split(",")]
subjects = [int(s) for s in subjs.split(",")]
# Seed all random functions
seed = 42  # Set your seed value
utils.seed_everything(seed)


# # Collate across subjects if provided

# In[ ]:


if subjs == '-1':
    # Default behavior: use the methods as provided
    processed_methods = methods.copy()
else:
    # Modify methods by replacing 'subj0X' with 'subj0{S}' for each S in subjects
    processed_methods = []
    for method in methods:
        if "subj0" in method:
            base_method = method.split("subj0")[0]
            suffix = method.split("subj0")[1]  # e.g., "1_40sess_hypatia_vd2"
            for S in subjects:
                subj_str = f"subj0{S}"
                new_method = f"{base_method}{subj_str}{suffix[len(str(S)):]}"

                # **Check if the new_method exists in the methods list**
                # Construct the expected metrics file paths for all modes
                for mode in ['imagery', 'vision']:
                    metrics_file = f"tables/{new_method}_all_recons_{mode}.csv"
                    if not os.path.exists(metrics_file):
                        print(f"Error: Metrics file for method '{new_method}' and mode '{mode}' not found at {metrics_file}")
                        sys.exit(1)
                processed_methods.append(new_method)
        else:
            # Method does not contain 'subj0X', include it as is
            processed_methods.append(method)

    if not processed_methods:
        print("Error: No processed methods found after applying subjects.")
        sys.exit(1)

    # Update the methods list to the processed methods
    methods = processed_methods


# # Loading tables

# In[4]:


# Loading tables for both modes
modes = ['imagery', 'vision']
dfs = []
for method in methods:
    for mode in modes:
        metrics_file = f"tables/{method}_all_recons_{mode}.csv"
        if not os.path.exists(metrics_file):
            print(f"Metrics file for method '{method}' and mode '{mode}' not found at {metrics_file}")
            sys.exit(1)
        df = pd.read_csv(metrics_file, sep="\t")
        df['method'] = method
        df['mode'] = mode
        dfs.append(df)

# **If averaging over subjects, group by the base method names**
if subjs != '-1':
    # Extract base method names by removing 'subj0X' parts
    def get_base_method(method_name):
        import re
        return re.sub(r'subj0\d+', 'subj0X', method_name)
    methods = list(dict.fromkeys(get_base_method(method) for method in methods))
    all_metrics = pd.concat(dfs, ignore_index=True)
    all_metrics['base_method'] = all_metrics['method'].apply(get_base_method)
    # Group by base_method and mode, then average the metrics
    grouped_metrics = all_metrics.groupby(['base_method', 'mode', 'sample', 'repetition'])[columns_to_normalize].mean().reset_index()

    # Replace 'base_method' with the actual base method name (with 'subj0X')
    grouped_metrics['method'] = grouped_metrics['base_method']
    grouped_metrics = grouped_metrics.drop(columns=['base_method'])

    all_metrics = grouped_metrics
else:
    all_metrics = pd.concat(dfs, ignore_index=True)
    
if stimtype == 'simple':
    all_metrics = all_metrics[all_metrics['sample'] < 6]
elif stimtype == 'complex':
    all_metrics = all_metrics[all_metrics['sample'] >= 6]


# In[6]:


# Check that columns_to_normalize exist in DataFrame
missing_columns = [col for col in columns_to_normalize if col not in all_metrics.columns]
if missing_columns:
    print(f"Error: The following columns to normalize are missing from the data: {missing_columns}")
    sys.exit(1)

all_metrics['EffNet-B'] = 1 - all_metrics['EffNet-B']
all_metrics['SwAV'] = 1 - all_metrics['SwAV']

# Normalize specified columns across the entire dataset
scaler = MinMaxScaler()
all_metrics[columns_to_normalize] = scaler.fit_transform(all_metrics[columns_to_normalize])
# Determine metrics to average
if criteria == 'all':
    metrics_to_average = columns_to_normalize
else:
    metrics_to_average = [col.strip() for col in criteria.split(",")]

# Check that metrics_to_average exist in DataFrame
missing_columns = [col for col in metrics_to_average if col not in all_metrics.columns]
if missing_columns:
    print(f"Error: The following metrics are missing from the data: {missing_columns}")
    sys.exit(1)

# Ensure 'method' is a categorical variable with the specified order
all_metrics['method'] = pd.Categorical(all_metrics['method'], categories=methods, ordered=True)
# Compute average normalized metric performance per method and mode
method_mode_scores = all_metrics.groupby(['method', 'mode'])[metrics_to_average].mean()
method_mode_scores['average_score'] = method_mode_scores.mean(axis=1)

# Create a pivot table with methods as index and modes as columns
average_scores = method_mode_scores['average_score'].unstack()

# Reindex the pivot table to match the original 'methods' order
average_scores = average_scores.reindex(methods)

# Ensure that both 'imagery' and 'vision' modes are present for all methods
average_scores = average_scores.dropna()


# In[ ]:


# Plot scatter plot
plt.figure(figsize=(15, 10))
ax = plt.subplot(111)
if gradient:
    cmap = plt.cm.viridis  # You can change this to 'plasma', 'inferno', 'magma', 'hsv', etc.
    # Generate a list of colors using the colormap
    colors = cmap(np.linspace(0, 1, len(average_scores.index)))
else:
    colors = plt.cm.tab10.colors  if len(average_scores.index) <= 10 else plt.cm.tab20.colors
for i, method in enumerate(average_scores.index):
    x = average_scores.loc[method, 'vision']
    y = average_scores.loc[method, 'imagery']
    plt.scatter(x, y, color=colors[i % len(colors)], label=method, s=100)
    
highest_method = average_scores['imagery'].idxmax()
print(f"The method with the highest y coordinate for imagery is: {highest_method}")

plt.xlabel('Vision Performance')
plt.ylabel('Imagery Performance')
plt.title(f'Imagery vs. Vision Performance\n{output_file}')
box = ax.get_position()
ax.set_position([box.x0 - 0.06, box.y0, box.width * 0.6, box.height])
# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(True)
# plt.tight_layout()
output_file = os.path.join(output_path, f'{output_file}.png')
print(f"Saving scatter plot to {output_file}")
plt.savefig(output_file, dpi=300)
plt.show()


# ### 
