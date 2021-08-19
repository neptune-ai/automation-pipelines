from model import *
from data import get_dataloader
import neptune.new as neptune
import os
import subprocess
import sys

project_name = 'common/pytorch-integration'

# Fetch project 
project = neptune.get_project(
    name = project_name,
    api_token=os.getenv("NEPTUNE_API_TOKEN")
)

# Champion
# Filter champion run
champion_runs_table_df = project.fetch_runs_table(tag='champion').to_pandas()
champion_run_id = champion_runs_table_df['sys/id'].values[0]

# Retrieve run in read mode
champion_run = neptune.init(
    project=project_name,
    api_token=os.getenv("NEPTUNE_API_TOKEN"),
    run = champion_run_id,
    mode = 'read-only'
)

# Challenger
# Filter challenger run
challenger_runs_table_df = project.fetch_runs_table(tag='challenger').to_pandas()
challenger_run_id = challenger_runs_table_df['sys/id'].values[0]

# Retrieve run
challenger_run = neptune.init(
    project=project_name,
    api_token=os.getenv("NEPTUNE_API_TOKEN"),
    run = challenger_run_id,
    mode = 'read-only'
)

# Fetch and download metadata from Neptune

# Feth dataset path
data_dir = champion_run['config/dataset/path'].fetch()

# Load and validation set
validloader = get_dataloader(data_dir, parameters['bs'])
images, labels = next(iter(validloader))

# Download and load model weights
champion_model = build_model(champion_run)
challenger_model = build_model(challenger_run)

# Run inference on validation set and get score
champion_score = get_model_score(champion_model, images,labels)
challenger_score = get_model_score(challenger_model, images, labels)

# Test challenger model score against champion model score
assert challenger_score >= champion_score, \
    f'The challenger model accuracy {round(challenger_score*100,2)} lower than threshold {round(champion_score*100,2)}%'

print(f'The challenger model with run_id = {challenger_run_id} has accuracy of {challenger_score*100}% that is greater than the current champion is promoted to production')

print("------------Evaluation test passed!!!------------")

print('Deploying Model...')

def deployment(new_best_run_id):
    # here you can write you deployment logic!
    print(f'Challenger model with Run ID {new_best_run_id} deployed successfully!!!')


deployment(challenger_run_id)
