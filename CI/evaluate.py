from model import *
from data import get_dataloader
import neptune.new as neptune
import os
import random

project_name = 'common/pytorch-integration'

# fetch project 
project = neptune.get_project(
    name = project_name,
    api_token=os.getenv("NEPTUNE_API_TOKEN"))

# filter in-prod run
runs_table_df = project.fetch_runs_table(tag='in-prod').to_pandas()
run_id = runs_table_df['sys/id'].values[0]

# retrieve run in read mode
run = neptune.init(
    project=project_name,
    api_token=os.getenv("NEPTUNE_API_TOKEN"),
    run = run_id,
    mode = 'read-only'
)

# fetching and downloading data from Neptune
parameters = run['config/hyperparameters'].fetch()
parameters['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_dir = run['config/dataset/path'].fetch()

model_fname = 'model.pth'
model_weights = run['io_files/artifacts/basemodel'].download(f'./{model_fname}')

# Load model checkpoint and validation set
model = get_model(parameters, model_fname)
validloader = get_dataloader(data_dir, parameters['bs'])
images, labels = next(iter(validloader))

# Run inference on validation set
output = model(images)
_, preds = torch.max(output, dim=1)

# Get Metric
acc = (torch.sum(preds == labels)) / len(images)

# Threshold
threshold = round(random.uniform(0.30, 0.65), 2)

# Test metric against threshold
assert acc >= threshold, f'Model accuracy {acc*100}%  is lower than threshold {threshold*100}%'

print(f'Model accuracy {acc*100}%  is higer than threshold {threshold*100}%')
print('Test Passed!!!')


