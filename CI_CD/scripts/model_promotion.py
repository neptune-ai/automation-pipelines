from model import *
from data import get_dataloader
import neptune.new as neptune
import os

project_name = 'common/pytorch-integration'

# Fetch project 
project = neptune.get_project(
    name = project_name,
    api_token=os.getenv("NEPTUNE_API_TOKEN")
)

# In-prod
# Filter in-prod run
prod_runs_table_df = project.fetch_runs_table(tag='in-prod').to_pandas()
prod_run_id = prod_runs_table_df['sys/id'].values[0]

# Retrieve run in read mode
prod_run = neptune.init(
    project=project_name,
    api_token=os.getenv("NEPTUNE_API_TOKEN"),
    run = prod_run_id,
    mode = 'read-only'
)

# Stagging
# filter stagging run
stagging_runs_table_df = project.fetch_runs_table(tag='stagging').to_pandas()
stagging_run_id = stagging_runs_table_df['sys/id'].values[0]

# Retrieve run
stagging_run = neptune.init(
    project=project_name,
    api_token=os.getenv("NEPTUNE_API_TOKEN"),
    run = stagging_run_id,
    mode = 'read-only'
)

# Fetching and downloading data from Neptune
parameters = prod_run['config/hyperparameters'].fetch()
parameters['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_dir = prod_run['config/dataset/path'].fetch()

# In-prod model weights
prod_model_fname = './prod_model.pth'
prod_model_weights = prod_run['io_files/artifacts/basemodel'].download(prod_model_fname)

# Stagging model weights
stagging_model_fname = './stagging_model.pth'
stagging_model_weights = stagging_run['io_files/artifacts/basemodel'].download(stagging_model_fname)

# Load model checkpoints 
prod_model = get_model(parameters, prod_model_fname)
stagging_model = get_model(parameters, stagging_model_fname)

# Load and validation set
validloader = get_dataloader(data_dir, parameters['bs'])
images, labels = next(iter(validloader))

# Run in-prod model inference on validation set
prod_output = prod_model(images)
_, prod_preds = torch.max(prod_output, dim=1)

prod_score = (torch.sum(prod_preds == labels)) / len(images)

# Run in-prod model inference on validation set
stagging_output = stagging_model(images)
_, stagging_preds = torch.max(stagging_output, dim=1)

stagging_score = (torch.sum(stagging_preds == labels)) / len(images)

# Test stagging model score against production model score
assert stagging_score >= prod_score, \
    f'Staging model accuracy {round(stagging_score*100,2)} lower than threshold {round(prod_score*100,2)}%'

print(f'Staging model with run_id = {stagging_run_id} has accuracy of {stagging_score*100}% that is greater than the current production-model was promoted to production')



