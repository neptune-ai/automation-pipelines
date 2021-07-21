from model import *
from data import get_dataloader
import neptune.new as neptune
import os
import torchdrift

project_name = 'common/pytorch-integration'

# Fetch project 
project = neptune.get_project(
    name = project_name,
    api_token=os.getenv("NEPTUNE_API_TOKEN"))

# In-prod
# Filter in-prod run
prod_runs_table_df = project.fetch_runs_table(tag='in-prod').to_pandas()
prod_run_id = prod_runs_table_df['sys/id'].values[0]

# Retrieve run in read mode
prod_run = neptune.init(
    project=project_name,
    api_token='ANONYMOUS',
    run = prod_run_id,
    mode = 'read-only'
)

# Adding Guassian blur to simulate drift
severity =  prod_run['config/dataset/drift_severity'].fetch()
def drift_data_function(x: torch.Tensor, severity):
    return torchdrift.data.functional.gaussian_blur(x, severity=severity)

# Fetching and downloading data from Neptune
parameters = prod_run['config/hyperparameters'].fetch()
parameters['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_dir = prod_run['config/dataset/path'].fetch()

# In-prod model weights
prod_model_fname = './prod_model.pth'
prod_model_weights = prod_run['io_files/artifacts/basemodel'].download(prod_model_fname)

# Load model checkpoints 
prod_model = get_model(parameters, prod_model_fname)

# Load data
validloader = get_dataloader(data_dir, parameters['bs'])
images, labels = next(iter(validloader))

# Creatinng drifted data
drifted_images = drift_data_function(images, severity)

# Run in-prod model inference on live drifted data
prod_output = prod_model(drifted_images)
_, prod_preds = torch.max(prod_output, dim=1)

prod_score = (torch.sum(prod_preds == labels)) / len(drifted_images)

validation_score = prod_run['validation/best_acc'].fetch()

# Test production model score agaisnt threshold
assert prod_score >= validation_score, \
f'Warning: Production model accuracy {round(prod_score.item()*100,2)}% is lower than normal. Running retrain.py'
print('Test Passed!!!')
print('No drift detected in the production model')