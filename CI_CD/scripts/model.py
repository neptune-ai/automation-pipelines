import torch
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, input_sz, hidden_dim, n_classes):
        super(BaseModel, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_sz, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_classes),
        )

    def forward(self, input):
        x = input.view(-1, 32 * 32 * 3)
        return self.main(x)


def get_model(parameters, model_fname):

    model = BaseModel(
        parameters["input_sz"], parameters["input_sz"], parameters["n_classes"]
    ).to(parameters["device"])

    checkpoint = torch.load(model_fname, map_location=parameters['device'])
    model.load_state_dict(checkpoint)
    model.eval()

    return model

def build_model(run):
    parameters = run['config/hyperparameters'].fetch()
    parameters['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Downloading model weights
    if 'champion' in run['sys/tags'].fetch(): 
        model_fname = './champion_model.pth'
    else:
        model_fname = './challenger_model.pth'
    model_weights = run['io_files/artifacts/basemodel'].download(model_fname)

    # Loading model weights
    model = get_model(parameters, champion_model_fname)
    return model


def get_model_score(model, images, labels):
    output = model(images)
    _, preds = torch.max(output, dim=1)
    return (torch.sum(preds == labels)) / len(images)