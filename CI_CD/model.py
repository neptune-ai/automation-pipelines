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