import torch
from torchvision import transforms, datasets


def get_dataloader(data_dir, bs):
    data_tfms = {
        "val": transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    }

    validset = datasets.CIFAR10(
        data_dir, train=False, transform=data_tfms["val"], download=True
    )
    validloader = torch.utils.data.DataLoader(
        validset, batch_size=bs, num_workers=2
    )


    return validloader