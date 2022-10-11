import os
import torch
from torchinfo import summary
from torch.optim import Adam
from models.GarbageEffNetModelV0 import GarbageEffNetModelV0
from utils.training import *
from data.data_lib import *

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    create_dir_extract()
    # split_data = split_data('data/dataset/garbage', 'data/dataset/output')

    train_dir = 'data/dataset/output/train'
    val_dir = 'data/dataset/output/val'
    test_dir = 'data/dataset/output/test'

    model = GarbageEffNetModelV0()
    summary(model)

    # hyperparameter
    batch_size = 16
    test_batch_size = 16
    num_epochs = 10
    learning_rate = 1e-4
    loss_fn = torch.nn.CrossEntropyLoss
    num_cpu_workers = os.cpu_count()
    optimizer = Adam(params=model.parameters(), lr=learning_rate)
    experiment_name = f"experiment_{model.model_backbone}_1"

    train_dl, val_dl, test_dl = dataloaders(train_dir=train_dir,
                                            val_dir=val_dir,
                                            test_dir=test_dir,
                                            batch_size=batch_size,
                                            test_batch_size=test_batch_size,
                                            num_cpu_workers=num_cpu_workers)

    parameters = {
        'dataset_version': 'v1.0',
        'device': device,
        'model': model.model_backbone,
        'num_classes': model.num_classes,
        'num_cpu_workers': num_cpu_workers,
        'batch_size': batch_size,
        'test_batch_size': test_batch_size,
        'epochs': num_epochs,
        'loss_fn': loss_fn,
        'optimizer_name': 'Adam',
        'amsgrad': optimizer.param_groups[0]['amsgrad'],
        'betas': optimizer.param_groups[0]['betas'],
        'capturable': optimizer.param_groups[0]['capturable'],
        'eps': optimizer.param_groups[0]['eps'],
        'maximize': optimizer.param_groups[0]['maximize'],
        'weight_decay': optimizer.param_groups[0]['weight_decay'],
        'learning_rate': optimizer.param_groups[0]['lr']
    }

    train(model, train_dl, val_dl, test_dl,
          model_path="track/model",
          experiment=experiment_name,
          device=device,
          epochs=parameters['epochs'],
          optimizer=optimizer,
          loss_fn=loss_fn,
          parameters=parameters)

    print("Training End!")
