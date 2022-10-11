import os
import torch
from torchinfo import summary
from pathlib import Path
from models.GarbageEffNetModelV0 import GarbageEffNetModelV0
from utils.training import *
from data.data_lib import *
from torch.optim import Adam

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    create_data = create_dir_extract()
    split_data = split_data('data/dataset/garbage', 'data/dataset/output')

    # train-val-test dir
    train_dir = 'data/dataset/output/train'
    val_dir = 'data/dataset/output/val'
    test_dir = 'data/dataset/output/test'

    # hyperparameter
    batch_size = 32
    test_batch_size = 16
    model = GarbageEffNetModelV0()
    summary(model)
    num_epochs = 10
    learning_rate = 1e-4
    criterion = torch.nn.CrossEntropyLoss
    optimizer = Adam(params=model.parameters(), lr=learning_rate)
    experiment_name = f"{model.model_backbone}_experiment_0"
    num_cpu_workers = os.cpu_count()

    train_dl, val_dl, test_dl = dataloaders(train_dir,
                                            val_dir,
                                            test_dir,
                                            batch_size=batch_size,
                                            test_batch_size=test_batch_size)

    parameters = {
        'dataset_version': 'v1.0',
        'epochs': num_epochs,
        'lr': learning_rate,
        'criterion': criterion,
        'model': model.model_backbone,
        'num_classes': model.num_classes,
        'optimizer_name': 'Adam',
        'batch_size': batch_size,
        'test_batch_size': test_batch_size,
        'device': device,
        'num_cpu_workers': num_cpu_workers
    }

    train(model, train_dl, val_dl, test_dl,
          model_path="track/model",
          experiment=experiment_name,
          device=device,
          epochs=parameters['epochs'],
          optimizer=optimizer,
          loss_fn=criterion,
          parameters=parameters)
    print("Training End!")
