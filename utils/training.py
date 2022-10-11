import time
import torch
import mlflow
from tqdm.auto import tqdm
from timeit import default_timer as timer


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               optimizer: torch.optim.Optimizer,
               device,
               loss_fn: torch.nn.Module,
               ):
    model.train()
    train_loss, train_acc = 0, 0

    for _, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn()(y_pred, y)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def val_step(model: torch.nn.Module,
             dataloader: torch.utils.data.DataLoader,
             device,
             loss_fn: torch.nn.Module,
             ):
    model.eval()
    val_loss, val_acc = 0, 0
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            val_pred_logits = model(X)
            loss = loss_fn()(val_pred_logits, y)
            val_loss += loss.item()
            val_pred_labels = val_pred_logits.argmax(dim=1)
            val_acc += ((val_pred_labels == y).sum().item() /
                        len(val_pred_labels))
    val_loss = val_loss / len(dataloader)
    val_acc = val_acc / len(dataloader)
    return val_loss, val_acc


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          val_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          model_path,
          experiment,
          device,
          epochs: int,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = torch.nn.CrossEntropyLoss(),
          parameters=None):
    model.to(device)

    try:
        experiment_id = mlflow.create_experiment(experiment)
    except:
        current_experiment = dict(mlflow.get_experiment_by_name(experiment))
        experiment_id = current_experiment['experiment_id']

    with mlflow.start_run(experiment_id=experiment_id):
        start_time = timer()
        t0 = time.time()
        for epoch in tqdm(range(epochs)):
            train_loss, train_accuracy = train_step(model=model,
                                                    dataloader=train_dataloader,
                                                    loss_fn=loss_fn,
                                                    optimizer=optimizer,
                                                    device=device)
            val_loss, val_accuracy = val_step(model=model,
                                              dataloader=val_dataloader,
                                              loss_fn=loss_fn,
                                              device=device)
            print(
                f"Epoch: {epoch+1}/{epochs} - "
                f"train_loss: {train_loss:.4f} - "
                f"train_accuracy: {train_accuracy:.4f} - "
                f"val_loss: {val_loss:.4f} - "
                f"val_accuracy: {val_accuracy:.4f} - "
                f"time: {time.time() - t0:.0f}s"
            )
            mlflow.log_metrics({
                'train_loss': train_loss,
                'train_acc': train_accuracy,
                'val_loss': val_loss,
                'val_acc': val_accuracy,
            }, step=epoch)
            mlflow.pytorch.log_state_dict(model.state_dict(),
                                          f"{model_path}/weights_epoch_{epoch}.pt")
        end_time = timer()
        test_loss, test_accuracy = val_step(model=model,
                                            dataloader=test_dataloader,
                                            loss_fn=loss_fn,
                                            device=device)
        mlflow.pytorch.log_model(model, "model")
        mlflow.log_metrics({
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "time": end_time - start_time
        })

        mlflow.log_params(parameters)
        mlflow.end_run()
