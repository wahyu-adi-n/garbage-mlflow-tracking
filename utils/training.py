import os
import shutil
import time
import torch
import mlflow
import mlflow.pytorch
from mlflow import MlflowClient
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
          model_dir,
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

    with mlflow.start_run(experiment_id=experiment_id) as run:
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

            if epoch == 0:
                shutil.rmtree(model_dir, ignore_errors=True)
                os.makedirs(model_dir, exist_ok=False)

                with open(f"{model_dir}/train_logs.csv", 'w', newline='\n', encoding='utf-8') as file:
                    file.write("train_loss,train_acc,val_loss,val_acc\n")

            with open(f"{model_dir}/train_logs.csv", 'a', newline='\n', encoding='utf-8') as file:
                file.write(f'{train_loss:.4f},{train_accuracy:.4f},{val_loss:.4f},{val_accuracy:.4f}\n')
      
        end_time = timer()
        test_loss, test_accuracy = val_step(model=model,
                                            dataloader=test_dataloader,
                                            loss_fn=loss_fn,
                                            device=device)
        mlflow.log_metrics({
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "time": end_time - start_time
        })
        mlflow.log_params(parameters)

        mlflow.pytorch.log_state_dict(model.state_dict(),'best_state_dict')
        mlflow.pytorch.log_model(model, 'final_model')
        scripted_pytorch_model = torch.jit.script(model)  
        mlflow.pytorch.log_model(scripted_pytorch_model, "scripted_model") 

        mlflow.log_artifact(f'{model_dir}/train_logs.csv')

        mlflow.end_run()

    # Fetch the logged model artifacts  
    print("run_id: {}".format(run.info.run_id))  
    for artifact_path in ["final_model/data", "scripted_model/data"]:  
        artifacts = [f.path for f in MlflowClient().list_artifacts(run.info.run_id,  
                    artifact_path)]  
        print("artifacts: {}".format(artifacts)) 