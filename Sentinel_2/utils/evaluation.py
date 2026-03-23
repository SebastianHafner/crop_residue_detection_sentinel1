import torch
from torch.utils import data as torch_data
import wandb
from datasets import crop_datasets
from utils import loss_functions
import numpy as np

EPS = 10e-05


def classification(model, cfg, device, run_type: str, epoch: float, step: int, max_samples: int = None):
    ds = crop_datasets.CropDataset(cfg, run_type, no_augmentations=True)

    criterion = loss_functions.get_criterion(cfg.trainer.loss)

    model.to(device)
    model.eval()

    dataloader = torch_data.DataLoader(ds, batch_size=cfg.trainer.batch_size, num_workers=0, shuffle=False,
                                       drop_last=False)

    loss_vals = []
    all_predictions = []
    all_labels = []

    for step, item in enumerate(dataloader):
        x, mask = item['x'].to(device), item['msk'].to(device)

        with torch.no_grad():
            y_hat = model(x, mask).squeeze(1)

            y = item['y'].to(device)
            loss = criterion(y_hat, y)

            # Calculate predictions for accuracy
            predicted = torch.argmax(y_hat, dim=1)

            # Store predictions and labels
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

        loss_vals.append(loss.item())

    # Calculate metrics
    loss_mean = np.mean(loss_vals)

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    accuracy = (all_predictions == all_labels).mean()

    # Log both metrics
    wandb.log({
        f'{run_type} loss_epoch': loss_mean,
        f'{run_type} accuracy_epoch': accuracy,
        'step': step,
        'epoch': epoch,
    })

    print(f'{run_type} - Epoch {epoch:.2f}: Loss={loss_mean:.4f}, Accuracy={accuracy * 100:.2f}%')

    return loss_mean