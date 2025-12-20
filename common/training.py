import torch
from vae_ludo.base_vae import VAEOutput

#TODO this could be converted to a overridden method in a Trainer class
def convert_prediction_output(pred):
    if isinstance(pred, tuple):
        pred = pred.output
    elif isinstance(pred, VAEOutput):
        pred = pred.x_recon
    return pred

def evaluate_tm(model, data_loader, metric, device, reconstruct=False):
    model.eval()
    metric.reset()
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            if reconstruct:
                target = X_batch
            else:
                target = y_batch
            pred = model(X_batch)
            pred = convert_prediction_output(pred)
            metric.update(pred, target)
    return metric.compute()

def train(model, optimizer, loss_fn, metric, train_loader, valid_loader,
          n_epochs, device, patience=2, factor=0.5, epoch_callback=None, reconstruct=False):
    
    # Reduce learning rate when a metric has stopped improving for a 'patience' number of epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=patience, factor=factor)

    history = {"train_losses": [], "train_metrics": [], "valid_metrics": []}
    
    for epoch in range(n_epochs):
        total_loss = 0.0
        metric.reset()
        model.train()
        if epoch_callback is not None:
            epoch_callback(model, epoch)

        # - For each batch
        for index, (X_batch, y_batch) in enumerate(train_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            # - Compute output from the model
            pred = model(X_batch)
            # choose the right target for current task
            if reconstruct:
                target = X_batch
            else:
                target = y_batch
            # - Compute loss and backpropagate it
            loss = loss_fn(pred, target)
            total_loss += loss.item()
            loss.backward()
            # clip gradient norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            optimizer.zero_grad()

            # - compute train metric
            pred = convert_prediction_output(pred)
            metric.update(pred, target)
            train_metric = metric.compute().item()
            print(f"\rBatch {index + 1}/{len(train_loader)}", end="")
            print(f", loss={total_loss/(index+1):.4f}", end="")
            print(f", {train_metric=:.3f}", end="")
        
        # - On epoch end
        history["train_losses"].append(total_loss / len(train_loader))
        history["train_metrics"].append(train_metric)
        # compute validation metric
        val_metric = evaluate_tm(model, valid_loader, metric, device, reconstruct).item()
        history["valid_metrics"].append(val_metric)
        # check for early stopping (TODO?)
        scheduler.step(val_metric)
        print(f"\rEpoch {epoch + 1}/{n_epochs},                      "
              f"train loss: {history['train_losses'][-1]:.4f}, "
              f"train metric: {history['train_metrics'][-1]:.3}, "
              f"valid metric: {history['valid_metrics'][-1]:.3}")
    return history