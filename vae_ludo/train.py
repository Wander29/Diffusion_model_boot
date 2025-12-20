import numpy as np
import torch
from tqdm.auto import tqdm

from ludo_vae.utils import helpers as hel
from ludo_vae.models import base_vae as base
from ludo_vae.training import losses as ls
from ludo_vae.utils.types import Tensor, ModelClassType

def train(model:base.BaseVAE, x_tr, y_tr,
          optimizer, loss_fn, n_epochs, 
          batch_size, patience, x_val=torch.Tensor([]), y_val=torch.Tensor([]), 
          min_grad_norm=1e-6, permute_data=False, verbosity=1):
  """
  Trains the given model on the given dataset

  Args:
    @model (nn.Module) Model to train
    @x_tr (Tensor) Training data
    @optimizer (torch.optim.Optimizer)
    @n_epochs (int)
    @batch_size (int)
    @permute_data (bool) permute data at the start of each epoch
  """
  def do_train(batch_size=batch_size):
    # Setup training batches  
    N = x_tr.shape[0]
    if batch_size > N:
      batch_size = N
    n_batches = N // batch_size
    tr_samples_idx = np.arange(N)

    if permute_data:
      idx_perm = torch.randperm(x_tr.size(0))
      idx_batches = np.array_split(idx_perm, n_batches)
    else:
      idx_batches = np.array_split(tr_samples_idx, n_batches)
    
    tot_loss = 0
    tot_grad_norm = 0
    for n_upd, batch in enumerate(tqdm(idx_batches)):
      x_data = x_tr[batch].clone().detach()
      y_data = y_tr[batch].clone().detach()
      
      # clear gradients
      optimizer.zero_grad()

      # Forward
      result = model(x_data)
      # call the right loss based on overloading with the lambda functions
      # note: some parameters are ignored for some losses, this is a trick to have a common interface
      loss_result = loss_fn(x=x_data, y=y_data, out=result)
      loss = loss_result['loss']
      
      # Backward
      loss.backward()

      # gradient clipping (it returns the norm before clipping)
      grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 50.0)
      tot_grad_norm += grad_norm

      # Print info
      if verbosity > 1:
        if n_upd % 200 == 0:
          print_loss(loss_result, n_upd, batch_size, grad_norm)

      # Update model parameters
      optimizer.step()

      tot_loss += loss.detach().item()
    
    tot_loss /= n_batches
    tot_grad_norm /= n_batches

    return tot_loss, tot_grad_norm

  def validate():
    tot_loss = 0
    with torch.no_grad():
      result = model(x_val)
      loss_result = loss_fn(x=x_val, y=y_val, out=result)
      loss = loss_result['loss']
      tot_loss += loss.detach().item()
    return tot_loss
  
  ###
  history = {'train': [], 'val': []}
  patience_cnt = 0 #cnt of epochs for validation loss not decreasing  
  grad_too_low_cnt = 0
  patience_grad_too_low = 3 
  best_val_loss = float('inf')
  early_stopped = False 

  for e in range(n_epochs):
    tr_loss = 0
    val_loss = 0

    ### TRAINING
    model.train() # Set model to training phase
    tr_loss, tot_grad_norm = do_train()
    
    if verbosity > 0:
      print(f' Epoch: {e + 1}\t Avg Loss: {tr_loss}')
    history['train'].append(tr_loss)

    # VALIDATION
    model.eval()

    if x_val.numel() != 0 and y_val.numel() != 0:
      val_loss = validate()
      if verbosity > 0:
        print(f' Epoch: {e + 1}\t Avg validation Loss: {val_loss}')
      history['val'].append(val_loss)

    # --- Early stop check: validation loss not decreasing ---
      if hel.fuzzy_gte(val_loss, best_val_loss):
        patience_cnt += 1
      else:
        patience_cnt = 0
        best_val_loss = val_loss
        best_model_state = model.state_dict()
        history['epochs_done'] = e+1

      if patience > 0 and patience_cnt >= patience:
        print("STOP: Early stopping due to validation loss plateau.")
        early_stopped = True
        break

    # --- Early stop check: gradient too low ---
    if tot_grad_norm < min_grad_norm:
        grad_too_low_cnt += 1
        print(f"WARN: Grad norm too low ({tot_grad_norm:.2e}) — {grad_too_low_cnt}/{patience_grad_too_low}")
    else:
        grad_too_low_cnt = 0

    if grad_too_low_cnt >= patience_grad_too_low:
        print("STOP: Early stopping due to consistently small gradients")
        history['epochs_done'] = e+1
        early_stopped = True
        break
  
  if not early_stopped:
    history['epochs_done'] = n_epochs
  
  print(f"Min validation loss: {best_val_loss}")

  # load best model on validation always (if present)
  if x_val.numel() != 0 and y_val.numel() != 0:
    print("Loading best model on validation...")
    model.load_state_dict(best_model_state)

  return history


def evaluate_vae_model(model:base.BaseVAE, x_test, y_test, threshold_anomaly=0.05, use_kld=False):
  y_pred = torch.Tensor()
  auc_pr = 0.0
  with torch.no_grad():
    y_pred = find_anomalies(model, x_test, threshold_anomaly, use_kld)

  metrics = hel.compute_metrics(y_pred, y_test.squeeze())
  metrics.auc_pr = auc_pr
  
  return metrics

def evaluate_model(model:base.BaseVAE, model_t:ModelClassType, 
                   x_test, y_test):
  y_pred = torch.Tensor()
  auc_pr = -1.0
  with torch.no_grad():
    result = model(x_test)

    # Find class prediction
    if model_t == ModelClassType.Hybrid_VAE_class or model_t == ModelClassType.Transfer_VAE_class:
      auc_pr, best_f1_thr = hel.compute_auc_pr(y_true=y_test.squeeze(), y_scores=result.y_pred)
      y_pred = (result.y_pred > best_f1_thr).float().squeeze()  

  metrics = hel.compute_metrics(y_pred, y_test.squeeze())
  metrics.auc_pr = auc_pr
  
  return metrics
  
def vae_compute_anomalies_errors(model, x, use_kld=False):
    out = model(x)
    if use_kld:
      errors = ls.kl_divergence(out)
    else:
      errors = ls.reconstruction_error(x=x, x_recon=out.x_recon)
    errors = errors.detach().numpy()

    return errors

def find_anomalies(model, x, threshold=0.001, use_kld=False):
  errors = vae_compute_anomalies_errors(model, x, use_kld)
  # error above threshold means anomaly
  y_pred = torch.tensor(errors > threshold, dtype=torch.float32).squeeze()
  return y_pred

def vae_find_anomalies_threshold(model, x, use_kld=False):
    errors = vae_compute_anomalies_errors(model, x, use_kld)
    threshold = ls.compute_anomalies_threshold_gaussian(errors)
    
    return threshold

def print_vae_anomalies(model, test_x, test_y, threshold, use_kld=False):
  print(f"Anomalies threshold (mean + 3σ): {threshold:.5f}")

  # Plot distribution of errors also for test set (only for assessment)
  errors = vae_compute_anomalies_errors(model, test_x, use_kld)
  hel.plot_reconstruction_errors_dist(reconstruction_errors=errors, 
                                      labels=test_y.squeeze().numpy(),
                                      threshold=threshold)

def print_loss(loss_result, n_upd, batch_size, grad_norm):
  if 'class' in loss_result:
    if 'recon' in loss_result and 'kld' in loss_result:
      print(f'Step {n_upd:,}, Loss: {loss_result['loss'].item():.6f} (Recon: {loss_result['recon'].item():.6f}, DKL: {loss_result['kld'].item():.6f}, class: {loss_result['class'].item():.6f}), Grad: {grad_norm:.6f}, (N samples: {n_upd*batch_size:,})')
    else:
      print(f'Step {n_upd:,}, Loss: {loss_result['loss'].item():.6f} (Class), Grad: {grad_norm:.6f},  (N samples: {n_upd*batch_size:,})')
  else:
    print(f'Step {n_upd:,}, Loss: {loss_result['loss'].item():.6f} (Recon: {loss_result['recon'].item():.6f}, DKL: {loss_result['kld'].item():.6f}), Grad: {grad_norm:.6f},  (N samples: {n_upd*batch_size:,})')