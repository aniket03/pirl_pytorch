import os
import pandas as pd

from common_constants import PAR_OBSERVATIONS_DIR


def log_experiment(exp_name, n_epochs, train_losses, val_losses, train_accs, val_accs):
    observations_df = pd.DataFrame()
    observations_df['epoch count'] = [i for i in range(1, n_epochs + 1)]
    observations_df['train loss'] = train_losses
    observations_df['val loss'] = val_losses
    observations_df['train acc'] = train_accs
    observations_df['val acc'] = val_accs
    observations_file_path = os.path.join(PAR_OBSERVATIONS_DIR, exp_name + '_observations.csv')
    observations_df.to_csv(observations_file_path)
