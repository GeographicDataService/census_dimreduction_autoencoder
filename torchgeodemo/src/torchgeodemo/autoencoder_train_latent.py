# Libraries
import sys
import os
import argparse
import yaml
import numpy as np
import pandas as pd
import math
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

# This project's modules
from .models import AutoEncoder

# Utils -------------------------------------------------------------------

# Generate autoencoder sizes if none are provided
def generate_ae_sizes(input_size, latent_size, depth):
    if depth < 1:
        depth = 1
    return list(reversed([math.ceil(i) for i in np.linspace(latent_size, input_size, depth+1)]))


# Main --------------------------------------------------------------------

def main(config, train_ae=True, create_latent=True, save_reco_error=True, verbose=True):

    # Configuration -------------------------------------------------------

    with open(config, 'r') as file:
        geodemo_config = yaml.safe_load(file)
    print(f'\n{geodemo_config=}\n', flush=True) if verbose else None

    if 'random_seed' in geodemo_config:
        torch.manual_seed(geodemo_config['random_seed'])

    dataset_path      = os.path.expanduser(
                        geodemo_config['data']['source'])
    dataset_nickname  = geodemo_config['data']['nickname']

    model_nickname    = geodemo_config['autoencoder']['nickname']
    model_version     = geodemo_config['autoencoder']['version']
    model_save_latent = geodemo_config['autoencoder']['save_latent'] if 'save_latent' in geodemo_config['autoencoder'] else None
    
    # Files and directories -----------------------------------------------

    dir_project       = os.path.expanduser(
                        geodemo_config['working_dir'])
    dir_logs          = os.path.join(dir_project, 'logs')

    if not os.path.exists(dir_project):
        os.makedirs(dir_project)

    latent_csv_path  = os.path.join(dir_project, f'{dataset_nickname}__{model_nickname}_v{model_version}__latent.csv')
    latent_prq_path  = os.path.join(dir_project, f'{dataset_nickname}__{model_nickname}_v{model_version}__latent.parquet')
    model_path       = os.path.join(dir_project, f'{dataset_nickname}__{model_nickname}_v{model_version}__model.pth')
    model_info_path  = os.path.join(dir_project, f'{dataset_nickname}__{model_nickname}_v{model_version}__model__info.txt')


    # Load data -----------------------------------------------------------

    if dataset_path.endswith('.csv'):
        input_dataset = pd.read_csv(dataset_path)
    elif dataset_path.endswith('.parquet'):
        input_dataset = pd.read_parquet(dataset_path)
    else:
        raise ValueError('Unsupported file format. Only CSV and Parquet are currently supported.')

    # Extract ID
    dataset_id_col        = geodemo_config['data']['id_col']
    ids                   = input_dataset[dataset_id_col]
    input_dataset         = input_dataset.drop(columns=[dataset_id_col])
    # Extract excluded columns
    if 'exclude_cols' in    geodemo_config['data']:
        dataset_excl_cols = geodemo_config['data']['exclude_cols']
    else:
        dataset_excl_cols = []
    if len(dataset_excl_cols) > 0:
        excl_cols         = input_dataset[dataset_excl_cols]
        input_dataset     = input_dataset.drop(columns=dataset_excl_cols)

    # Create tensor and data loader
    data_tensor      = torch.tensor(input_dataset.values).float()
    data_tensor_nrow = data_tensor.shape[0]
    data_tensor_ncol = data_tensor.shape[1]


    if train_ae:

        # Model ---------------------------------------------------------------

        # Training parameters
        model_max_epochs      = geodemo_config['autoencoder']['max_epochs'] if 'max_epochs' in geodemo_config['autoencoder'] else 100
        model_batch_size      = geodemo_config['autoencoder']['batch_size'] if 'batch_size' in geodemo_config['autoencoder'] else -1
        # Set batch size
        if model_batch_size   <= 0.0:
            data_batch_size   = data_tensor_nrow
        elif model_batch_size <= 1.0:
            data_batch_size   = math.ceil(data_tensor_nrow*model_batch_size)
        elif model_batch_size <= data_tensor_nrow:
            data_batch_size   = model_batch_size
        else:
            data_batch_size   = data_tensor_nrow

        # Autoencoder parameters
        ae_args            = {}
        ae_args['verbose'] = verbose

        # Encoder sizes
        if 'encoder' in geodemo_config['autoencoder']:
            ae_encoder_sizes  = geodemo_config['autoencoder']['encoder']['sizes']
            ae_encoder_sizes  = [data_tensor_ncol] + ae_encoder_sizes
        else:
            if 'depth' in       geodemo_config['autoencoder']:
                ae_depth      = geodemo_config['autoencoder']['depth']
            else:
                ae_depth      = 2
                print(f'No encoder sizes or depth specified. Using default depth of {ae_depth}.', flush=True)
            if 'latent' in      geodemo_config['autoencoder']:
                ae_latent     = geodemo_config['autoencoder']['latent']
            else:
                ae_latent     = 8
                print(f'No latent size specified. Using default latent size of {ae_latent}.', flush=True)
            ae_encoder_sizes  = generate_ae_sizes(data_tensor_ncol, ae_latent, ae_depth)
            print(f'Using computed sizes: {ae_encoder_sizes}', flush=True)

        # Other autoencoder parameters
        if 'encoder' in                               geodemo_config['autoencoder']:
            if 'activation' in                        geodemo_config['autoencoder']['encoder']:
                ae_args['encoder_activation']       = geodemo_config['autoencoder']['encoder']['activation']
            if 'sparse' in                            geodemo_config['autoencoder']['encoder']:
                ae_args['encoder_sparse']           = True
                ae_args['encoder_sparse_topk_k']    = geodemo_config['autoencoder']['encoder']['sparse']['topk_k']
                if 'sparsity_loss_weight' in          geodemo_config['autoencoder']['encoder']:
                    ae_args['sparsity_loss_weight'] = geodemo_config['autoencoder']['encoder']['sparse']['loss_weight']
        # Add covariance loss config
        if 'use_covariance_loss' in geodemo_config['autoencoder']:
            ae_args['use_covariance_loss'] = geodemo_config['autoencoder']['use_covariance_loss']
        if 'dcc' in                                   geodemo_config['autoencoder']:
            ae_args['dcc_from_epoch']               = geodemo_config['autoencoder']['dcc']['from_epoch']
            if 'neighbours' in                        geodemo_config['autoencoder']['dcc']:
                ae_args['dcc_neighbours']           = geodemo_config['autoencoder']['dcc']['neighbours']
            if 'nn_lambda' in                         geodemo_config['autoencoder']['dcc']:
                ae_args['dcc_nn_lambda']            = geodemo_config['autoencoder']['dcc']['nn_lambda']

        if 'decoder' in                               geodemo_config['autoencoder']:
            if 'sizes' in                             geodemo_config['autoencoder']['decoder']:
                ae_args['decoder_sizes']            = geodemo_config['autoencoder']['decoder']['sizes']
                ae_args['decoder_sizes']            = ae_args['decoder_sizes'] + [data_tensor_ncol]
            if 'activation' in                        geodemo_config['autoencoder']['decoder']:
                ae_args['decoder_activation']       = geodemo_config['autoencoder']['decoder']['activation']

        print(f'\n{ae_args=}\n', flush=True) if verbose else None


        # Training ----------------------------------------------------------------

        # Create data loader
        data_tensor_loader = torch.utils.data.DataLoader(
            data_tensor, 
            batch_size=data_batch_size, 
            shuffle=True
            )

        # Create loggers
        logger_test_name = f'log_{dataset_nickname}_{model_nickname}_v{model_version}'
        logger_folder = dir_logs
        logger_tb = TensorBoardLogger(logger_folder, name=logger_test_name)
        logger_csv = CSVLogger(logger_folder, name=logger_test_name)

        # Create model
        geodemo_ae = AutoEncoder(encoder_sizes = ae_encoder_sizes, **ae_args)
        # Save model info
        model_str = str(geodemo_ae)
        print(model_str, flush=True) if verbose else None
        with open(model_info_path, 'w') as f:
            f.write(model_str)

        # Train the model
        accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
        trainer = Trainer(
            devices=1, 
            accelerator=accelerator,
            logger=[logger_tb, logger_csv], 
            max_epochs=model_max_epochs,
            enable_progress_bar=verbose
            )
        trainer.fit(
            model=geodemo_ae, 
            train_dataloaders=data_tensor_loader
            )

        # Save model
        torch.save(geodemo_ae, model_path)
    

    # Load model and create latent ----------------------------------------
    else:

        if not os.path.exists(model_path):
            raise FileNotFoundError(f'The model file {model_path} does not exist.')

        # Load trained model
        print('\n>>> WARNING <<<\nLoading model from disk.\nThis can result in **arbitrary code execution**. Do it only if you got the file from a **trusted** source!')
        response = input('Do you want to continue loading the model? (y/N): ')
        if response.lower() != 'y':
            print('Aborting operation.')
            sys.exit(0)
        geodemo_ae = torch.load(model_path, weights_only=False)


    # Create latent -------------------------------------------------------
    if create_latent:

        for f in [latent_csv_path, latent_prq_path]:
            if os.path.exists(f):
                raise FileExistsError(f'The file {f} already exists.')

        # Generate latent in batches and combine results
        with torch.no_grad():
            geodemo_ae.eval()

            latent_batches = []
                
            # Create tensor and data loader
            data_tensor = torch.tensor(input_dataset.values).float()
            data_tensor_nrow = data_tensor.shape[0]
            data_loader_for_encoder = torch.utils.data.DataLoader(
                data_tensor, 
                batch_size=math.ceil(data_tensor_nrow*0.01),
                shuffle=False # IMPORTANT: must be False to match the ids
                )
            for batch in data_loader_for_encoder:
                    latent_batch = geodemo_ae.encode(batch).cpu().detach().numpy()
                    latent_batches.append(latent_batch)
            latent = np.vstack(latent_batches)

            # Write reduced_data to a parquet file in the OAC folder
            latent_df = pd.DataFrame(latent)
            latent_df = latent_df.rename(columns=lambda x: f'EMB_{x:03d}')
            # Create output df
            if len(dataset_excl_cols) > 0:
                output_df = pd.concat([ids, excl_cols, latent_df], axis=1)
            else:
                output_df = pd.concat([ids, latent_df], axis=1)
            # Save output df
            if model_save_latent == 'csv':
                output_df.to_csv(latent_csv_path, index=False)
            elif model_save_latent == 'parquet':
                output_df.to_parquet(latent_prq_path, index=False)
            else:
                print(f'No dataset output format specified. Saving as CSV by default.') if verbose else None
                output_df.to_csv(latent_csv_path, index=False)

    if save_reco_error:
        data_tensor_loader_no_shuffle = torch.utils.data.DataLoader(
            data_tensor, 
            batch_size=data_batch_size, 
            shuffle=False
            )

        # Save reconstruction error, inputs, and reconstructed outputs separately
        with torch.no_grad():
            geodemo_ae.eval()
            reconstructed_outputs = []

            for batch in data_tensor_loader_no_shuffle:
                batch = batch.to(geodemo_ae.device)
                reconstructed_batch = geodemo_ae(batch)
                
                if isinstance(reconstructed_batch, tuple):
                    reconstructed_batch = reconstructed_batch[1] 
                reconstructed_outputs.extend(reconstructed_batch.cpu().numpy())

        # Get column names from the original dataset
        feature_columns = input_dataset.columns.tolist()
        # Save reconstructed outputs with correct column names
        reconstructed_outputs_df = pd.DataFrame(reconstructed_outputs, columns=feature_columns)
        reconstructed_outputs_df.insert(0, 'ID', ids)  # Add IDs as the first column
        reconstructed_outputs_path = os.path.join(dir_project, f'{dataset_nickname}__{model_nickname}_v{model_version}__reconstructed_outputs.csv')
        reconstructed_outputs_df.to_csv(reconstructed_outputs_path, index=False)
        print(f'Reconstructed outputs saved to {reconstructed_outputs_path}')


if __name__ == '__main__':
     
    # Parse arguments --------------------------------------------------------

    parser = argparse.ArgumentParser(description='Train AutoEncoder model.')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('config', type=str, help='Path to the configuration YAML file.')
    args = parser.parse_args()

    main(args.config, verbose=args.verbose)