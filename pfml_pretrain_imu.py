# -*- coding: utf-8 -*-
"""
Author: Einari Vaaras, einari.vaaras@tuni.fi, Tampere University
Speech and Cognition Research Group, https://webpages.tuni.fi/specog/index.html

Code for pre-training neural networks using PFML for multi-sensor IMU data and for
using a PFML pre-trained model to extract features.

"""

import numpy as np
import time
import os
import sys

from pathlib import Path
from scipy.io import loadmat
from importlib.machinery import SourceFileLoader
from copy import deepcopy
from torch import cuda, no_grad, save, load, stack
from torch.utils.data import DataLoader

from py_conf_file_into_text import convert_py_conf_file_to_text


# Read the configuration file
if len(sys.argv) > 2:
    sys.exit('\nUsage: \n1) python pfml_pretrain_imu.py \nOR \n2) python pfml_pretrain_imu.py <configuration_file>')
if len(sys.argv) == 2:
    conf = SourceFileLoader('', sys.argv[1]).load_module()
    conf_file_name = sys.argv[1]
else:
    try: 
        import conf_pfml_pretrain_imu as conf
        conf_file_name = 'conf_pfml_pretrain_imu.py'
    except ModuleNotFoundError:
        sys.exit('\nUsage: \n1) python pfml_pretrain_imu.py \nOR \n2) python pfml_pretrain_imu.py <configuration_file>\n\n' \
        'By using the first option, you need to have a configuration file named "conf_pfml_pretrain_imu.py" in the same ' \
        'directory as "pfml_pretrain_imu.py"')


# Import our models
pfml_encoder = getattr(__import__('pfml_model', fromlist=[conf.encoder_name]), conf.encoder_name)
pfml_transformer = getattr(__import__('pfml_model', fromlist=[conf.transformer_name]), conf.transformer_name)
pfml_decoder = getattr(__import__('pfml_model', fromlist=[conf.decoder_name]), conf.decoder_name)

# Import our dataset for our data loader
pfml_dataset = getattr(__import__('pfml_data_loader', fromlist=[conf.dataset_name]), conf.dataset_name)

# Import our loss function
pfml_loss = getattr(__import__('torch.nn', fromlist=[conf.pfml_loss_name]), conf.pfml_loss_name)

# Import our optimization algorithm
optimization_algorithm = getattr(__import__('torch.optim', fromlist=[conf.optimization_algorithm]), conf.optimization_algorithm)

# Import our learning rate scheduler
if conf.use_lr_scheduler:
    scheduler = getattr(__import__('torch.optim.lr_scheduler', fromlist=[conf.lr_scheduler]), conf.lr_scheduler)



def frame_sig(X, winlen, hop):
    Nframes = int(np.floor(((X.shape[0] - winlen)/hop) + 1))
    numchans = X.shape[1]
    X_framed = np.zeros([Nframes, numchans, winlen], dtype=np.float32) # [Nframes, Nchans, winlen]
    for i in range(0, Nframes):
        start = i * hop
        stop = start + winlen
        X_framed[i,:,:] = np.transpose(X[start:stop,:])

    return X_framed





if __name__ == '__main__':
    
    # We make sure that we are able to write the logging file
    textfile_path, textfile_name = os.path.split(conf.name_of_log_textfile)
    if not os.path.exists(textfile_path):
        if textfile_path != '':
            os.makedirs(textfile_path)
    file = open(conf.name_of_log_textfile, 'w')
    file.close()
    
    # Read the text in the configuration file and add it to the logging file
    if conf.print_conf_contents:
        conf_file_lines = convert_py_conf_file_to_text(conf_file_name)
        with open(conf.name_of_log_textfile, 'a') as f:
            f.write(f'The configuration settings in the file {conf_file_name}:\n\n')
            for line in conf_file_lines:
                f.write(f'{line}\n')
            f.write('\n########################################################################################\n\n\n\n')
        
    
    # Use CUDA if it is available, else use CPU
    device = 'cuda' if cuda.is_available() else 'cpu'
    #device= 'cpu'
    with open(conf.name_of_log_textfile, 'a') as f:
        f.write(f'Process on {device}\n\n')
    
    # Read data
    with open(conf.name_of_log_textfile, 'a') as f:
        f.write('Generating data...\n')
    
    # Tämä korvataan oikealla datalla 
    """     
    Data = []
    for iBaby in range(conf.num_randomly_generated_babydata):
        babyData = {}
        
        # We generate random signals to simulate having MAIJU recordings
        num_samples = np.random.randint(50000, high=300000)
        num_channels = 9
        x = np.linspace(0, num_samples, num_samples)
        
        acc_data = [] # Randomly generated accelerometer data (12 channels)
        gyro_data = [] # Randomly generated gyroscope data (12 channels)
        for i in range(num_channels):
            for data_list in [acc_data, gyro_data]:
                if np.random.rand() < 0.5:
                    data_list.append(np.random.rand() * np.sin(x) + np.random.normal(scale=0.1, size=len(x)))
                else:
                    data_list.append(np.random.rand() * np.cos(x) + np.random.normal(scale=0.1, size=len(x)))
        
        # Has the shape [num_samples, num_channels*2]
        x_r = np.concatenate((acc_data, gyro_data), axis=0).T
        
        # We frame the signals
        x_r = frame_sig(x_r, conf.window_len, conf.hop_len)
        
        # We randomly generate a mask to simulate sections of the data in which unwanted phenomena occurred, such as
        # if the baby was out of screen or if the baby was being carried by a caregiver.
        # 1 = frame is masked, 0 = frame is not masked
        mask = (np.random.rand(len(x_r)) < 0.1).astype(int)
        
        # Here x_r has the shape [num_frames, num_channels, window_len], and mask has the shape [num_frames]. 
        babyData['X'] = x_r
        babyData['Mask'] = mask

        Data.append(babyData) 
     """
    
    # Load real data 
    
    Data = [] 
    mat_folder = Path("/home/rqb592/dippa/all_data_mat/")
    mat_files = list(mat_folder.glob("*.mat"))
    for f in mat_files: 
        data = loadmat(f)
        acc = data["acc_data"]
        gyro = data["gyro_data"]  
        X = np.concatenate((acc, gyro), axis=1)
        X_framed = frame_sig(X, conf.window_len, conf.hop_len)
        data['X'] = X_framed
        data['Mask'] = np.zeros(len(X_framed))
        Data.append(data) 
    
    

    with open(conf.name_of_log_textfile, 'a') as f:
        f.write('Done!\n\n')
    
    
    # Initialize our models, pass the models to the available device
    Encoder = pfml_encoder(**conf.encoder_model_params).to(device)
    Transformer = pfml_transformer(**conf.transformer_params).to(device)
    Decoder = pfml_decoder(**conf.decoder_params).to(device)
    
    # Give the parameters of our models to an optimizer
    model_parameters = list(Encoder.parameters()) + list(Transformer.parameters()) + list(Decoder.parameters())
    optimizer = optimization_algorithm(params=model_parameters, **conf.optimization_algorithm_params)
    
    # Get our learning rate for later use
    learning_rate = optimizer.param_groups[0]['lr']
    
    # Give the optimizer to the learning rate scheduler
    if conf.use_lr_scheduler:
        lr_scheduler = scheduler(optimizer, **conf.lr_scheduler_params)
    
    # Initialize our loss functions
    loss_function_pfml = pfml_loss(**conf.pfml_loss_params)

    # Variables for early stopping
    lowest_validation_loss = 1e10
    best_validation_epoch = 0
    patience_counter = 0
    
    if conf.load_model:
        try:
            with open(conf.name_of_log_textfile, 'a') as f:
                f.write('Loading model from file...\n')
                f.write(f'Loading model {conf.encoder_best_model_name}\n')
                f.write(f'Loading model {conf.transformer_best_model_name}\n')
                f.write(f'Loading model {conf.decoder_best_model_name}\n')
            Encoder.load_state_dict(load(conf.encoder_best_model_name, map_location=device))
            Transformer.load_state_dict(load(conf.transformer_best_model_name, map_location=device))
            Decoder.load_state_dict(load(conf.decoder_best_model_name, map_location=device))
            best_model_encoder = deepcopy(Encoder.state_dict())
            best_model_transformer = deepcopy(Transformer.state_dict())
            best_model_decoder = deepcopy(Decoder.state_dict())
            
            with open(conf.name_of_log_textfile, 'a') as f:
                f.write('Done!\n\n')
        except FileNotFoundError:
            with open(conf.name_of_log_textfile, 'a') as f:
                f.write('An error occurred while loading the files! Continuing with randomly initialized models...\n\n')
    else:
        best_model_encoder = None
        best_model_transformer = None
        best_model_decoder = None
    
    # Initialize the data loaders
    if conf.train_model:
        with open(conf.name_of_log_textfile, 'a') as f:
            f.write('Initializing training set...\n')
        training_set = pfml_dataset(Data, train_val_test='train', **conf.params_train_dataset)
        train_data_loader = DataLoader(training_set, **conf.params_train)
        with open(conf.name_of_log_textfile, 'a') as f:
            f.write('Done!\n')
            f.write('Initializing validation set...\n')
        validation_set = pfml_dataset(Data, train_val_test='validation', **conf.params_validation_dataset)
        validation_data_loader = DataLoader(validation_set, **conf.params_train)
        with open(conf.name_of_log_textfile, 'a') as f:
            f.write('Done!\n')
    if conf.extract_features:
        with open(conf.name_of_log_textfile, 'a') as f:
            f.write('Initializing full dataset for feature extraction...\n')
        test_set = pfml_dataset(Data, train_val_test='test', **conf.params_feature_extraction_dataset)
        test_data_loader = DataLoader(test_set, **conf.params_feature_extraction)
        with open(conf.name_of_log_textfile, 'a') as f:
            f.write('Done!\n\n')
    
    # Check that we don't have a mistake in the config settings
    if conf.max_epochs < conf.min_train_epochs:
        sys.exit(f'max_epochs cannot be smaller than min_train_epochs ({conf.max_epochs} < {conf.min_train_epochs})')
    
    # Flag for indicating if max epochs are reached
    max_epochs_reached = True
    
    # Start training our model
    if conf.train_model:
        with open(conf.name_of_log_textfile, 'a') as f:
            f.write('Starting training...\n')
        
        for epoch in range(1, conf.max_epochs + 1):
            
            start_time = time.time()
    
            # Lists containing the losses of each epoch
            epoch_loss_training = []
            epoch_loss_validation = []
            
            # Lists containing the variances (over the time dimension) of our model inputs, embeddings, and outputs
            epoch_variance_inputs = []
            epoch_variance_embeddings = []
            epoch_variance_outputs = []
    
            # Indicate that we are in training mode, so e.g. dropout will function
            Encoder.train()
            Transformer.train()
            Decoder.train()
            
            # Loop through every batch of our training data
            for train_data in train_data_loader:
                
                # Get the minibatches. We remove each input sequence that contains over 80% of masked data.
                X_input_init, _, data_masks_init, target_functionals_init = [element.to(device) for element in train_data]
                target_functionals_init = target_functionals_init.float()
                X_input = []
                data_masks = []
                target_functionals = []
                for sequence_index in range(len(X_input_init)):
                    if not data_masks_init[sequence_index].sum() > 0.80 * X_input_init.size()[1]:
                        X_input.append(X_input_init[sequence_index])
                        data_masks.append(data_masks_init[sequence_index])
                        target_functionals.append(target_functionals_init[sequence_index])
                if len(X_input) == 0:
                    continue
                
                X_input = stack(X_input, dim=0)
                data_masks = stack(data_masks, dim=0)
                target_functionals = stack(target_functionals, dim=0)
                padding_masks = data_masks.bool()
                
                del X_input_init
                del data_masks_init
                del target_functionals_init
                
                # Zero the gradient of the optimizer
                optimizer.zero_grad()
                
                # Pass our data through the encoder           
                Embedding = Encoder(X_input.float())
                
                # Pass our embeddings to the Transformer encoder
                X_output, mask_indices = Transformer(Embedding, src_key_padding_mask=padding_masks, mask_embeddings=True)
                
                # Pass our Transformer outputs to the Decoder
                pred_functionals = Decoder(X_output)
                
                # Compute the variance (over the time dimension, ignoring paddings) of our model inputs, embeddings, and outputs
                X_input_variance = []
                Embedding_variance = []
                X_output_variance = []
                for i in range(X_output.size()[0]):
                    if conf.compute_variance_for_unmasked_parts:
                        X_input_variance.append(X_input[i, ~padding_masks[i], :][~mask_indices[i, ~padding_masks[i]], :].var(dim=0) + 1e-6)
                        Embedding_variance.append(Embedding[i, ~padding_masks[i], :][~mask_indices[i, ~padding_masks[i]], :].var(dim=0) + 1e-6)
                        X_output_variance.append(X_output[i, ~padding_masks[i], :][~mask_indices[i, ~padding_masks[i]], :].var(dim=0) + 1e-6)
                    else:
                        X_input_variance.append(X_input[i, ~padding_masks[i], :].var(dim=0) + 1e-6)
                        Embedding_variance.append(Embedding[i, ~padding_masks[i], :].var(dim=0) + 1e-6)
                        X_output_variance.append(X_output[i, ~padding_masks[i], :].var(dim=0) + 1e-6)
                
                X_input_variance = stack(X_input_variance, dim=0).nanmean()
                Embedding_variance = stack(Embedding_variance, dim=0).nanmean()
                X_output_variance = stack(X_output_variance, dim=0).nanmean()
                
                epoch_variance_inputs.append(X_input_variance.item())
                epoch_variance_embeddings.append(Embedding_variance.item())
                epoch_variance_outputs.append(X_output_variance.item())
                
                if conf.compute_loss_only_for_masked_embeddings:
                    # We compute the prediction loss only for the masked embeddings
                    pred_functionals_masked_section = pred_functionals[mask_indices]
                    target_functionals_masked_section = target_functionals[mask_indices]
                    loss = loss_function_pfml(pred_functionals_masked_section, target_functionals_masked_section) * conf.pfml_loss_scaler
                else:
                    # We compute the prediction loss for the non-padded parts of the embeddings
                    loss = []
                    for i in range(pred_functionals.size()[0]):
                        pred_functionals_non_padded = pred_functionals[i, ~padding_masks[i], :]
                        target_functionals_non_padded = target_functionals[i, ~padding_masks[i], :]
                        loss.append(loss_function_pfml(pred_functionals_non_padded, target_functionals_non_padded))
                    loss = (sum(loss) / len(loss)) * conf.pfml_loss_scaler
                
                # Perform the backward pass
                loss.backward()
                
                # Update the weights
                optimizer.step()

                # Add the loss to the total loss of the batch
                epoch_loss_training.append(loss.item())
                
            
            # Indicate that we are in evaluation mode, so e.g. dropout will not function
            Encoder.eval()
            Transformer.eval()
            Decoder.eval()
    
            # Make PyTorch not calculate the gradients, so everything will be much faster.
            with no_grad():
                
                # Loop through every batch of our validation data and perform a similar process as for the training data
                for validation_data in validation_data_loader:
                    X_input_init, _, data_masks_init, target_functionals_init = [element.to(device) for element in validation_data]
                    target_functionals_init = target_functionals_init.float()
                    X_input = []
                    data_masks = []
                    target_functionals = []
                    for sequence_index in range(len(X_input_init)):
                        if not data_masks_init[sequence_index].sum() > 0.80 * X_input_init.size()[1]:
                            X_input.append(X_input_init[sequence_index])
                            data_masks.append(data_masks_init[sequence_index])
                            target_functionals.append(target_functionals_init[sequence_index])
                    if len(X_input) == 0:
                        continue
                    X_input = stack(X_input, dim=0)
                    data_masks = stack(data_masks, dim=0)
                    target_functionals = stack(target_functionals, dim=0)
                    padding_masks = data_masks.bool()
                    del X_input_init
                    del data_masks_init
                    del target_functionals_init
                    Embedding = Encoder(X_input.float())
                    X_output, mask_indices = Transformer(Embedding, src_key_padding_mask=padding_masks, mask_embeddings=True)
                    pred_functionals = Decoder(X_output)
                    X_input_variance = []
                    Embedding_variance = []
                    X_output_variance = []
                    for i in range(X_output.size()[0]):
                        if conf.compute_variance_for_unmasked_parts:
                            X_input_variance.append(X_input[i, ~padding_masks[i], :][~mask_indices[i, ~padding_masks[i]], :].var(dim=0) + 1e-6)
                            Embedding_variance.append(Embedding[i, ~padding_masks[i], :][~mask_indices[i, ~padding_masks[i]], :].var(dim=0) + 1e-6)
                            X_output_variance.append(X_output[i, ~padding_masks[i], :][~mask_indices[i, ~padding_masks[i]], :].var(dim=0) + 1e-6)
                        else:
                            X_input_variance.append(X_input[i, ~padding_masks[i], :].var(dim=0) + 1e-6)
                            Embedding_variance.append(Embedding[i, ~padding_masks[i], :].var(dim=0) + 1e-6)
                            X_output_variance.append(X_output[i, ~padding_masks[i], :].var(dim=0) + 1e-6)
                    X_input_variance = stack(X_input_variance, dim=0).nanmean()
                    Embedding_variance = stack(Embedding_variance, dim=0).nanmean()
                    X_output_variance = stack(X_output_variance, dim=0).nanmean()
                    epoch_variance_inputs.append(X_input_variance.item())
                    epoch_variance_embeddings.append(Embedding_variance.item())
                    epoch_variance_outputs.append(X_output_variance.item())
                    if conf.compute_loss_only_for_masked_embeddings:
                        pred_functionals_masked_section = pred_functionals[mask_indices]
                        target_functionals_masked_section = target_functionals[mask_indices]
                        loss = loss_function_pfml(pred_functionals_masked_section, target_functionals_masked_section) * conf.pfml_loss_scaler
                    else:
                        loss = []
                        for i in range(pred_functionals.size()[0]):
                            pred_functionals_non_padded = pred_functionals[i, ~padding_masks[i], :]
                            target_functionals_non_padded = target_functionals[i, ~padding_masks[i], :]
                            loss.append(loss_function_pfml(pred_functionals_non_padded, target_functionals_non_padded))
                        loss = (sum(loss) / len(loss)) * conf.pfml_loss_scaler
                    epoch_loss_validation.append(loss.item())
    
            # Calculate mean losses and variances
            epoch_loss_training = np.nanmean(np.array(epoch_loss_training))
            epoch_loss_validation = np.nanmean(np.array(epoch_loss_validation))
            epoch_variance_inputs = np.nanmean(np.array(epoch_variance_inputs))
            epoch_variance_embeddings = np.nanmean(np.array(epoch_variance_embeddings))
            epoch_variance_outputs = np.nanmean(np.array(epoch_variance_outputs))
    
            # Check early stopping conditions
            if epoch_loss_validation < lowest_validation_loss and epoch_loss_validation > 0.0001:
                lowest_validation_loss = epoch_loss_validation
                patience_counter = 0
                best_model_encoder = deepcopy(Encoder.state_dict())
                best_model_transformer = deepcopy(Transformer.state_dict())
                best_model_decoder = deepcopy(Decoder.state_dict())
                best_validation_epoch = epoch
                if conf.save_best_model:
                    # We first make sure that we are able to write the files
                    save_names = [conf.encoder_best_model_name, conf.transformer_best_model_name, conf.decoder_best_model_name]
                    for model_save_name in save_names:
                        model_path, model_filename = os.path.split(model_save_name)
                        if not os.path.exists(model_path):
                            if model_path != '':
                                os.makedirs(model_path)
                    
                    save(best_model_encoder, conf.encoder_best_model_name)
                    save(best_model_transformer, conf.transformer_best_model_name)
                    save(best_model_decoder, conf.decoder_best_model_name)
            else:
                patience_counter += 1
            
            end_time = time.time()
            epoch_time = end_time - start_time
            
            with open(conf.name_of_log_textfile, 'a') as f:
                f.write(f'Epoch: {epoch:04d} | Mean training loss: {epoch_loss_training:6.4f} | '
                  f'Mean validation loss: {epoch_loss_validation:6.4f} (lowest: {lowest_validation_loss:6.4f}) | '
                  f'Mean input variance: {epoch_variance_inputs:6.4f} | Mean embedding variance: {epoch_variance_embeddings:6.4f} | '
                  f'Mean output variance: {epoch_variance_outputs:6.4f} | Duration: {epoch_time:4.2f} seconds\n')
                
            # We check that do we need to update the learning rate based on the validation loss
            if conf.use_lr_scheduler:
                if conf.lr_scheduler == 'ReduceLROnPlateau':
                    lr_scheduler.step(epoch_loss_validation)
                else:
                    lr_scheduler.step()
                current_learning_rate = optimizer.param_groups[0]['lr']
                if current_learning_rate != learning_rate:
                    learning_rate = current_learning_rate
                    with open(conf.name_of_log_textfile, 'a') as f:
                        f.write(f'Updated learning rate after epoch {epoch} based on learning rate scheduler, now lr={learning_rate}\n')
            
            # If patience counter is fulfilled, stop the training
            if patience_counter >= conf.patience:
                max_epochs_reached = False
                break
            
            # If the variance of our embeddings falls too low, we stop training
            if epoch >= conf.min_train_epochs and epoch_variance_embeddings < conf.min_embedding_variance:
                max_epochs_reached = 0
                break
            
        if max_epochs_reached:
            with open(conf.name_of_log_textfile, 'a') as f:
                f.write('\nMax number of epochs reached, stopping training\n\n')
        else:
            if patience_counter >= conf.patience:
                with open(conf.name_of_log_textfile, 'a') as f:
                    f.write('\nExiting due to early stopping\n\n')
            else:
                with open(conf.name_of_log_textfile, 'a') as f:
                    f.write('\nThe variance of our embeddings fell too low, stopping training\n\n')
        
        if best_model_encoder is None:
            with open(conf.name_of_log_textfile, 'a') as f:
                f.write('\nNo best model. The criteria for the lowest acceptable validation loss not satisfied!\n\n')
            sys.exit('No best model, exiting...')
        else:
            with open(conf.name_of_log_textfile, 'a') as f:
                f.write(f'\nBest epoch {best_validation_epoch} with validation loss {lowest_validation_loss}\n\n')
        
        

    # Perform feature extraction using a trained model
    if conf.extract_features:
        with open(conf.name_of_log_textfile, 'a') as f:
            f.write('\n\nExtracting features using a trained PFML model...\n\n')
        
        for output_type in [None, 'ff_outputs', 'ff_residual_outputs', 'end_of_block',
                            'ff_output_second_last', 'ff_residual_output_second_last', 'end_of_block_second_last']:
        
            # Initialize the best version of our model
            try:
                Encoder.load_state_dict(load(conf.encoder_best_model_name, map_location=device))
                Transformer.load_state_dict(load(conf.transformer_best_model_name, map_location=device))
                Decoder.load_state_dict(load(conf.decoder_best_model_name, map_location=device))
            except (FileNotFoundError, RuntimeError):
                Encoder.load_state_dict(best_model_encoder)
                Transformer.load_state_dict(best_model_transformer)
                Decoder.load_state_dict(best_model_decoder)
                    
            Encoder.eval()
            Transformer.eval()
            with no_grad():
                X_outputs_array = []
                padding_mask_indices_array = []
                for test_data in test_data_loader:
                    X_input, _, data_masks, _ = [element.to(device) for element in test_data]
                    padding_masks = data_masks.bool()
                    Embedding = Encoder(X_input.float())
                    X_output = Transformer(Embedding, src_key_padding_mask=padding_masks, output_type=output_type)
                    X_outputs_array.append(X_output.cpu().numpy())
                    padding_mask_indices_array.append(padding_masks.cpu().numpy())
            
            X_outputs_array = np.vstack(X_outputs_array)
            padding_mask_indices_array = np.vstack(padding_mask_indices_array)
            
            # We make sure that we are able to write the files before we save the files
            save_names = [f'{conf.feature_extraction_model_output_savefile_basename}_{output_type}.npy']
            for file_save_name in save_names:
                file_path, filename = os.path.split(file_save_name)
                if not os.path.exists(file_path):
                    if file_path != '':
                        os.makedirs(file_path)
            # We compute the mean of the output features over the time dimension
            feats = []
            for i in range(X_outputs_array.shape[0]):
                non_padded_sample = X_outputs_array[i][~padding_mask_indices_array[i,:]]
                feats.append(np.mean(non_padded_sample, axis=0))
            feats = np.nan_to_num(np.array(feats))
            np.save(f'{conf.feature_extraction_model_output_savefile_basename}_{output_type}.npy', feats)
            
            with open(conf.name_of_log_textfile, 'a') as f:
                f.write(f'Done! Data written to the file {conf.feature_extraction_model_output_savefile_basename}_{output_type}.npy\n')
            
