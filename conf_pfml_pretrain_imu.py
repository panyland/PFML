#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Einari Vaaras, einari.vaaras@tuni.fi, Tampere University
Speech and Cognition Research Group, https://webpages.tuni.fi/specog/index.html

The configuration file for pfml_pretrain_imu.py.

"""

experiment_num = 1

"""The hyperparameters for our training and feature extraction processes"""

# The maximum number of training epochs
max_epochs = 10000

# The patience counter for early stopping
patience = 100

# Dropout rate of the encoder model
dropout_encoder_model = 0.1

# The learning rate of our model training
learning_rate = 1e-4

# The number of frames in each input sequence for our model (Fs=52 Hz, 60-sample hop length --> 260 frames is 5 minutes)
train_sequence_length = 260

# The minibatch size
batch_size = 64

# Window length (in samples)
window_len = 120

# Hop length (in samples)
hop_len = 60

# Flag for running PFML pre-training
train_model = True

# Flag for using our PFML pre-trained model to extract features
extract_features = False

# Flag for loading the weights for our model, i.e. flag for continuing a previous training process
load_model = False

# Flag for saving the best model (according to validation loss) after each training epoch where the
# validation loss is lower than before
save_best_model = True

# The name of the text file into which we log the output of the training process
name_of_log_textfile = f'trainlogs_pfml_pretraining/pfml_trainlog_simulated_imu_data_{experiment_num}.txt'

# A flag for determining whether we want to print the contents of the configuration file into the
# logging file
print_conf_contents = True

# Define our models that we want to use from the file pfml_model.py
encoder_name = 'SENSOR_MODULE_v3'
transformer_name = 'pfml_transformer_encoder'
decoder_name = 'pfml_decoder_linear'

# Defines the minimum number of training epochs, just to make sure that we don't stop training too soon if
# the variance of the embeddings is too low in the beginning of the training process
min_train_epochs = 20

# The minimum acceptable variances of our embeddings. If we don't care about the minimum acceptable
# variance, we can set this variable to some negative value (variance cannot be less than 0)
min_embedding_variance = -9999.0

# A flag for computing the prediction loss only for the masked embeddings (True). If set to False,
# the loss is computed for the non-padded parts of the embeddings, including the non-masked parts.
compute_loss_only_for_masked_embeddings = True

# Define our loss function that we want to use from torch.nn
pfml_loss_name = 'MSELoss'

# The scaling multipliers for the loss functions, a value of 1.0 means no scaling
pfml_loss_scaler = 1.0

# The hyperparameters for the loss functions
pfml_loss_params = {}

# A flag for defining whether we want to compute the variance over the time dimension only for non-padded
# and unmasked parts of our predicted outputs and training targets (True), or only for the non-padded
# parts (False). Since the embedding masks can bring additional variance to the outputs, it might be a good
# choice to ignore them when computing the variance.
compute_variance_for_unmasked_parts = True

# Define the optimization algorithm we want to use from torch.optim
optimization_algorithm = 'RAdam'

# The hyperparameters for our optimization algorithm
optimization_algorithm_params = {'lr': learning_rate}

# A flag to determine if we want to use a learning rate scheduler
use_lr_scheduler = False

# Define which learning rate scheduler we want to use from torch.optim.lr_scheduler
lr_scheduler = 'ReduceLROnPlateau'

# The hyperparameters for the learning rate scheduler
lr_scheduler_params = {'mode': 'min',
                       'factor': 0.5,
                       'patience': 40}

# The names of the model weight files of the best models (according to validation loss) for
# loading/saving model weights
encoder_best_model_name = f'pfml_pretrained_models/pfml_Encoder_imu_best_model_{experiment_num}.pt'
transformer_best_model_name = f'pfml_pretrained_models/pfml_Transformer_imu_best_model_{experiment_num}.pt'
decoder_best_model_name = f'pfml_pretrained_models/pfml_Decoder_imu_best_model_{experiment_num}.pt'

# The base name of the files containing the output of the feature extraction process
feature_extraction_model_output_savefile_basename = f'feature_extraction_output/simulated_imu_data_pfml_feats_{experiment_num}'



"""The hyperparameters for our encoder model"""

num_input_channels = 18
encoder_num_latent_channels = 80
encoder_num_output_channels = 160

encoder_model_params = {'s_channels': num_input_channels,
                        'input_channels': window_len,
                        'latent_channels': encoder_num_latent_channels,
                        'output_channels': encoder_num_output_channels,
                        'dropout': dropout_encoder_model}


"""The hyperparameters for our Transformer encoder"""
# The dimensionality of the input embedding sequences for the Transformer encoder
embedding_dim = encoder_num_output_channels

# The size of the hidden dimension of the feed-forward neural network part of the Transformer encoder blocks
transformer_hidden_dim = 640

# The number of attention heads for each multi-head self-attention
num_attention_heads = 10

# The number of Transformer encoder blocks
num_transformer_encoder_layers = 6

# The dropout of the Transformer encoder blocks
dropout_transformer = 0.2

# The activation function for the Transformer feed-forward neural network part. Options: 'relu' and 'gelu'.
# ReLU was used in the original Transformer paper, whereas GELU was used in e.g. wav2vec 2.0 and data2vec.
transformer_activation_function = 'gelu'

# Defines whether we want to have the same number of embedding masks in each batch element (as was used in e.g. the
# original data2vec implementation in Fairseq). If set to True: After computing the embedding mask indices, the
# minimum number of embedding masks in a batch element is first defined. Then, for the rest of the batch elements
# containing more embedding masks, mask indices are randomly removed until each batch element has the same number
# of embedding masks. Please note that this might be problematic if there are large differences between the lengths
# of the batch elements (e.g. a long sample might have very few masks compared to the length of the sample).
require_same_num_embedding_masks = False

# The probability of a frame being the start of an embedding mask when masking embeddings
# for the student network
prob_frame_is_start_of_embedding_mask = 0.15

# The length of the embedding masks (in frames) when masking the embeddings for the student network
embedding_mask_length_frames = 3

# The minimum number of embedding mask starting frames in each embedding (the embedding mask start indices are
# chosen randomly, so without this parameter there is a chance that there might be # no masked frames at all
min_num_mask_start_frames = 1

# Defines whether we want to use a learnable mask embedding (as in e.g. the data2vec paper). If set to False,
# the masked parts of the embeddings are replaced with a mask token (see next hyperparameter)
learnable_mask_embedding = False

# Defines the type of the mask token. Options: 'random' / 'ones' / 'zeros'. This hyperparameter is neglected
# if learnable_mask_embedding = True
mask_type = 'ones'

# Defines what output of the Transformer encoder blocks we want to use as our training targets (the targets are
# instance-normalized and averaged, except for the option None). There are three possible options, all present in 
# the data2vec paper (Table 4):
#     None: The output of the last Transformer encoder block, without instance-normalizing or averaging
#    'ff_outputs': The output of the feed-forward (FFN) part of the Transformer encoder
#    'ff_residual_outputs': The output of the FFN of the Transformer encoder after adding the residual
#    'end_of_block': The output of the FFN of the Transformer encoder after the residual connection and LayerNorm
target_output_type = None

# Defines whether our Transformer encoder is bidirectional (False) or left-to-right (True). In e.g. data2vec
# and BERT, a bidirectional version was used.
only_attend_to_previous_context = False

# Defines whether we want to multiply the embeddings with the square root of the model dimensionality
# before we compute the positional encodings. In the original Transformer paper, this was done to make
# the positional encodings less dominant compared to the embeddings.
use_sqrt = False

# Defines whether we want to apply a linear projection to the embeddings after the positional encoding.
use_embedding_projection = False

# Defines whether we want to apply a linear projection after the final Transformer encoder block
use_final_projection = True

# Defines whether we want to use absolute positional encodings (using sinusoids) or relative positional
# encodings (using a CNN layer) for our embeddings. Relative positional encoding was used in e.g. the
# data2vec paper, whereas absolute positional encodings were used in the original Transformer paper.
#     Options: 'absolute' or 'relative'
positional_encoding_type = 'relative'

# Defines the dropout of our positional encodings (applies to both the absolute and relative positional
# encodings). In the original Transformer paper, a dropout of 0.1 was used as a regularization technique
dropout_pos_encoding = 0.0

# (Only related to absolute positional encodings) Defines the maximum sequence length in frames
abs_pos_encoding_max_sequence_length = train_sequence_length

# (Only related to relative positional encodings)
rel_pos_encoding_conv_in_dim = encoder_num_output_channels # The input dimensionality of the positional encodings
rel_pos_encoding_conv_out_dim = encoder_num_output_channels # The output dimensionality of the positional encodings
rel_pos_encoding_conv_kernel_size = 13 # The CNN kernel size of the positional encodings
rel_pos_encoding_conv_stride = 1 # The CNN stride of the positional encodings
rel_pos_encoding_conv_padding = 6 # The CNN padding of the positional encodings
rel_pos_encoding_conv_bias = False # The CNN bias of the pos. encodings (not used in wav2vec 2.0 and data2vec papers)
rel_pos_encoding_use_layernorm = True # Defines whether we want to apply LayerNorm after the positional encoding

# The hyperparameters for constructing the Transformer model. Empty dictionary = use default hyperparameters
transformer_params = {'dim_model': embedding_dim,
                      'dim_feedforward': transformer_hidden_dim,
                      'num_heads': num_attention_heads,
                      'num_encoder_layers': num_transformer_encoder_layers,
                      'dropout': dropout_transformer,
                      'transformer_activation_function': transformer_activation_function,
                      'require_same_num_embedding_masks': require_same_num_embedding_masks,
                      'prob_frame_is_start_of_embedding_mask': prob_frame_is_start_of_embedding_mask,
                      'embedding_mask_length_frames': embedding_mask_length_frames,
                      'min_num_mask_start_frames': min_num_mask_start_frames,
                      'learnable_mask_embedding': learnable_mask_embedding,
                      'mask_type': mask_type,
                      'only_attend_to_previous_context': only_attend_to_previous_context,
                      'use_sqrt': use_sqrt,
                      'use_embedding_projection': use_embedding_projection,
                      'use_final_projection': use_final_projection,
                      'positional_encoding_type': positional_encoding_type,
                      'dropout_pos_encoding': dropout_pos_encoding,
                      'abs_pos_encoding_max_sequence_length': abs_pos_encoding_max_sequence_length,
                      'rel_pos_encoding_conv_in_dim': rel_pos_encoding_conv_in_dim,
                      'rel_pos_encoding_conv_out_dim': rel_pos_encoding_conv_out_dim,
                      'rel_pos_encoding_conv_kernel_size': rel_pos_encoding_conv_kernel_size,
                      'rel_pos_encoding_conv_stride': rel_pos_encoding_conv_stride,
                      'rel_pos_encoding_conv_padding': rel_pos_encoding_conv_padding,
                      'rel_pos_encoding_conv_bias': rel_pos_encoding_conv_bias,
                      'rel_pos_encoding_use_layernorm': rel_pos_encoding_use_layernorm}


"""The hyperparameters for our PFML decoders"""
# The hyperparameters of our decoder model. Empty dictionary = use default hyperparameters
decoder_params = {'input_dim': embedding_dim,
                  'output_dim': num_input_channels * 11}




"""The hyperparameters for training data augmentation"""

# Select whether we want to use data augmentation for our training data or not
use_augmentation = False

# Probability for additive noise augmentation
aug_p_noise = 0.0

# If we perform additive noise augmentation, the probability for adding noise to samples
aug_p_dropout = 0.0

# Probability for performing a random rotation
aug_p_rotation = 0.3

# Probability for sensor dropout
aug_p_chandropout = 0.1

# Probability for time warping
aug_p_time_warping = 0.0




"""The hyperparameters for our dataset and data loaders"""

# The number of randomly generated multi-sensor IMU recordings of babies
num_randomly_generated_babydata = 20

# Define our dataset for our data loader that we want to use from the file pfml_data_loader.py
dataset_name = 'random_imu_data_dataset'

# The ratio in which we split our training data into training and validation sets. For example, a ratio
# of 0.8 will result in 80% of our training data being in the training set and 20% in the validation set.
train_val_ratio = 0.8

# Select whether we want to shuffle our training data
shuffle_training_data = True

# Select if we want to split our training and validation data so that baby-specific data is included
# in both sets.
mix_train_val_babies = False

# The hyperparameters for our data loaders
params_train_dataset = {'train_sequence_length': train_sequence_length,
                        'train_val_ratio': train_val_ratio,
                        'window_len': window_len,
                        'hop_len': hop_len,
                        'mix_train_val_babies': mix_train_val_babies,
                        'augment_train_data': use_augmentation,
                        'aug_p_noise': aug_p_noise,
                        'aug_p_dropout': aug_p_dropout,
                        'aug_p_rotation': aug_p_rotation,
                        'aug_p_chandropout': aug_p_chandropout,
                        'aug_p_time_warping': aug_p_time_warping}

params_validation_dataset = {'train_sequence_length': train_sequence_length,
                             'train_val_ratio': train_val_ratio,
                             'window_len': window_len,
                             'hop_len': hop_len,
                             'mix_train_val_babies': mix_train_val_babies}

params_feature_extraction_dataset = {'train_sequence_length': train_sequence_length,
                                     'window_len': window_len,
                                     'hop_len': hop_len}

# The hyperparameters for training and validation (arguments for torch.utils.data.DataLoader object)
params_train = {'batch_size': batch_size,
                'shuffle': shuffle_training_data,
                'drop_last': False}

# The hyperparameters for using our trained PFML model to extract features (arguments for
# torch.utils.data.DataLoader object)
params_feature_extraction = {'batch_size': batch_size,
                             'shuffle': False,
                             'drop_last': False}
