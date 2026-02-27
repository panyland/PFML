# -*- coding: utf-8 -*-
"""
Author: Einari Vaaras, einari.vaaras@tuni.fi, Tampere University
Speech and Cognition Research Group, https://webpages.tuni.fi/specog/index.html

The data loaders for PFML pre-training and fine-tuning for three different data
modalities (speech data, multi-sensor IMU data, and EEG data). The data augmentation
scripts for multi-sensor IMU data have been implemented by Manu Airaksinen.

NOTE: For detailed descriptions regarding the input variables for the data loaders,
see the configuration files.

"""

import numpy as np
from torch.utils.data import Dataset
import os
import sys
import librosa
import scipy




class pfml_raw_audio_dataset_librispeech(Dataset):
    """
    Dataloader for PFML pre-training using the Librispeech (https://www.openslr.org/12) dataset.
    
    """
    
    def __init__(self, train_val_test='train', max_length_seconds=3.0, train_val_ratio=0.8, random_seed=22,
                 file_dir='./LibriSpeech', normalize_waveform=True, window_len_seconds=0.03, hop_len_seconds=0.01,
                 target_fs=16000, apply_smooth_windowing=False, normalize_functionals_sample_level=False,
                 normalize_functionals_corpus_level=True, functionals_include_mean=True,
                 functionals_include_var=True, functionals_include_skew=True, functionals_include_kurtosis=True,
                 functionals_include_min=True, functionals_include_max=True, functionals_include_zcr=True,
                 functionals_include_acf_mean=True, functionals_include_acf_var=True,
                 functionals_include_acf_skew=True, functionals_include_acf_kurtosis=True):
        super().__init__()
        
        # Find out our FLAC files in the given directory
        try:
            # This is used to spot nonexisting directories since os.walk() is silent about them
            error_variable = os.listdir(file_dir)
            del error_variable
            
            filenames_flac = []
            for dir_path, dir_names, file_names in os.walk(file_dir):
                if len(file_names) > 0:
                    for file_name in file_names:
                        filenames_flac.append(os.path.join(dir_path, file_name))
        except FileNotFoundError:
            sys.exit(f'Given .flac file directory {file_dir} does not exist!')
        
        # Clean the list if there are other files than .flac files
        flac_file_names = [filename for filename in filenames_flac if filename.endswith('.flac')]
        flac_file_names = sorted(flac_file_names, key=lambda x: (int(x.split(os.sep)[-1].split('.')[0].split('-')[0]),
                                                                 int(x.split(os.sep)[-1].split('.')[0].split('-')[1]),
                                                                 int(x.split(os.sep)[-1].split('.')[0].split('-')[2])))
        flac_file_names = np.array(flac_file_names)
        del filenames_flac
        
        # We go through each WAV file and we frame the signals
        feats = []
        for file in flac_file_names:
            x, fs = librosa.core.load(file, sr=target_fs)
            
            # Normalize to zero mean, unit variance
            if normalize_waveform:
                x = (x - x.mean()) / x.std()
            
            # We frame our signal
            frame_len = int(window_len_seconds * fs)
            shift = int(hop_len_seconds * fs)
            
            # x_framed is of size [num_frames, frame_len]
            x_framed = librosa.util.frame(x, frame_length=frame_len, hop_length=shift, axis=0)
            
            if apply_smooth_windowing:
                # We apply a Hann window for our frames
                window = scipy.signal.hann(frame_len, sym=False)
                x_framed_windowed = np.zeros_like(x_framed)
                for i in range(x_framed.shape[0]):
                    x_framed_windowed[i,:] = x_framed[i,:] * window
                x_framed = x_framed_windowed
            
            feats.append(x_framed)
        
        # We convert the list of variable-length features into a Numpy object
        feats = np.array(feats, dtype=object)
        
        # We define the longest sample length (in frames)
        x_zeros = np.zeros(int(max_length_seconds*target_fs))
        self.x_zeros_framed = librosa.util.frame(x_zeros, frame_length=frame_len, hop_length=shift, axis=0)
        self.longest_sample_length = len(self.x_zeros_framed)
        
        # We compute functionals of the features
        feats_functionals = []
        for feat in feats:
            functionals = []
            if functionals_include_mean:
                functionals.append(np.mean(feat, axis=1))
            if functionals_include_var:
                functionals.append(np.var(feat, axis=1))
            if functionals_include_skew:
                functionals.append(scipy.stats.skew(feat, axis=1))
            if functionals_include_kurtosis:
                functionals.append(scipy.stats.kurtosis(feat, axis=1))
            if functionals_include_min:
                functionals.append(feat.min(axis=1))
            if functionals_include_max:
                functionals.append(feat.max(axis=1))
            if functionals_include_zcr:
                functionals.append(librosa.zero_crossings(feat, axis=1).sum(axis=1) / frame_len)
            if functionals_include_acf_mean or functionals_include_acf_var or functionals_include_acf_skew or functionals_include_acf_kurtosis:
                ac = estimated_autocorrelation(feat)
                if functionals_include_acf_mean:
                    functionals.append(np.mean(ac, axis=1))
                if functionals_include_acf_var:
                    functionals.append(np.var(ac, axis=1))
                if functionals_include_acf_skew:
                    functionals.append(scipy.stats.skew(ac, axis=1))
                if functionals_include_acf_kurtosis:
                    functionals.append(scipy.stats.kurtosis(ac, axis=1))
            functionals = np.stack(functionals, axis=1)
            if normalize_functionals_sample_level:
                feats_functionals.append(normalize_sample(functionals))
            else:
                feats_functionals.append(functionals)
        
        if normalize_functionals_corpus_level:
            feats_functionals = normalize_dataset(feats_functionals)
        
        feats_functionals = np.array(feats_functionals, dtype=object)
                
        # Split our data into a train, validation, and test set
        np.random.seed(random_seed)
        mask_trainval_split = np.random.rand(len(flac_file_names)) <= train_val_ratio
        
        # train_val_test has three options: 'train', 'validation' and 'test'. We use 'test' when we want to extract
        # features using a trained PFML model, i.e. we use all of our data with the option 'test'.
        if train_val_test == 'train':
            self.feats = feats[mask_trainval_split]
            self.feats_functionals = feats_functionals[mask_trainval_split]
        elif train_val_test == 'validation':
            self.feats = feats[~mask_trainval_split]
            self.feats_functionals = feats_functionals[~mask_trainval_split]
        else:
            self.feats = feats
            self.feats_functionals = feats_functionals
        
        self.train_val_test = train_val_test
        
    def __len__(self) -> int:
        return len(self.feats)

    def __getitem__(self, index):
        
        framed_signal_orig = self.feats[index]
        functionals = self.feats_functionals[index]
        
        # If our sample is shorter than the longest acceptable sample, we add a zero-padded part to the end
        if len(framed_signal_orig) < self.longest_sample_length:
            num_missing_frames = self.longest_sample_length - len(framed_signal_orig)
            framed_signal = np.concatenate((framed_signal_orig, self.x_zeros_framed[:num_missing_frames, :]))
            num_zero_padded_frames = num_missing_frames
            functionals_zeropad = np.zeros((len(framed_signal), functionals.shape[1]))
            functionals = np.concatenate((functionals, functionals_zeropad[:num_zero_padded_frames, :]))
            
        # If our sample is longer than the longest acceptable sample, we take a random segment of the same
        # length as the longest acceptable sample length
        elif len(framed_signal_orig) > self.longest_sample_length:
            if self.train_val_test == 'test':
                np.random.seed(12)
            part_index = np.random.randint(len(framed_signal_orig) - self.longest_sample_length + 1)
            framed_signal = framed_signal_orig[part_index:(part_index + self.longest_sample_length)]
            num_zero_padded_frames = 0
            functionals = functionals[part_index:(part_index + self.longest_sample_length)]
        else:
            framed_signal = framed_signal_orig
            num_zero_padded_frames = 0
            
        # The indices of zero padded frames are tagged with True, whereas non-padded frames are tagged with False
        zero_padding_mask = np.full(len(framed_signal), False)
        if num_zero_padded_frames != 0:
            zero_padding_mask[-num_zero_padded_frames:] = True
        
        return framed_signal, zero_padding_mask, functionals






    

class random_imu_data_dataset(Dataset):
    """
    Dataloader for PFML pre-training and pre-trained model fine-tuning using randomly generated multi-sensor IMU data.
    
    """

    def __init__(self, data_list, train_val_test = 'train', train_sequence_length = 260, train_val_ratio = 0.8,
                 random_seed = 42, window_len = 120, hop_len = 60, mix_train_val_babies = False,
                 augment_train_data = False, aug_p_noise = 0.0, aug_p_dropout = 0.1, aug_p_rotation = 0.3,
                 aug_p_chandropout = 0.3, aug_p_time_warping = 0.0, data_sampling_rate=1.0,
                 include_artificial_labels=False, normalize_functionals_sample_level=False,
                 normalize_functionals_dataset_level=True, functionals_include_mean=True,
                 functionals_include_var=True, functionals_include_skew=True, functionals_include_kurtosis=True,
                 functionals_include_min=True, functionals_include_max=True, functionals_include_zcr=True,
                 functionals_include_acf_mean=True, functionals_include_acf_var=True,
                 functionals_include_acf_skew=True, functionals_include_acf_kurtosis=True):
        super().__init__()
        
        if train_val_test == 'train' and augment_train_data:
            self.augment = augment_train_data
            self.aug_p_noise = aug_p_noise
            self.aug_p_dropout = aug_p_dropout
            self.aug_p_rotation = aug_p_rotation
            self.aug_p_chandropout = aug_p_chandropout
            self.aug_p_time_warping = aug_p_time_warping
            self.window_len = window_len
            self.hop_len = hop_len
        else:
            self.augment = False
        
        X = []
        data_masks = []
        if not mix_train_val_babies and train_val_test != 'test':
            # We split our training and validation data so that baby-specific data is not included in both sets.
            num_train_babies = int(np.round(train_val_ratio*len(data_list)))
            train_val_babies_permutation = np.random.RandomState(seed=random_seed*2).permutation(len(data_list))
            if train_val_test == 'train':
                data_list = [data_list[i] for i in train_val_babies_permutation[:num_train_babies]]
            else:
                data_list = [data_list[i] for i in train_val_babies_permutation[num_train_babies:]]
        
        # We go through the data sequences one at a time and we append them to their appropriate lists.
        for baby_data in data_list:
            data_in = baby_data['X']
            data_mask = baby_data['Mask']
            num_sequences = data_in.shape[0] // train_sequence_length
            leftover_sequence_len = data_in.shape[0] % train_sequence_length
            if not mix_train_val_babies or train_val_test == 'test':
                for i in range(num_sequences):
                    X.append(data_in[i*train_sequence_length:(i+1)*train_sequence_length,:,:])
                    data_masks.append(data_mask[i*train_sequence_length:(i+1)*train_sequence_length])
            else:
                num_train_seq = int(np.round(train_val_ratio*num_sequences)) # The number of training data sequences
                train_val_permutation = np.random.RandomState(seed=random_seed).permutation(num_sequences)
                if train_val_test == 'train':
                    sequences = train_val_permutation[:num_train_seq]
                else:
                    sequences = train_val_permutation[num_train_seq:]
                
                for i in sequences:
                    X.append(data_in[i*train_sequence_length:(i+1)*train_sequence_length,:,:])
                    data_masks.append(data_mask[i*train_sequence_length:(i+1)*train_sequence_length])
                
            if leftover_sequence_len != 0 and (train_val_test != 'validation' or not mix_train_val_babies):
                # We add the last sequence that is shorter than others and pad it to be of equal length
                X_leftover = np.copy(data_in[i*train_sequence_length:(i+1)*train_sequence_length,:,:])
                X_leftover[:leftover_sequence_len] = data_in[-leftover_sequence_len:, :, :]
                X.append(X_leftover)
                leftover_mask = np.ones_like(data_mask[i*train_sequence_length:(i+1)*train_sequence_length])
                leftover_mask[:leftover_sequence_len] = data_mask[-leftover_sequence_len:]
                data_masks.append(leftover_mask)
        
        self.X = np.array(X)
        self.data_masks = np.array(data_masks)
        
        # We compute functionals of the features
        feats_functionals = []
        for feat in X:
            functionals = []
            if functionals_include_mean:
                functionals.append(np.mean(feat, axis=2))
            if functionals_include_var:
                functionals.append(np.var(feat, axis=2))
            if functionals_include_skew:
                functionals.append(scipy.stats.skew(feat, axis=2))
            if functionals_include_kurtosis:
                functionals.append(scipy.stats.kurtosis(feat, axis=2))
            if functionals_include_min:
                functionals.append(feat.min(axis=2))
            if functionals_include_max:
                functionals.append(feat.max(axis=2))
            if functionals_include_zcr:
                functionals.append(librosa.zero_crossings(feat, axis=2).sum(axis=2) / window_len)
            if functionals_include_acf_mean or functionals_include_acf_var or functionals_include_acf_skew or functionals_include_acf_kurtosis:
                if functionals_include_acf_mean:
                    ac_channel_mean = []
                if functionals_include_acf_var:
                    ac_channel_var = []
                if functionals_include_acf_skew:
                    ac_channel_skew = []
                if functionals_include_acf_kurtosis:
                    ac_channel_kurtosis = []
                for i in range(feat.shape[1]):
                    feat_channel = feat[:, i, :]
                    ac = estimated_autocorrelation(feat_channel)
                    if functionals_include_acf_mean:
                        ac_channel_mean.append(np.mean(ac, axis=1))
                    if functionals_include_acf_var:
                        ac_channel_var.append(np.var(ac, axis=1))
                    if functionals_include_acf_skew:
                        ac_channel_skew.append(scipy.stats.skew(ac, axis=1))
                    if functionals_include_acf_kurtosis:
                        ac_channel_kurtosis.append(scipy.stats.kurtosis(ac, axis=1))
                if functionals_include_acf_mean:
                    ac_channel_mean = np.transpose(np.array(ac_channel_mean))
                    functionals.append(ac_channel_mean)
                if functionals_include_acf_var:
                    ac_channel_var = np.transpose(np.array(ac_channel_var))
                    functionals.append(ac_channel_var)
                if functionals_include_acf_skew:
                    ac_channel_skew = np.transpose(np.array(ac_channel_skew))
                    functionals.append(ac_channel_skew)
                if functionals_include_acf_kurtosis:
                    ac_channel_kurtosis = np.transpose(np.array(ac_channel_kurtosis))
                    functionals.append(ac_channel_kurtosis)
            functionals = np.stack(functionals, axis=2)
            
            # We reshape the functional array from the shape [train_sequence_length, num_channels, num_functionals]
            # into the shape [train_sequence_length, num_channels * num_functionals]
            functionals = functionals.reshape(functionals.shape[0], -1)
            
            if normalize_functionals_sample_level:
                feats_functionals.append(normalize_sample(functionals))
            else:
                feats_functionals.append(functionals)
        
        if normalize_functionals_dataset_level:
            feats_functionals = normalize_dataset(feats_functionals)
        
        self.feats_functionals = np.array(feats_functionals)
        
        # We create artificial labels for our randomly generated dataset. There are nine different labels
        # for movement in MAIJU data.
        # Tämä muutetaan niin, että tuodaan oikeat labelit data_listista
        """         
        if include_artificial_labels:
            Y = np.zeros((len(self.X), train_sequence_length, 9))
            for i in range(len(Y)):
                for j in range(train_sequence_length):
                    random_vec = np.random.rand(Y.shape[2])
                    max_ind = np.argmax(random_vec)
                    Y[i, j, max_ind] = 1.0
        
            self.Y = Y
         """
        if include_artificial_labels:
            Y = []
            

            # We need to slice B1 in the same way X was sliced
            idx = 0
            for baby_data in data_list:
                labels_in = baby_data['B1']  # shape (T, num_classes)
                T = labels_in.shape[0]

                num_sequences = T // train_sequence_length
                leftover = T % train_sequence_length

                # Same logic as X slicing
                if not mix_train_val_babies or train_val_test == 'test':
                    sequences = range(num_sequences)
                else:
                    num_train_seq = int(np.round(train_val_ratio * num_sequences))
                    perm = np.random.RandomState(seed=random_seed).permutation(num_sequences)
                    sequences = perm[:num_train_seq] if train_val_test == 'train' else perm[num_train_seq:]

                # Full sequences
                for i in sequences:
                    start = i * train_sequence_length
                    end = (i + 1) * train_sequence_length
                    Y.append(labels_in[start:end, :])

                # Leftover sequence (padded)
                if leftover != 0 and (train_val_test != 'validation' or not mix_train_val_babies):
                    start = sequences[-1] * train_sequence_length
                    end = (sequences[-1] + 1) * train_sequence_length

                    Y_left = np.copy(labels_in[start:end, :])
                    Y_left[:leftover] = labels_in[-leftover:, :]
                    Y.append(Y_left)

            self.Y = np.array(Y)

        
        self.include_artificial_labels = include_artificial_labels
        
        if data_sampling_rate < 1.00 and train_val_test != 'test':
            # We randomly select a subset of the data
            num_sampled = int(data_sampling_rate * len(X))
            np.random.seed(3*random_seed)
            sampling_indices = np.random.choice(np.arange(len(X)), num_sampled, replace=False)
            self.X = self.X[sampling_indices, :, :, :]
            self.data_masks = self.data_masks[sampling_indices, :]
            self.feats_functionals = self.feats_functionals[sampling_indices, :, :]
            if include_artificial_labels:
                self.Y = self.Y[sampling_indices, :, :]

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index):
        
        if self.augment:
            X = data_augmentation(self.X[index], self.aug_p_noise, self.aug_p_dropout, self.aug_p_rotation,
                                  self.aug_p_chandropout, self.aug_p_time_warping, self.window_len, self.hop_len)
        else:
            X = self.X[index]
        
        if self.include_artificial_labels:
            target_labels = self.Y[index]
        else:
            target_labels = 0
        
        return X, target_labels, self.data_masks[index], self.feats_functionals[index]




class sleep_edf_expanded_dataset_pfml(Dataset):
    """
    Dataloader for PFML pre-training using the pre-processed Sleep-EDF Database Expanded dataset
    (https://github.com/emadeldeen24/AttnSleep).
    
    """

    def __init__(self, data_dir = './sleep_edf_78', preprocess_data = False,
                 preprocessed_data_dir = './preprocessed_sleep_edf_exp_files_framed',
                 precompute_functionals = False, functionals_save_dir = './precomputed_sleep_edf_exp_functionals',
                 train_val_test = 'train', train_val_ratio = 0.8, random_seed = 42, fs=100,
                 window_len_seconds=4.0, hop_len_seconds=2.0, normalize_functionals_sample_level=False, 
                 normalize_functionals_dataset_level=True, functionals_include_mean=True,
                 functionals_include_var=True, functionals_include_skew=True, functionals_include_kurtosis=True,
                 functionals_include_min=True, functionals_include_max=True, functionals_include_zcr=True,
                 functionals_include_acf_mean=True, functionals_include_acf_var=True,
                 functionals_include_acf_skew=True, functionals_include_acf_kurtosis=True, data_sampling_rate=1.0):
        super().__init__()
        
        # Preprocess the data
        if not os.path.exists(preprocessed_data_dir):
            os.makedirs(preprocessed_data_dir)
            preprocess_data = True
        else:
            if preprocess_data and len(os.listdir(preprocessed_data_dir)) != 0:
                # Remove old files from the given directory
                filenames_old_files = os.listdir(preprocessed_data_dir)
                for filename in filenames_old_files:
                    os.remove(os.path.join(preprocessed_data_dir, filename))
            
        if preprocess_data or len(os.listdir(preprocessed_data_dir)) == 0:
            # Find out our EDF files in the given directory
            try:
                filenames_edf = os.listdir(data_dir)
            except FileNotFoundError:
                sys.exit(f'Given EDF file directory {data_dir} does not exist!')
            
            # Remove other files that EDF files
            edf_file_names = [filename for filename in filenames_edf if filename.endswith('.npz')]
            del filenames_edf
            
            # Go through each MAT file and preprocess the data
            for filename in edf_file_names:
                X = np.load(os.path.join(data_dir, filename))['x'].squeeze()
                Y = np.load(os.path.join(data_dir, filename))['y']
                
                # X is now of shape [num_sequences, sequence_length]. We z-score normalize each sequence
                # to have zero mean and unit variance.
                for i in range(len(X)):
                    X[i,:] = (X[i,:] - X[i,:].mean()) / X[i,:].std()
                
                # We frame each sequence
                frame_len = int(window_len_seconds * fs)
                shift = int(hop_len_seconds * fs)
                
                data_framed = frame_sig_eeg(X, frame_len, shift)
                del X
                
                # Save the sequences in .npy format.
                for i in range(len(data_framed)):
                    savedata = data_framed[i,:,:,:]
                    label = Y[i]
                    savename = os.path.join(preprocessed_data_dir, f'{filename.split(".")[0]}_framed_{i}_{label}.npy')
                    np.save(savename, savedata)
        
        # List all of our preprocessed files
        preprocessed_files = [filename for filename in os.listdir(preprocessed_data_dir) if filename.endswith('.npy')]
        preprocessed_files = np.array(sorted(preprocessed_files))
        
        # Split our data into separate sets
        np.random.seed(random_seed)
        mask_trainval_split = np.random.rand(len(preprocessed_files)) <= train_val_ratio
        
        # train_val_test has three options: 'train', 'validation' and 'test'. We use 'test' when we want to extract
        # features using a trained data2vec model, i.e. we use all of our data with the option 'test'.
        if train_val_test == 'train':
            self.feat_files = preprocessed_files[mask_trainval_split]
        elif train_val_test == 'validation':
            self.feat_files = preprocessed_files[~mask_trainval_split]
        else:
            self.feat_files = preprocessed_files
        
        self.preprocessed_data_dir = preprocessed_data_dir
        
        
        # Pre-compute the functionals
        if not os.path.exists(functionals_save_dir):
            os.makedirs(functionals_save_dir)
            precompute_functionals = True
        else:
            if precompute_functionals and len(os.listdir(functionals_save_dir)) != 0:
                # Remove old files from the given directory
                filenames_old_files = os.listdir(functionals_save_dir)
                for filename in filenames_old_files:
                    os.remove(os.path.join(functionals_save_dir, filename))
            
        if precompute_functionals or len(os.listdir(functionals_save_dir)) == 0:
            # We go through each file one at a time and compute its functionals
            feats_functionals = []
            for i in range(len(preprocessed_files)):
                feat = np.load(os.path.join(preprocessed_data_dir, preprocessed_files[i])).squeeze()
                functionals = []
                if functionals_include_mean:
                    functionals.append(np.mean(feat, axis=1))
                if functionals_include_var:
                    functionals.append(np.var(feat, axis=1))
                if functionals_include_skew:
                    functionals.append(scipy.stats.skew(feat, axis=1))
                if functionals_include_kurtosis:
                    functionals.append(scipy.stats.kurtosis(feat, axis=1))
                if functionals_include_min:
                    functionals.append(feat.min(axis=1))
                if functionals_include_max:
                    functionals.append(feat.max(axis=1))
                if functionals_include_zcr:
                    functionals.append(librosa.zero_crossings(feat, axis=1).sum(axis=1) / frame_len)
                if functionals_include_acf_mean or functionals_include_acf_var or functionals_include_acf_skew or functionals_include_acf_kurtosis:
                    ac = estimated_autocorrelation(feat)
                    if functionals_include_acf_mean:
                        functionals.append(np.mean(ac, axis=1))
                    if functionals_include_acf_var:
                        functionals.append(np.var(ac, axis=1))
                    if functionals_include_acf_skew:
                        functionals.append(scipy.stats.skew(ac, axis=1))
                    if functionals_include_acf_kurtosis:
                        functionals.append(scipy.stats.kurtosis(ac, axis=1))
                functionals = np.stack(functionals, axis=1)
                if normalize_functionals_sample_level:
                    feats_functionals.append(normalize_sample(functionals))
                else:
                    feats_functionals.append(functionals)
            
            if normalize_functionals_dataset_level:
                feats_functionals = normalize_dataset(feats_functionals)
            
            # We save the functionals using .npy format
            for i in range(len(preprocessed_files)):
                savedata = feats_functionals[i]
                name_parts = preprocessed_files[i].split('_')
                savename = os.path.join(functionals_save_dir, f'{name_parts[0]}_{name_parts[1]}_{name_parts[2].split(".")[0]}_functionals.npy')
                np.save(savename, savedata)
        
        # List all of our preprocessed functional files
        preprocessed_functional_files = [filename for filename in os.listdir(functionals_save_dir) if filename.endswith('.npy')]
        preprocessed_functional_files = np.array(sorted(preprocessed_functional_files))
        
        # Split our data into separate sets
        if train_val_test == 'train':
            self.functional_files = preprocessed_functional_files[mask_trainval_split]
        elif train_val_test == 'validation':
            self.functional_files = preprocessed_functional_files[~mask_trainval_split]
        else:
            self.functional_files = preprocessed_functional_files
        
        if data_sampling_rate < 1.00 and train_val_test != 'test':
            # We randomly select a subset of the data
            num_sampled = int(data_sampling_rate * len(self.functional_files))
            np.random.seed(3*random_seed)
            sampling_indices = np.random.choice(np.arange(len(self.functional_files)), num_sampled, replace=False)
            self.feat_files = self.feat_files[sampling_indices]
            self.functional_files = self.functional_files[sampling_indices]
        
        self.functionals_save_dir = functionals_save_dir

    def __len__(self) -> int:
        return len(self.feat_files)

    def __getitem__(self, index):
        
        X = np.load(os.path.join(self.preprocessed_data_dir, self.feat_files[index]))
        feats_functionals = np.load(os.path.join(self.functionals_save_dir, self.functional_files[index]))
        data_mask = np.zeros((len(X)))
        
        return X, data_mask, feats_functionals







class random_speech_data_dataset(Dataset):
    """
    Dataloader for PFML pre-training and pre-trained model fine-tuning using randomly generated speech data.
    
    """ 

    def __init__(self, data_list, train_val_test = 'train', train_val_ratio = 0.8, random_seed = 42, 
                 data_sampling_rate=1.0, normalize_waveform=True, window_len_seconds=0.03, hop_len_seconds=0.01,
                 fs=16000, max_length_seconds=3.0, include_artificial_labels=True):
        super().__init__()
        
        
        X = []
        max_num_samples = int(max_length_seconds * fs)
        frame_len = int(window_len_seconds * fs)
        shift = int(hop_len_seconds * fs)
        
        for x in data_list:
            
            # Normalize to zero mean, unit variance
            if normalize_waveform:
                x = (x - x.mean()) / x.std()
            
            # We either truncate or zero-pad our signal to be of the length max_length_seconds
            if len(x) != max_num_samples:
                x = librosa.util.fix_length(x, size=max_num_samples)
            
            # We frame our signal. x_framed is of size [num_frames, frame_len]
            x_framed = librosa.util.frame(x, frame_length=frame_len, hop_length=shift, axis=0)
            
            X.append(x_framed)
        
        self.X = np.array(X)
        
        # We create artificial binary labels for our randomly generated dataset.
        if include_artificial_labels:
            Y = np.zeros((len(self.X), 2))
            for i in range(len(Y)):
                random_vec = np.random.rand(Y.shape[1])
                max_ind = np.argmax(random_vec)
                Y[i, max_ind] = 1.0
            self.Y = Y
        
        self.include_artificial_labels = include_artificial_labels
        
        if train_val_test != 'test':
            mask_trainval_split = np.random.rand(len(self.X)) <= train_val_ratio
            if train_val_test == 'train':
                self.X = self.X[mask_trainval_split]
                if include_artificial_labels:
                    self.Y = self.Y[mask_trainval_split]
            else:
                self.X = self.X[~mask_trainval_split]
                if include_artificial_labels:
                    self.Y = self.Y[~mask_trainval_split]
        
        if data_sampling_rate < 1.00 and train_val_test != 'test':
            # We randomly select a subset of the data
            num_sampled = int(data_sampling_rate * len(X))
            np.random.seed(3*random_seed)
            sampling_indices = np.random.choice(np.arange(len(X)), num_sampled, replace=False)
            self.X = self.X[sampling_indices, :, :]
            if include_artificial_labels:
                self.Y = self.Y[sampling_indices, :]

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index):
        
        if self.include_artificial_labels:
            return self.X[index], self.Y[index]
        else:
            return self.X[index], 0










class sleep_edf_expanded_dataset_pfml_finetuning(Dataset):
    """
    Dataloader for fine-tuning PFML pre-trained models using the pre-processed Sleep-EDF Database Expanded dataset
    (https://github.com/emadeldeen24/AttnSleep).
    
    """

    def __init__(self, test_subject_index_list, preprocessed_data_dir = './preprocessed_sleep_edf_exp_files_framed',
                 train_val_test = 'train', train_val_ratio = 0.8, random_seed = 42, mix_train_val_subjects = False,
                 data_sampling_rate=1.0):
        super().__init__()
        
        # Find out our EDF files in the given directory
        try:
            filenames_eeg = os.listdir(preprocessed_data_dir)
        except FileNotFoundError:
            sys.exit(f'Given EEG file directory {preprocessed_data_dir} does not exist!')
        
        # Remove other files that EEG files
        eeg_file_names = [filename for filename in filenames_eeg if filename.endswith('.npy')]
        del filenames_eeg
        
        X = []
        Y = []
        
        if not mix_train_val_subjects and train_val_test != 'test':
            # We split our training and validation data so that test subject-specific data is not included in both sets.
            num_train_test_subjects = int(np.round(train_val_ratio*len(test_subject_index_list))) # The number of training data sequences
            train_val_test_subjects_permutation = np.random.RandomState(seed=random_seed*2).permutation(len(test_subject_index_list))
            if train_val_test == 'train':
                test_subject_index_list = [test_subject_index_list[i] for i in train_val_test_subjects_permutation[:num_train_test_subjects]]
            else:
                test_subject_index_list = [test_subject_index_list[i] for i in train_val_test_subjects_permutation[num_train_test_subjects:]]
        
        # We go through the data sequences one at a time and we append them to their appropriate lists.
        for test_subject_index in test_subject_index_list:
            test_subject_data = []
            test_subject_labels = []
            for i in range(len(eeg_file_names)):
                if eeg_file_names[i][3:5] == test_subject_index:
                    test_subject_data.append(np.load(os.path.join(preprocessed_data_dir, eeg_file_names[i])))
                    test_subject_labels.append(float(eeg_file_names[i].split('.')[0].split('_')[-1]))
            X += test_subject_data
            Y += test_subject_labels
            
        if not mix_train_val_subjects or train_val_test == 'test':
            self.X = np.array(X)
            self.Y = np.array(Y)
        else:
            np.random.seed(random_seed*5)
            mask_trainval_split = np.random.rand(len(eeg_file_names)) <= train_val_ratio
            if train_val_test == 'train':
                self.X = np.array(X)[mask_trainval_split]
                self.Y = np.array(Y)[mask_trainval_split]
            else:
                self.X = np.array(X)[~mask_trainval_split]
                self.Y = np.array(Y)[~mask_trainval_split]

        
        if data_sampling_rate < 1.00 and train_val_test != 'test':
            # We randomly select a subset of the data
            num_sampled = int(data_sampling_rate * len(X))
            np.random.seed(3*random_seed)
            sampling_indices = np.random.choice(np.arange(len(X)), num_sampled, replace=False)
            self.X = self.X[sampling_indices, :, :, :]
            self.Y = self.Y[sampling_indices]

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index):
        
        data_mask = np.zeros((len(self.X[index])))
        
        return self.X[index], self.Y[index], data_mask





# Normalize the 2D input sample to have zero mean and unit variance along each feature. The dimensions of
# the input are (frame_index, feature_index).
def normalize_sample(feats) -> np.ndarray:
    
    normalized = (feats - feats.mean(axis=0)) / feats.std(axis=0)
    
    # Remove NaN values by converting them to zero
    normalized = np.nan_to_num(normalized)
    
    return normalized


# Normalize the 3D input features (can be different-length) to have zero mean and unit variance
# -> the input is a list with samples of dimensions (frame_index, feature_index).
def normalize_dataset(feat_list):
    
    feats_unrolled = np.nan_to_num(np.concatenate(feat_list, axis=0))
    feat_mean = feats_unrolled.mean(axis=0)
    feat_std = feats_unrolled.std(axis=0)
    del feats_unrolled
    
    for i in range(len(feat_list)):
        feat_list[i] = (feat_list[i] - feat_mean) / feat_std
        feat_list[i] = np.nan_to_num(feat_list[i]) # Remove NaN values by converting them to zero
    
    return feat_list



def estimated_autocorrelation(frames):
    
    ac = []
    for x in frames:
        n = len(x)
        variance = x.var()
        x = x - x.mean()
        r = np.correlate(x, x, mode = 'full')[-n:]
        result = r/(variance*(np.arange(n, 0, -1)))
        ac.append(result)
        
    return np.array(ac)



def time_warping(data, p=1.0, winlen=120):
    basevec = np.arange(winlen) + 1.0
    Nframes = int(np.floor(((data.shape[0] - winlen)/winlen) + 1))
    for iFrame in range(Nframes):
        # Randomly warp p*100% of frames
        if np.random.random_sample() <= p:
        # Random sinusoid with random phase, amplitude [0.5, 1.5], frequency
            freq = np.random.random_sample() * basevec / basevec.shape[0]
            phase = 2 * np.pi * np.random.random_sample()
            amplitude = np.random.random_sample()
            sinusoid = amplitude * np.sin(2 * np.pi * freq + phase) + 2
            sinusoid /= np.mean(sinusoid)

            newbase = np.cumsum(sinusoid)
            start = iFrame * winlen
            stop = start + winlen
            for iChan in range(data.shape[1]):
                data[start:stop,iChan] = np.interp(newbase, basevec, data[start:stop, iChan])

    return data



def rotationMatrix(a_x, a_y, a_z, angle_type='deg'):
    if angle_type == 'deg':
        a_x *= np.pi / 180.0
        a_y *= np.pi / 180.0
        a_z *= np.pi / 180.0

    M = np.array([[np.cos(a_y) * np.cos(a_z), 
                   -np.cos(a_x) * np.sin(a_z) + np.sin(a_x) * np.sin(a_y) * np.cos(a_z), 
                   np.sin(a_x) * np.sin(a_z) + np.cos(a_x) * np.sin(a_y) * np.cos(a_z)],
                  [np.cos(a_y) * np.sin(a_z), 
                   np.cos(a_x) * np.cos(a_z) + np.sin(a_x) * np.sin(a_y) * np.sin(a_z),
                   -np.sin(a_x) * np.cos(a_z) + np.cos(a_x) * np.sin(a_y) * np.sin(a_z)],
                  [-np.sin(a_y),
                   np.sin(a_x) * np.cos(a_y),
                   np.cos(a_x) * np.cos(a_y)]])

    return M




def random_rotation(data, angle=15.0):
    # Get rotation matrix, random rotation for each sensor
    range_x = [-angle, angle]
    range_y = [-angle, angle]
    range_z = [-angle, angle]
    Nsens = data.shape[1] // 6
    n = data.shape[-1] // 2
    acc = data[:,:n]
    gyro = data[:,n:]
    for i in range(Nsens):
        a_x = np.random.random_sample() * (range_x[1] - range_x[0]) + range_x[0]
        a_y = np.random.random_sample() * (range_y[1] - range_y[0]) + range_y[0]
        a_z = np.random.random_sample() * (range_z[1] - range_z[0]) + range_z[0]
        M = rotationMatrix(a_x, a_y, a_z)
        
        acc[:,i*3:(i+1)*3] = np.matmul(acc[:,i*3:(i+1)*3], M)
        gyro[:,i*3:(i+1)*3] = np.matmul(gyro[:,i*3:(i+1)*3], M)

    data = np.concatenate([acc, gyro], axis=-1)

    return data




def dropout_noise(data, p):
    mask = np.random.binomial(1, 1.0 - p, data.shape)
    
    return data * mask



def channel_dropout(data, num_chans=1, tot_chans=4):
    chans_to_drop = np.random.permutation(tot_chans)
    chans_to_drop = chans_to_drop[:num_chans]
    N = data.shape[-1] // 2
    for i in chans_to_drop:
        data[:,(3*i):(3*i+3)] *= 0.0 # Accelerometer signals
        data[:,(N+3*i):(N+3*i+3)] *= 0.0 # Gyroscope signals

    return data



def frame_sig(X, winlen, hop):
    Nframes = int(np.floor(((X.shape[0] - winlen)/hop) + 1))
    numchans = X.shape[1]
    X_framed = np.zeros([Nframes, numchans, winlen], dtype=np.float32) # [Nframes, Nchans, winlen]
    for i in range(0, Nframes):
        start = i * hop
        stop = start + winlen
        X_framed[i,:,:] = np.transpose(X[start:stop,:])

    return X_framed





def data_augmentation(data, aug_p_noise, aug_p_dropout, aug_p_rotation, aug_p_chandropout,
                      aug_p_time_warping, window_len, hop_len):
    
    # Augmentation to frames, assume data is 50% overlapped
    N = data.shape[-1] // 2
    data = np.concatenate([np.reshape(np.transpose(data[:,:,:N], [0,2,1]), [-1, data.shape[1]]),
                           np.transpose(data[-1,:,N:])], axis=0)

    # Time warping
    if np.random.random_sample() < aug_p_time_warping:
        data = time_warping(data, p=1.0, winlen=window_len)

    # Random rotation
    if np.random.random_sample() < aug_p_rotation:
        data = random_rotation(data)

    # Additive noise augmentation
    if np.random.random_sample() < aug_p_noise:
        data = dropout_noise(data, aug_p_dropout)

    # Sensor dropout
    if np.random.random_sample() < aug_p_chandropout:
        data = channel_dropout(data, num_chans=1)

    # Retain framed format
    data = frame_sig(data, window_len, hop_len)

    return data



def frame_sig_eeg(X, winlen, hop):
    """
    The input data should be either of size [num_sequences, num_channels, sequence_length]
    or of size [num_sequences, sequence_length]
    
    Output is of size [num_sequences, Nframes, num_channels, winlen]
    """
    
    if len(X.shape) < 3:
        # We add a dummy channel to the data
        X = np.expand_dims(X, axis=1)
    
    Nframes = int(np.floor(((X.shape[2] - winlen)/hop) + 1))
    num_channels = X.shape[1]
    num_sequences = X.shape[0]
    
    X_framed = np.zeros([num_sequences, Nframes, num_channels, winlen], dtype=np.float32)
    for i in range(num_sequences):
        for j in range(0, Nframes):
            start = j * hop
            stop = start + winlen
            X_framed[i,j,:,:] = X[i,:,start:stop]

    return X_framed




