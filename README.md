# A PyTorch implementation of the PFML algorithm for speech, EEG, and multi-sensor inertial measurement unit (IMU) data

This repository contains code for pre-training models using the [Prediction of Functionals from Masked Latents (PFML) algorithm](https://ieeexplore.ieee.org/document/10947019) for speech, EEG, and multi-sensor IMU data, and also code for fine-tuning the pre-trained models using labeled data. The code has been implemented using PyTorch. For a thorough description of the PFML algorithm, see Section III of [the publication](https://ieeexplore.ieee.org/document/10947019). The arXiv pre-print version of the paper is available [here](https://arxiv.org/abs/2411.10087).

**The present PFML implementation has been used in the following publication:**
[E. Vaaras, M. Airaksinen, and O. Räsänen, "PFML: Self-Supervised Learning of Time-Series Data Without Representation Collapse", _IEEE Access_, vol. 13, pp. 60233–60244, 2025](https://ieeexplore.ieee.org/document/10947019).

If you use the present code or its derivatives, please cite the [repository URL](https://github.com/SPEECHCOG/PFML) and/or the [aforementioned publication](https://ieeexplore.ieee.org/document/10947019).

## Requirements
Any `PyTorch` version newer than version 1.9.0 should work fine. You can find out how to install PyTorch here: https://pytorch.org/get-started/locally/. You also need to have `NumPy`, `scikit-learn`, `Librosa`, and `SciPy` installed.

## Repository contents
- `conf_finetune_pfml_pretrained_eeg_models.py`: Example configuration file for fine-tuning pre-trained models for EEG data, using the same configuration settings that were used in the [present paper](https://ieeexplore.ieee.org/document/10947019).
- `conf_finetune_pfml_pretrained_imu_models.py`: Example configuration file for fine-tuning pre-trained models for multi-sensor IMU data, using the same configuration settings that were used in the [present paper](https://ieeexplore.ieee.org/document/10947019).
- `conf_finetune_pfml_pretrained_speech_models.py`: Example configuration file for fine-tuning pre-trained models for speech data, using the same configuration settings that were used in the [present paper](https://ieeexplore.ieee.org/document/10947019).
- `conf_pfml_pretrain_eeg.py`: Example configuration file for PFML pre-training for EEG data, using the same configuration settings that were used in the [present paper](https://ieeexplore.ieee.org/document/10947019).
- `conf_pfml_pretrain_imu.py`: Example configuration file for PFML pre-training for multi-sensor IMU data, using the same configuration settings that were used in the [present paper](https://ieeexplore.ieee.org/document/10947019).
- `conf_pfml_pretrain_speech.py`: Example configuration file for PFML pre-training for speech data, using the same configuration settings that were used in the [present paper](https://ieeexplore.ieee.org/document/10947019).
- `finetune_pfml_pretrained_eeg_models.py`: A script for fine-tuning a pre-trained model using labeled EEG data.
- `finetune_pfml_pretrained_imu_models.py`: A script for fine-tuning a pre-trained model using labeled multi-sensor IMU data.
- `finetune_pfml_pretrained_speech_models.py`: A script for fine-tuning a pre-trained model using labeled speech data.
- `pfml_data_loader.py`: A file containing data loaders for PFML pre-training and fine-tuning for all three different data modalities (speech, multi-sensor IMU, and EEG data).
- `pfml_model.py`: A file containing the neural network model implementations of the [present paper](https://ieeexplore.ieee.org/document/10947019), including data modality-specific encoders for framed speech, multi-sensor IMU, and EEG data.
- `pfml_pretrain_eeg.py`: A script for running PFML pre-training and/or using a pre-trained model to extract features for EEG data.
- `pfml_pretrain_imu.py`: A script for running PFML pre-training and/or using a pre-trained model to extract features for multi-sensor IMU data.
- `pfml_pretrain_speech.py`: A script for running PFML pre-training and/or using a pre-trained model to extract features for speech data.
- `py_conf_file_into_text.py`: An auxiliary script for converting _.py_ configuration files into lists of text that can be used for printing or writing the configuration file contents into a text file.
- `transformer_encoder_pytorch.py`: A file containing a slightly modified version of PyTorch's Transformer encoder implementation.


## Examples of how to use the code


### How to run PFML pre-training:
For example for speech data, you can either use the command
```
python pfml_pretrain_speech.py
```
or
```
python pfml_pretrain_speech.py <configuration_file>
```
in order to run PFML pre-training. Using the former of these options requires having a configuration file named _conf_pfml_pretrain_speech.py_ in the same directory as the file _pfml_pretrain_speech.py_. In the latter option, _<configuration_file>_ is a _.py_ configuration file containing the hyperparameters you want to use during pre-training. By default, the configuration file _conf_pfml_pretrain_speech.py_ uses the [Librispeech](https://www.openslr.org/12) dataset.

### How to fine-tune pre-trained models:
For example for speech data, you can either use the command
```
python finetune_pfml_pretrained_speech_models.py
```
or
```
python finetune_pfml_pretrained_speech_models.py <configuration_file>
```
in order to fine-tune pre-trained models. Using the former of these options requires having a configuration file named _conf_finetune_pfml_pretrained_speech_models.py_ in the same directory as the file _finetune_pfml_pretrained_speech_models.py_. In the latter option, _<configuration_file>_ is a _.py_ configuration file containing the hyperparameters you want to use during fine-tuning.
