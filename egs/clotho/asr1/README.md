# Clotho Recipe

## Data preparation
* Data preparation during `stage 0` can be performed by appropriately setting the boolean variables in below command. By default, all variables are set to `false`. Description of each variable is also detailed below.
  
  ```bash
  ./run.sh --stage 0 --stop_stage 0  \
           --download_clothov2 true  \
           --download_audiocaps false \
           --augment_audiocaps false  \
           --augment_speedperturbation false \
           --download_evalmetrics false
  ```

#### Setting up Clotho-V2 dataset
* Download and prepare the *Clotho-V2* dataset using below command. This should prepare `data` and `clothov2_data` directories in the current recipe's root directory. The `data` directory should have `{dev,val,eval,recog_val,recog_eval}_clothov2` directories. The `clothov2_data` should have `clotho_{audio,csv}_files` directories.

  ```bash
  ./run.sh --stage 0 --stop_stage 0 --download_clothov2 true
  ```
* Among the `data/{dev,val,eval,recog_val,recog_eval}_clothov2` directories, `dev_clothov2` is used for training, `val_clothov2` is used for validation, and `recog_{val,eval}_clothov2` are used for decoding captions.
* The `clothov2_data/clotho_audio_files` directory contains the audio files from development, validation and evaluation sets, however renamed to `{dev,val,eval}file_{file-ID}.wav` filenames respectively. A mapping of renamed filenames to the original filenames can be found in `data/{dev,val,eval,recog_val,recog_eval}_clothov2/original_filenames.txt`.

#### Setting up AudioCaps dataset
* Download and prepare the *AudioCaps* dataset using below command.
  
  ```bash
  ./run.sh --stage 0 --stop_stage 0 --download_audiocaps true
  ```
* To augment the *AudioCaps* dataset during stages 1 to 5 (i.e. for generation of features, dict, json, and for training and decoding), please add `--augment_audiocaps true` when executing `./run.sh`.

#### Performing speed perturbation augmentation
* For speed perturbation based data-augmentation, please add `--augment_speedperturbation true` during data preparation.

#### Setting up COCO Evaluation Metrics
* Download and setup the evaluation framework using below command. **WARNING:** This performs `pip3 install scikit-image`
  
  ```bash
  ./run.sh --stage 0 --stop_stage 0 --download_evalmetrics true
  ```

## Decoding
#### Using best 10 validation epochs
* By default, stage 5 decoding averages the model parameters saved from the last 10 training epochs. To instead average the model parameters saved from the training epochs with best 10 validation scores, please add `--use_valbest_average true`.
