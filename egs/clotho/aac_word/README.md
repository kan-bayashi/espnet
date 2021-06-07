# Clotho Recipe

## Data preparation
* Data preparation during `stage 0` can be performed by appropriately setting the boolean variables in below command. By default, all variables are set to `false` except for `--augment_audiocaps` which is set to `true`. Description of each variable is also detailed below.
  
  ```bash
  ./run.sh --stage 0 --stop_stage 0  \
           --download_clothov2 true  \
           --download_audiocaps true \
           --augment_audiocaps true  \
           --augment_speedperturbation false \
           --download_evalmetrics true
  ```

#### Setting up Clotho-V2 dataset
* Download and prepare the *Clotho-V2* dataset using below command. This should prepare `data` and `clothov2_data` directories in the current recipe's root directory. The `data` directory should have `{dev,val,eval,recog_val,recog_eval}_clothov2` directories. The `clothov2_data` should have `clotho_{audio,csv}_files` directories.

  ```bash
  ./run.sh --stage 0 --stop_stage 0 --download_clothov2 true
  ```
* Among the `data/{dev,val,eval,recog_val,recog_eval}_clothov2` directories, `dev_clothov2` is used for training, `val_clothov2` is used for validation, and `recog_{val,eval}_clothov2` are used for decoding captions.
* Since each audio sample in this dataset has 5 captions, the `wav.scp` and `text` files in `data/{dev,val,eval}_clothov2` directories contain 5 lines for each audio sample, mapping to its 5 captions.
* To aviod decoding the same audio sample 5 times during the decoding stage, the `wav.scp` and `text` files in `data/recog_{val,eval}_clothov2` directories contain just one line for each audio sample, mapping to its first caption. Additionally, a `groundtruth_captions.txt` file is created in each directory which providing all the 5 ground truth captions for each audio sample.
* The `clothov2_data/clotho_audio_files` directory contains the audio samples from development, validation and evaluation sets, however renamed to `{dev,val,eval}file_{file-ID}.wav` filenames respectively. A mapping of renamed filenames to the original filenames can be found in `data/{dev,val,eval,recog_val,recog_eval}_clothov2/original_filenames.txt`.

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
#### Caption evaluation
* By default, stage 5 decoding evaluates the decoded captions and saves a summary `caption_evaluation_summary.txt` file and a detailed `caption_evaluation_results.txt` file to the experiment's decoding directory (ex: `exp/dev_clothov2_pytorch_train_specaug/decode_recog_val_clothov2_decode_lm_last10/`).
* Alternatively to evaluate the decoded captions, please execute `local/evaluate_decoded_captions.py`. This method takes two inputs: `decoded_json_path`, `groundtruth_captions_path`, and outputs a textfile: `caption_evaluation_results.txt` to the same directory as `decoded_json_path`. This output file tabulates the individual metric scores of each decoded audio sample. An example execution is provided below.
  
  ```bash
  python local/evaluate_decoded_captions.py \
      exp/dev_clothov2_pytorch_train_specaug/decode_recog_val_clothov2_decode_lm_last10/data.json \
      data/recog_eval_clothov2/groundtruth_captions.txt
  ```

#### Using best 10 validation epochs
* By default, stage 5 decoding averages the model parameters saved from the last 10 training epochs. To instead average the model parameters saved from the training epochs with best 10 validation scores, please add `--use_valbest_average true`.
