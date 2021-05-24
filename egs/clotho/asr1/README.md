# Clotho Recipe

* Download `data.tar.gz` and `clotho_data.tar.gz` files from [Google Drive Folder](https://drive.google.com/drive/folders/1F2T0k5QRmq403a4KcQt-VOELkqHK0PQb?usp=sharing) 
* Extract both files, you should get `data/` and `clotho_data/` directories respectively. Place both of them in the `egs/clotho/asr1/`
* `clotho_data/clotho_audio_files_16Hz` should contain the Clotho dataset audio files, however resampled to 16kHz from original 44.1kHz. A mapping of resampled filenames to the original filenames can be found in `data/{dev,eval}_clotho/original_filenames.txt`
* To download and setup *AudioCaps* dataset, execute `./run.sh --stage 0 --stop_stage 0`
* For speed perturbation, uncomment corresponding code in `stage 0` and execute `./run.sh --stage 0 --stop_stage 0`
