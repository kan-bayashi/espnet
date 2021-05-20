# Clotho Recipe

* Download `data.tar.gz` and `clotho_data.tar.gz` files from [Google Drive Folder](https://drive.google.com/drive/folders/1F2T0k5QRmq403a4KcQt-VOELkqHK0PQb?usp=sharing) 
* Extract both files, you should get `data/` and `clotho_data/` directories respectively. Place both of them in the `egs/clotho/asr1/`
* For speed perturbation, uncomment corresponding code in `stage 0` and execute `./run.sh --stage 0 --stop_stage 0`
* For feature extraction, execute `./run.sh --stage 1 --stop_stage 1`
