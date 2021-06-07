import argparse
import json
import os
from eval_metrics import evaluate_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("decoded_json_path", type=str)
    parser.add_argument("groundtruth_captions_path", type=str)
    args = parser.parse_args()

    gts = {}
    with open(args.groundtruth_captions_path) as f:
        lines = [line[:-1] for line in f.readlines()]
        for line in lines:
            key = line.split(' ')[0]
            caption = ' '.join(line.split(' ')[1:])
            fileid = key.split('_')[0]
            capid = int(key[-1])
            if fileid not in gts:
                gts[fileid] = {'file_name': fileid}
            gts[fileid][f'caption_{capid+1}'] = caption.lower()

    preds = {}
    with open(args.decoded_json_path) as f:
        json_data = json.load(f)
        for key, val in json_data['utts'].items():
            fileid = key.split('_')[0]
            if fileid not in preds:
                preds[fileid] = {'file_name': fileid, 'caption_predicted': ''}
            pred_caption = val['output'][0]['rec_text'].replace(' <eos>','').replace('<eos>','').lower().replace('▁',' ')
            preds[fileid][f'caption_predicted'] = pred_caption

    captions_gts = [val for _,val in gts.items()]
    captions_preds = [val for _,val in preds.items()]
    
    metrics = evaluate_metrics(captions_preds, captions_gts)
    metrics_individual = {}
    for metric in metrics:
        for fileid,val in metrics[metric]['scores'].items():
            if fileid not in metrics_individual: metrics_individual[fileid] = []
            metrics_individual[fileid].append(round(val,3))

    dashes = '|'.join(['{:-^10}'.format('')]*10)
    headers = ['fileID','BLEU_1','BLEU_2','BLEU_3','BLEU_4','METEOR','ROUGE_L','CIDEr','SPICE','SPIDEr']
    def tabled_row(arr): return '|'.join(['{:^10}'.format(x) for x in arr])

    decode_dirpath = os.path.dirname(args.decoded_json_path)
    caption_evalpath = os.path.join(decode_dirpath,'caption_evaluation_results.txt')
    with open(caption_evalpath, 'w') as f:
        f.write(f'|{dashes}|\n')
        f.write(f'|{tabled_row(headers)}|\n')
        f.write(f'|{dashes}|\n')
        metrics_summary = ['overall']+[round(metrics[metric]['score'],3) for metric in metrics]
        f.write(f'|{tabled_row(metrics_summary)}|\n')
        f.write(f'|{dashes}|\n')
        for fileid,score_list in metrics_individual.items():
            metrics_fileid = [fileid]+score_list
            f.write(f'|{tabled_row(metrics_fileid)}|\n')
        f.write(f'|{dashes}|\n')

if __name__ == "__main__":
    main()