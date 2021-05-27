#!/usr/bin/env python3

"""Perform tokenization with pretrained BERT."""

import argparse
import logging
import os

import transformers


def download_pretrained_bert(
    cache_dir,
    tokenizer_type="BertTokenizer",
    pretrained_bert_tag="bert-base-uncased",
):
    """Get pretrained BERT model and save it.

    Args:
        cacher_dir (str): Directory to cache pretrained bert and tokenizer model.
        tokenizer_type (str): Tokenizer type.
            E.g.: "BertTokenizer"
        pretrained_bert_tag (str): Pretrained BERT model tag.
            E.g.: "bert-base-uncased"


    """
    os.makedirs(cache_dir, exist_ok=True)
    tokenizer_class = getattr(transformers, tokenizer_type)
    tokenizer = tokenizer_class.from_pretrained(pretrained_bert_tag)
    tokenizer.save_pretrained(cache_dir)

    if tokenizer_type in ["BertTokenizer", "BertJapaneseTokenizer"]:
        model = transformers.BertModel.from_pretrained(pretrained_bert_tag)
    else:
        raise ValueError(f"{tokenizer_type} is not supported.")
    model.save_pretrained(cache_dir)
    logging.info(f"Model and tokenizer related files are saved in {cache_dir}")


def main():
    """Run main process."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_bert_tag", type=str, default="bert-base-uncased")
    parser.add_argument("--root_cache_dir", type=str, default="exp")
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("mode", type=str, choices=["encode", "decode"])
    parser.add_argument("in_text", type=str)
    parser.add_argument("out_text", type=str)
    args = parser.parse_args()

    if args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )

    bert_cache_dir = os.path.join(args.root_cache_dir, args.pretrained_bert_tag)
    if not os.path.exists(bert_cache_dir):
        download_pretrained_bert(
            bert_cache_dir, pretrained_bert_tag=args.pretrained_bert_tag
        )

    tokenizer = transformers.BertTokenizer.from_pretrained(bert_cache_dir)

    with open(args.in_text) as f:
        lines = [line.strip() for line in f.readlines()]
    text = {line.split()[0]: " ".join(line.split()[1:]) for line in lines}

    with open(args.out_text, "w") as f:
        for utt_id, txt in text.items():
            if args.mode == "encode":
                new_txt = ' '.join(tokenizer.tokenize(txt))
            else:
                new_txt = tokenizer.decode(tokenizer.convert_tokens_to_ids(txt.split()))
            f.write(f"{utt_id} {new_txt}\n")


if __name__ == "__main__":
    main()
