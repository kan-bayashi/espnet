#!/usr/bin/env python3

"""Back-translate the text."""

import argparse
import logging
import time

import editdistance

from textblob import TextBlob

DELAY_ALPHA = 0.01


def main():
    """Run back-translation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--tgt_lang", type=str, default="en")
    parser.add_argument("--inter_lang", type=str, default="es")
    parser.add_argument("--edit_distance_threshold", type=int, default=-1)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("in_text")
    parser.add_argument("out_text")
    args = parser.parse_args()

    if args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )

    with open(args.in_text) as f:
        lines = [line.strip() for line in f.readlines()]
    text = {line.split()[0]: " ".join(line.split()[1:]) for line in lines}

    with open(args.out_text, "w") as f:
        for utt_id, txt in text.items():
            blob = TextBlob(txt.lower())
            # NOTE(kan-bayashi): Sleep to avoid too many request
            time.sleep(len(txt) * DELAY_ALPHA)
            blob = blob.translate(to=args.inter_lang, from_lang=args.tgt_lang)
            # NOTE(kan-bayashi): Sleep to avoid too many request
            time.sleep(len(txt) * DELAY_ALPHA)
            blob = blob.translate(to=args.tgt_lang, from_lang=args.inter_lang)
            new_txt = str(blob).upper()
            logging.info(f"original  : {txt}")
            logging.info(f"translated: {new_txt}")
            if args.edit_distance_threshold > 0:
                distance = editdistance.eval(txt, new_txt)
                if distance > args.edit_distance_threshold:
                    continue
            f.write(f"{utt_id}_{args.inter_lang}2{args.tgt_lang} {new_txt}\n")


if __name__ == "__main__":
    main()
