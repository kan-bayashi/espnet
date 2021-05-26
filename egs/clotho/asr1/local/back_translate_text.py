#!/usr/bin/env python3

"""Back-translate the text."""

import argparse
import logging

from textblob import TextBlob


def main():
    """Run back-translation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--tgt_lang", type=str, default="en")
    parser.add_argument("--inter_lang", type=str, default="es")
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
            blob = blob.translate(to=args.inter_lang)
            blob = blob.translate(to=args.tgt_lang)
            new_txt = str(blob).upper()
            f.write(f"{utt_id}_{args.inter_lang}_to_{args.tgt_lang} {new_txt}\n")
            logging.info(f"original  : {txt}")
            logging.info(f"translated: {new_txt}")


if __name__ == "__main__":
    main()
