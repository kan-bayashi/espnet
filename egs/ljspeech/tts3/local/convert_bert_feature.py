#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import logging

from distutils.util import strtobool

import numpy as np

from espnet.utils.cli_utils import FileWriterWrapper


def load_json(json_name):
    """Load json file

    This function enables to load following format json file:
    ```
    {
        "foo1": "bar1"
    }
    {
        "foo2": "bar2"
    }
    ```

    Arg:
        json_name (str): filename of json to be loaded

    Return:
        (list): list of dictionaries
    """
    with open(json_name, "r") as f:
        raw_text = f.read()
    size = len(raw_text)
    decoder = json.JSONDecoder()

    end = 0
    n_jss = 0
    jss = []
    while True:
        idx = json.decoder.WHITESPACE.match(raw_text[end:]).end()
        i = end + idx
        if i >= size:
            break
        js, end = decoder.raw_decode(raw_text, i)
        jss += [js]
        n_jss += 1
        logging.info("process %d" % (n_jss))

    return jss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compress', type=strtobool, default=False,
                        help='Save in compressed format')
    parser.add_argument('--compression-method', type=int, default=2,
                        help='Specify the method(if mat) or gzip-level(if hdf5)')
    parser.add_argument('--write-num-frames', type=str,
                        help='Specify wspecifer for utt2num_frames')
    parser.add_argument('--filetype', type=str, default='mat',
                        choices=['mat', 'hdf5'],
                        help='Specify the file format for output. '
                             '"mat" is the matrix format in kaldi')
    parser.add_argument("text", type=str, help="kaldi format text")
    parser.add_argument("json", type=str, help="json extracted by bert")
    parser.add_argument('wspecifier', type=str, help='Write specifier')
    args = parser.parse_args()

    # logging info
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")

    # load text and json
    with open(args.text, "r") as f:
        text = f.readlines()
        uids = [text_.split()[0] for text_ in text]
    with open(args.json, "r") as f:
        line = f.readline()
        jss = []
        while line:
            js = json.loads(line)
            jss += [js]
            line = f.readline()

    assert len(uids) == len(jss)

    with FileWriterWrapper(args.wspecifier,
                           filetype=args.filetype,
                           write_num_frames=args.write_num_frames,
                           compress=args.compress,
                           compression_method=args.compression_method
                           ) as writer:
        for idx, (uid, js) in enumerate(zip(uids, jss), 1):
            n_tokens = len(js["features"])
            bert_feats = [np.array(js["features"][idx]["layers"][0]["values"]) for idx in range(1, n_tokens)]
            bert_feats = np.stack(bert_feats).astype(np.float32)
            writer[uid] = bert_feats
            logging.info("(%d/%d) %s" % (idx, len(uids), uid))



if __name__ == '__main__':
    main()
