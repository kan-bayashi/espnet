#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import multiprocessing as mp
import os

import numpy as np

import kaldi_io_py


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


def _write_json_as_arkscp(uids, jss, outid):
    # write to ark and scp file (see https://github.com/vesis84/kaldi-io-for-python)
    arkscp = 'ark:| copy-vector --print-args=false ark:- ark,scp:%s.ark,%s.scp' % (outid, outid)

    # convert to ark, scp, and num_frame files
    with kaldi_io_py.open_or_fd(arkscp, 'wb') as f:
        for idx, (uid, js) in enumerate(zip(uids, jss), 1):
            feats = np.array(js["features"][0]["layers"][0]["values"]).reshape(-1).astype(np.float32)
            kaldi_io_py.write_vec_flt(f, feats, uid)
            logging.info("(%d/%d) %s" % (idx, len(uids), uid))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nj", type=int, default=32, help="number of jobs")
    parser.add_argument("text", type=str, help="kaldi format text")
    parser.add_argument("json", type=str, help="json extracted by bert")
    parser.add_argument("out", type=str, help="output file id")
    args = parser.parse_args()

    # logging info
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")

    # chech direcitory
    outdir = os.path.dirname(args.out)
    if len(outdir) != 0 and not os.path.exists(outdir):
        os.makedirs(outdir)

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

    # split uids and json
    uids_list = np.array_split(uids, args.nj)
    uids_list = [uids_.tolist() for uids_ in uids_list]
    jss_list = np.array_split(jss, args.nj)
    jss_list = [jss_.tolist() for jss_ in jss_list]

    # convert with multi-processing
    processes = []
    for idx, (uids_, jss_) in enumerate(zip(uids_list, jss_list), 1):
        outid = args.out + ".%d" % (idx)
        p = mp.Process(target=_write_json_as_arkscp, args=(uids_, jss_, outid))
        p.start()
        processes.append(p)

    # wait for all process
    for p in processes:
        p.join()


if __name__ == '__main__':
    main()
