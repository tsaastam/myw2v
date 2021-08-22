import json
import os
import pathlib
import re
import time
from typing import List

import requests
import tqdm
from gensim import utils
from gensim.models.callbacks import CallbackAny2Vec
from gensim.test.utils import datapath
from gensim.models import KeyedVectors, Word2Vec
from gensim.scripts import segment_wiki

import myw2v

OUT_PATH_ROOT = "e:/data/wikip_for_myw2v_oh_yeah"
OUT_PATH_TXT_DIR = os.path.join(OUT_PATH_ROOT, "txt")
OUT_PATH_VECTORS = os.path.join(OUT_PATH_ROOT, "vectors")
# see https://dumps.wikimedia.org/enwiki/20210801/
URLS = ["https://dumps.wikimedia.org/enwiki/20210801/enwiki-20210801-pages-articles-multistream6.xml-p958046p1483661.bz2"]
FILENAMES = [re.sub(r".*/([^/]+)$", r"\1", url) for url in URLS]


def maybe_download(urls: List[str], filenames: List[str], out_path_root: str) -> List[str]:
    for i, url in enumerate(urls):
        out_filename = filenames[i]
        out_path = os.path.join(out_path_root, out_filename)
        if os.path.isfile(out_path):
            print(f"File '{out_filename}' already exists locally, skipping")
        else:
            print(f"Downloading from url #{i+1} to file '{out_filename}'...")
            with requests.get(url, stream=True) as g:
                g.raise_for_status()
                size_maybe = int(g.headers.get("content-length", 0))
                prgb = tqdm.tqdm(total=size_maybe, unit="B", unit_scale=True)
                with open(out_path, "wb") as o:
                    for data in g.iter_content(chunk_size=16384):
                        prgb.update(len(data))
                        o.write(data)
                prgb.close()
    return filenames


def maybe_process_bz2_into_json(path: str, files: List[str]) -> List[str]:
    outfiles = []
    for filename in files:
        out_filename = re.sub(r"\.bz2$", ".json.gz", filename)
        full_path = os.path.join(path, filename)
        full_out_path = os.path.join(path, out_filename)
        if not os.path.isfile(full_out_path):
            print(f"Processing file '{filename}' into '{out_filename}'...")
            start = time.time()
            segment_wiki.segment_and_write_all_articles(full_path, full_out_path)
            print(f"  -> Done in {time.time() - start} s")
        else:
            print(f"File '{out_filename}' exists - not re-processing")
        outfiles.append(full_out_path)
    return outfiles


def maybe_clean_up_json(json_files: List[str], out_path: str, batch_size_min: int = 10000) -> None:
    if os.path.isdir(out_path):
        print(f"Text output path '{out_path}' seems to exist already - skipping!")
        return
    pathlib.Path(out_path).mkdir(parents=True, exist_ok=True)
    out_i = 0
    batch: List[List[str]] = []
    for i, file in enumerate(json_files):
        start = time.time()
        print(f"Processing json.gz file #{i+1}/{len(json_files)}: '{file}'...")
        with utils.open(file, 'rb') as jf:
            for line in jf:
                # decode each JSON line into a Python dictionary object
                article = json.loads(line)
                batch.extend(_clean_up(article["title"]))
                for section_title, section_text in zip(article['section_titles'], article['section_texts']):
                    batch.extend(_clean_up(section_title))
                    batch.extend(_clean_up(section_text))
                if len(batch) >= batch_size_min:
                    _write_batch(batch, out_path, out_i)
                    batch = []
                    out_i += 1
        _write_batch(batch, out_path, out_i + 1)
        print(f"  -> Processed in {time.time() - start} s")


def maybe_train_myw2v_model(input_txt_path: str, outfile_path: str, epochs: int):
    if os.path.isfile(outfile_path):
        print(f"The file '{outfile_path}' exists already - not training myw2v model...")
        return
    print(f"Training myw2v model, input txt path '{input_txt_path}', output file '{outfile_path}'...")
    myw2v.do_it(input_txt_path, outfile_path, epochs=epochs)


def check_accuracy(vectors_path: str):
    acc_path = vectors_path + "_accuracy.json"
    details_path = vectors_path + "_accuracy_details.json"
    if os.path.isfile(acc_path) or os.path.isfile(details_path):
        print(f"Accuracy check results at '{acc_path}' and/or '{details_path}' seem to exist already... skipping")
        return
    eval_start = time.time()
    print(f"Loading vectors from path: '{vectors_path}' and running word analogy test...")
    vecs = KeyedVectors.load_word2vec_format(vectors_path, binary=False)
    acc, details_dict = vecs.evaluate_word_analogies(datapath("questions-words.txt"), case_insensitive=True)
    details_lite = [(d["section"], len(d["correct"]) / len(d["correct"] + d["incorrect"])) for d in details_dict]
    print(f"Accuracy: {round(100 * acc, 2)} %")
    print(f"Saving model accuracy to '{acc_path}'...")
    with open(acc_path, "w") as f:
        f.write(json.dumps(details_lite) + "\n")
    print(f"Saving model accuracy details to '{details_path}'...")
    with open(details_path, "w") as f:
        f.write(json.dumps(details_dict) + "\n")
    print(f"Accuracy check took {time.time()-eval_start} s")


def maybe_train_gensim_model(input_txt_path: str, outfile_path: str, epochs: int):
    if os.path.isfile(outfile_path):
        print(f"The file '{outfile_path}' exists already - not training gensim model...")
        return
    print(f"Training gensim model, input txt path '{input_txt_path}', output file '{outfile_path}'...")

    # TODO maybe parameterise the params heh
    params = {
        "sg": 1,  # 1 = skip-gram, 0 = cbow
        "hs": 0,  # 0 = negative sampling, 1 = hierarchical softmax
        "compute_loss": True,
        "size": 100,  # aka embedding dimension
        "window": 5,  # c: (max) window size
        "alpha": 0.025,  # initial learning rate
        "min_alpha": 0.0025,  # minimum (eventual) learning rate
        "min_count": 3,  # vocabulary pruning of fewer-than-this-many-times-occurring words
        "negative": 5,  # k: how many noise words to sample (when negative sampling)
    }

    print(f"Reading all data into memory...")
    start = time.time()
    txt_files = myw2v.get_data_file_names(input_txt_path, seed=12345)
    sents: List[List[str]] = []
    for txt_file in txt_files:
        with open(os.path.join(input_txt_path, txt_file), encoding="utf-8") as f:
            for line in f:
                words = [word for word in re.split(r"[ .]+", line.strip()) if word]
                if len(words) < 2:
                    continue
                sents.append(words)
    print(f"Data read in {time.time()-start} s. Building vocab...")
    vocab_start = time.time()
    my_model: Word2Vec = Word2Vec(workers=12, **params)
    my_model.build_vocab(sents)
    print(f"Vocab built in {time.time()-vocab_start} s. Training gensim model...")
    train_start = time.time()
    mlc = EpochTimer()
    my_model.train(sents,
                   total_examples=my_model.corpus_count,
                   compute_loss=True,
                   epochs=epochs,
                   callbacks=[mlc])
    print(f"Trained in {time.time()-train_start} s! Writing vectors to '{outfile_path}'...")
    my_model.wv.save_word2vec_format(outfile_path)
    print(f"DONE! Total time {time.time()-start} s")


def _clean_up(s: str) -> List[List[str]]:
    # TODO: doesn't quite work for things like "1.6 km"
    sentences_probably = re.split(r"[?\n.]+", s)
    clean_sents = []
    for sent_str in sentences_probably:
        a = re.sub(r"[\t\n]", " ", sent_str)
        b = re.sub(r"[ ]{2,}", " ", a)
        # yeah super quick & dirty here
        c = re.sub(r"[^a-zA-Z ]", "", b)
        words = [w.lower() for w in c.split(" ") if w]
        if words:
            clean_sents.append(words)
    return clean_sents


def _write_batch(batch: List[List[str]], out_path: str, i: int) -> None:
    outf = "{:05d}".format(i)
    print(f"Writing {len(batch)} sentences to file: '{outf}'...")
    with open(os.path.join(out_path, outf), "w", encoding="utf-8") as of:
        for words in batch:
            of.write(" ".join(words))
            of.write("\n")


class EpochTimer(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 1
        self.stats = []
        self.previous_epoch_time = time.time()

    def on_epoch_end(self, model):
        epoch_seconds = time.time() - self.previous_epoch_time
        print(f"-> Epoch {self.epoch} took {epoch_seconds} s")
        self.stats.append({"epoch": self.epoch, "epoch_time_seconds": epoch_seconds})
        self.epoch += 1
        self.previous_epoch_time = time.time()


if __name__ == "__main__":
    # total data size: approx 2.5 GB for one source file of 500 MB
    # total time to run this: approx 46 min
    #
    # data download & parse: 8-10 min (1.5 GB output)
    # word2vec model build & check: 16 min (500 MB output file)
    # gensim model build & check: 21 min (500 MB output file)

    # about 2 min per file (approx 500 MB)
    filenames = maybe_download(URLS, FILENAMES, OUT_PATH_ROOT)
    # about 3.5 min per file (approx 300 MB)
    jsons = maybe_process_bz2_into_json(OUT_PATH_ROOT, filenames)
    # about 2.5 min per json file (approx 660 MB) - multiple output files (1000+)
    maybe_clean_up_json(jsons, OUT_PATH_TXT_DIR)

    # about 10 minutes for 10 epochs - 40 s per epoch, rest overhead (TODO: other settings)
    maybe_train_myw2v_model(OUT_PATH_TXT_DIR, OUT_PATH_VECTORS, epochs=10)
    # about 6 minutes, funnily enough
    check_accuracy(OUT_PATH_VECTORS)

    # about 15 minutes for 10 epochs - 70 s per epoch, rest overhead
    TODO_gensim_path = OUT_PATH_VECTORS+"_gensim_joo"

    maybe_train_gensim_model(OUT_PATH_TXT_DIR, TODO_gensim_path, epochs=10)
    # about 6 minutes, funnily enough
    check_accuracy(TODO_gensim_path)
