# Copyright 2021 Taneli Saastamoinen <taneli.saastamoinen@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import json
import math
import os
import pathlib
import re
import time
from collections import defaultdict
from typing import List, Tuple, Dict, Any

from numba import cuda
from numba.cuda import random as c_random
import numpy as np
from numpy import linalg, ndarray


MYW2V_VERSION = "0.1"
BLANK_TOKEN = "<BLANK>"


def build_vocab(data_path: str) -> List[Tuple[str, int, int]]:
    files = [fn for fn in os.listdir(data_path) if fn.startswith("0")]
    sentences_per_word = defaultdict(int)
    totals_per_word = defaultdict(int)
    for file in files:
        with open(os.path.join(data_path, file), encoding="utf-8") as f:
            for line in f:
                less_spacey = re.sub(r"[ ]{2,}", " ", line.strip())
                words = less_spacey.split(" ")
                if len(words) > 1:
                    uniques = set()
                    for word in words:
                        uniques.add(word)
                        totals_per_word[word] += 1
                    for deduped in uniques:
                        sentences_per_word[deduped] += 1
    r = []
    for word, total in totals_per_word.items():
        sent = sentences_per_word[word]
        r.append((word, total, sent))
    return r


def sort_vocab(my_vocab: List[Tuple[str, int, int]]) -> List[Tuple[str, int, int]]:
    vs = [(BLANK_TOKEN, 0, 0)] + sorted(my_vocab, key=lambda t: (-t[1], t[0]))
    return vs


def prune_vocab(min_occrs: int, my_vocab: List[Tuple[str, int, int]]) -> List[Tuple[str, int]]:
    """
    Returns only total counts (first count of input); prunes according to sentence counts (second count of input).
    """
    if min_occrs > 1:
        totals = [(wrd, total_count) for wrd, total_count, sentence_count in my_vocab if sentence_count >= min_occrs or wrd == BLANK_TOKEN]
        return totals
    else:
        return [(word, total) for word, total, _ in my_vocab]


def bias_freq_counts(vocab: List[Tuple[str, int]], exponent: float) -> List[Tuple[str, float]]:
    totalsson = sum(count for _, count in vocab)
    plain = [(word, count / totalsson) for word, count in vocab]
    if exponent == 1.0:
        return plain
    exped = [(word, math.pow(count, exponent)) for word, count in plain]
    sum_exped = sum([q for _, q in exped])
    jooh = [(word, f/sum_exped) for word, f in exped]
    return jooh


def handle_vocab(data_path: str, min_occurs_by_sentence: int, freq_exponent: float):
    vocab: List[Tuple[str, int, int]] = build_vocab(data_path)
    sorted_vocab: List[Tuple[str, int, int]] = sort_vocab(vocab)
    pruned_vocab: List[Tuple[str, int]] = prune_vocab(min_occurs_by_sentence, sorted_vocab)
    biased_vocab: List[Tuple[str, float]] = bias_freq_counts(pruned_vocab, freq_exponent)
    w_to_i: Dict[str, int] = { word: idx for idx, (word, _) in enumerate(biased_vocab) }
    return biased_vocab, w_to_i


def get_subsampling_weights_and_negative_sampling_array(vocab: List[Tuple[str, float]], t: float) -> (ndarray, ndarray):
    # subsampling
    tot_wgt: int = sum([c for _, c in vocab])
    freqs: List[float] = [c/tot_wgt for _, c in vocab]
    # consider t/freq: if freq << t, formula might result in negative "probability", so need to clamp to zero
    probs: List[float] = [max(0.0, 1-math.sqrt(t/freq)) if freq > 0 else 0.0 for freq in freqs]

    # negative sampling array - precalc for efficient (but possibly slightly inaccurate) sampling
    arr_len = 200000
    w2 = [round(f*arr_len) for f in freqs]
    neg_arr = []
    for i, scaled in enumerate(w2):
        neg_arr.extend([i]*scaled)

    return np.asarray(probs, dtype=np.float32), np.asarray(neg_arr, dtype=np.int32)


def get_data_file_names(path: str, seed: int) -> List[str]:
    rng = np.random.default_rng(seed=seed)
    qq = [fn for fn in os.listdir(path) if fn.startswith("0")]
    # gotta sort first so that the random shuffle will shuffle the same list, regardless of os.listdir() order
    data_files = sorted(qq)
    rng.shuffle(data_files)
    return data_files


def read_all_data_files_ever(dat_path: str, file_names: List[str], w_to_i: Dict[str, int]) -> Tuple[List[int], List[int], List[int]]:
    start = time.time()
    inps, offs, lens = [], [], []
    offset_total = 0
    stats = defaultdict(int)
    for fn in file_names:
        fp = os.path.join(dat_path, fn)
        ok_lines = 0
        too_short_lines = 0
        with open(fp, encoding="utf-8") as f:
            for line in f:
                words = [word for word in re.split(r"[ .]+", line.strip()) if word]
                if len(words) < 2:
                    too_short_lines += 1
                    continue
                idcs = [w_to_i[w] for w in words if w in w_to_i]
                le = len(idcs)
                # note that idcs might be missing some words which are too rare to be in the vocab -
                # the data isn't pre-pruned, but the vocab is. this is fine
                # - TODO: if len(idcs) < 2, continue?
                ok_lines += 1
                offs.append(offset_total)
                lens.append(le)
                inps.extend(idcs)
                offset_total += le
        stats["file_read_lines_ok"] += ok_lines
        stats["one_word_sentence_lines_which_were_ignored"] += too_short_lines

    print(f"read_all_data_files_ever() STATS: {stats}")
    tot_tm = time.time()-start
    print(f"read_all_data_files_ever() Total time {tot_tm} s for {len(file_names)} files (avg {tot_tm/len(file_names)} s/file)")
    return inps, offs, lens


def init_weight_matrices(vocab_size: int, embed_dim: int, seed: int) -> (ndarray, ndarray):
    rng = np.random.default_rng(seed=seed)
    rows, cols = vocab_size, embed_dim
    sigma: float = math.sqrt(1.0/cols)
    zs = rng.standard_normal(size=(rows, cols), dtype=np.float32)
    xs = sigma * zs
    # first row all zero since it represents the blank token
    xs[0, :] = 0.0
    zs2 = rng.standard_normal(size=(rows, cols), dtype=np.float32)
    xs2 = sigma * zs2
    xs2[0, :] = 0.0
    return xs, xs2


def print_norms(weights_cuda):
    w = weights_cuda.copy_to_host()
    norms = [linalg.norm(v) for v in w]
    a, med, b = np.percentile(norms, [2.5, 50, 97.5])
    avg = float(sum(norms) / len(norms))
    print(f"Vector norms (count {len(norms)}) 2.5% median mean 97.5%: {a:0.4f}  {med:0.4f}  {avg:0.4f}  {b:0.4f}")


def write_vectors(weights_cuda, vocab: List[Tuple[str, float]], out_path: str):
    w = weights_cuda.copy_to_host()
    pathlib.Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        # len-1: skip first which is the blank token & all zero
        f.write(f"{len(w)-1} {len(w[0])}\n")
        for i, v in enumerate(w):
            # skip first which is the blank token & all zero
            if i == 0:
                continue
            v_str = " ".join([str(f) for f in v])
            word, _ = vocab[i]
            f.write(f"{word} {v_str}\n")


def write_json(to_jsonify: Dict[str, Any], json_path: str):
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(to_jsonify))
        f.write("\n")
        f.flush()


@cuda.jit
def calc(
        rows: int,
        c: int,
        k: int,
        learning_rate: float,
        w1,
        w2,
        calc_aux,
        random_states,
        subsample_weights,
        negsample_array,
        inp,
        offsets,
        lengths):
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if idx >= rows:
        return
    le = lengths[idx]
    off = offsets[idx]
    for centre in range(0, le):
        word_idx = inp[off + centre]
        prob_to_reject = subsample_weights[word_idx]
        rnd = c_random.xoroshiro128p_uniform_float32(random_states, idx)
#            aux_out[word_idx, 0] = rnd
#            aux_out[word_idx, 1] = prob_to_reject
        if rnd > prob_to_reject:
            r_f = c_random.xoroshiro128p_uniform_float32(random_states, idx)
            r: int = math.ceil(r_f * c)
#            aux_out[idx, 0] = r_f
#            aux_out[idx, 1] = r
            for context_pre in range(max(0, centre-r), centre):
                step(idx, w1, w2, calc_aux, inp[off+centre], inp[off+context_pre], k, learning_rate, negsample_array, random_states)#, aux_out)
            for context_post in range(centre + 1, min(le, centre + 1 + r)):
                step(idx, w1, w2, calc_aux, inp[off+centre], inp[off+context_post], k, learning_rate, negsample_array, random_states)#, aux_out)


@cuda.jit(device=True)
def step(thread_idx, w1, w2, calc_aux, x, y, k, learning_rate, negsample_array, random_states):#, aux_out):
    # calc_aux should be a matrix of size (thread_count, emb_dim)
    emb_dim = w1.shape[1]
    negs_arr_len = len(negsample_array)
    # d/dx -log sigmoid(x dot y)  = -1/sigmoid(x dot y) * sigmoid(x dot y) * (1-sigmoid(x dot y) * y
    #                             = (sigmoid(x dot y) - 1) * y
    #                             = -(1 - sigmoid(x dot y)) * y
    #                             = -(sigmoid(-x dot y)) * y
    # d/dx -log sigmoid(-x dot q) = -1/sigmoid(-x dot q) * sigmoid(-x dot q) * (1-sigmoid(-x dot q) * -q
    #                             = -(1-sigmoid(-x dot q) - 1) * -q
    #                             = -(sigmoid(x dot q) * -q
    #                             = sigmoid(x dot q) * q
    # d/dy -log sigmoid(x dot y)  = (sigmoid(x dot y) - 1) * x
    # d/dq -log sigmoid(-x dot q) = sigmoid(x dot q) * x
    # -> calculate:
    # sigmoid(x dot y) - 1  and use for adjusting x, y
    # for each q:
    #   sigmoid(x dot q)  and use for adjusting x, this q

    # sigmoid(x dot y) - 1
    dot_xy = 0.
    for i in range(0, emb_dim):
        dot_xy += w1[x, i] * w2[y, i]
    s_xdy_m1 = 1. / (1. + math.exp(-dot_xy)) - 1
    # (sigmoid(x dot y) - 1) * y  -> store in aux for adjusting x with later
    # (sigmoid(x dot y) - 1) * x  -> adjust y immediately
    for i in range(0, emb_dim):
        calc_aux[thread_idx, i] = -learning_rate * s_xdy_m1 * w2[y, i]
        w2[y, i] -= learning_rate * s_xdy_m1 * w1[x, i]

    # sigmoid(x dot q)
    for neg_sample in range(0, k):
        rnd = c_random.xoroshiro128p_uniform_float32(random_states, thread_idx)
        q_idx: int = int(math.floor(negs_arr_len * rnd))
        neg = negsample_array[q_idx]
        dot_xq = 0.
        for i in range(0, emb_dim):
            dot_xq += w1[x, i] * w2[neg, i]
        s_dxq = 1. / (1. + math.exp(-dot_xq))
        # sigmoid(x dot q) * q  -> store in aux for adjusting x with later
        # sigmoid(x dot q) * x  -> adjust q immediately
        for i in range(0, emb_dim):
            calc_aux[thread_idx, i] -= learning_rate * s_dxq * w2[neg, i]
            w2[neg, i] -= learning_rate * s_dxq * w1[x, i]
        # just for debugging
#        aux_out[x, 0] = neg
#        aux_out[x, 1] = rnd
#        aux_out[x, 2] = thread_idx
#        aux_out[x, 3] = emb_dim
    # adjust x from stored
    for i in range(0, emb_dim):
        w1[x, i] += calc_aux[thread_idx, i]


# TODO: NOTE: will overwrite any existing vectors + params + stats in "out_file_path{,*}"
def do_it(data_path: str,
          out_file_path: str,
          epochs: int,
          embed_dim: int = 100,
          min_occurs: int = 3,
          c: int = 5,
          k: int = 5,
          t: float = 1e-5,
          vocab_freq_exponent: float = 0.75,
          lr_max: float = 0.025,
          lr_min: float = 0.0025,
          cuda_threads_per_block: int = 32):

    params = {
        "myw2v_version": MYW2V_VERSION,
        "data_path": data_path,
        "out_file_path": out_file_path,
        "epochs": epochs,
        "embed_dim": embed_dim,
        "min_occurs": min_occurs,
        "c": c,
        "k": k,
        "t": t,
        "vocab_freq_exponent": vocab_freq_exponent,
        "lr_max": lr_max,
        "lr_min": lr_min,
        "cuda_threads_per_block": cuda_threads_per_block
    }
    stats = {
    }
    params_path = out_file_path + "_params.json"
    stats_path = out_file_path + "_stats.json"

    # TODO: note: same seed can give different results - this is probably because of kernel execution order differences
    # - seed can still be useful for unit test
    seed = 12345
    lr_step = (lr_max-lr_min) / (epochs-1)

    print(f"Seed: {seed}")
    print(f"Word2vec params: c {c}, k {k}, learning rate from {lr_max} to {lr_min} by steps of {lr_step}...")
    print(f"Full params: {params}")

    print(f"Data path: '{data_path}'. Building vocabulary first (going through full data)...")
    start = time.time()

    vocab, w_to_i = handle_vocab(data_path, min_occurs, freq_exponent=vocab_freq_exponent)
    ssw, negs = get_subsampling_weights_and_negative_sampling_array(vocab, t=t)
    vocab_size = len(vocab)
    print(f"Vocabulary build took {time.time() - start} s. Vocab size {vocab_size}, embedding dimension {embed_dim}")

    data_files = get_data_file_names(data_path, seed=seed)
    print(f"Gonna process data. First few data files btw: {data_files[0:10]}")
    inps_, offs_, lens_ = read_all_data_files_ever(data_path, data_files, w_to_i)
    inps, offs, lens = np.asarray(inps_, dtype=np.int32), np.asarray(offs_, dtype=np.int32), np.asarray(lens_, dtype=np.int32)
    sentence_count = len(lens)
    blocks: int = math.ceil(sentence_count / cuda_threads_per_block)
    print(f"inps: {inps[0:10]}")
    print(f"offs: {offs[0:10]}")
    print(f"lens: {lens[0:10]}")
    print(f"CUDA kernel launch params: {cuda_threads_per_block} threads per block, {blocks} blocks ({sentence_count} sentences/threads)")

    data_init_start = time.time()
    w1, w2 = init_weight_matrices(vocab_size, embed_dim, seed=seed)
    calc_aux = np.zeros((sentence_count, embed_dim), dtype=np.float32)
    data_size_weights = 4 * (w1.size + w2.size)
    data_size_inputs = 4 * (inps.size + offs.size + lens.size + ssw.size + negs.size)
    data_size_aux = 4 * calc_aux.size
    print(f"Data init took {time.time()-data_init_start} s")
    print(f"Sentence count {sentence_count:,} ({len(lens):,}), total data length {len(inps):,}")
    print(f"Sentence lengths, a selection: {lens[0:20]}")
    print(f"Subsampling weights, some of (length {len(ssw)}): {ssw[0:10]} {ssw[len(ssw)//6:len(ssw)//6+10]} {ssw[-10:]}")
    print(f"Negative sampling array, some of (length {len(negs)}): {negs[0:10]} {negs[len(negs)//6:len(negs)//6+10]} {negs[-10:]}")

    print(f"Transferring datas to GPU...")
    data_transfer_start = time.time()
    inps_cuda, offs_cuda, lens_cuda = cuda.to_device(inps), cuda.to_device(offs), cuda.to_device(lens)
    ssw_cuda, negs_cuda = cuda.to_device(ssw), cuda.to_device(negs)
    w1_cuda, w2_cuda = cuda.to_device(w1), cuda.to_device(w2)
    calc_aux_cuda = cuda.to_device(calc_aux)
    print(f"Data transfer in {time.time()-data_transfer_start} s - data size in bytes {data_size_weights:,} weights + {data_size_inputs:,} inputs + {data_size_aux:,} aux = {data_size_weights+data_size_inputs+data_size_aux:,} total")

    stats["sentence_count"] = len(lens)
    stats["word_count"] = len(inps)
    stats["approx_data_size_weights"] = data_size_weights
    stats["approx_data_size_inputs"] = data_size_inputs
    stats["approx_data_size_aux"] = data_size_aux
    stats["approx_data_size_total"] = data_size_weights + data_size_inputs + data_size_aux

    print(f"Initing CUDA random states for {sentence_count} sentences/threads...")
    random_init_start = time.time()
    random_states_cuda = c_random.create_xoroshiro128p_states(sentence_count, seed=seed)
    print(f"CUDA random state init in {time.time()-random_init_start} s")

    print_norms(w1_cuda)
    print(f"Running calc - {epochs} epochs...")
    epoch_times = []
    calc_start = time.time()
    for epoch in range(0, epochs):
        lr = lr_max - (epoch*lr_step)
        epoch_start = time.time()
        calc[blocks, cuda_threads_per_block](sentence_count, c, k, lr, w1_cuda, w2_cuda, calc_aux_cuda, random_states_cuda, ssw_cuda, negs_cuda, inps_cuda, offs_cuda, lens_cuda)
        print(f"  Kernel launch in {time.time()-epoch_start} s. Synchronising (btw: learning rate was {lr})...")
        sync_start = time.time()
        cuda.synchronize()
        epoch_times.append(time.time()-epoch_start)
        print(f"  Synchronised in {time.time()-sync_start} s")
        print(f"--> Epoch {epoch+1} took {epoch_times[-1]} s")
    print(f"DONE! Times per epoch min/avg/max: {min(epoch_times):0.2f} {np.mean(epoch_times):0.2f} {max(epoch_times):0.2f}")
    print(f"Total training time: {time.time()-calc_start}")
    print(f"Total time: {time.time()-start}")
    print_norms(w1_cuda)
    stats["epoch_time_min_seconds"] = min(epoch_times)
    stats["epoch_time_avg_seconds"] = np.mean(epoch_times)
    stats["epoch_time_max_seconds"] = max(epoch_times)
    stats["epoch_time_total_seconds"] = sum(epoch_times)
    stats["epoch_times_all_seconds"] = epoch_times
    print(f"Writing vectors to file: '{out_file_path}'...")
    write_vectors(w1_cuda, vocab, out_file_path)
    print(f"Writing parameters to file: '{params_path}'...")
    write_json(params, params_path)
    print(f"Writing statistics to file: '{stats_path}'...")
    write_json(stats, stats_path)
    print("DONE!")
