import math
import os
import sys
from typing import List

import numpy as np

import pytest
from pytest import fail

# https://numba.readthedocs.io/en/stable/cuda/simulator.html
os.environ["NUMBA_ENABLE_CUDASIM"] = "1"
from numba import cuda
from numba.cuda import random as c_random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')))
from myw2v.myw2v import BLANK_TOKEN
from myw2v import myw2v

test_data_path = os.path.join(os.path.dirname(__file__), "data")


def test_build_vocab():
    vocab = myw2v.build_vocab(test_data_path)
    pleasures, total, sentences = list(filter(lambda t: t[0] == "pleasures", vocab))[0]
    # cat 0* | fgrep -c -i pleasures  # 2
    # cat 0* | perl -pe 's/ /\n/g' | fgrep -c -i pleasures  # 3
    assert total == 3
    assert sentences == 2


def test_sort_vocab():
    orig = [("foo", 1, 2), ("bar", 55, 2), ("asdf", 3, 3), ("another_word", 4, 1)]
    exp = [(BLANK_TOKEN, 0, 0), ("bar", 55, 2), ("another_word", 4, 1), ("asdf", 3, 3), ("foo", 1, 2)]
    assert exp == myw2v.sort_vocab(orig)


def test_prune_vocab():
    orig = [("foo", 1, 2), ("bar", 55, 2), ("asdf", 3, 3), ("another_word", 4, 1)]
    result1 = myw2v.prune_vocab(min_occrs=1, my_vocab=orig)
    assert result1 == [(word, total) for word, total, _ in orig]
    result2 = myw2v.prune_vocab(min_occrs=2, my_vocab=orig)
    assert result2 == [("foo", 1), ("bar", 55), ("asdf", 3)]


def test_bias_freq_counts():
    def assert_words_and_close_enough_freqs(act, exp):
        assert [word for word, _ in act] == [word for word, _ in exp]
        for i in range(0, len(exp)):
            _, fact = act[i]
            _, fexp = exp[i]
            assert close_enough(fact, fexp), f"At index {i} actual was {fact}, expected {fexp}"
    words = [BLANK_TOKEN, "pie", "apple", "mince", "beef", "delicious"]
    freqs = [0, 3, 2, 2, 1, 1]
    vocab = list(zip(words, freqs))
    freqs_basic = list(zip(words, [0, 3/9, 2/9, 2/9, 1/9, 1/9]))
    result1 = myw2v.bias_freq_counts(vocab, exponent=1.0)
    print(result1)
    assert_words_and_close_enough_freqs(result1, freqs_basic)
    result2 = myw2v.bias_freq_counts(vocab, exponent=0.75)
    print(result2)
    q2 = [math.pow(q, 0.75) for q in freqs]
    assert_words_and_close_enough_freqs(result2, list(zip(words, [f/sum(q2) for f in q2])))
    result3 = myw2v.bias_freq_counts(vocab, exponent=0.444)
    print(result3)
    q3 = [math.pow(q, 0.444) for q in freqs]
    assert_words_and_close_enough_freqs(result3, list(zip(words, [f/sum(q3) for f in q3])))


def test_calc_subsampling_weights():
    vocab = [('pie', 3), ('apple', 2), ('mince', 2), ('beef', 1), ('delicious', 1)]
    s = sum([c for _, c in vocab])
    freqs = [c/s for _, c in vocab]

    t1 = 1e-5
    expected1 = [1-math.sqrt(t1/freq) for freq in freqs]
    weights1, _ = myw2v.get_subsampling_weights_and_negative_sampling_array(vocab, t=t1)
    print(f"EXPECTED 1: {expected1}")
    print(f"WEIGHTS 1:  {weights1}")
    assert_all_close_enough(weights1, expected1)

    t2 = 1e-5
    f = 500000
    vocab2 = [("pie", f), ("apple", 1)]
    expected2_ = [1-math.sqrt(t2/freq) for freq in [f/(f+1), 1/(f+1)]]
    # when freq < t, might go negative, check that we guard against that
    expected2 = [q if q >= 0 else 0 for q in expected2_]
    weights2, _ = myw2v.get_subsampling_weights_and_negative_sampling_array(vocab2, t=t2)
    print(f"EXPECTED 2: {expected2}")
    print(f"WEIGHTS 2:  {weights2}")
    assert_all_close_enough(weights2, expected2)


def test_weight_init():
    vocab_size = 12345
    for seed in 1234, 12345, 123456, 999:
        for emb_dim in [1, 10, 100]:
            # numpy, no cuda, btw
            w1, w2 = myw2v.init_weight_matrices(vocab_size, emb_dim, seed=seed)
            assert_weights(w1, w2, emb_dim)


def test_step():
    # if the test data is all roughly the same magnitude, then the learning will proceed as expected -
    # if one or two of these vectors is say 0.01 or something, then one or more gradients will get inverted.
    # this is fine of course, but in a unit test it can be a bit awkward heh
    emb = np.array([
        [0, 0],
        [-0.1, 0.1],
        [0.1, -0.1],
        [0.2, 0.4],
        [-0.3, -0.2]
    ], dtype=np.float32)
    emb2 = np.array([
        [0, 0],
        [0.2, -0.5],
        [-0.2, 0.3],
        [0.1, 0.7],
        [-0.4, 0.1]
    ], dtype=np.float32)
    emb_dim = emb.shape[1]  # 3 vectors of dim 2 each, so this is 2
    k = 2
    thread_idx = 0
    vocab_size = emb.shape[0]
    neg_smpl_arr = [1, 1, 1, 1, 2, 2]
    # with this seed the first rnd will be 0.26615933 which ^ results in 1 from this arr
    # second: 0.81275994 -> 2
    q_expected = [1, 2]
    x = 3
    y = 4
    lr = 0.2
    calc_aux = cuda.to_device(np.zeros((vocab_size, emb_dim), dtype=np.float32))
    random_states_init_cuda = c_random.create_xoroshiro128p_states(vocab_size, seed=12345)
    neg_smpl_arr_cuda = cuda.to_device(neg_smpl_arr)

    w1 = cuda.to_device(emb)
    w2 = cuda.to_device(emb2)
    print(f"Initial: w1, shape {w1.shape}:\n{w1}")
    print(f"Initial: w2, shape {w2.shape}:\n{w2}")

    # thread_idx, w1, w2, calc_aux, x, y, k, learning_rate, negsample_array, random_states  # btw
    myw2v.step(thread_idx, w1, w2, calc_aux, x, y, k, lr, neg_smpl_arr_cuda, random_states_init_cuda)
    w1 = w1.copy_to_host()
    w2 = w2.copy_to_host()
    ca = calc_aux.copy_to_host()
    print(f"Then: ca, shape {ca.shape}:\n{ca}")

    dot_pos = np.dot(emb[x,], emb2[y,])
    dot_neg_1 = np.dot(emb[x,], emb2[q_expected[0],])
    dot_neg_2 = np.dot(emb[x,], emb2[q_expected[1],])
    s_xdq_1 = 1./(1. + math.exp(-dot_neg_1))
    s_xdq_2 = 1./(1. + math.exp(-dot_neg_2))
    print(f"x, y {emb[x,]} {emb2[y,]}, dot(x,y) = {dot_pos}")
    print(f"x, q {emb[x,]} {emb2[q_expected,]}, dot(x,q1) = {dot_neg_1}, dot(x,q2) = {dot_neg_2}")
    print(f"sigmoid(x dot q1) = {s_xdq_1}, -''- q2 = {s_xdq_2}")
    print(f"q orig = {emb2[q_expected,]}")
    print(f"q1 * sigmoid(x dot q1) = {emb2[q_expected[0],] * s_xdq_1}")
    print(f"q2 * sigmoid(x dot q2) = {emb2[q_expected[1],] * s_xdq_2}")
    neg_samples_contrib_x = emb2[q_expected[0],] * s_xdq_1 + emb2[q_expected[1],] * s_xdq_2
    pos_sample_contrib_x = (1./(1. + math.exp(-np.dot(emb[x,], emb2[y,]))) - 1) * emb2[y,]
    gradient_x = neg_samples_contrib_x+pos_sample_contrib_x
    gradient_y = (1./(1. + math.exp(-np.dot(emb[x,], emb2[y,]))) - 1) * emb[x,]
    gradient_q1 = s_xdq_1 * emb[x,]
    gradient_q2 = s_xdq_2 * emb[x,]
    print(f"neg_samples_contrib_x: {neg_samples_contrib_x}")
    print(f"pos_sample_contrib_x: {pos_sample_contrib_x}")
    print(f"GRADIENT, x: {gradient_x}")
    print(f"GRADIENT, y: {gradient_y}")
    print(f"GRADIENT, q1: {gradient_q1}")
    print(f"GRADIENT, q2: {gradient_q2}")
    print(f"x dot y,  orig {np.dot(emb[x,], emb2[y,])} -> now {np.dot(w1[x,], w2[y,])}")
    print(f"x dot q1, orig {np.dot(emb[x,], emb2[q_expected[0],])} -> now {np.dot(w1[x,], w2[q_expected[0],])}")
    print(f"x dot q2, orig {np.dot(emb[x,], emb2[q_expected[1],])} -> now {np.dot(w1[x,], w2[q_expected[1],])}")
    print(f"x orig {emb[x,]} -> now {w1[x,]}: adjustment of {w1[x,]-emb[x,]} vs. lr*gradient {-lr*gradient_x}")
    print(f"y orig {emb2[y,]} -> now {w2[y,]}: adjustment of {w2[y,]-emb2[y,]} vs. lr*gradient {-lr*gradient_y}")
    print(f"q1 orig {emb2[q_expected[0],]} -> now {w2[q_expected[0],]}: adjustment of {w2[q_expected[0],]-emb2[q_expected[0],]} vs. lr*gradient {-lr*gradient_q1}")
    print(f"q2 orig {emb2[q_expected[1],]} -> now {w2[q_expected[1],]}: adjustment of {w2[q_expected[1],]-emb2[q_expected[1],]} vs. lr*gradient {-lr*gradient_q2}")
    assert close_enough_(w1[x,]-emb[x,], -lr*gradient_x, 1e-5)
    assert close_enough_(w2[q_expected[0],]-emb2[q_expected[0],], -lr*gradient_q1, 1e-5)
    assert close_enough_(w2[q_expected[1],]-emb2[q_expected[1],], -lr*gradient_q2, 1e-5)
    assert np.dot(w1[x,], w2[y,]) > np.dot(emb[x,], emb2[y,])
    assert np.dot(w1[x,], w2[q_expected[0],]) < np.dot(emb[x,], emb2[q_expected[0],])
    assert np.dot(w1[x,], w2[q_expected[1],]) < np.dot(emb[x,], emb2[q_expected[1],])


def test_word2vec():
    # if the test data is all roughly the same magnitude, then the learning will proceed as expected -
    # if one or two of these vectors is say 0.01 or something, then one or more gradients will get inverted.
    # this is fine of course, but in a unit test it can be a bit awkward heh
    emb = np.array([
        [0, 0],
        [-0.1, 0.1],
        [0.1, -0.1],
        [0.2, 0.4],
        [-0.3, -0.2]
    ], dtype=np.float32)
    emb2 = np.array([
        [0, 0],
        [0.2, -0.5],
        [-0.2, 0.6],
        [0.1, 0.7],
        [-0.4, 0.1]
    ], dtype=np.float32)
    # no i don't necessarily trust myself with generating "random" test data why do you ask
    for i in range(1, emb.shape[0]):
        for j in range(1, emb.shape[0]):
            dot = float(np.dot(emb[i,], emb2[j,]))
            if close_enough(dot, 0, 1e-6):
                fail(f'Expecting test data to be "naturally non-zeroing" but got zero dot for emb[{i},] emb2[{j},] = {emb[i,]}, {emb2[j,]}')

    vocab_size = emb.shape[0]
    emb_dim = emb.shape[1]
    c = 1
    k = 1
    lr = 0.1
    threads_per_block = 16
    a, b = 3, 4
    subs_weights = [0.0, 0.6, 0.2, 0.15, 0.1]
    # TODO NOTE: depending on random seed & how many rng calls are made, the following could cause this test to fail:
    # TODO NOTE: see "aux_out" stuff for how to "debug" where the randoms fall on each run...
    neg_smpl_arr = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2]
    q1, q2 = 2, 1
    inp = [a, b]
    rows = 1
    blocks: int = math.ceil(rows / threads_per_block)
    random_states_init_cuda = c_random.create_xoroshiro128p_states(vocab_size, seed=12345)
    w1 = cuda.to_device(emb)
    w2 = cuda.to_device(emb2)

    calc_aux = cuda.to_device(np.zeros((len(inp), emb_dim), dtype=np.float32))
    aux_out = cuda.to_device(np.zeros((threads_per_block, 2), dtype=np.float32))
    myw2v.calc[blocks, threads_per_block](rows, c, k, lr, w1, w2, calc_aux, random_states_init_cuda,
                                         cuda.to_device(subs_weights), cuda.to_device(neg_smpl_arr),
                                         cuda.to_device(inp), cuda.to_device([0]), cuda.to_device([2]))#,
#                                         aux_out)
    w1 = w1.copy_to_host()
    w2 = w2.copy_to_host()
    calc_aux = calc_aux.copy_to_host()
    print(f"After: w1, shape {w1.shape}:\n{w1}")
    print(f"After: w2, shape {w2.shape}:\n{w2}")
    print(f"After: calc_aux, shape {calc_aux.shape}:\n{calc_aux}")
    print(f"After: aux_out, shape {aux_out.shape}:\n{aux_out}")

    # DATA [3, 4] & q [1, 2] ->
    # 3 -> 4, q = 1: adjust w1[3]; adjust w2[4]; adjust w2[1]
    # 4 -> 3, q = 2: adjust w1[4]; adjust w2[3]; adjust w2[2]

    gradient_a1, gradient_b1, gradient_q1 = calc_contrib(emb, emb2, a, b, q1)
    print(f"GRADIENT x={a} #1: {gradient_a1}")
    print(f"GRADIENT y={b} #1: {gradient_b1}")
    print(f"GRADIENT q={q1} #1: {gradient_q1}")
    gradient_b2, gradient_a2, gradient_q2 = calc_contrib(emb, emb2, b, a, q2)
    print(f"GRADIENT x={b} #2: {gradient_b2}")
    print(f"GRADIENT y={a} #2: {gradient_a2}")
    print(f"GRADIENT q={q2} #2: {gradient_q2}")

    print(f"{a} dot {b}, orig {np.dot(emb[a,], emb2[b,])} -> now {np.dot(w1[a,], w2[b,])}")
    print(f"{a} dot {q1}, orig {np.dot(emb[a,], emb2[q1,])} -> now {np.dot(w1[a,], w2[q1,])}")
    print(f"{b} dot {q2}, orig {np.dot(emb[b,], emb2[q2,])} -> now {np.dot(w1[b,], w2[q2,])}")
    print(f"{a} #1 orig {emb[a,]} -> now {w1[a,]}: adjustment of {w1[a,] - emb[a,]} vs. lr*gradient {-lr * gradient_a1}")
    print(f"{b} #2 orig {emb2[b,]} -> now {w2[b,]}: adjustment of {w2[b,] - emb2[b,]} vs. lr*gradient {-lr * gradient_b1}")
    print(f"{b} #1 orig {emb[b,]} -> now {w1[b,]}: adjustment of {w1[b,] - emb[b,]} vs. lr*gradient {-lr * gradient_b2}")
    print(f"{a} #2 orig {emb2[a,]} -> now {w2[a,]}: adjustment of {w2[a,] - emb2[a,]} vs. lr*gradient {-lr * gradient_a2}")
    print(f"{q1} orig {emb2[q1,]} -> now {w2[q1,]}: adjustment of {w2[q1,] - emb2[q1,]} vs. lr*gradient {-lr * gradient_q1}")
    print(f"{q2} orig {emb2[q2,]} -> now {w2[q2,]}: adjustment of {w2[q2,] - emb2[q2,]} vs. lr*gradient {-lr * gradient_q2}")
    assert close_enough_(w1[a,] - emb[a,], -lr * gradient_a1, 1e-5)
    assert close_enough_(w2[q1,] - emb2[q1,], -lr * gradient_q1, 1e-5)
    assert close_enough_(w2[q2,] - emb2[q2,], -lr * gradient_q2, 1e-5)
    assert np.dot(w1[a,], w2[b,]) > np.dot(emb[a,], emb2[b,])
    assert np.dot(w1[a,], w2[q1,]) < np.dot(emb[a,], emb2[q1,])
    assert np.dot(w1[b,], w2[q2,]) < np.dot(emb[b,], emb2[q2,])


def calc_contrib(emb, emb2, inp, outp, q1):#, q2):
    dot_pos = np.dot(emb[inp,], emb2[outp,])
    dot_neg_1 = np.dot(emb[inp,], emb2[q1,])
    s_xdq_1 = 1. / (1. + math.exp(-dot_neg_1))
    neg_samples_contrib_x = emb2[q1,] * s_xdq_1
    pos_sample_contrib_x = (1. / (1. + math.exp(-dot_pos)) - 1) * emb2[outp,]
    gradient_x = neg_samples_contrib_x + pos_sample_contrib_x
    gradient_y = (1. / (1. + math.exp(-dot_pos)) - 1) * emb[inp,]
    gradient_q1 = s_xdq_1 * emb[inp,]
    return gradient_x, gradient_y, gradient_q1


def get_file_contents(path) -> List[str]:
    with open(path, "r") as f:
        l = f.readlines()
        l2 = [line.strip() for line in l]
        l3 = [line for line in l2 if line]
        return l3


def close_enough(x: float, tgt: float, tol: float = 1e-6) -> bool:
    return abs(x-tgt) <= tol


def assert_all_close_enough(act, exp):
    for i in range(0, len(exp)):
        fact = act[i]
        fexp = exp[i]
        assert close_enough(fact, fexp), f"At index {i} actual was {fact}, expected {fexp}"


def close_enough_(x: np.ndarray, tgt: np.ndarray, tol: float) -> bool:
    return (abs(x-tgt) <= tol).all()


def assert_weights(w1, w2, emb_dim):
    muw1 = float(np.mean(w1[1:, :]))
    muw2 = float(np.mean(w2[1:, :]))
    varw1 = float(np.var(w1[1:, :]))
    varw2 = float(np.var(w2[1:, :]))
    print(f"Mean w1 {muw1} w2 {muw2} / variance w1 {varw1} w2 {varw2} (NOTE: emb_dim {emb_dim})")
    # first vector = blank token
    assert (w1[0, :] == 0).all()
    assert (w2[0, :] == 0).all()
    assert close_enough(muw1, 0, 1/(3.3333*emb_dim))
    assert close_enough(muw2, 0, 1/(3.3333*emb_dim))
    assert close_enough(varw1, 1/emb_dim, 1/(3.3333*emb_dim))
    assert close_enough(varw2, 1/emb_dim, 1/(3.3333*emb_dim))
