# myw2v
A word2vec implementation with Cuda &amp; Python

## What?

A simple word2vec implementation written in Python using Cuda. Requires a Cuda-capable GPU (e.g. NVidia GPU).

Features implemented:
- negative sampling
- skip-gram
- basic data loading and vocabulary handling
- for negative sampling, biasing the word frequency counts with a given exponent, e.g. 0.75 (see paper)
- given a context window size `C`, the actual context window size is random `R` with `R ~ Unif(1, C)`
- basic gradient descent with a linearly decreasing learning rate

Currently all data is loaded into GPU memory first, after which the model is trained. Afterwards the first weight matrix is written to disk in text format ("word2vec format" in gensim).

## How's the code?

    conda env create -f env_demo.yaml
    conda activate myw2v-demo
    pytest -s

Check out the unit tests (and the rest of the code) to see how things work and what's going on.

## How to demo it/check it out?

    conda env create -f env_demo.yaml
    conda activate myw2v-demo
    python mywv2/demo.py

The demo code will first download & process a partial Wikipedia dump and then train a myw2v model on it. Notice that this will require a few gigabytes of disk and will also take a while. Please see `demo.py` for details, or just run it and it will print the details and wait for a key press.

After training, the demo code uses gensim to do the standard "word analogy" task on the trained vectors, to gauge accuracy.

In addition, the demo code will also train another word2vec model on the same data, using gensim, which is a well-known and reliable word2vec implementation. Afterwards the same analogy task is done on the gensim vectors as well.

## TODO

- Parameterise all the model parameters properly
- Report demo results maybe

