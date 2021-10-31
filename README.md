# myw2v
A word2vec implementation with Cuda &amp; Python

## What?

A simple word2vec implementation written in Python using Cuda. Requires a Cuda-capable GPU (e.g. NVidia GPU).

Features implemented:
- negative sampling
- skip-gram
- basic data loading and vocabulary handling
- for negative sampling, biasing the word frequency counts with a given exponent, e.g. 0.75 (see paper)
- given a maximum context window size `C`, the actual context window size is random `R` with `R ~ Unif(1, C)`
- basic gradient descent with a linearly decreasing learning rate

Currently all data is loaded into GPU memory first, after which the model is trained. Afterwards the first weight matrix is written to disk in plain text format (called "word2vec format" in gensim).

## How's the code?

    conda env create -f env_demo.yaml
    conda activate myw2v-demo
    pytest -s

Check out the unit tests (and the rest of the code) to see how things work and what's going on.

## How to demo it/check it out?

    conda env create -f env_demo.yaml
    conda activate myw2v-demo
    python mywv2/demo.py [-d <data_dir>]

The `-d <data_dir>` parameter is optional. The default is `./demo_data`.

Requirements: 2 GB of disk space; CUDA-capable GPU (e.g. NVidia) with, probably, 5 GB or more of GPU RAM; for gensim model, maybe 8 GB of main RAM to read data into memory first (TODO: optimise this part maybe).

The demo code will first download & process a partial Wikipedia dump and then train a myw2v model on it. Notice that this will require a few gigabytes of disk and will also take a while. Please see `demo.py` for details, or just run it and it will print the details and wait for a key press.

After training, the demo code uses gensim to do the standard "word analogy" task on the trained vectors, to gauge accuracy.

In addition, the demo code will also train another word2vec model on the same data, using gensim, which is a well-known and reliable word2vec implementation. Afterwards the same analogy task is done on the gensim vectors as well.

## Demo outputs

Under the given data directory, the following will be created:

    enwiki-20210801-pages-articles-multistream6.xml-p958046p1483661.bz2      # original Wikipedia dump, as downloaded
    enwiki-20210801-pages-articles-multistream6.xml-p958046p1483661.json.gz  # with-gensim-processed Wikipedia dump
    txt/                                                                     # Wikipedia data processed into plain text sentences

    vectors                                                                  # output of myw2v, in plain text format
    vectors_accuracy.json                                                    # gensim's word analogy task results for myw2v output (summary only)
    vectors_accuracy_details.json                                            # gensim's word analogy task results for myw2v output (full details)
    vectors_params.json                                                      # myw2v parameters used for training
    vectors_stats.json                                                       # myw2v statistics about source data

    vectors_gensim                                                           # output of gensim, in plain text format
    vectors_gensim_accuracy.json                                             # gensim's word analogy task results for gensim output (summary only)
    vectors_gensim_accuracy_details.json                                     # gensim's word analogy task results for gensim output (summary only)
    vectors_gensim_params.json                                               # gensim parameters used for training
    vectors_gensim_stats.json                                                # gensim statistics about source data

## Results

Here are the approximate results of the demo code above, when evaluated on the standard word analogy test:

* gensim: approx. 48-49%
* myw2v: approx. 44-45% (~ 10% lower)

The training times of the demo (on my PC) were:

* gensim: approx. 11.5 minutes
* myw2v: approx. 6.5 minutes (~ 40% faster)

See also [results](doc/results.md) of running another comparison of gensim vs myw2v.

## Remarks and thoughts on the implementation

See [blog post here](https://lobotomys.blogspot.com/2021/09/implementing-word2vec-on-gpu.html).

## TODO

- Parameterise all the model parameters properly
- "Streaming" data handling would be nice (as opposed to loading everything into memory all at once)
- Perhaps try to optimise the code more

## Licence

Copyright 2021 Taneli Saastamoinen <taneli.saastamoinen@gmail.com>

See also [COPYING](COPYING), [COPYING.LESSER](COPYING.LESSER)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
