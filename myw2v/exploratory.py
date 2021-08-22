import time
from typing import List

from gensim.models import KeyedVectors

from myw2v.myword2vec.demo import OUT_PATH_VECTORS


def load(vectors_path: str) -> KeyedVectors:
    print(f"Loading vectors from path: '{vectors_path}'...")
    start = time.time()
    v = KeyedVectors.load_word2vec_format(vectors_path, binary=False)
    print(f"Loaded in {time.time()-start} s")
    return v


def print_most_similars(vecs: KeyedVectors, words: List[str]) -> None:
    for word in words:
        print(f"--- {word} ---")
        for neighb, dist in vecs.most_similar(word):
            print(f"{neighb} {dist}")
        print()


if __name__ == '__main__':
    my_vec = load(OUT_PATH_VECTORS)
    print_most_similars(my_vec, ["ocean", "network", "parliament", "algorithm", "fusion", "history", "robot"])
