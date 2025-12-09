"""
Data loading and preprocessing module for sentiment analysis.
Based on NLP1 2025 Practical 2.
"""

import os
import re
import random
import zipfile
import urllib.request
import numpy as np
from collections import namedtuple, Counter, OrderedDict
from typing import List, Iterator, Optional, Tuple
from nltk import Tree

# Constants
SHIFT = 0
REDUCE = 1

# URLs for downloading data
SST_URL = "http://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip"
GLOVE_URL = "https://gist.githubusercontent.com/bastings/b094de2813da58056a05e8e7950d4ad1/raw/3fbd3976199c2b88de2ae62afc0ecc6f15e6f7ce/glove.840B.300d.sst.txt"
WORD2VEC_URL = "https://gist.githubusercontent.com/bastings/4d1c346c68969b95f2c34cfbc00ba0a0/raw/76b4fefc9ef635a79d0d8002522543bc53ca2683/googlenews.word2vec.300d.txt"

# Data structures
Example = namedtuple("Example", ["tokens", "tree", "label", "transitions"])


def download_file(url: str, dest_path: str) -> str:
    """Download a file from URL if it doesn't exist."""
    if os.path.exists(dest_path):
        print(f"File already exists: {dest_path}")
        return dest_path
    
    print(f"Downloading {url}...")
    os.makedirs(os.path.dirname(dest_path) if os.path.dirname(dest_path) else ".", exist_ok=True)
    urllib.request.urlretrieve(url, dest_path)
    print(f"Downloaded to {dest_path}")
    return dest_path


def download_and_extract_sst(data_dir: str = "data") -> str:
    """
    Download and extract the Stanford Sentiment Treebank dataset.
    
    Args:
        data_dir: directory to save the data
    
    Returns:
        path to the extracted trees directory
    """
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "trainDevTestTrees_PTB.zip")
    trees_dir = os.path.join(data_dir, "trees")
    
    # Check if already extracted
    if os.path.exists(os.path.join(trees_dir, "train.txt")):
        print(f"SST data already exists in {trees_dir}")
        return trees_dir
    
    # Download
    download_file(SST_URL, zip_path)
    
    # Extract
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    print(f"Extracted to {trees_dir}")
    
    return trees_dir


def download_embeddings(embedding_type: str = "glove", data_dir: str = "data") -> str:
    """
    Download pre-trained word embeddings.
    
    Args:
        embedding_type: "glove" or "word2vec"
        data_dir: directory to save embeddings
    
    Returns:
        path to the embeddings file
    """
    os.makedirs(data_dir, exist_ok=True)
    
    if embedding_type == "glove":
        filename = "glove.840B.300d.sst.txt"
        url = GLOVE_URL
    elif embedding_type == "word2vec":
        filename = "googlenews.word2vec.300d.txt"
        url = WORD2VEC_URL
    else:
        raise ValueError(f"Unknown embedding type: {embedding_type}")
    
    dest_path = os.path.join(data_dir, filename)
    return download_file(url, dest_path)


class OrderedCounter(Counter, OrderedDict):
    """Counter that remembers the order elements are first seen"""
    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))
    
    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)


class Vocabulary:
    """A vocabulary, assigns IDs to tokens"""
    
    def __init__(self):
        self.freqs = OrderedCounter()
        self.w2i = {}
        self.i2w = []
    
    def count_token(self, t: str):
        self.freqs[t] += 1
    
    def add_token(self, t: str):
        self.w2i[t] = len(self.w2i)
        self.i2w.append(t)
    
    def build(self, min_freq: int = 0):
        """
        Build vocabulary from counted tokens.
        
        Args:
            min_freq: minimum number of occurrences for a word to be included
        """
        self.add_token("<unk>")  # reserve 0 for <unk> (unknown words)
        self.add_token("<pad>")  # reserve 1 for <pad>
        
        tok_freq = list(self.freqs.items())
        tok_freq.sort(key=lambda x: x[1], reverse=True)
        for tok, freq in tok_freq:
            if freq >= min_freq:
                self.add_token(tok)
    
    def __len__(self):
        return len(self.w2i)


def filereader(path: str) -> Iterator[str]:
    """Read lines from a file, fixing backslash issues."""
    with open(path, mode="r", encoding="utf-8") as f:
        for line in f:
            yield line.strip().replace("\\", "")


def tokens_from_treestring(s: str) -> List[str]:
    """Extract the tokens (leaves) from a sentiment tree string."""
    return re.sub(r"\([0-9] |\)", "", s).split()


def transitions_from_treestring(s: str) -> List[int]:
    """Extract transition sequence (SHIFT/REDUCE) from a tree string."""
    # Replace leaf nodes (label + word) with 0 (SHIFT)
    s = re.sub(r"\([0-5] ([^)]+)\)", "0", s)
    # Add space before closing parens
    s = re.sub(r"\)", " )", s)
    # Keep removing opening parens with labels until none remain
    while re.search(r"\([0-5]\s+", s):
        s = re.sub(r"\([0-5]\s+", "", s)
    # Replace closing parens with 1 (REDUCE)
    s = re.sub(r"\)", "1", s)
    return list(map(int, s.split()))


def examplereader(path: str, lower: bool = False) -> Iterator[Example]:
    """Returns all examples in a file one by one."""
    for line in filereader(path):
        line = line.lower() if lower else line
        tokens = tokens_from_treestring(line)
        tree = Tree.fromstring(line)
        label = int(line[1])
        trans = transitions_from_treestring(line)
        yield Example(tokens=tokens, tree=tree, label=label, transitions=trans)


def load_data(data_dir: str, lower: bool = False) -> Tuple[List[Example], List[Example], List[Example]]:
    """
    Load train, dev, and test datasets.
    
    Args:
        data_dir: directory containing train.txt, dev.txt, test.txt
        lower: whether to lowercase tokens
    
    Returns:
        train_data, dev_data, test_data
    """
    train_path = os.path.join(data_dir, "train.txt")
    dev_path = os.path.join(data_dir, "dev.txt")
    test_path = os.path.join(data_dir, "test.txt")
    
    train_data = list(examplereader(train_path, lower=lower))
    dev_data = list(examplereader(dev_path, lower=lower))
    test_data = list(examplereader(test_path, lower=lower))
    
    return train_data, dev_data, test_data


def build_vocabulary(train_data: List[Example], min_freq: int = 0) -> Vocabulary:
    """Build vocabulary from training data."""
    v = Vocabulary()
    for ex in train_data:
        for token in ex.tokens:
            v.count_token(token)
    v.build(min_freq=min_freq)
    return v


def load_pretrained_embeddings(embedding_path: str, embedding_dim: int = 300) -> Tuple[Vocabulary, np.ndarray]:
    """
    Load pre-trained word embeddings and build vocabulary.
    
    Args:
        embedding_path: path to embeddings file (GloVe or word2vec format)
        embedding_dim: dimension of embeddings
    
    Returns:
        vocabulary, embedding matrix
    """
    v = Vocabulary()
    v.add_token('<unk>')
    v.add_token('<pad>')
    
    vectors = []
    # Random init for <unk> and <pad>
    vectors.append(np.random.randn(embedding_dim).astype(np.float32) * 0.01)
    vectors.append(np.zeros(embedding_dim, dtype=np.float32))
    
    with open(embedding_path, 'r', encoding='utf-8') as f:
        for line in f:
            items = line.strip().split()
            if len(items) < embedding_dim + 1:
                continue
            word = items[0]
            vector = np.array(items[1:embedding_dim+1], dtype=np.float32)
            v.add_token(word)
            vectors.append(vector)
    
    vectors = np.stack(vectors, axis=0)
    return v, vectors


def extract_all_subtrees(tree: Tree) -> List[Tree]:
    """
    Extract all subtrees from a tree, treating each non-leaf node as a separate tree.
    Each subtree represents a phrase in the sentence with its own sentiment label.
    
    Args:
        tree: An NLTK Tree object
        
    Returns:
        A list of all subtrees (including the root tree)
    """
    subtrees = []
    
    def collect_subtrees(node):
        if isinstance(node, Tree):
            subtrees.append(node)
            for child in node:
                collect_subtrees(child)
    
    collect_subtrees(tree)
    return subtrees


def build_augmented_dataset(dataset: List[Example]) -> List[Example]:
    """
    Build augmented dataset where each node in each tree becomes a separate training example.
    This allows the model to learn sentiment at the phrase level, not just sentence level.
    
    Args:
        dataset: List of Example objects with original sentences
        
    Returns:
        List of Example objects where each represents a phrase (subtree)
    """
    augmented_examples = []
    
    for example in dataset:
        subtrees = extract_all_subtrees(example.tree)
        
        for subtree in subtrees:
            subtree_label = int(subtree.label())
            subtree_tokens = subtree.leaves()
            subtree_transitions = transitions_from_treestring(str(subtree))
            
            augmented_example = Example(
                tokens=subtree_tokens,
                tree=subtree,
                label=subtree_label,
                transitions=subtree_transitions
            )
            augmented_examples.append(augmented_example)
    
    return augmented_examples


# Sentiment label mappings
SENTIMENT_LABELS = ["very negative", "negative", "neutral", "positive", "very positive"]
LABEL2IDX = {label: i for i, label in enumerate(SENTIMENT_LABELS)}
IDX2LABEL = {i: label for i, label in enumerate(SENTIMENT_LABELS)}
NUM_CLASSES = len(SENTIMENT_LABELS)


def pad(tokens: List[int], length: int, pad_value: int = 1) -> List[int]:
    """Add padding to a sequence to reach desired length."""
    return tokens + [pad_value] * (length - len(tokens))


def get_examples(data: List[Example], shuffle: bool = True, **kwargs) -> Iterator[Example]:
    """Shuffle data set and return 1 example at a time."""
    if shuffle:
        random.shuffle(data)
    for example in data:
        yield example


def get_minibatch(data: List[Example], batch_size: int = 25, shuffle: bool = True) -> Iterator[List[Example]]:
    """Return minibatches, optional shuffling."""
    if shuffle:
        random.shuffle(data)
    
    batch = []
    for example in data:
        batch.append(example)
        if len(batch) == batch_size:
            yield batch
            batch = []
    
    if len(batch) > 0:
        yield batch


if __name__ == "__main__":
    # Test the data loading
    print("Testing data loading...")
    
    data_dir = "../trees"
    if os.path.exists(data_dir):
        train_data, dev_data, test_data = load_data(data_dir)
        print(f"Train: {len(train_data)}, Dev: {len(dev_data)}, Test: {len(test_data)}")
        
        v = build_vocabulary(train_data)
        print(f"Vocabulary size: {len(v)}")
        
        # Test augmentation
        augmented = build_augmented_dataset(train_data[:10])
        print(f"Augmented examples from 10 sentences: {len(augmented)}")
    else:
        print(f"Data directory {data_dir} not found")
