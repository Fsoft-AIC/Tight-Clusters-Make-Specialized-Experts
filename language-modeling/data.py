import os, sys
import argparse
import math, random
import torch
import tqdm
#from textattack.transformations import WordSwapRandomCharacterDeletion, WordSwapRandomCharacterInsertion, WordSwapRandomCharacterSubstitution, CompositeTransformation
#from textattack.constraints.pre_transformation import StopwordModification
#from textattack.constraints.pre_transformation.min_word_length import MinWordLength
#from textattack.augmentation import Augmenter
#from textattack.transformations import WordSwap
#from textattack.augmentation.recipes import EasyDataAugmenter
import mmap

def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines

# class AAAWordSwap(WordSwap):
#     def _get_replacement_words(self, word):
#         return ["AAA"]

# def _tokenize_attack(text_path, dictionary_to_update):
#     """Tokenizes a text file."""
#     print("Tokenizing {}".format(text_path))
#     assert os.path.exists(text_path)

#     nb_tokens_in_dictionary = len(dictionary_to_update)

#     # Count nb of tokens in text and update the dictionary
#     with open(text_path, "r", encoding="utf8") as f:
#         for line in f:
#             tokens = line.split() + ["<eos>"]
#             for token in tokens:
#                 if token not in dictionary_to_update:
#                     dictionary_to_update[token] = nb_tokens_in_dictionary
#                     nb_tokens_in_dictionary += 1
#     if "train" not in text_path:
#         ######## Attack!!!
#         transformation = AAAWordSwap()
#         constraints = [StopwordModification()]
#         # initiate augmenter
#         augmenter = Augmenter(
#             transformation=transformation,
#             constraints=constraints,
#             pct_words_to_swap=0.025,
#             transformations_per_example=1,
#             fast_augment=True
#         )

#     # Assign to each token its identifier
#     ids = []
#     with open(text_path, "r", encoding="utf8") as f:
#         for line in tqdm.tqdm(f, total=get_num_lines(text_path)):
#             if "train" not in text_path:
#                 line = augmenter.augment(line)[0]
#             tokens = line.split() + ["<eos>"]
#             for token in tokens:
#                 if token not in dictionary_to_update:
#                     ids.append(dictionary_to_update["<unk>"])
#                 else:
#                     ids.append(dictionary_to_update[token])
#     ids = torch.LongTensor(ids)
#     return ids

def _tokenize_clean(text_path, dictionary_to_update):
    """Tokenizes a text file."""
    print("Tokenizing {}".format(text_path))
    assert os.path.exists(text_path)

    nb_tokens_in_dictionary = len(dictionary_to_update)

    # Count nb of tokens in text and update the dictionary
    with open(text_path, "r", encoding="utf8") as f:
        for line in f:
            tokens = line.split() + ["<eos>"]
            for token in tokens:
                if token not in dictionary_to_update:
                    dictionary_to_update[token] = nb_tokens_in_dictionary
                    nb_tokens_in_dictionary += 1

    # Assign to each token its identifier
    ids = []
    with open(text_path, "r", encoding="utf8") as f:
        for line in f:
            tokens = line.split() + ["<eos>"]
            for token in tokens:
                ids.append(dictionary_to_update[token])
    ids = torch.LongTensor(ids)
    return ids

class Corpus: # create clean corpus
    def __init__(self, data_path):
        self._dictionary = {}
        self.train = _tokenize_clean(
            text_path=os.path.join(data_path, "train.txt"),
            dictionary_to_update=self._dictionary,
        )
        # self.train = "ONLY ATTACK VAL AND TEST"
        self.valid = _tokenize_clean(
            text_path=os.path.join(data_path, "valid.txt"),
            dictionary_to_update=self._dictionary,
        )
        self.test = _tokenize_clean(
            text_path=os.path.join(data_path, "test.txt"),
            dictionary_to_update=self._dictionary,
        )

    @property
    def vocab_size(self):
        return len(self._dictionary)
    
class Corpus_attack:
    def __init__(self, data_path):
        self._dictionary = {}
        self.train = _tokenize_clean(
            text_path=os.path.join(data_path, "train.txt"),
            dictionary_to_update=self._dictionary,
        )
        # self.train = "ONLY ATTACK VAL AND TEST"
        self.valid = _tokenize_clean(
            text_path=os.path.join(data_path, "valid.txt"),
            dictionary_to_update=self._dictionary,
        )
        self.test = _tokenize_clean(
            text_path=os.path.join(data_path, "test.txt"),
            dictionary_to_update=self._dictionary,
        )

    @property
    def vocab_size(self):
        return len(self._dictionary)


def _batchify(data_tensor, batch_size):
    nb_batches = data_tensor.size(0) // batch_size
    # trim away some tokens to make whole batches
    data_tensor = data_tensor.narrow(0, 0, nb_batches * batch_size)
    data_tensor = data_tensor.view(batch_size, -1).contiguous()
    return data_tensor


def _build_corpus(data_path, env_params, wt103_attack = False, data_name=None):
    # save the corpus to a file so that it's faster next time
    #corpus_path = os.path.join(data_path, "corpus_a1.pt") # looks like a1 is same as regular corpus. Need to make a new textattack dataset
    if wt103_attack:
        corpus_path = os.path.join(data_path, "corpus_attack_smoe.pt") # use word attack
        #corpus_path = os.path.join(data_path, "cache_a4.pt")
    else:
        corpus_path = os.path.join(data_path, "corpus.pt") # use clean wikitext
    if os.path.exists(corpus_path):
        print("Loading an existing corpus file from {}".format(corpus_path))
        corpus = torch.load(corpus_path)
    else:
        if not wt103_attack:    
            print("Creating a corpus file at {}".format(corpus_path))
            if env_params["distributed"]:
                # only one process need to create a corpus file
                if env_params["rank"] == 0:
                    corpus = Corpus(data_path)
                    torch.save(corpus, corpus_path)
                    # sync with other processes
                    torch.distributed.broadcast(torch.zeros(1).cuda(), src=0)
                else:
                    print("Waiting rank0 to create a corpus file.")
                    # sync with rank0
                    torch.distributed.broadcast(torch.zeros(1).cuda(), src=0)
                    corpus = torch.load(corpus_path)
            else:
                corpus = Corpus(data_path)
                torch.save(corpus, corpus_path)
        else:
            print("Creating an attack corpus file at {}".format(corpus_path))
            if env_params["distributed"]:
                # only one process need to create a corpus file
                if env_params["rank"] == 0:
                    corpus = Corpus_attack(data_path)
                    #raise ValueError('STOP: Running textattack with text attack code commented out. Makes sure env is fastmoe-attack and uncomment out text blocks.') 
                    torch.save(corpus, corpus_path)
                    # sync with other processes
                    torch.distributed.broadcast(torch.zeros(1).cuda(), src=0)
                else:
                    print("Waiting rank0 to create a corpus file.")
                    # sync with rank0
                    torch.distributed.broadcast(torch.zeros(1).cuda(), src=0)
                    corpus = torch.load(corpus_path)
            else:
                corpus = Corpus_attack(data_path)
                #raise ValueError('STOP: Running textattack with text attack code commented out. Makes sure env is fastmoe-attack and uncomment out text blocks.') 
                torch.save(corpus, corpus_path)
            

    return corpus


def _get_train_val_test_data(corpus, batch_size):
    return [
        _batchify(corpus.train, batch_size),
        _batchify(corpus.valid, batch_size),
        _batchify(corpus.test, batch_size),
    ]

def _get_val_test_data(corpus, batch_size):
    return [
        _batchify(corpus.valid, batch_size),
        _batchify(corpus.test, batch_size),
    ]


def get_train_val_test_data(data_params, env_params, batch_size, device):
    corpus = _build_corpus(**data_params, env_params=env_params)
    data_params["vocab_size"] = corpus.vocab_size
    train_data, val_data, test_data = _get_train_val_test_data(
        corpus=corpus, batch_size=batch_size
    )

    if env_params["distributed"]:
        # split the data into equal parts
        assert batch_size % env_params["world_size"] == 0
        device_batch_size = batch_size // env_params["world_size"]
        slice_data = slice(
            device_batch_size * env_params["rank"],
            device_batch_size * (env_params["rank"] + 1),
        )
        train_data = train_data[slice_data]
        val_data = val_data[slice_data]
        test_data = test_data[slice_data]

    train_data = train_data.to(device)
    val_data = val_data.to(device)
    test_data = test_data.to(device)
    return train_data, val_data, test_data

def get_val_test_data(data_params, env_params, batch_size, device, attack = False):
    assert attack == data_params["wt103_attack"]
    corpus = _build_corpus(**data_params, env_params=env_params)
    data_params["vocab_size"] = corpus.vocab_size
    val_data, test_data = _get_val_test_data(
        corpus=corpus, batch_size=batch_size
    )

    if env_params["distributed"]:
        # split the data into equal parts
        assert batch_size % env_params["world_size"] == 0
        device_batch_size = batch_size // env_params["world_size"]
        slice_data = slice(
            device_batch_size * env_params["rank"],
            device_batch_size * (env_params["rank"] + 1),
        )
        val_data = val_data[slice_data]
        test_data = test_data[slice_data]

    val_data = val_data.to(device)
    test_data = test_data.to(device)
    return val_data, test_data
    