import numpy as np
import torch
from BenchKit.Data.Helpers import get_dataloader
import torch.nn as nn
import torch.optim as opt
from tqdm import tqdm
from datasets import Dataset

from Datasets.convert_dataset import get_bench_dataloader, to_hf_dataset, to_hf_iterable


def main():
    train, val = to_hf_dataset("./alpaca_data.json",
                          "decapoda-research/llama-7b-hf",
                          validation_samples=1000)
    # print("here")
    print(train[0])
    # print("here")

    print("*" * 80)

    train = to_hf_iterable(get_bench_dataloader("vanilla_ds"),
                           "decapoda-research/llama-7b-hf")

    for i in train:
        print(i)
        break


if __name__ == '__main__':
    main()
