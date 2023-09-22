"""
Adapted from https://github.com/tloen/alpaca-lora/blob/main/finetune.py
"""

import datasets
import transformers
from datasets import load_dataset, IterableDataset
from transformers import LlamaTokenizer
from .prompter import Prompter
from BenchKit.Data.Helpers import get_dataloader
from Datasets.ProjectDatasets import AlpacaChunker


class Tokenizer:
    """
    Tokenizes the prompts to be model understandable
    """

    CUT_OFF_LEN = 256

    def __init__(self,
                 base_model: str,
                 template_name: str = "alpaca",
                 train_on_inputs=True,
                 add_eos_token=True):

        """
        :param base_model: the model you are training (we need the same tokenizer)
        :param template_name: the template being used for int the prompt
        :param train_on_inputs: whether we should train on the input field of the template
        :param add_eos_token: whether to add end of sequence token
        """

        self.tokenizer = LlamaTokenizer.from_pretrained(base_model)

        self.tokenizer.pad_token_id = (
            0  # unk. we want this to be different from the eos token
        )

        self.tokenizer.padding_side = "left"  # Allow batched inference

        self.prompter = Prompter(template_name)
        self.train_on_inputs = train_on_inputs
        self.add_eos_token = add_eos_token

    @property
    def seq_to_seq_collator(self):
        return transformers.DataCollatorForSeq2Seq(self.tokenizer,
                                                   pad_to_multiple_of=8,
                                                   return_tensors="pt",
                                                   padding=True)

    def _add_eos(self, result: dict) -> dict:
        """
        add eos if condition is met

        :param result: the dict to modify
        :return: the modified dict
        """
        ends_without_eos = (result["input_ids"][-1] != self.tokenizer.eos_token_id)
        less_than_cutoff = (len(result["input_ids"]) < self.CUT_OFF_LEN)

        if ends_without_eos and less_than_cutoff:
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        return result

    def _tokenize(self,
                  prompt: str) -> dict:

        """
        tokenizes a prompt

        :param prompt: the prompt to tokenize
        :return: the tokenized prompt
        """

        result = self.tokenizer(prompt,
                                truncation=True,
                                max_length=self.CUT_OFF_LEN,
                                padding=False,
                                return_tensors=None)

        result = self._add_eos(result) if self.add_eos_token else result
        result["labels"] = result["input_ids"].copy()

        return result

    def __call__(self, prompt) -> dict:

        """
        Tokenizes the prompt with preferred settings

        :param prompt: The prompt to tokenize
        :return: the token dict
        """

        full_prompt = self.prompter.generate_prompt(prompt["instruction"],
                                                    inpt=prompt["input"],
                                                    label=prompt["output"])

        tokenized_full_prompt = self._tokenize(full_prompt)

        if not self.train_on_inputs:
            user_prompt = self.prompter.generate_prompt(prompt["instruction"],
                                                        prompt["input"])

            tokenized_user_prompt = self._tokenize(user_prompt)

            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if self.add_eos_token:
                user_prompt_len -= 1

            tok_lab = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]
            tokenized_full_prompt["labels"] = tok_lab

        return tokenized_full_prompt


def to_hf_dataset(data_path,
                  base_model: str,
                  validation_samples=0,
                  template_name: str = "alpaca",
                  train_on_inputs=True,
                  add_eos_token=True) -> tuple[datasets.Dataset, datasets.Dataset | None]:
    """
    If you wish to use the hugging face dataset this method will return the split dataset ready for
    training

    :param data_path: the path to the json prompt file
    :param base_model: the model being trained from hugging face
    :param validation_samples: the amount of samples wished to be used for validation
    :param template_name: what template is being used for prompt structuring
    :param train_on_inputs: ~
    :param add_eos_token: ~
    :return: Train dataset or Validation and a Train Dataset
    """

    data = load_dataset("json", data_files=data_path)

    tokenizer = Tokenizer(base_model,
                          template_name=template_name,
                          train_on_inputs=train_on_inputs,
                          add_eos_token=add_eos_token)

    if validation_samples > 0:
        train_val = data["train"].train_test_split(test_size=validation_samples, shuffle=True, seed=42)
        train_data = train_val["train"].shuffle().map(tokenizer)
        val_data = train_val["test"].shuffle().map(tokenizer)
    else:
        train_data = data["train"].shuffle().map(tokenizer)
        val_data = None

    return train_data, val_data


def apply_transformations(data: IterableDataset,
                          tokenizer: Tokenizer) -> IterableDataset:
    """
    Maps the output of the HuggingFace Iterable and turns it into a structured prompt
    :param data: the IterableDataset
    :param tokenizer: the tokenizer wished to be used
    """

    data = data.map(tokenizer)

    return data


def get_bench_hf_iterable_ds(ds_name: str,
                             train=True) -> datasets.IterableDataset:
    """

    :param ds_name: The name of the dataset
    :param train: gets the val portion of the dataset if false, train if true
    :return: The Iterable dataset split
    """

    ds_name = f"TRAIN_{ds_name}" if train else f"VAL_{ds_name}"
    chunk_loader = AlpacaChunker()
    data_loader = get_dataloader(chunk_loader,
                                 ds_name,
                                 batch_size=1)

    def my_gen():
        for batch in data_loader:
            for key in list(batch.keys()):
                if isinstance(batch[key], list):
                    batch[key] = batch[key][0]
            yield batch

    my_iterable_dataset = IterableDataset.from_generator(my_gen)
    return my_iterable_dataset
