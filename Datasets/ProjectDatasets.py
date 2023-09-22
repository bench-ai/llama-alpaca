from sklearn.model_selection import train_test_split
from BenchKit.Data.Datasets import ProcessorDataset, IterableChunk
from BenchKit.Data.FileSaver import JsonFile
import pathlib
import json


# Write your datasets or datapipes here


class AlpacaDataset(ProcessorDataset):
    """
    Processor Class used to upload json files to cloud
    """

    def __init__(self, prompt_list: list[dict]):
        """
        :param prompt_list: The list of prompts

        Should be structured as list of prompt dicts
        [
            {
                "instruction": str
                "input": str
                "output": str
            },
            ...
        ]
        """
        super().__init__()
        self.json_file = JsonFile()  # The file saver

        self.prompt_list = prompt_list

    def _get_savers(self) -> JsonFile:
        """
        returns all the file savers used, required for the ProcessorDataset to function

        :return: All the file savers used in the processor
        """
        return self.json_file

    def _get_data(self, idx: int):
        """
        returns the dict at `index `idx` of prompt_list, required for the ProcessorDataset to function
        :param idx: int
        """
        self.json_file.append(self.prompt_list[idx])

    def __len__(self):
        return len(self.prompt_list)


class AlpacaChunker(IterableChunk):
    """
    Chunks Alpaca data into one zip file
    """

    def unpack_data(self,
                    idx) -> dict[str, str]:
        """
        returns the unzipped data, required for the IterableChunk to function
        :param idx: int
        :return: the prompted dict
        """
        text_dict: dict = super().unpack_data(idx)
        return text_dict


def get_processor_list(json_path: str,
                       name: str) -> list[tuple[AlpacaDataset, AlpacaChunker, str]]:

    """
    processes and splits dataset for upload

    :param json_path: Path to the json file containing prompts
    :param name: the name you wish to apply to the dataset
    :return: Tuple of a processor dataset, chunker, and dataset_name
    """

    with open(json_path, "r") as file:
        file_prompt: list = json.load(file)

        train, test = train_test_split(file_prompt,
                                       test_size=0.03,
                                       random_state=108,
                                       shuffle=True)

        train_processor = AlpacaDataset(train)
        val_processor = AlpacaDataset(test)

    return [
        (train_processor, AlpacaChunker(), f"TRAIN_{name}"),
        (val_processor, AlpacaChunker(), f"VAL_{name}"),
    ]


def main():
    """
    This method returns all the necessary components to build your dataset
    You will return a list of tuples, each tuple represents a different dataset
    The elements of the tuple represent the components to construct your dataset
    Element one will be your ProcessorDataset object
    Element two will be your IterableChunk class
    Element three will be the name of your Dataset
    Element four will be all the args needed for your Iterable Chunk as a list
    Element five will be all the kwargs needed for your Iterable Chunk as a Dict
    """

    base_path = pathlib.Path(__file__).resolve().parent.parent

    path_list =[
        base_path / "alpaca_data.json",
        base_path / "alpaca_data_cleaned_archive.json",
        base_path / "alpaca_data_gpt4.json"
    ]

    name_list = [
        "vanilla",
        "clean",
        "gpt4"
    ]

    result = list(map(get_processor_list, path_list, name_list))

    flattened_result = [item for row in result for item in row]

    return flattened_result
