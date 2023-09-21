from BenchKit.Data.Datasets import ProcessorDataset, IterableChunk
from BenchKit.Data.FileSaver import JsonFile, BaseFile
import pathlib
import json


# Write your datasets or datapipes here


class AlpacaDataset(ProcessorDataset):

    """
    Processor Class used to upload json files to cloud
    """

    def __init__(self, json_path: str):
        """
        :param json_path: The path of the json file

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
        self.json_file = JsonFile() # The file saver

        self.prompt_list = []
        with open(json_path, "r") as file:
            file_prompt = json.load(file)
            self.prompt_list.extend(file_prompt)

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

    vanilla_data_path = base_path / "alpaca_data.json"
    clean_data_path = base_path / "alpaca_data_cleaned_archive.json"
    gpt4_data_path = base_path / "alpaca_data_gpt4.json"

    vanilla_proc = AlpacaDataset(str(vanilla_data_path))
    clean_proc = AlpacaDataset(str(clean_data_path))
    gpt4_proc = AlpacaDataset(str(gpt4_data_path))

    vanilla_chunker = AlpacaChunker()
    clean_chunker = AlpacaChunker()
    gpt4_chunker = AlpacaChunker()

    return [
        (vanilla_proc, vanilla_chunker, "vanilla_ds"),
        (clean_proc, clean_chunker, "clean_ds"),
        (gpt4_proc, gpt4_chunker, "gpt4_ds")
    ]
