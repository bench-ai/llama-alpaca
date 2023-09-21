import json
import os
from pathlib import Path

"""
Adapted from code in https://github.com/tloen/alpaca-lora/blob/main/utils/prompter.py#L10
"""


class Prompter:

    """
    Generates prompts based on text
    """

    __slots__ = ("template", "_verbose")

    def __init__(self,
                 template_name: str = "alpaca"):

        """
        :param template_name: The name of the template in the template folder
        """

        template_path = Path(__file__).resolve().parent / "templates" / "alpaca.json"

        file_name = os.path.join(template_path)
        if not os.path.exists(file_name):
            raise ValueError(f"Can't read {file_name}")

        with open(file_name) as fp:
            self.template = json.load(fp)

    def generate_prompt(self,
                        instruction: str,
                        inpt: None | str = None,
                        label: None | str = None) -> str:

        """
        Formats the prompt with provided values

        :param instruction: The instruction to give the agent
        :param inpt: optional input for the instruction
        :param label: optional label for the instruction
        :return: the prompt with properly formatted with all the values
        """

        if inpt:
            res = self.template["prompt_input"].format(instruction=instruction,
                                                       input=inpt)
        else:
            res = self.template["prompt_no_input"].format(instruction=instruction)
        if label:
            res = f"{res}{label}"

        return res

    def get_response(self, output: str) -> str:
        """
        Splits the prompt based on the specifier and returns the response

        :param output: the prompt to split
        :return: the response
        """
        return output.split(self.template["response_split"])[1].strip()
