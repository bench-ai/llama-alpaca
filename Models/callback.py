import os.path

from BenchKit.tracking.config import Config
from transformers import TrainerCallback


class BenchKitCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that sends that logs to the Bench-Kit Tracker

    Args:
        tb_writer (`SummaryWriter`, *optional*):
            The writer to use. Will instantiate one if not set.
    """

    def __init__(self,
                 hyper_parameters: dict,
                 evaluation_criteria_name: str,
                 mode: str = min):

        self.hyper_parameters = hyper_parameters
        self.evaluation_criteria_name = evaluation_criteria_name
        self.mode = mode

        self.model_config: Config | None = None

    def _init_config(self):

        self.model_config = Config(self.hyper_parameters,  # hyperparams we are using for this model
                                   self.evaluation_criteria_name,
                                   # We will be evaluating this model based on validation loss
                                   self.mode)

    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return

        if not self.model_config:
            self._init_config()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero:
            return

        if not self.model_config:
            self._init_config()

        for k, v in logs.items():
            print(f"The key is {k} and correlates to {v}")

    def on_save(self, args, state, control, **kwargs):

        output_dir = args.output_dir

        if os.path.isdir(args.output_dir):
            print(os.listdir(args.output_dir))

        self.model_config.save_model_and_state(
            lambda: output_dir,
            lambda: output_dir,
            state.global_step,
            evaluation_value=100
        )
