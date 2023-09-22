import torch
import transformers
from peft import prepare_model_for_int8_training, LoraConfig, get_peft_model, get_peft_model_state_dict
from transformers import LlamaForCausalLM
from Models.callback import BenchKitCallback

from Datasets.convert_dataset import get_bench_hf_iterable_ds, Tokenizer, apply_transformations


def main():
    hyper_parameters = {
        # model/data params
        "base_model": "decapoda-research/llama-7b-hf",
        "dataset_name": "clean",
        # training hyperparams
        "batch_size": 128,
        "micro_batch_size": 4,
        "num_epochs": 3,
        "learning_rate": 3e-4,
        "cutoff_len": 256,
        # lora hyperparams
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "lora_target_modules": ["q_proj", "v_proj"],
        # llm hyperparams
        "train_on_inputs": True,
        "add_eos_token": False,
        "group_by_length": False,
        "prompt_template_name": "alpaca"
    }

    save_dir = "./save_dir"

    gradient_accumulation_steps = hyper_parameters["batch_size"] // hyper_parameters["micro_batch_size"]

    model = LlamaForCausalLM.from_pretrained(hyper_parameters["base_model"],
                                             load_in_8bit=True,
                                             torch_dtype=torch.float16)

    train_ds = get_bench_hf_iterable_ds(hyper_parameters["dataset_name"],
                                        True)

    val_ds = get_bench_hf_iterable_ds(hyper_parameters["dataset_name"],
                                      False)

    tokenizer = Tokenizer(hyper_parameters["base_model"],
                          template_name=hyper_parameters["prompt_template_name"],
                          train_on_inputs=hyper_parameters["train_on_inputs"],
                          add_eos_token=hyper_parameters["add_eos_token"])

    train_ds = apply_transformations(train_ds, tokenizer)
    val_ds = apply_transformations(val_ds, tokenizer)

    model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=hyper_parameters["lora_r"],
        lora_alpha=hyper_parameters["lora_alpha"],
        target_modules=hyper_parameters["lora_target_modules"],
        lora_dropout=hyper_parameters["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config)

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        callbacks=[BenchKitCallback(hyper_parameters, "val_loss")],
        args=transformers.TrainingArguments(
            per_device_train_batch_size=hyper_parameters["micro_batch_size"],
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=hyper_parameters["num_epochs"],
            learning_rate=hyper_parameters["learning_rate"],
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=200,
            save_steps=200,
            output_dir=save_dir,
            overwrite_output_dir=True,
            save_total_limit=3,
            load_best_model_at_end=True,
            group_by_length=hyper_parameters["group_by_length"],
        ),
        data_collator=tokenizer.seq_to_seq_collator
    )

    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    if torch.__version__ >= "2":
        model = torch.compile(model)

    trainer.train()

    model.save_pretrained(save_dir)


if __name__ == '__main__':
    main()
