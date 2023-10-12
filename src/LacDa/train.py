from ast import arg
import os, torch, logging
from datasets import load_dataset
from traitlets import default
import transformers
from transformers import (
    Trainer,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    GenerationConfig,
    TextIteratorStreamer,
)
from peft import LoraConfig, PeftModel, get_peft_model
from peft.utils import prepare_model_for_kbit_training
from peft.tuners.lora import LoraLayer
from trl import SFTTrainer
import bitsandbytes as bnb
from datasets import load_dataset
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, List
import argparse
from os.path import exists, join, isdir
from torch.utils.data import IterableDataset
import torch
import random
import warnings
import torch.nn.functional as F
import re


class ConstantLengthDataset(IterableDataset):
    """
    Iterable dataset that returns constant length chunks of tokens from stream of text files.
    The dataset also formats the text before tokenization with a specific format that is provided
    by the user.

        Args:
            tokenizer (`transformers.PreTrainedTokenizer`):
                The processor used for processing the data.
            dataset (`dataset.Dataset`):
                Dataset with text files.
            dataset_text_field (`str`, **optional**):
                Name of the field in the dataset that contains the text. Used only if `formatting_func` is `None`.
            formatting_func (`Callable`, **optional**):
                Function that formats the text before tokenization. Usually it is recommended to have follows a certain
                pattern such as `"### Question: {question}\n ### Answer: {answer}\n"`
            infinite (`bool`, *optional*, defaults to `False`):
                If True the iterator is reset after dataset reaches end else stops.
            seq_length (`int`, *optional*, defaults to `1024`):
                Length of token sequences to return.
            num_of_sequences (`int`, *optional*, defaults to `1024`):
                Number of token sequences to keep in buffer.
            chars_per_token (`int`, *optional*, defaults to `3.6`):
                Number of characters per token used to estimate number of tokens in text buffer.
            eos_token_id (`int`, *optional*, defaults to `0`):
                Id of the end of sequence token if the passed tokenizer does not have an EOS token.
            shuffle ('bool', *optional*, defaults to True)
                Shuffle the examples before they are returned
    """

    def __init__(
        self,
        tokenizer,
        dataset,
        dataset_text_field=None,
        formatting_func=None,
        infinite=False,
        seq_length=1024,
        num_of_sequences=1024,
        chars_per_token=4,
        eos_token_id=0,
        shuffle=True,
        add_special_tokens=False,
        add_concat_token=False,
    ):
        self.tokenizer = tokenizer
        self.add_special_tokens = add_special_tokens
        self.add_concat_token = add_concat_token
        if tokenizer.eos_token_id is None:
            warnings.warn(
                "The passed tokenizer does not have an EOS token. We will use the passed eos_token_id instead which corresponds"
                f" to {eos_token_id}. If this is not the correct EOS token, make sure to pass the correct eos_token_id."
            )

        self.concat_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id else eos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.infinite = infinite
        self.current_size = 0
        self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
        self.shuffle = shuffle
        if formatting_func is None:
            self.formatting_func = lambda x: x[dataset_text_field]
        else:
            self.formatting_func = formatting_func

        if formatting_func is not None:
            formatting_func_signature = formatting_func.__code__.co_varnames
            if len(formatting_func_signature) > 1:
                warnings.warn(
                    "The passed formatting_func has more than one argument. Usually that function should have a single argument `example`"
                    " which corresponds to the dictionary returned by each element of the dataset. Make sure you know what you are doing."
                )

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    buffer.append(self.formatting_func(next(iterator)))
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                        warnings.warn("The dataset reached end and the iterator is reset to the start.")
                    else:
                        more_examples = False
                        break
            if len(buffer) > 0:
                tokenized_inputs = self.tokenizer(
                    buffer, truncation=False, add_special_tokens=self.add_special_tokens
                )["input_ids"]
                all_token_ids = []
                if self.add_concat_token:
                    for tokenized_input in tokenized_inputs:
                        all_token_ids.extend(tokenized_input + [self.concat_token_id])
                else:
                    for tokenized_input in tokenized_inputs:
                        all_token_ids.extend(tokenized_input)
                examples = []
                i = 0
                last_bos_index = 0
                while i < len(all_token_ids):
                    input_ids = all_token_ids[i : i + self.seq_length]
                    if input_ids[0] != self.tokenizer.bos_token_id:
                        last_tokens = all_token_ids[last_bos_index : last_bos_index + self.seq_length]
                        try:
                            last_i_bos_index = max(
                                [
                                    index
                                    for index, item in enumerate(last_tokens)
                                    if item == self.tokenizer.bos_token_id
                                ]
                            )
                            if last_i_bos_index != 0:
                                last_bos_index += last_i_bos_index
                                input_ids = all_token_ids[last_bos_index : last_bos_index + self.seq_length]
                            else:
                                for index, item in enumerate(input_ids):
                                    if item == self.tokenizer.bos_token_id:
                                        last_i_bos_index = index
                                        break
                                else:
                                    last_bos_index = i + last_i_bos_index
                                    input_ids = all_token_ids[last_bos_index : last_bos_index + self.seq_length]
                                    i = last_bos_index
                        except:
                            warnings.warn("No bos_token_id found")
                            last_bos_index = i

                    if len(input_ids) <= self.seq_length and input_ids[0] == self.tokenizer.bos_token_id:
                        if len(input_ids) < self.seq_length:
                            input_ids = F.pad(
                                torch.LongTensor(input_ids),
                                (0, self.seq_length - len(input_ids)),
                                value=self.tokenizer.pad_token_id,
                            )
                        examples.append(input_ids)
                    i += self.seq_length
                if self.shuffle:
                    random.shuffle(examples)
                for example in examples:
                    self.current_size += 1
                    yield {
                        "input_ids": torch.LongTensor(example),
                        "labels": torch.LongTensor(example),
                    }
            else:
                if self.infinite:
                    iterator = iter(self.dataset)
                    warnings.warn("The dataset reached end and the iterator is reset to the start.")
                else:
                    more_examples = False
                    warnings.warn("The dataset reached end.")


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="/kaggle/input/llama-2/pytorch/7b-hf/1"
        # default="meta-llama/Llama-2-7b-chat-hf"
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."},
    )
    use_auth_token: bool = field(
        default=False, metadata={"help": "Enables using Huggingface auth token from Git Credentials."}
    )


@dataclass
class DataArguments:
    max_seq_length: int = field(
        default=1024,
        metadata={
            "help": "Maximum model length (input and output).  Sequences will be right padded (and possibly truncated)."
        },
    )
    dataset_name: str = field(
        default="timdettmers/openassistant-guanaco",
        metadata={
            "help": "Which dataset to finetune on. For now only support `timdettmers/openassistant-guanaco` and `mlabonne/guanaco-llama2-1k`."
        },
    )
    dataset_text_field: str = field(default="text", metadata={"help": "Which column field contain the instruction."})


@dataclass
class CustomTrainingArguments(transformers.Seq2SeqTrainingArguments):
    # The training data hyperparameters is inspired from https://github.com/karpathy/llama2.c and https://github.com/jondurbin/qlora/
    cache_dir: Optional[str] = field(default=None, metadata={"help": "Tokenization cache dir"})
    double_quant: bool = field(
        default=True, metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_dtype: str = field(
        default="nf4", metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(default=4, metadata={"help": "How many bits to use."})
    lora_r: int = field(default=64, metadata={"help": "Lora R dimension."})
    lora_alpha: float = field(default=16, metadata={"help": " Lora alpha."})
    lora_dropout: float = field(default=0.05, metadata={"help": "Lora dropout."})
    lora_bias: str = field(default="none", metadata={"help": "Lora linear bias."})
    max_memory_MB: int = field(default=15000, metadata={"help": "Free memory per gpu."})
    report_to: str = field(default="none", metadata={"help": "To use wandb or something else for reporting."})
    torch_compile: bool = field(
        default=False,
        metadata={"help": "Enable model compiled by using torch.compile. Now only support Ampere GPU or higher"},
    )
    save_model_dir: str = field(
        default="./save_model_dir", metadata={"help": "The output dir for logs and checkpoints"}
    )
    output_dir: str = field(default="./output", metadata={"help": "The final output directory."})
    checkpoint_dir: Optional[str] = field(
        default=None, metadata={"help": "The trained checkpoint directory. Can be same with working dir"}
    )
    optim: str = field(
        default="paged_adamw_32bit",
        metadata={"help": "The optimizer to be used `paged_adamw_8bit` or `paged_adamw_32bit`"},
    )
    per_device_train_batch_size: int = field(
        default=4, metadata={"help": "The training batch size per GPU. Increase for better speed."}
    )
    per_device_eval_batch_size: int = field(
        default=1, metadata={"help": "The eval batch size per GPU. Increase for better speed."}
    )
    gradient_accumulation_steps: int = field(
        default=16, metadata={"help": "How many gradients to accumulate before to perform an optimizer step"}
    )
    num_train_epochs: int = field(default=1, metadata={"help": "Number of training epochs."})
    weight_decay: float = field(
        default=0.0, metadata={"help": "The L2 weight decay rate of AdamW"}
    )  # use lora dropout instead for regularization if needed
    learning_rate: float = field(default=2e-4, metadata={"help": "The learning rate"})
    max_grad_norm: float = field(
        default=0.3,
        metadata={"help": "Gradient clipping max norm. This is tuned and works well for all models tested."},
    )
    gradient_checkpointing: bool = field(
        default=True, metadata={"help": "Use gradient checkpointing. You want to use this."}
    )
    do_train: bool = field(default=True, metadata={"help": "To train or not to train, that is the question?"})
    lr_scheduler_type: str = field(
        default="cosine",
        metadata={"help": "Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis"},
    )
    warmup_ratio: float = field(default=0.03, metadata={"help": "Fraction of steps to do a warmup for"})
    logging_steps: int = field(
        default=1, metadata={"help": "The frequency of update steps after which to log the loss"}
    )
    group_by_length: bool = field(
        default=False,
        metadata={
            "help": "Group sequences into batches with same length. Saves memory and speeds up training considerably."
        },
    )
    save_steps: int = field(default=1, metadata={"help": "How often to save a model"})
    save_total_limit: int = field(
        default=2, metadata={"help": "How many checkpoints to save before the oldest is overwritten"}
    )
    device_map: str = field(default="auto", metadata={"help": "device mapping for training cuda or cpu"})
    bf16: bool = field(
        default=False,
        metadata={
            "help": "Whether to use bf16 16-bit (mixed) precision training instead of 32-bit training. \
            Requires Ampere or higher NVIDIA architecture or using CPU (no_cuda). This is an experimental API and it may change."
        },
    )
    fp16: bool = field(
        default=True,
        metadata={"help": "Whether to use fp16 16-bit (mixed) precision training instead of 32-bit training."},
    )
    low_cpu_mem_usage: Optional[bool] = field(
        default=None, metadata={"help": "Enable low memory to offload CPU memory"}
    )
    save_safetensors: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Use [safetensors](https://huggingface.co/docs/safetensors) saving and loading for state dicts instead of\
            default `torch.load` and `torch.save`."
        },
    )


@dataclass
class GenerationArguments:
    # For more hyperparameters check:
    # https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
    # Length arguments
    max_new_tokens: int = field(
        default=512,
        metadata={
            "help": "Maximum number of new tokens to be generated in evaluation or prediction loops"
            "if predict_with_generate is set."
        },
    )
    # Generation strategy
    do_sample: bool = field(default=True)
    num_beams: int = field(default=1)
    num_beam_groups: int = field(default=1)
    use_cache: bool = field(default=True, metadata={"help": "True for model inference only"})
    # Hyperparameters for logit manipulation
    # How to understand temperature, top_k, top_p
    # https://peterchng.com/blog/2023/05/02/token-selection-strategies-top-k-top-p-and-temperature
    temperature: float = field(default=1.0)
    top_k: int = field(default=50)
    top_p: float = field(default=0.9)
    typical_p: float = field(default=1.0)
    diversity_penalty: float = field(default=0.0)
    repetition_penalty: float = field(default=1.0)
    length_penalty: float = field(default=1.0)
    no_repeat_ngram_size: int = field(default=0)


def find_all_linear_names(args, model):
    cls = bnb.nn.Linear4bit if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def get_last_checkpoint(checkpoint_dir):
    if isdir(checkpoint_dir):
        is_completed = exists(join(checkpoint_dir, 'completed'))
        if is_completed: return None, True # already finished
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if isdir(join(checkpoint_dir, filename)) and filename.startswith('checkpoint'):
                max_step = max(max_step, int(filename.replace('checkpoint-', '')))
        if max_step == 0: return None, is_completed # training started, but no checkpoint
        checkpoint_dir = join(checkpoint_dir, f'checkpoint-{max_step}')
        print(f"Found a previous checkpoint at: {checkpoint_dir}")
        return checkpoint_dir, is_completed # checkpoint found!
    return None, False # first training


def transform_conversation(example):
    conversation_text = example["text"]
    segments = conversation_text.split("###")

    reformatted_segments = []

    # Iterate over pairs of segments
    for i in range(1, len(segments) - 1, 2):
        human_text = segments[i].strip().replace("Human:", "").strip()

        # Check if there is a corresponding assistant segment before processing
        if i + 1 < len(segments):
            assistant_text = segments[i + 1].strip().replace("Assistant:", "").strip()

            # Apply the new template
            reformatted_segments.append(f"<s>[INST] {human_text} [/INST] {assistant_text} </s>")
        else:
            # Handle the case where there is no corresponding assistant segment
            reformatted_segments.append(f"<s>[INST] {human_text} [/INST] </s>")

    return {"text": "".join(reformatted_segments)}


def setup_data(args, tokenizer):
    """Only support HF Dataset"""
    # TODO Add more custom dataset
    if "guanaco-llama2-1k" not in args.dataset_name and "openassistant-guanaco" not in args.dataset_name:
        raise (ValueError("Only `guanaco-llama2-1k` and `openassistant-guanaco` are valid for now"))

    training_data = load_dataset(args.dataset_name, split="train")
    if "openassistant-guanaco" in args.dataset_name:
        training_data = training_data.map(transform_conversation)

    training_data = ConstantLengthDataset(
        tokenizer,
        training_data,
        dataset_text_field=args.dataset_text_field,
        formatting_func=None,
        seq_length=args.max_seq_length,
        infinite=False,
        num_of_sequences=1024,
        chars_per_token=4,
        eos_token_id=tokenizer.eos_token_id,
        shuffle=False,
        add_special_tokens=False,
        add_concat_token=False,
    )
    return training_data


def setup_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        use_fast=True,
        padding_side="right",
        tokenizer_type="llama" if "llama" in args.model_name_or_path else None,  # Needed for HF name change
        trust_remote_code=args.trust_remote_code,
        use_auth_token=args.use_auth_token,
    )
    tokenizer.pad_token_id = 0
    return tokenizer


def setup_model(args):
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()

    max_memory = f"{args.max_memory_MB}MB"
    max_memory = {i: max_memory for i in range(n_gpus)}
    # if we are in a distributed setting, we need to set the device map and max memory per device
    if os.environ.get("LOCAL_RANK") is not None:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device_map = {"": local_rank}
        max_memory = {"": max_memory[local_rank]}

    device_map = args.device_map
    compute_dtype = torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)
    # If a network requires more precision it may need to use float16,
    # and if a network requires more dynamic range it may need to use bfloat16, whose dynamic range is equal to that of float32.
    # If overflows are observed, for example, then we suggest trying bfloat16.
    # https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=args.bits == 8,
        load_in_4bit=args.bits == 4,  # Enable 4-bit quantization
        bnb_4bit_quant_type=args.quant_dtype,  # "nf4/fp4" Specify the type of 4-bit quantization to use
        bnb_4bit_compute_dtype=compute_dtype,  # Specify the data type to use for computations during 4-bit quantization bfloat16 claim that faster training than float16
        bnb_4bit_use_double_quant=args.double_quant,  # Quant weight twice -> memory efficiently, enable the use of double quantization
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        quantization_config=bnb_config,
        torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)),
        device_map=device_map,
    )

    setattr(model, "model_parallel", True)
    setattr(model, "is_parallelizable", True)
    # for training turn off use_cache, transformer can"t set use_cache=False by default
    model.config.use_cache = False  # for training
    model.config.pretraining_tp = 1  # >1 more accurate but slower (experimental)
    # `gradient_checkpointing_enable` Optimize memory but run slowlydue to recomputing parts of the graph during back-propagation
    # The slowdown will depend on the model but quite often it is around 20-30%.
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    if args.checkpoint_dir:
        print("Loading adapters from checkpoint.")
        checkpoint_dir, completed_training = get_last_checkpoint(args.checkpoint_dir)
        model = PeftModel.from_pretrained(model, checkpoint_dir, is_trainable=True)
    else:
        # Qlora has it own function to find all linear layer
        # For more information pls refer to https://www.databricks.com/blog/efficient-fine-tuning-lora-guide-llms
        lora_target_modules = find_all_linear_names(args, model)
        print("lora_target_modules", lora_target_modules)
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=args.lora_dropout,
            bias=args.lora_bias,
            task_type="CAUSAL_LM",
            inference_mode=False,
        )
        model.enable_input_require_grads()
        # this step is done internally in SFTTrainer but normal Trainer not
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        print(model)
        print(model.config)
    return model


def train():
    hfparser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, CustomTrainingArguments, GenerationArguments)
    )
    model_args, data_args, training_args, generation_args, extra_args = hfparser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )
    training_args.generation_config = GenerationConfig(**vars(generation_args))
    args = argparse.Namespace(**vars(model_args), **vars(data_args), **vars(training_args))
    print(args)

    model = setup_model(args)
    tokenizer = setup_tokenizer(args)
    training_data = setup_data(args, tokenizer)

    # If you create a model outside the trainer, make sure to not pass to the trainer any additional keyword arguments that are relative to from_pretrained() method.
    trainer = SFTTrainer(
        model=model,
        train_dataset=training_data,
        tokenizer=tokenizer,
        args=training_args,
        max_seq_length=args.max_seq_length,
        packing=True,
    )
    # Training
    trainer.train()


if __name__ == "__main__":
    train()
