import argparse
import torch
import json
from peft.utils import _get_submodules
import os
import bitsandbytes as bnb
from bitsandbytes.functional import dequantize_4bit
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import copy


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str)
    parser.add_argument("--peft_model", type=str)
    parser.add_argument("--save_dequant_model",default=None, type=str)
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--dtype", default="float16")
    return parser.parse_args()


def dequantize_model(model, tokenizer, to, dtype="float16", device="cuda", save_dequant_model=None):
    """
    'model': the peftmodel you loaded with qlora.
    'tokenizer': the model's corresponding hf's tokenizer.
    'to': directory to save the dequantized model
    'dtype': dtype that the model was trained using
    'device': device to load the model to
    """
    if os.path.exists(to):
        return AutoModelForCausalLM.from_pretrained(to, torch_dtype=dtype, device_map="auto")
    cls = bnb.nn.Linear4bit
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, cls):
                print(f"Dequantizing `{name}`...")
                quant_state = copy.deepcopy(module.weight.quant_state)
                quant_state[2] = dtype
                weights = dequantize_4bit(module.weight.data, quant_state=quant_state, quant_type="nf4").to(dtype)
                new_module = torch.nn.Linear(module.in_features, module.out_features, bias=None, dtype=dtype)
                new_module.weight = torch.nn.Parameter(weights)
                new_module.to(device=device, dtype=dtype)
                parent, target, target_name = _get_submodules(model, name)
                setattr(parent, target_name, new_module)
        model.is_loaded_in_4bit = False
        if save_dequant_model:
            os.makedirs(to, exist_ok=True)
            print("Saving dequantized model...")
            model.save_pretrained(to)
            tokenizer.save_pretrained(to)
            config_data = json.loads(open(os.path.join(to, "config.json"), "r").read())
            config_data.pop("quantization_config", None)
            config_data.pop("pretraining_tp", None)
            with open(os.path.join(to, "config.json"), "w") as config:
                config.write(json.dumps(config_data, indent=2))
        return model


def merge():
    args = get_args()
    model_path = args.base_model
    adapter_path = args.peft_model
    if not torch.cuda.is_available():
        raise("Cuda is not available in your machine")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, cache_dir=None, use_fast=False, padding_side="right", tokenizer_type="llama"
    )
    tokenizer.pad_token_id = 0

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # Enable 4-bit quantization
        bnb_4bit_quant_type="nf4",  # "nf4/fp4" Specify the type of 4-bit quantization to use
        bnb_4bit_compute_dtype=args.dtype,  # "bfloat16/float16"Specify the data type to use for computations during 4-bit quantization bfloat16 claim that faster training than float16
        bnb_4bit_use_double_quant=True,  # Quant weight twice -> memory efficiently, enable the use of double quantization
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path, load_in_4bit=True, torch_dtype=args.dtype, quantization_config=bnb_config, device_map="auto"
    )
    model = dequantize_model(model, tokenizer, args.save_dequant_model)
    model = PeftModel.from_pretrained(model, adapter_path)
    if args.push_to_hub:
        print(f"Saving to hub ...")
        model.save_pretrained(args.out_dir, safe_serialization=True, push_to_hub=True)
        tokenizer.save_pretrained(args.out_dir, safe_serialization=True, push_to_hub=True)
        print("Model successfully pushed to hf.")


if __name__ == "__main__":
    merge()
