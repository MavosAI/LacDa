# LacDa - A Vietnamese Chat Bot
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
<a target="_blank" href="https://colab.research.google.com/drive/1akx1fZihH5hspLBhJzL3EDHp1PwGCSCC">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

LacDa is a fine-tuned natural language processing model derived from LLama2, tailored for specific domain or application needs. This repository contains information about the model, its capabilities, licensing, and how to use it effectively.

## Model Description

- **Model Name:** LacDa
- **Fine-tuned from:** [LLama2-7B](https://huggingface.co/meta-llama/Llama-2-7b-hf)
- **Domain:** Vietnamese Chat Bot

| [Huggingface Model](https://huggingface.co/willnguyen/lacda-2-7B-chat-v0.1) | [Colab Demo](https://colab.research.google.com/drive/1akx1fZihH5hspLBhJzL3EDHp1PwGCSCC) | 

LacDa leverages LLama2's advanced language capabilities and extends them to excel in domain-specific tasks.
## What is different?

- 10K instructions data: [openassistant-guanaco](https://huggingface.co/datasets/timdettmers/openassistant-guanaco)

- Modified [ConstantLengthDataset](https://github.com/huggingface/trl/blob/main/trl/trainer/utils.py#L482) in SFTTrainer
  - `add_special_tokens=False`
  - Make sure instruction in every batch start with `<s>` and full instruction
  - Convert other chat template to LLama2 chat template

## Benchmark 

- In progress

## LacDa Repository To-Do List

A checklist of tasks to manage and improve the LacDa project repository.

### Model Accuracy
- [ ] Train with large instruction dataset such as Open-Orca, Dolly-15k, Alpaca, ...
- [ ] Use [Mistral](https://github.com/mistralai/mistral-src) model
### CPU
- [ ] Implement quantization using [llama.cpp](https://github.com/ggerganov/llama.cpp)

### GPU
- [ ] Implement quantization using [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ)
- [ ] Implement quantization using [llm-awq](https://github.com/mit-han-lab/llm-awq)

### Chat bot interface with 
- [ ] [text-generation-webui](https://github.com/oobabooga/text-generation-webui) the most widely used web UI, with many features and powerful extensions. Supports GPU acceleration.

## Licensing

Before using the LacDa model, it's essential to acknowledge and adhere to LLama2's licensing terms, as your use of LacDa implicitly agrees to LLama2's license. Please consult LLama2's documentation for detailed licensing information.

## Usage

LacDa can be utilized for a variety of natural language processing tasks within its designated domain. Ensure that your usage aligns with the intended purpose and scope of the model.

## Precautions

- Ensure that your usage complies with the licensing terms of LLama2.
- Understand LacDa's strengths and limitations, as it may be more suitable for certain tasks within its domain.
