# IndicQA-Benchmark
> [INDIC QA BENCHMARK: A Multilingual Benchmark to Evaluate Question Answering capability of LLMs for Indic Languages](https://arxiv.org/abs/2407.13522)  
> Abhishek Kumar Singh, Vishwajeet kumar, Rudra Murthy, Jaydeep Sen, Ashish Mittal and Ganesh Ramakrishnan  
> The 2025 Annual Conference of the Nations of the Americas Chapter of the ACL, (__NAACL__) 2025

## Overview

The **IndicQA-Benchmark** repository is designed to evaluate various models and pipelines for question-answering tasks, particularly focused on Indic languages. The repository includes paths where one can download the benchmark locally, along with scripts for evaluating base models, fine-tuned instruction models, and translation tasks.

To download the benchmark, use the following link:  
**[Download IndicQA-Benchmark](your-download-link-here)**

## Scripts

- **Base Model Evaluation with VLLM**:  
  To evaluate the base model using the VLLM library, use the `Base_model.py` script.

- **Base Model Evaluation with Hugging Face**:  
  To evaluate the base model using Hugging Face, use the `Hugging_face_inference.py` script.

- **Instruction-Finetuned Model**:  
  For evaluating instruction-finetuned models, use the specific prompts corresponding to the models provided.

- **Translation Test Pipeline**:  
  The `Trans_test.py` script is used for translation tasks. For this you also need to setup envirnment for Indic Tranv2 which we Had used as Translation system(https://github.com/AI4Bharat/IndicTrans2/tree/main). It translates from a source language to English, back-translates it, and then evaluates the translated output.

## How to Use

1. Clone this repository to your local machine.
2. Install necessary dependencies.
3. Run the appropriate script based on your evaluation needs.

For more details on how to run each script and set up the environment, refer to the individual script documentation.
