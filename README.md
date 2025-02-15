# IndicQA-Benchmark

**A Multilingual Benchmark to Evaluate Question Answering Capability of LLMs for Indic Languages**

> **Abhishek Kumar Singh, Vishwajeet Kumar, Rudra Murthy, Jaydeep Sen, Ashish Mittal, and Ganesh Ramakrishnan**  
> *The 2025 Annual Conference of the Nations of the Americas Chapter of the ACL (NAACL) 2025*  
> [Link to Paper](https://arxiv.org/abs/2407.13522)

---

## Overview

The **IndicQA-Benchmark** repository is designed to evaluate various models and pipelines for question-answering tasks, particularly focused on Indic languages. It provides tools for evaluating base models, fine-tuned instruction models, and translation tasks. 

You can easily download and set up the benchmark locally, along with evaluation scripts for various tasks.

### Download the Benchmark

To download the IndicQA-Benchmark, click the link below:

[Download IndicQA-Benchmark](https://huggingface.co/datasets/ayushayush591/IndicQA_Benchmark)

---

## Scripts

The repository includes several scripts for evaluating different types of models and tasks. Below are the available scripts:

- **Base Model Evaluation with VLLM**  
  Use the `Base_model.py` script to evaluate the base model using the VLLM library.

- **Base Model Evaluation with Hugging Face**  
  Use the `hugging_face_inference.py` script to evaluate the base model using Hugging Face.

- **Instruction-Finetuned Model**  
  For evaluating instruction-finetuned models, use the specific prompts provided for these models.

- **Translation Test Pipeline**  
  The `Trans_test.py` script is used for translation tasks. You'll need to set up the IndicTrans2 system as a translation system, which is available at [IndicTrans2 GitHub](https://github.com/AI4Bharat/IndicTrans2/tree/main). This will be used for translating from a source language to English, back-translates it, and then evaluates the translated output.

---

## How to Use

Follow the steps below to set up and run the evaluation scripts:

1. Clone this repository to your local machine.
   ```bash
   git clone https://github.com/your-repo/IndicQA-Benchmark.git
   cd IndicQA-Benchmark
2. Install necessary dependencies.
   ```bash
   pip install -r requirements.txt 
3. Run the appropriate script based on your evaluation needs.
   ```bash
   python Base_model.py --model_name<Model path> --dataset<Dataset Path> --local_dir<cache path where you want to store model>

For more details on how to run each script and set up the environment, refer to the individual script documentation.
