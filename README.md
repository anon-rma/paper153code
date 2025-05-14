# RMA Paper153 Code Repository

This repository contains the official implementation of our research project. It provides scripts and guidance for dataset generation, model training, merging adapters with pretrained models, and performing inference tests.

## Quick Start

### 1. Data Generation
Generate dataset using OpenAI's `o4-mini`:
```bash
bash o4_pipeline.sh  # openAI, o4-mini
```

### 2. Model Training
Navigate to the training directory:
```bash
cd train
```
#### 2.1 RMA Model Training
```bash
python3 train_rma.py
```

#### 2.2 Planner Model Training
```bash
python3 train_llama.py  # or train_qwen.py, train_phi.py
```

### 3. Model Merge
Merge adapter weights with the pretrained model:
```bash
cd train
python3 model_merge.py
```
Note: After merging, a separate environment setup is required for inference testing. We used the default `ollama` setup.

### 4. Model Inference

#### 4.1 RMA Model Inference
```bash
python3 rma_inferece.py
```

#### 4.2 Planner Model Inference

- **Closed-source LLM inference**:
```bash
python3 cloudllm_inference.py
```

- **Open-source LLM inference (ollama)**:
```bash
python3 ollama_inference.py
```

### 5. Evaluation Analysis
Run the evaluation analysis script:
```bash
python3 evaluation_analysis.py
```

Ensure all dependencies are properly installed as specified in respective scripts for smooth operation.
## License

This repository is released under the Creative Commons Attribution 4.0 International (CC BY 4.0) license.  
You are free to **share** and **adapt** the material for any purpose provided that you give appropriate credit.  