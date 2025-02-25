# LLAMA Fine-Tuning with QLORA using MLX Library

Medium Article: https://medium.com/@ardakskc/fine-tuning-llama-with-qlora-using-the-mlx-library-15f1ed3c61fd

In this research, it was investigated how to fine-tune an LLM model using the MLX library. The study specifically focuses on utilizing the QLORA (Quantized Low-Rank Adaptation) technique to efficiently fine-tune the LLaMA model on Apple Silicon hardware. This approach aims to reduce memory consumption while maintaining model performance, making it suitable for resource-constrained environments.

The document covers the entire fine-tuning process, from setting up the MLX environment to converting the trained model to GGUF format for deployment. Additionally, it highlights the challenges encountered during the process, such as compatibility issues when integrating QLORA with MLX-LLAMA and the limitations of the GGUF format regarding dequantization.

To make this process more accessible, a Jupyter Notebook has been prepared to serve as a step-by-step guide for implementing QLORA with MLX. This notebook provides practical examples and code snippets to help users seamlessly follow along with each stage of the fine-tuning workflow.

This research aims to provide valuable insights for practitioners interested in LLaMA fine-tuning with MLX, especially those working on Apple Silicon devices.

## What is LLM Fine Tuning?

LLM fine-tuning is the process of adjusting the parameters or weights of a pre-trained large language model (LLM) to tailor it for a specific task or domain. While models like GPT possess extensive general language knowledge, they often lack the specialized expertise needed for niche applications. Fine-tuning bridges this gap by training the model on domain-specific data, allowing it to capture the nuances of a particular subject area.

This process involves updating the model’s weights based on task-specific examples, effectively transforming a general-purpose model into a specialized tool. Although fine-tuning can significantly enhance accuracy and performance, it typically requires substantial computational resources, such as GPUs, to manage the training workload efficiently.

To optimize this process, various fine-tuning techniques have been developed. Traditional methods adjust all model parameters, but more efficient approaches—like Low-Rank Adapters (LoRA) and Quantized LoRA (QLoRA)—introduce a small number of trainable parameters while keeping the original weights frozen.

## What is MLX?

MLX is an open-source library from Apple that lets Mac users more efficiently run programs with large tensors in them. Naturally, when we want to train or fine-tune a model, this library comes in handy.

The way MLX works is by being very efficient with memory transfers between your Central Processing Unit (CPU), Graphics Processing Unit (GPU), and the Memory Management Unit (MMU). For every system architecture, the most time-intensive operations are when you are moving memory between registers. On Nvidia GPUs, they minimize memory transfers by creating huge amounts of SRAM on their devices. For Apple, they designed their silicon so that the GPU and the CPU have access to the same memory via the MMU. Consequently, the GPU won’t have to load data into its memory before acting on it. This architecture is called System on Chip (SOC), and it typically requires you to build your chip internally rather than combine other manufacturers pre-built parts.

Because Apple now designs its own silicon, it can write low-level software that makes highly efficient use of it. This however means that anyone using a Mac with an Intel processor will not be able to make use of this library.

## Setting Up MLX Environment

To begin fine-tuning LLaMA with MLX, let’s create a dedicated Python environment and install the necessary libraries using a `requirements.txt` file. Follow these steps:

1. **Create a Project Directory**
    
    Open your terminal and create a new folder for your project:
    
    ```bash
    mkdir llama-finetune
    cd llama-finetune
    
    ```
    
2. **Set Up a Python Virtual Environment**
    
    Initialize a virtual environment named `venv`:
    
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    
    ```
    
3. **Create a `requirements.txt` File**
    
    Using a text editor, create a `requirements.txt` file with the following contents:
    
    ```
    mlx
    mlx-lm
    pandas
    pyarrow
    huggingface-hub
    huggingface_hub[cli]
    langchain
    ollama
    bitsandbytes>=0.39.0
    accelerate
    transformers
    ```
    
4. **Install Dependencies from `requirements.txt`**
    
    Run the following command to install all required libraries:
    
    ```bash
    pip install -r requirements.txt
    
    ```
    
5. **Verify the Installation**
    
    Check if MLX, MLX-LM, and Hugging Face CLI are installed correctly:
    
    ```bash
    python -c "import mlx; import mlx_lm; import ollama; from huggingface_hub import login; print('All libraries installed successfully!')"
    
    ```
    

Your environment is now ready for LLaMA fine-tuning using MLX.

## **Model Loading and Generation**

In this section, two different approaches are explained for loading and generating text using the LLM model:

1. Using Hugging Face Hub & LangChain
2. Using MLX-LM

### **Option 1: Loading the Model from Hugging Face and Using LangChain**

Hugging Face Hub provides an easy way to download and load pre-trained models, including quantized versions. This approach is particularly useful when integrating with LangChain for structured prompt execution. At this stage, we cannot import the 'mlx-community/Llama-3.2-3B-Instruct-4bit' model using AutoModelForCausalLM. You may try this part with a different model. The load and generate part of the model we will use will be explained in section 2.

**Steps:**

1. **Authenticate & Download Model**
    - Use an API key to authenticate with Hugging Face Hub.
    - Download the model snapshot locally.
2. **Load the Model with Transformers**
    - The model is loaded with **AutoModelForCausalLM** and **AutoTokenizer** from `transformers`.
    - The `BitsAndBytesConfig` is used to enable efficient 4-bit quantization.
3. **Generate Text with LangChain**
    - A `HuggingFacePipeline` is created to wrap the model.
    - A prompt template is defined and executed via LangChain.

```python
# Login Hugging Face with API Key
import huggingface_hub
from huggingface_hub import hf_hub_download
import os

MODEL_ID = "ENTER MODEL NAME HERE"
HUGGING_FACE_API_KEY = os.environ.get("HUGGING_FACE_API_KEY")

huggingface_hub.login(HUGGING_FACE_API_KEY)
huggingface_hub.snapshot_download(repo_id=MODEL_ID)

```

```python
# Load model with 4-bit quantization
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", quantization_config=quantization_config
)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=128)
```

```python
# Generate text using LangChain
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain

local_llm = HuggingFacePipeline(pipeline=pipe)

prompt = PromptTemplate(
    input_variables=["name"],
    template="Can you answer this question: '''{name}''' "
)

chain = LLMChain(prompt=prompt, llm=local_llm)
chain.run("What are competitors to Apache Kafka?")

```

### **Option 2: Loading the Model Using MLX-LM**

MLX is an optimized machine learning framework designed for Apple Silicon. Unlike the Hugging Face approach, this method loads the model directly into MLX's memory-efficient pipeline, avoiding additional framework dependencies.

**Steps:**

1. **Load the Model using MLX-LM**
    - The `mlx_lm.load()` function loads the quantized model without requiring explicit quantization steps.
2. **Generate Responses with MLX**
    - A chat template is applied to format the user query.
    - The `mlx_lm.generate()` function is used to produce an output.

```python
import mlx_lm

MODEL_ID = "mlx-community/Llama-3.2-3B-Instruct-4bit"
model, tokenizer = mlx_lm.load(MODEL_ID)

```

```python
# Generate response
user_content = "What are competitors to Apache Kafka?"

if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
    messages = [{"role": "user", "content": user_content}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

print(prompt)

response = mlx_lm.generate(model, tokenizer, prompt=prompt, verbose=True)

```

**Comparison of Methods**

| Feature | Hugging Face + LangChain | MLX-LM |
| --- | --- | --- |
| **Ease of Use** | Requires Hugging Face login & pipeline setup | Direct model loading |
| **Integration** | Works well with LangChain for structured prompts | Optimized for Apple Silicon |
| **Performance** | 4-bit quantization via BitsAndBytes | Native MLX optimization |

Both methods provide effective ways to load and generate text from the large language models. If you're looking for a **plug-and-play** approach with **Hugging Face compatibility and LangChain integration**, Option 1 is a great choice. However, if you're running the model on an **Apple Silicon device and want the most efficient performance**, Option 2 using MLX-LM is the recommended approach. 

## **Data Preparation**

In this section, we will prepare our dataset for fine-tuning the LLM using **synthetic SQL-related data**. The dataset is downloaded from Hugging Face, preprocessed, and formatted for training and validation.

- **About the Dataset**
    
    The dataset we are using is **"gretelai/synthetic_text_to_sql"**, which is a synthetic dataset designed for **Text-to-SQL** tasks. It consists of:
    
    - **`sql_prompt`** → A natural language question that requires an SQL query as an answer.
    - **`sql_context`** → The schema of the database tables related to the query.
    - **`sql`** → The generated SQL query for the given prompt.
    - **`sql_explanation`** → A textual explanation of how the SQL query is structured.
    
    This dataset is particularly useful for training LLMs to generate SQL queries based on natural language inputs, making it valuable for **NL2SQL** (Natural Language to SQL) applications.
    
- **Downloading the Dataset**
    
    First, we fetch the dataset from Hugging Face Hub and save it to a local directory:
    
    ```python
    # Load dataset from HuggingFace
    dataset_name = "gretelai/synthetic_text_to_sql"
    save_dir = "./data/synthetic_text_to_sql/"
    
    huggingface_hub.snapshot_download(repo_id=dataset_name, repo_type="dataset", local_dir=save_dir)
    
    ```
    
- **Preparing the Training and Validation Data**
    
    Once the dataset is downloaded, we preprocess it into a format suitable for fine-tuning.
    
    The preprocessing steps include:
    
    - Combining `sql_prompt` with `sql_context` to create a complete prompt.
    - Structuring the `sql` output as the model's expected response.
    - Saving the data in JSONL format (`prompt`, `completion` pairs).
    - Splitting the dataset into train, test, and validation sets.
- **Preprocessing Functions:**
    
    ```python
    import pandas as pd
    
    def prepare_train():
        df = pd.read_parquet('./data/synthetic_text_to_sql/train.snappy.parquet')
    
        df['prompt'] = df['sql_prompt'] + " with given SQL schema " + df['sql_context']
        df['completion'] = "SQL: " + df['sql'] + " Explanation: " + df['sql_explanation']
        df = df[['prompt', 'completion']]
    
        print(df.head(10))
    
        # Save as JSONL format
        df.to_json('train.jsonl', orient='records', lines=True)
    
    def prepare_test_valid():
        df = pd.read_parquet('./data/synthetic_text_to_sql/test.snappy.parquet')
    
        df['prompt'] = df['sql_prompt'] + " with given SQL schema " + df['sql_context']
        df['completion'] = "SQL: " + df['sql'] + " Explanation: " + df['sql_explanation']
        df = df[['prompt', 'completion']]
    
        # Splitting test and validation sets (2:1 ratio)
        split_index = int(len(df) * 2 / 3)
        test_df = df[:split_index]
        valid_df = df[split_index:]
    
        print(test_df.head(10))
        print(valid_df.head(10))
    
        # Save as JSONL format
        test_df.to_json('test.jsonl', orient='records', lines=True)
        valid_df.to_json('valid.jsonl', orient='records', lines=True)
    
    ```
    
    Once the functions are defined, we execute them to generate the processed dataset files:
    
    ```python
    from data.prepare import prepare_train, prepare_test_valid
    
    prepare_train()
    prepare_test_valid()
    ```
    
    This will generate three files:
    
    `train.jsonl` → Training data
    
    `test.jsonl` → Testing data
    
    `valid.jsonl` → Validation data
    
    These three files should be in the same directory because their paths will be directly provided to the processes in the later stages.
    

## Quantization Stage for Unquantized Models

In this stage, we will perform quantization using the `mlx-lm` library. Quantization is a crucial step that reduces the model's memory footprint and improves inference efficiency by lowering the precision of model weights (e.g., from 16-bit floating point to 8-bit integer). This enables large models to run efficiently on hardware with limited memory.

### Why Are We Doing This?

Quantization is essential for optimizing large language models for deployment, especially on hardware with limited memory and computing power. The main goal of quantization for us is:

- **Reducing Memory Usage:** Full-precision models (e.g., 16-bit or 32-bit) require large amounts of memory. By quantizing to lower-bit representations (e.g., 8-bit or 4-bit), we can significantly reduce the model’s size.

In our case, **we do not need to quantize our specific model** (`mlx-community/Llama-3.2-3B-Instruct-4bit`) because it has already been quantized to 4-bit. However, for users working with unquantized models, this step is necessary to ensure efficient performance on MLX.

To demonstrate the quantization process, we use another model (`meta-llama/Llama-3.1-8B-Instruct`) and apply 8-bit quantization using the `mlx.convert` API.

### Quantizing a Hugging Face Model

The `mlx-lm.convert` module provides functionality for converting and quantizing models from the Hugging Face Hub or  MLX Community to be used efficiently in MLX. Below is the code snippet used for quantization:

```python
# Quantizing a Hugging Face model using the mlx.convert API
# Our model is already quantized, so we will use a different model for demonstration.
import argparse
import sys

args = argparse.Namespace(
    hf_path="meta-llama/Llama-3.1-8B-Instruct",
    q_bits=8,
)

# Set args with sys.argv to simulate terminal execution
sys.argv = [
    "convert.py",
    "--hf-path", str(args.hf_path),
    "--q-bits", str(args.q_bits),
    "--quantize",
]

convert.main()

```

### Explanation of the Code

- The `mlx_lm.convert` module is allows quantization by specifying the number of bits (`q_bits`).
- **Key Parameters**
    - `hf_path`: The path to the Hugging Face model that will be quantized.
    - `q_bits`: The desired quantization bit depth (in this case, 8-bit).
- **Using `sys.argv`**
    - MLX-LM's conversion script is designed to be executed from the command line.
    - To simulate a command-line execution inside a Python script, we manually set `sys.argv` with the required parameters before calling `convert.main()`.
    - This ensures that the script receives the correct arguments as if it were run in the terminal.

By following this method, we can efficiently quantize models and making them more memory-efficient for fine-tune or inference on MLX.

## **Fine-Tuning with LoRA**

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that allows us to update only a small subset of model parameters, reducing memory usage and computational cost. Since our model is already quantized (4-bit), we can directly apply LoRA without needing further quantization.

Below is the setup for fine-tuning:

```python
import argparse
import sys

args = argparse.Namespace(
    model="mlx-community/Llama-3.2-3B-Instruct-4bit",
    data="./data/",
    train=True,
    test=False,
    batch_size=2,
    num_layers=16,
    iters=1000,
    fine_tune_type="lora",
    resume_adapter_file=None,
    adapter_path="adapters_llama_3b",
    seed=42
)

# Simulate command-line arguments for the LoRA script
sys.argv = [
    "lora.py",
    "--model", str(args.model),
    "--data", str(args.data),
    "--train",
    "--batch-size", str(args.batch_size),
    "--num-layers", str(args.num_layers),
    "--iters", str(args.iters),
    "--fine-tune-type", str(args.fine_tune_type),
    "--adapter-path", str(args.adapter_path),
    "--seed", str(args.seed)
]

lora.main()

```

### **Key Parameters**

- **`-model`**
    - Specifies the pre-trained model we want to fine-tune.
    - Example: `"mlx-community/Llama-3.2-3B-Instruct-4bit"`
- **`-data`**
    - Points to the dataset directory for fine tuning process containing `train.jsonl`, `valid.jsonl`, and `test.jsonl` files.
    - Example: `"./data/"`
- **`-train`**
    - Indicates that training should be performed.
    - If this flag is set, the script will enter the training phase.
- **`-batch-size`**
    - Defines how many examples are processed in a single training step.
    - Example: `2`
    - Lower values reduce memory usage but may slow down training.
- **`-num-layers`**
    - Number of layers for model will be fine-tuned.
    - Example: `16`

The reason for using `sys.argv` and calling LoRA with the `main` function is the same as the explanations provided in the **Quantization** section.

By using LoRA, we significantly reduce computational overhead while effectively adapting the model to our specific task.

## **Evaluation**

In this section, we evaluate the fine-tuned model using the test dataset. The evaluation is performed by setting arguments with `sys.argv` and calling the `lora.main()` function, similar to the fine-tuning process. This ensures that the script runs with the correct parameters.

- **Loss Function**
    
    The model is evaluated using the **cross-entropy loss**, a common loss function for language models. It measures how well the predicted probability distribution aligns with the actual words in the dataset. Lower loss values indicate better predictions.
    
- **Perplexity (PPL) Metric**
    
    The primary evaluation metric used here is **Perplexity (PPL)**, which quantifies how well the model predicts the next word in a sequence. A lower perplexity suggests better generalization:
    
    - **PPL < 10** is considered good.
    - **PPL ≈ 2** is near perfect.

**Key Parameters**

The evaluation is triggered by passing arguments using `sys.argv`, including:

- **`-model`**: Specifies the pre-trained model.
- **`-data`**: Defines the dataset location.
- **`-test`**: Ensures the model is tested instead of trained.
- **`-adapter-path`**: Loads the fine-tuned LoRA adapter weights.

```python
sys.argv = [
    "lora.py",
    "--model", str(args.model),
    "--data", str(args.data),
    "--test",
    "--adapter-path", str(args.adapter_path),
]

lora.main()

```

### **Metric Results**

- **Test loss:** 0.620 (Lower values indicate better generalization).
- **Test PPL:** 1.859 (The model predicts each word by selecting from ~1.859 different words on average, an excellent result).

These results indicate that the fine-tuned model has effectively learned from the dataset and can generate structured outputs with high accuracy.

## **Evaluation of Base Model vs Fine-Tuned Model**

**Custom Chat Template**

To ensure a structured and controlled input format for the model, we implemented a **custom chat template** instead of using the default formatting. This function ensures that system, user, and assistant messages are properly structured, which can impact the model’s performance and consistency.

```python
# Custom Template
def custom_chat_template(messages):
    """Creates LLaMA 3.2 custom chat template."""
    chat_str = "<|begin_of_text|>"
    for msg in messages:
        if msg["role"] == "system":
            chat_str += f"<|start_header_id|>system<|end_header_id|>\n\n{msg['content']}\n\n<|eot_id|>\n"
        elif msg["role"] == "user":
            chat_str += f"<|start_header_id|>user<|end_header_id|>\n\n{msg['content']}<|eot_id|>\n"
        elif msg["role"] == "assistant":
            chat_str += f"<|start_header_id|>assistant<|end_header_id|>\n\n{msg['content']}<|eot_id|>\n"
    chat_str += "<|start_header_id|>assistant<|end_header_id|>"
    return chat_str

```

---

### **Base Model Performance**

**Code Execution:**

```python
# Base Model
messages_base = [
    {"role": "system", "content": "Cutting Knowledge Date: December 2023\nToday Date: 14 Feb 2025\nYou are the support assistant for questions that are asked to you."},
    {"role": "user", "content": "List all transactions and customers from the 'Africa' region."}
]

prompt_base = custom_chat_template(messages_base)
response = mlx_lm.generate(model, tokenizer, prompt=prompt_base, verbose=True)

```

**Base Model Output:**

```
==========

I don't have any information about transactions or customers from the 'Africa' region. I'm a text-based AI assistant and do not have access to any specific data or databases. If you could provide more context or clarify what you are referring to, I'll do my best to help.
==========
Prompt: 61 tokens, 71.435 tokens-per-sec
Generation: 60 tokens, 68.330 tokens-per-sec
Peak memory: 1.875 GB

```

**Analysis:**

- The base model lacks domain-specific knowledge and returns a generic response indicating that it does not have access to structured transaction data.
- The response does not attempt to generate an SQL query, showing that the model was not fine-tuned for such tasks.

---

### **Fine-Tuned Model Performance**

**Code Execution:**

```python
# Fine-Tuned Model
import mlx_lm.generate as generate
import argparse
import sys

messages_ft = [
    {"role": "system", "content": "Cutting Knowledge Date: December 2023\nToday Date: 14 Feb 2025\nYou are the support assistant for questions that are asked to you."},
    {"role": "user", "content": "List all transactions and customers from the 'Africa' region."}
]

prompt_ft = custom_chat_template(messages_ft)

args = argparse.Namespace(
    model=MODEL_ID,
    adapter_path = "./adapters_llama_3b/",
    prompt = prompt_ft,
    max_tokens = 500,
    verbose = True,
)

# Set args with sys.argv
sys.argv = [
    "generate.py",
    "--model", str(args.model),
    "--ignore-chat-template",
    "--adapter-path", str(args.adapter_path),
    "--prompt",str(args.prompt),
    "--max-tokens",str(args.max_tokens),
    "--verbose",str(args.verbose)
]

generate.main()

```

**Fine-Tuned Model Output:**

```
==========

SQL: SELECT transactions.id, customers.name, transactions.amount, customers.country FROM transactions JOIN customers ON transactions.customer_id = customers.id WHERE customers.region = 'Africa';
==========
Prompt: 62 tokens, 160.189 tokens-per-sec
Generation: 34 tokens, 51.437 tokens-per-sec
Peak memory: 3.686 GB

```

**Analysis:**

- The fine-tuned model **correctly generates an SQL query**, which aligns with the user prompt.
- This suggests that fine-tuning has provided the model with **domain-specific knowledge** about database queries.
- Although the fine-tuned model uses more memory and has a slightly lower token generation speed, the output is **far more relevant and actionable**.

### **Compare Result**

The comparison between the base and fine-tuned models demonstrates the **effectiveness of fine-tuning** in adapting the model to a domain-specific task. The base model struggles to provide a relevant response, while the fine-tuned model successfully **constructs an SQL query**.

Although fine-tuning increases **memory usage**, the improvement in **response quality** justifies the trade-off, making it a powerful technique for specialized applications.

## **Fusing Adapters to Model**

Model fusion is the process of merging fine-tuned adapters (such as LoRA) with the base model. This ensures that the model no longer requires separate adapter files during inference, leading to:

- **Reduced computational overhead** by integrating adapters directly into the model weights,
- **Easier deployment** as a standalone model,
- **Optimized storage** when exporting to formats like GGUF.

In MLX-LM, this process is handled using the `mlx.fuse` function or an external script like **`fuse.py`**. Below is an example of executing the fusing process using a script:

```python
args = argparse.Namespace(
    model=MODEL_ID,
    adapter_path = "./adapters_llama_3b/",
    prompt = prompt_ft,
    max_tokens = 500,
    verbose = True,
    save_path = "./sql_agent/",
    gguf_path = "./sql_agent/gguf_model/ggml-model-sql-agent.gguf"
)

sys.argv = [
    "fuse.py",
    "--model", str(args.model),
    "--adapter-path", str(args.adapter_path),
    "--de-quantize",  # Required since GGUF format does not support quantized models.
    "--save-path", str(args.save_path),
    "--export-gguf",
    # "--gguf-path", str(args.gguf_path)  # Optional path for GGUF model output.
]

fuse.main()

```

**Key Parameters:**

- `-model`: Specifies the base model to be fused.
- `-adapter-path`: Defines the location of fine-tuned adapters.
- `-de-quantize`: Ensures dequantization for GGUF compatibility.
- `-save-path`: Sets the directory for saving the fused model.
- `-export-gguf`: Enables conversion of the fused model into GGUF format.
- `-gguf-path`: Optional path to define where the GGUF model should be stored.

Once the fusing process is complete, the model is **fully integrated with adapters**, making it more efficient for inference and deployment in production environments with GGUF format.

## Build Ollama Model

Ollama is a powerful tool that simplifies the deployment and management of large language models. In this section, we will walk through the process of building a custom Ollama model from a **GGUF** file. This includes setting up the environment, creating a model definition, and verifying the model's availability.

- **Step 1: Start Ollama Service**
    
    Before creating a model, ensure that the Ollama API is running:
    
    ```bash
    !ollama serve
    
    ```
    
    This command keeps the Ollama service active, allowing models to be loaded and executed.
    
- **Step 2: Define the Model with a Model File**
    
    Ollama uses a **Modelfile**, similar to a Dockerfile, to define models. Create a new Modelfile:
    
    ```bash
    !nano models/Modelfile
    
    ```
    
    Inside this file, specify the base model using the **FROM** directive, pointing to the GGUF file we converted earlier:
    
    ```
    FROM ../sql_agent/ggml-model-f16.gguf
    ```
    
    Save and close the file.
    
- **Step 3: Create the Ollama Model**
    
    With the Modelfile in place, build the model using:
    
    ```bash
    !ollama create sql-agent -f models/Modelfile
    
    ```
    
    This process will parse the GGUF file, copy model components, and register it within Ollama.
    
- **Step 4: Verify Model Creation**
    
    To check if the model was successfully added, list the available models:
    
    ```bash
    !ollama list
    ```
    
    This should display the newly created model, e.g., `sql-agent:latest`.
    
- **Step 5: Run the Model**
    
    Now, the model is ready to use. Run it with:
    
    ```bash
    !ollama run sql-agent
    
    ```
    
    This launches the model, allowing it to process queries and generate responses.
    

By following these steps, you can integrate and run your fine-tuned model with Ollama, making it easily accessible for various applications.

## Result

In this article, we have thoroughly examined the LLM fine-tuning process from start to finish. We touched on various topics such as data preparation, model loading, training, and evaluation. We have demonstrated the success of the fine-tuned model in a domain-specific area through both its generated output and metrics. Additionally, we explored how the LoRA method works in practice with the MLX library. By leveraging the MLX library, we were able to effectively fine-tune our model using Apple Silicon chips. Thank you for reading.

## Contact Information

LinkedIn → [ardakskc](https://www.linkedin.com/in/ardakskc/)

Github → [ardakskc](https://github.com/ardakskc)

## Reference

1-[https://arxiv.org/abs/2305.14314](https://arxiv.org/abs/2305.14314)

2- [https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)

3-[https://medium.com/rahasak/fine-tuning-llms-on-macos-using-mlx-and-run-with-ollama-182a20f1fd2c](https://medium.com/rahasak/fine-tuning-llms-on-macos-using-mlx-and-run-with-ollama-182a20f1fd2c)
4-[https://towardsdatascience.com/lora-fine-tuning-on-your-apple-silicon-macbook-432c7dab614a/](https://towardsdatascience.com/lora-fine-tuning-on-your-apple-silicon-macbook-432c7dab614a/)

5-[https://medium.com/@mustangs007/mlx-building-fine-tuning-llm-model-on-apple-m3-using-custom-dataset-9209813fd38e](https://medium.com/@mustangs007/mlx-building-fine-tuning-llm-model-on-apple-m3-using-custom-dataset-9209813fd38e)

6-[https://medium.com/@elijahwongww/how-to-finetune-llama-3-model-on-macbook-4cb184e6d52e](https://medium.com/@elijahwongww/how-to-finetune-llama-3-model-on-macbook-4cb184e6d52e)

7-[https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/LORA.md](https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/LORA.md)

8-[https://medium.com/@zilliz_learn/lora-explained-low-rank-adaptation-for-fine-tuning-llms-066c9bdd0b32](https://medium.com/@zilliz_learn/lora-explained-low-rank-adaptation-for-fine-tuning-llms-066c9bdd0b32)

9-[https://huggingface.co/mlx-community/Llama-3.2-3B-Instruct-4bit/blob/main/README.md](https://huggingface.co/mlx-community/Llama-3.2-3B-Instruct-4bit/blob/main/README.md)

10-[https://huggingface.co/datasets/gretelai/synthetic_text_to_sql](https://huggingface.co/datasets/gretelai/synthetic_text_to_sql)
