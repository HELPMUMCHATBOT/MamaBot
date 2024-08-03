# Model Card for MamaBot-Llama-1

MamaBot-Llama-1 is an opensource fine-tuned large language model developed by HelpMum to assist with maternal healthcare by providing accurate and reliable answers to questions about pregnancy and childbirth. The model has been fine-tuned on Llama 3.1 8b-instruct using a dataset of maternal healthcare questions and answers.

## Model Details

- **Developed by:** HelpMum
- **Shared by :** HelpMum
- **Model type:** Causal Language Model (Llama 3.1 8b-instruct)
- **Language(s) (NLP):** English
- **License:** Apache-2.0
- **Finetuned from model:** Llama 3.1 8b-instruct

### Model Sources

- **Repository:** [MamaBot-Llama-1 on Hugging Face](https://huggingface.co/HelpMum-Personal/mamabot-llama-1)

## Uses

### Direct Use

MamaBot-Llama-1 can be directly used to provide answers to maternal healthcare questions, offering guidance and support to mothers during pregnancy and childbirth.

### Downstream Use

The model can be integrated into healthcare applications, chatbots, or other systems that aim to provide maternal healthcare support.

### Out-of-Scope Use

The model is not intended for use in medical diagnosis or treatment without the supervision of a qualified healthcare professional. It should not be used for malicious purposes or misinformation.

## Bias, Risks, and Limitations

The model was trained on a specific dataset related to maternal healthcare. While it aims to provide accurate and supportive information, users should be aware of the following:

- **Bias:** The model may reflect biases present in the training data, which could affect the quality and impartiality of the responses.
- **Risks:** Users should not rely solely on the model for critical medical decisions. Always consult with a healthcare professional for medical advice.
- **Limitations:** The model's responses are based on the data it was trained on and may not cover all possible scenarios or latest medical guidelines.

### Recommendations

Users (both direct and downstream) should be made aware of the risks, biases, and limitations of the model. It is recommended to use the model as a supplementary tool and not as a primary source of medical advice.

## How to Get Started with the Model

Use the code below to get started with the model.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "HelpMum-Personal/mamabot-llama-1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

messages = [
    {
        "role": "user",
        "content": "Why might mothers not realize they are already pregnant in the first two weeks?"
    }
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to("cuda")
outputs = model.generate(**inputs, max_length=100, num_return_sequences=1)
text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(text.split("assistant")[1])
```

## Training Details

### Training Data

The training data consists of a HelpMum-created dataset of maternal healthcare questions and answers covering all stages of pregnancy up to birth.

### Training Procedure

#### Preprocessing

The dataset was cleaned and formatted to align with the required input format for the model.

#### Training Hyperparameters

- **Training regime:** torch.bfloat16
- **Optimizer:** paged_adamw_32bit
- **Learning rate:** 2e-4

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The testing data is a subset of the training dataset, split into training and testing sets.

#### Factors

The evaluation considered the training and validation losses.

#### Metrics

The model was evaluated based on training loss and validation loss metrics.

### Results

- **Training Loss:** 0.4654
- **Validation Loss:** 0.5168

#### Summary

The model showed consistent performance with a training loss of 0.4654 and a validation loss of 0.5168, indicating its effectiveness in answering maternal healthcare questions.

## Environmental Impact

- **Hardware Type:** GPU

## Technical Specifications

### Model Architecture and Objective

The model is based on the Llama 3.1 8b-instruct architecture and aims to provide accurate and supportive responses to maternal healthcare questions.

### Compute Infrastructure

#### Hardware

The model was trained using GPUs to handle the computational load of fine-tuning a large language model.

#### Software

The training and inference were conducted using the Hugging Face Transformers library and other associated tools.

## Citation

**BibTeX:**

```bibtex
@misc{mamabot-llama-1,
  author = {HelpMum},
  title = {MamaBot-Llama-1},
  year = {2024},
  howpublished = {\url{https://huggingface.co/HelpMum-Personal/mamabot-llama-1}},
}
```

**APA:**

HelpMum. (2024). MamaBot-Llama-1. Retrieved from https://huggingface.co/HelpMum-Personal/mamabot-llama-1

## Model Card Contact

For more information, please contact [tech@helpmum.org](mailto:tech@helpmum.org).
