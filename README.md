# Support Ticket Classification Demo

This is a small demo project for classifying support tickets into
departments, tags, and priority levels using a language model (LLM).

The purpose of this project is to showcase:
- Prompt engineering with few-shot examples for LLM classification tasks
- JSON schema validation and output normalization with `pydantic`
- Simple evaluation using a confusion matrix and classification report
- Robust handling of model outputs including normalization and error handling

## How It Works

1. A list of example support tickets is provided, each with a ground-truth department, tag, and priority.
2. The script constructs prompts with few-shot classification examples to guide the LLM.
3. The model generates a JSON object with structured classification output matching a strict schema.
4. Outputs are validated and normalized to map predicted strings to canonical enum labels.
5. Predictions are compared to the ground truth, and evaluation metrics including confusion matrices and classification reports are generated.

## Example Output

    --- Ticket 1 ---
    Text: My invoice shows the wrong amount for last month.
    Department (pred): Billing | (true): Billing
    Tag: payment_issue
    Priority: 5
    Keywords: ['invoice', 'wrong amount', 'billing']
    ----------------------------------------

## Evaluation Metrics

The script outputs:

- **Classification report** for each target (department, tag, priority)
- **Confusion matrix** for each target

Example:

```
Department Classification Report:
              precision    recall  f1-score   support

    Billing       1.00      1.00      1.00         2
  Technical       0.50      1.00      0.67         2
   General        1.00      0.50      0.67         2

   accuracy                           0.83         6
```

## Installation

#### It is recommended to do this inside a Python virtual environment

Clone this repository:

    git clone https://github.com/Bredicus/support-ticket-classifier.git
    cd support-ticket-classifier

Install dependencies:

    pip install -r requirements.txt

## Running the Demo

    python classify_tickets.py

## Model Choice

The demo uses the model:

- `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` (1.5B parameters)

You can experiment with other models by changing `MODEL_NAME` in the script.

## Notes

- This project uses **synthetic example tickets** — no real customer data is included.
- Few-shot examples are included in the prompt to improve classification accuracy.
- The output JSON is validated and normalized before evaluation to handle variations in model-generated text.
- Because this uses a relatively small, instruction-tuned model without fine-tuning, accuracy may vary. Larger or fine-tuned models can yield better results.

