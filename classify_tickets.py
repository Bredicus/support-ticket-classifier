import outlines
from transformers import AutoTokenizer, AutoModelForCausalLM
from pydantic import BaseModel
from enum import Enum
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from datasets import Dataset

# ------------------------------
# CONFIGURATION
# ------------------------------
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# ------------------------------
# ENUMS & SCHEMA
# ------------------------------
class Department(str, Enum):
    billing = "Billing"
    technical = "Technical Support"
    sales = "Sales"
    abuse = "Abuse"
    general = "General Inquiry"
    support = "Support"
    security = "Security"

class Tag(str, Enum):
    payment_issue = "payment_issue"
    invoice_dispute = "invoice_dispute"
    refund_request = "refund_request"
    account_suspension = "account_suspension"
    dns_configuration = "dns_configuration"
    ssl_certificate = "ssl_certificate"
    server_performance = "server_performance"
    backup_failure = "backup_failure"
    email_delivery = "email_delivery"
    migration_assistance = "migration_assistance"
    software_bug = "software_bug"
    upgrade_plan = "upgrade_plan"
    new_service_inquiry = "new_service_inquiry"
    cancellation_request = "cancellation_request"
    spam_abuse = "spam_abuse"
    phishing_attempt = "phishing_attempt"
    DDoS_attack = "DDoS_attack"
    malware_distribution = "malware_distribution"
    content_violation = "content_violation"
    hacking_attempt = "hacking_attempt"
    general_inquiry = "general_inquiry"
    feedback = "feedback"
    complaint = "complaint"
    escalation = "escalation"
    account_update = "account_update"
    service_status = "service_status"
    policy_question = "policy_question"
    login_problem = "login_problem"
    control_panel_help = "control_panel_help"
    feature_request = "feature_request"
    general_question = "general_question"
    phishing_alert = "phishing_alert"
    spam_report = "spam_report"
    unauthorized_access = "unauthorized_access"
    malware_infection = "malware_infection"

class Priority(int, Enum):
    low = 1
    medium = 3
    high = 5

class TicketClassification(BaseModel):
    department: Department
    tag: Tag
    priority: Priority
    keywords: list[str]

# ------------------------------
# DATASET CREATION
# ------------------------------
ticket_data = {
    "text": [
        "My website is down and I can't access the control panel.",
        "I need to change my billing address on file.",
        "How do I install an SSL certificate on my domain?",
        "We’re seeing extremely high latency and packet loss on our server.",
        "I was charged twice this month, please investigate.",
        "My email is not syncing properly on my phone.",
        "What’s the process to upgrade my VPS plan?",
        "I received a phishing complaint about my IP address.",
        "The SSL certificate has expired and we need urgent renewal.",
        "I want to cancel my service and request a refund.",
        "I believe someone accessed my account without permission.",
        "Where can I find the status of your data center services?",
        "We detected malware on one of our hosted websites.",
        "I’d like to provide feedback on your customer service.",
        "There’s an ongoing DDoS attack on our network.",
        "I need help logging into my control panel.",
        "What is your policy on storing customer data?",
        "We suspect spam abuse coming from our hosted domain.",
        "Our internal backup process failed last night.",
        "Can I get assistance with migrating my website to your servers?"
    ],
    "true_department": [
        Department.technical,       # 1
        Department.billing,         # 2
        Department.technical,       # 3
        Department.technical,       # 4
        Department.billing,         # 5
        Department.technical,       # 6
        Department.sales,           # 7
        Department.abuse,           # 8
        Department.technical,       # 9
        Department.sales,           # 10
        Department.security,        # 11
        Department.general,         # 12
        Department.security,        # 13
        Department.general,         # 14
        Department.abuse,           # 15
        Department.support,         # 16
        Department.general,         # 17
        Department.abuse,           # 18
        Department.technical,       # 19
        Department.support          # 20
    ]
}

dataset = Dataset.from_dict(ticket_data)

# ------------------------------
# PROMPT CREATION
# ------------------------------
def create_prompt(ticket: str) -> str:
    departments_str = ", ".join(dept.value for dept in Department)
    tags_str = ", ".join(tag.value for tag in Tag)
    
    response_examples = """
Example 1:
--- Support Ticket ---
My credit card was charged twice this month and I need a refund.
----------------------
{
  "department": "Billing",
  "tag": "payment_issue",
  "priority": 5,
  "keywords": ["credit card", "charged twice", "refund"]
}

Example 2:
--- Support Ticket ---
There is a malware infection detected on one of our hosted websites.
----------------------
{
  "department": "Security",
  "tag": "malware_infection",
  "priority": 5,
  "keywords": ["malware", "infection", "hosted website"]
}

Example 3:
--- Support Ticket ---
How do I upgrade my VPS plan to get more resources?
----------------------
{
  "department": "Sales",
  "tag": "upgrade_plan",
  "priority": 3,
  "keywords": ["upgrade", "VPS plan", "resources"]
}
"""

    return (
        f"You are a classification assistant for a web hosting company's support system.\n"
        f"Your job is to analyze the following customer support ticket and classify it into the correct categories.\n\n"
        f"{response_examples}\n\n"
        f"--- Support Ticket ---\n"
        f"{ticket}\n"
        f"----------------------\n\n"
        f"### CLASSIFICATION RULES ###\n"
        f"1. Department: Choose exactly one department from the list below.\n"
        f"   Do NOT confuse departments — for example, 'Abuse' and 'Security' are distinct from 'Support' and 'General Inquiry'.\n"
        f"   Departments: {departments_str}\n\n"
        f"2. Tag: Choose exactly one tag from the list below that best describes the main issue.\n"
        f"   Tags: {tags_str}\n\n"
        f"3. Priority: Choose one integer value representing urgency:\n"
        f"   - 1 (Low): Informational or minor issues with no immediate impact\n"
        f"   - 3 (Medium): Issues affecting functionality but with workarounds\n"
        f"   - 5 (High): Critical outages, severe security threats, or payment issues blocking service\n\n"
        f"4. Keywords: Provide a short list (3-5) of concise, relevant keywords or phrases from the ticket.\n"
        f"   - Use only important terms related to the issue\n"
        f"   - No stopwords, punctuation, or duplicates\n\n"
        f"### RESPONSE FORMAT ###\n"
        f"Respond ONLY with a valid JSON object matching exactly this schema:\n"
        f'{{\n'
        f'  "department": "<one of the above department names>",\n'
        f'  "tag": "<one of the above tags>",\n'
        f'  "priority": <1 or 3 or 5>,\n'
        f'  "keywords": ["<keyword1>", "<keyword2>", "<keyword3>", ...]\n'
        f'}}\n\n'
        f"Do NOT include any explanations or text outside the JSON."
    )

# ------------------------------
# LOAD MODEL
# ------------------------------
print(f"Loading model: {MODEL_NAME}...")
model = outlines.from_transformers(
    AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto"),
    AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
)

# ------------------------------
# CLASSIFICATION FUNCTION
# ------------------------------
def classify_tickets(ds: Dataset):
    predictions = []
    truths = []
    errors = []

    for i, row in enumerate(ds, start=1):
        prompt = create_prompt(row["text"])
        try:
            raw_response = model(prompt, TicketClassification, max_new_tokens=200)
            parsed = TicketClassification.model_validate_json(raw_response)

            print(f"--- Ticket {i} ---")
            print(f"Text: {row['text']}")
            print(f"Department (pred): {parsed.department} | (true): {row['true_department']}")
            print(f"Tag: {parsed.tag}")
            print(f"Priority: {parsed.priority}")
            print(f"Keywords: {parsed.keywords}")
            print("-" * 50)

            predictions.append(parsed.department.value if isinstance(parsed.department, Enum) else str(parsed.department))
            truths.append(row["true_department"].value if isinstance(row["true_department"], Enum) else str(row["true_department"]))

        except Exception as e:
            errors.append((row["text"], str(e)))

    return predictions, truths, errors

# ------------------------------
# MAIN
# ------------------------------
if __name__ == "__main__":
    preds, truths, errs = classify_tickets(dataset)

    truths = [Department(t) if isinstance(t, str) else t for t in truths]
    preds = [Department(p) if isinstance(p, str) else p for p in preds]

    truths_str = [t.value for t in truths]
    preds_str = [p.value for p in preds]

    labels = [d.value for d in Department]

    if preds_str and truths_str and any(t in truths_str for t in labels):
        print("\nClassification Report:")
        print(classification_report(truths_str, preds_str, labels=labels, zero_division=0))

        try:
            cm = confusion_matrix(truths_str, preds_str, labels=labels)
            cm_df = pd.DataFrame(cm, index=labels, columns=labels)
            print("\nConfusion Matrix:")
            print(cm_df)
        except ValueError as e:
            print(f"\nCould not generate confusion matrix: {e}")
    else:
        print("\nNo valid predictions to compute classification report or confusion matrix.")

    if errs:
        print("\nErrors:")
        for text, err in errs:
            print(f"Ticket: {text} | Error: {err}")

