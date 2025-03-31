import pandas as pd
from evaluate import load

rouge = load("rouge")
response_df = pd.read_csv("./data/test_responses.csv")

responses = response_df["response"].tolist()
true_notes = response_df["label"].tolist()

# Calculate ROUGE scores
rouge_scores = rouge.compute(
    predictions=responses,
    references=true_notes,
    use_stemmer=True,
    rouge_types=["rouge1", "rouge2"],
)

print(rouge_scores)
