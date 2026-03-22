import ollama
import pandas as pd
import ast

df = pd.read_csv('sri_lanka_landmarks_final.csv')
df['Facts'] = df['Facts'].apply(ast.literal_eval)


def generate_narration(landmark, facts, significance):
    prompt = f"""Plan 5 scenes for a documentary about {landmark}. Facts: {facts}. Significance: {significance}. Format: Scene X | Video: [Visuals] | Audio: [Dialogue]
    """

    response = ollama.generate(model='llama3.1', prompt=prompt)
    return response['response']


from rouge_score import rouge_scorer
from bert_score import score as bert_score_calculation
import torch

def evaluate_model(generated_text, reference_facts):
    # 2. Calculate ROUGE Scores (1, 2, and L)
    # ROUGE-1: Single word matching
    # ROUGE-2: Two-word phrase matching
    # ROUGE-L: Flow/Sentence structure matching
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    # Join the 5 facts into one reference string
    reference_text = " ".join(reference_facts)

    rouge_results = scorer.score(reference_text, generated_text)

    # 3. Calculate BERTScore (Meaning Matching)
    P, R, F1 = bert_score_calculation([generated_text], [reference_text], lang="en", verbose=False)

    # 4. Gather everything into a nice dictionary
    results = {
        "ROUGE-1": rouge_results['rouge1'].fmeasure,
        "ROUGE-2": rouge_results['rouge2'].fmeasure,
        "ROUGE-L": rouge_results['rougeL'].fmeasure,
        "BERTScore": F1.item()  # .item() converts the math tensor to a simple number
    }

    return results

# Evaluation
all_results = []

print("Running evaluation for all 23 landmarks...")
for index, row in df.iterrows():
    # 1. Generate story
    generated_story = generate_narration(row['Landmark'], row['Facts'], row['Significance'])

    # 2. Evaluate
    scores = evaluate_model(generated_story, row['Facts'])
    scores['Landmark'] = row['Landmark']
    all_results.append(scores)

# 3. Create a summary table
results_df = pd.DataFrame(all_results)
print("\n--- Average Project Metrics ---")
print(results_df[['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BERTScore']].mean())