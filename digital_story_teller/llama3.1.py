import ollama
import pandas as pd
import ast

df = pd.read_csv('sri_lanka_landmarks_final.csv')
df['Facts'] = df['Facts'].apply(ast.literal_eval)


def generate_narration(landmark, facts, significance):
    prompt = f"""
    Act as a professional Sri Lankan tour guide. 
    Using the following facts and significance, create a 1-minute historical narration for a tourism video.

    Landmark: {landmark}
    Significance: {significance}
    Facts: {", ".join(facts)}

    Narration:
    """

    response = ollama.generate(model='llama3.1', prompt=prompt)
    return response['response']


# Test it for Sigiriya
sigiriya_row = df.iloc[1]
story = generate_narration(sigiriya_row['Landmark'], sigiriya_row['Facts'], sigiriya_row['Significance'])
print(f"--- Story for {sigiriya_row['Landmark']} ---\n{story}")

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

# Example evaluation
final_scores = evaluate_model(story, sigiriya_row['Facts'])
print("--- Evaluation Results ---")
for metric, value in final_scores.items():
    print(f"{metric}: {value:.4f}")