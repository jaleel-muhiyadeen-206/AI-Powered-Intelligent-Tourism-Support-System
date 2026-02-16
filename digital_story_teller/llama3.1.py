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

def evaluate_model(generated_text, reference_facts):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    # Join the 5 facts into one reference string
    reference_text = " ".join(reference_facts)
    scores = scorer.score(reference_text, generated_text)
    return scores['rougeL'].fmeasure

# Example evaluation
score = evaluate_model(story, sigiriya_row['Facts'])
print(f"Validation (ROUGE-L Accuracy): {score:.4f}")