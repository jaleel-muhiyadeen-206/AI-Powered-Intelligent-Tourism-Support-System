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