
import pandas as pd
import numpy as np

# Define the number of samples to take from each class
N_SAMPLES = 500

print("Loading datasets...")
# Load the original and translated datasets
try:
    df_original = pd.read_csv('multitude.csv')
    df_translated = pd.read_csv('multitude_portuguese_translated.csv')
    print("Datasets loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: {e}. Make sure 'multitude.csv' and 'multitude_portuguese_translated.csv' are in the directory.")
    exit()

# --- Prepare Class 1: Human ---
print("Preparing 'human' class...")
df_human = df_original[df_original['label'] == 0]
# Ensure we only use English text for a controlled comparison
df_human = df_human[df_human['language'] == 'en']
if len(df_human) > N_SAMPLES:
    df_human = df_human.sample(n=N_SAMPLES, random_state=42)
df_human['source_label'] = 'human'
print(f"Found {len(df_human)} human samples.")

# --- Prepare Class 2: AI Generated ---
print("Preparing 'ai_generated' class...")
df_ai = df_original[df_original['label'] == 1]
# Match the language with the human samples
df_ai = df_ai[df_ai['language'] == 'en']
if len(df_ai) > N_SAMPLES:
    df_ai = df_ai.sample(n=N_SAMPLES, random_state=42)
df_ai['source_label'] = 'ai_generated'
print(f"Found {len(df_ai)} AI generated samples.")

# --- Prepare Class 3: Machine Translated ---
print("Preparing 'machine_translated' class...")
# We use the entire translated dataset as the 'machine_translated' class
# The act of translation is the feature we want to capture
df_mt = df_translated.copy()
if len(df_mt) > N_SAMPLES:
    df_mt = df_mt.sample(n=N_SAMPLES, random_state=42)
df_mt['source_label'] = 'machine_translated'
print(f"Found {len(df_mt)} machine translated samples.")


# --- Combine and Save ---
print("Combining datasets...")
# Select relevant columns and combine
final_df = pd.concat([
    df_human[['text', 'source_label']],
    df_ai[['text', 'source_label']],
    df_mt[['text', 'source_label']]
])

# Shuffle the dataset
final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to a new CSV
output_filename = 'processed_xai_dataset.csv'
final_df.to_csv(output_filename, index=False)

print(f"\nSuccessfully created '{output_filename}' with {len(final_df)} total samples.")
print("\nClass distribution:")
print(final_df['source_label'].value_counts())
