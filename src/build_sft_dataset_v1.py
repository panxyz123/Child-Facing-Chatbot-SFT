import os
import json
import time
import pandas as pd
from datetime import datetime
from typing import Literal
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from datasets import load_dataset

# =============================================================================
# 1. Setup & Model Definition
# =============================================================================
load_dotenv("google_gemini_key.env")
client = genai.Client()


# Define the structured output schema with strict Intent categories
class ChildrenSFTData(BaseModel):
    age: Literal["0-3", "4-6", "7-9", "10-13", "14-18"]
    # Restrict intent to exactly these four categories
    intent: Literal['narrative', 'scientific question', 'non-scientific question', 'request storytelling']
    instruction: str
    response: str


CONFIG = {
    "local_csv": "sampled_data.csv",
    "hf_dataset": "nikister/children-assistant",
    "output_path": "combined_sft_data.jsonl",
    "model_id": "gemini-2.0-flash-lite",
    "total_samples": 1200,
    "delay_seconds": 2  # Recommended for Free Tier stability
}


# =============================================================================
# 2. Hybrid Data Loading & Sampling
# =============================================================================
def get_combined_samples():
    print("📥 Loading and mixing datasets into DataFrame...")

    # --- A. Local CSV Processing ---
    # Column 0: Intent, Column 1: Instruction
    try:
        df_local = pd.read_csv(CONFIG["local_csv"], header=None, skiprows=1)
        df_local = df_local[[0, 1]].rename(columns={0: 'intent', 1: 'instruction'})
        df_local['source'] = 'local'
        df_local['age'] = 'N/A'  # To be inferred by Gemini
    except Exception as e:
        print(f"⚠️ Local CSV error: {e}")
        df_local = pd.DataFrame(columns=['intent', 'instruction', 'source', 'age'])

    # --- B. Hugging Face Dataset Processing ---
    # Load and immediately convert to Pandas for shuffling/sampling
    ds_hf = load_dataset(CONFIG["hf_dataset"], split='train')
    df_hf_raw = ds_hf.to_pandas()

    # Take 800 samples as requested
    df_hf = df_hf_raw.sample(n=min(800, len(df_hf_raw)), random_state=42)

    # Map 'system' to 'age' and 'user' to 'instruction'
    df_hf = df_hf[['system', 'user']].rename(columns={
        'age':'system',
        'user': 'instruction'
    })
    df_hf['intent'] = 'N/A'  # To be inferred by Gemini
    df_hf['source'] = 'hf'

    # --- C. Combine, Shuffle, and Final Truncation ---
    # Combine both dataframes
    df_combined = pd.concat([df_local, df_hf], ignore_index=True)

    # Shuffle the entire combined set and select top 1200
    df_final = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
    df_final = df_final.head(1200)

    print(f"✅ Created a mixed DataFrame with {len(df_final)} rows.")
    return df_final


# =============================================================================
# 3. Execution Pipeline
# =============================================================================
def run_sft_pipeline():
    all_inputs = get_combined_samples()

    # Resume from checkpoint if file exists
    start_index = 0
    if os.path.exists(CONFIG["output_path"]):
        with open(CONFIG["output_path"], 'r') as f:
            start_index = sum(1 for _ in f)

    print(f"🚀 Processing {len(all_inputs)} samples starting from index {start_index}...")

    for i in range(start_index, len(all_inputs)):
        item = all_inputs.iloc[i]

        # Build context-aware prompt
        target_age = item.get('age', 'N/A')
        original_intent = item.get('intent', 'N/A')

        prompt = f"""
        TARGET AGE: {target_age}
        ORIGINAL HINT: {original_intent}
        INSTRUCTION: {item['instruction']}

        TASK:
        1. Classify the 'instruction' into the closest INTENT category from the allowed list.
        2. Classify the 'age' into the closest AGE category from the allowed list.
        3. When generate response, use a friendly, educational tone and proper knowledge for the TARGET AGE.
        """

        success = False
        while not success:
            try:
                response = client.models.generate_content(
                    model=CONFIG["model_id"],
                    contents=prompt,
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": ChildrenSFTData,  # Pydantic model enforces Literal
                    }
                )

                # Parse and save structured result
                structured_data = response.parsed.dict()

                with open(CONFIG["output_path"], "a", encoding="utf-8") as f:
                    f.write(json.dumps(structured_data, ensure_ascii=False) + "\n")

                print(
                    f"[{i + 1}/{len(all_inputs)}] Intent: {structured_data['intent']} | Age: {structured_data['age']}")
                success = True
                time.sleep(CONFIG["delay_seconds"])

            except Exception as e:
                if "429" in str(e):
                    print("⚠️ Quota exhausted. Sleeping for 60s...")
                    time.sleep(60)
                else:
                    print(f"❌ Error at index {i}: {e}")
                    time.sleep(2)
                    success = True  # Skip to next row on persistent error


if __name__ == "__main__":
    run_sft_pipeline()