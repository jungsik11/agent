import re
from datasets import load_dataset, Dataset
import os

def prepare_dataset():
    """
    Downloads the maywell/korean_textbooks dataset and processes it.
    """
    dataset_name = "maywell/korean_textbooks"
    config_name = "normal_instructions"
    save_path = "./data/korean_textbooks_qa" # New path for the processed dataset

    if os.path.exists(save_path):
        print(f"Processed dataset already exists at {save_path}. Skipping processing.")
        return

    print(f"Loading dataset: {dataset_name} ({config_name})...\n") # Added newline for better readability
    try:
        dataset = load_dataset(dataset_name, config_name, split="train")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        print("Please ensure you have enough disk space and internet connection.")
        return

    processed_samples = []

    print("Processing dataset samples...")
    for i, sample in enumerate(dataset):
        text = sample.get("text", "")
        if not text:
            continue

        # 1. Extract the initial question
        initial_question_match = re.match(r"\"(.*?)\"에 대한 토론 내용:", text, re.DOTALL)
        if not initial_question_match:
            continue # Skip if the initial question format is not found
        
        initial_question = initial_question_match.group(1).strip()
        dialogue_text = text[initial_question_match.end():].strip()

        # 2. Parse the dialogue into individual turns
        # Split by speaker patterns, keeping the speaker names in the result
        # The regex captures the speaker name (e.g., Phi, Epsilon)
        turns_raw = re.split(r"\n\n\*\*(.*?):\*\*", dialogue_text)
        
        # turns_raw will look like ['', 'Phi', 'content of Phi', 'Epsilon', 'content of Epsilon', ...]
        # The first element is usually empty or preamble before the first speaker
        if len(turns_raw) < 3: # Need at least one speaker and their content
            continue

        # Create a list of (speaker, content) tuples
        parsed_turns = []
        for j in range(1, len(turns_raw), 2):
            speaker = turns_raw[j].strip()
            content = turns_raw[j+1].strip() if j+1 < len(turns_raw) else ""
            parsed_turns.append((speaker, content))

        # 3. Generate Q&A pairs
        # First pair: Initial question + first Phi's answer
        first_phi_turn = next(((s, c) for s, c in parsed_turns if s == "Phi"), None)
        if first_phi_turn:
            processed_samples.append({
                "question": initial_question,
                "answer": f"**{first_phi_turn[0]}:** {first_phi_turn[1]}"
            })

        # Subsequent pairs: Phi's turn as question, followed by Epsilon's turn as answer
        for k in range(len(parsed_turns) - 1):
            current_turn = parsed_turns[k]
            next_turn = parsed_turns[k+1]

            if current_turn[0] == "Phi" and next_turn[0] == "Epsilon":
                processed_samples.append({
                    "question": f"**{current_turn[0]}:** {current_turn[1]}",
                    "answer": f"**{next_turn[0]}:** {next_turn[1]}"
                })
        
        if i % 1000 == 0:
            print(f"Processed {i} samples...")

    # Create a new Dataset object from the processed samples
    processed_dataset = Dataset.from_list(processed_samples)
    processed_dataset.save_to_disk(save_path)
    print(f"Processed dataset saved to {save_path}. Total samples: {len(processed_dataset)}")

if __name__ == "__main__":
    prepare_dataset()
