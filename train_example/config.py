import os

# Model Type Selection
MODEL_TYPE = "gemma3" # "gemini_nano" or "gemma3"

# Model Args - gemini_nano
GEMINI_NANO_ARGS = {
    "d_model": 2048,
    "num_heads": 16,
    "num_layers": 16,
    "ffn_dim_multiplier": 4,
    "rope_theta": 10000.0,
}

# Model Args - gemma3
GEMMA3_ARGS = {
    "model_type": "gemma3",
    "hidden_size": 2048,
    "num_hidden_layers": 20,
    "intermediate_size": 6144,
    "num_attention_heads": 4,
    "head_dim": 512,
    "rms_norm_eps": 1e-6,
    "num_key_value_heads": 1,
    "rope_traditional": False,
}

# Training Args
batch_size = 1 # Reduced batch size for larger model
learning_rate = 1e-4
num_epochs = 1
seq_len = 256

# Data Args
DATA_CACHE_DIR = "data"
korquad_data_dir = os.path.join(DATA_CACHE_DIR, "korquad_v2_1")

# Checkpoint Args
checkpoint_dir = 'model'