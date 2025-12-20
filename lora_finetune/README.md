# LoRA Fine-tuning Script (`train_lora.py`)

This script provides functionality for fine-tuning pre-trained language models using LoRA (Low-Rank Adaptation) with the MLX framework. It supports training, validation, testing, and text generation.

## Features

-   **LoRA Fine-tuning:** Efficiently fine-tune large language models by adapting only a small number of parameters.
-   **Flexible Model Loading:** Load models from local directories or directly from Hugging Face repositories.
-   **Dataset Management:** Utilizes `.jsonl` files for training, validation, and testing datasets, allowing for easy data preparation.
-   **Configurable Training:** Customize training parameters such as batch size, number of epochs, learning rate, and the specific layers to apply LoRA.
-   **Evaluation Metrics:** Evaluate the fine-tuned model's performance on validation and test sets.
-   **Adapter Weight Management:** Save and load LoRA adapter weights, enabling continuation of training or deployment of fine-tuned models.
-   **Text Generation:** Generate text based on a given prompt using the fine-tuned model.

## What is LoRA? (Detailed Explanation)

LoRA (Low-Rank Adaptation) is a highly efficient fine-tuning technique that adapts large pre-trained language models for specific tasks without modifying the original model's weights. The core idea is that the change in weights during fine-tuning (`ΔW`) can be approximated by a low-rank matrix, which has far fewer parameters than the original weights (`W`).

### How It Works

Instead of directly updating the entire, massive weight matrix `W` of a layer (e.g., an attention layer), LoRA introduces a small "adapter" module. This process involves:

1.  **Freezing Original Weights:** The weights of the pre-trained model are frozen and do not receive any updates during training. This preserves the vast knowledge already learned by the model.

2.  **Injecting Low-Rank Matrices:** For a target layer with a weight matrix `W`, LoRA adds two small, low-rank matrices: `A` (rank `r` x `d_in`) and `B` (`d_out` x rank `r`). These matrices are designed to learn the task-specific update, where `r` (the rank) is much smaller than the original dimensions. The update is represented by their product, `BA`.

3.  **Training the Adapter:** Only the parameters of matrices `A` and `B` are trained. Since `r` is small, the number of trainable parameters is drastically reduced. For example, for a layer with 4096 dimensions, a full fine-tuning would update `4096 * 4096 ≈ 16.7 million` parameters. With LoRA and a rank `r=8`, we only train `(4096 * 8) + (8 * 4096) ≈ 65,536` parameters.

4.  **Combining Outputs:** During inference, the output of the original layer (`Wx`) and the LoRA adapter (`BAx`) are added together: `(W + BA)x`. This means the adapter can be merged with the original weights after training, introducing no additional latency.

### Benefits of LoRA

-   **Parameter Efficiency:** Reduces the number of trainable parameters by up to 10,000 times, making training feasible on consumer hardware.
-   **Storage Savings:** Instead of saving a full model for each task (e.g., 7B parameters ≈ 14GB), you only need to save the small adapter weights (a few megabytes).
-   **Fast Task Switching:** A single pre-trained model can be shared across many tasks, and switching tasks is as simple as loading a different, small adapter file.
-   **No Inference Latency:** Because the adapter weights can be merged with the main model weights, there is no performance overhead during inference compared to a fully fine-tuned model.

### (Korean) LoRA란 무엇인가요? (상세 설명)

LoRA(Low-Rank Adaptation)는 사전 훈련된 대규모 언어 모델의 원본 가중치를 수정하지 않고 특정 작업에 맞게 미세 조정하는 매우 효율적인 파인튜닝 기술입니다. 핵심 아이디어는 파인튜닝 중 가중치의 변화량(`ΔW`)이 원본 가중치(`W`)보다 훨씬 적은 수의 파라미터를 가진 저차원 행렬(low-rank matrix)로 근사될 수 있다는 것입니다.

### 작동 방식

LoRA는 레이어의 거대한 가중치 행렬 `W` 전체를 직접 업데이트하는 대신, 작은 "어댑터" 모듈을 도입합니다. 이 과정은 다음과 같습니다.

1.  **원본 가중치 동결:** 사전 훈련된 모델의 가중치는 동결되어 훈련 중에 업데이트되지 않습니다. 이를 통해 모델이 이미 학습한 방대한 지식을 보존합니다.

2.  **저차원 행렬 주입:** 가중치 행렬 `W`가 있는 대상 레이어에 대해 LoRA는 두 개의 작고 저차원인 행렬 `A`(랭크 `r` x `d_in`)와 `B`(`d_out` x 랭크 `r`)를 추가합니다. 이 행렬들은 작업별 업데이트를 학습하도록 설계되었으며, 여기서 `r`(랭크)은 원래 차원보다 훨씬 작습니다. 업데이트는 이 두 행렬의 곱 `BA`로 표현됩니다.

3.  **어댑터 훈련:** 행렬 `A`와 `B`의 파라미터만 훈련됩니다. `r`이 작기 때문에 훈련 가능한 파라미터의 수가 크게 줄어듭니다. 예를 들어, 4096 차원의 레이어에서 전체 파인튜닝은 `4096 * 4096 ≈ 1,670만` 개의 파라미터를 업데이트합니다. LoRA와 랭크 `r=8`을 사용하면 `(4096 * 8) + (8 * 4096) ≈ 65,536` 개의 파라미터만 훈련합니다.

4.  **출력 결합:** 추론 시 원본 레이어의 출력(`Wx`)과 LoRA 어댑터의 출력(`BAx`)이 함께 더해집니다: `(W + BA)x`. 이는 훈련 후 어댑터를 원본 가중치와 병합할 수 있어 추가적인 지연 시간(latency)이 발생하지 않음을 의미합니다.

### LoRA의 장점

-   **파라미터 효율성:** 훈련 가능한 파라미터 수를 최대 10,000배까지 줄여 일반 소비자용 하드웨어에서도 훈련이 가능하게 합니다.
-   **저장 공간 절약:** 각 작업에 대해 전체 모델(예: 7B 파라미터 ≈ 14GB)을 저장하는 대신, 작은 어댑터 가중치(수 메가바이트)만 저장하면 됩니다.
-   **빠른 작업 전환:** 단일 사전 훈련 모델을 여러 작업에서 공유할 수 있으며, 작고 다른 어댑터 파일을 로드하는 것만으로 간단하게 작업을 전환할 수 있습니다.
-   **추론 지연 없음:** 어댑터 가중치를 주 모델 가중치와 병합할 수 있으므로, 완전히 파인튜닝된 모델에 비해 추론 시 성능 오버헤드가 없습니다.

## Dataset

The script is designed to work with instruction-based datasets in JSON Lines (`.jsonl`) format. The default data directory (`lora_finetune/data`) contains a Korean question-and-answer dataset. Each line in the `.jsonl` file is a JSON object with a `"text"` key. The value of this key is a string containing a dialogue between two speakers, "### 질문:" (Question) and "### 답변:" (Answer).

**Example from `train.jsonl`:**

```json
{
	"text": "### 질문:\n**Phi:** 하지만 동물에게 죽음에 대해 이야기하는 것은 그들을 혼란스럽게 하고 불안하게 만들 수 있습니다. 그들은 죽음 이 무엇인지 이해하지 못하기 때문에 우리가 죽음에 대해 이야기하면 그들은 우리가 무엇을 말하는지 모르고 불안해질 수 있습니다.\n\n### 답변:\n**Epsilon:** 동물에게 죽음에 대해 이야기할 때는 그들이 이해할 수 있는 방식으로 이야기하는 것이 중요합니다. 우리는 그들에게 죽음이 모든 생명체의 자연스러운 과정이라고 말하고, 그들이 죽어가고 있다는 사실에 대해 두려워할 필요가 없다고 말할 수 있습니다. 우리는 또한 그들에게 그들이 죽은 후에도 항상 그들의 마음속에 살아 있을 것이라고 말할 수 있습니다."
}
```

### (Korean) 데이터셋

이 스크립트는 JSON Lines(`.jsonl`) 형식의 지침 기반 데이터셋과 함께 작동하도록 설계되었습니다. 기본 데이터 디렉토리(`lora_finetune/data`)에는 한국어 질의응답 데이터셋이 포함되어 있습니다. `.jsonl` 파일의 각 줄은 `"text"` 키를 가진 JSON 객체입니다. 이 키의 값은 "### 질문:"과 "### 답변:"이라는 두 명의 화자 간의 대화를 포함하는 문자열입니다.

## Dataset Source

The `train.jsonl`, `valid.jsonl`, and `test.jsonl` files used for LoRA training are generated by the `prepare_lora_data.py` script. This script reads a local dataset named `korean_textbooks_qa` which is expected to be located in the `./data/korean_textbooks_qa` directory.

The project does not contain a script to download the `korean_textbooks_qa` dataset, so its original source is not specified. It is assumed that the user has already downloaded this dataset and placed it in the appropriate directory.

### (Korean) 데이터셋 출처

LoRA 훈련에 사용되는 `train.jsonl`, `valid.jsonl`, `test.jsonl` 파일은 `prepare_lora_data.py` 스크립트에 의해 생성됩니다. 이 스크립트는 `./data/korean_textbooks_qa` 디렉토리에 위치할 것으로 예상되는 `korean_textbooks_qa`라는 로컬 데이터셋을 읽어들입니다.

이 프로젝트에는 `korean_textbooks_qa` 데이터셋을 다운로드하는 스크립트가 포함되어 있지 않으므로, 원본 출처는 명시되어 있지 않습니다. 사용자가 이 데이터셋을 미리 다운로드하여 적절한 디렉토리에 배치한 것으로 간주됩니다.

## LoRA Parameter Configuration

In this script, LoRA is applied to the query and value projection layers (`q_proj` and `v_proj`) of the self-attention mechanism in the transformer layers. If the model includes a Mixture-of-Experts (MoE) block, the gate layer (`gate`) is also fine-tuned with LoRA.

The `--lora-layers` argument specifies how many of the final layers of the model will be adapted. For example, if you set `--lora-layers 16`, the last 16 layers will have their attention and/or MoE layers adapted with LoRA.

**Relevant Code Snippet:**

```python
# Freeze all layers other than LORA linears
model.freeze()
for l in model.model.layers[len(model.model.layers) - args.lora_layers :]:
    l.self_attn.q_proj = LoRALinear.from_base(l.self_attn.q_proj)
    l.self_attn.v_proj = LoRALinear.from_base(l.self_attn.v_proj)
    if hasattr(l, "block_sparse_moe"):
        l.block_sparse_moe.gate = LoRALinear.from_base(l.block_sparse_moe.gate)
```

### (Korean) LoRA 파라미터 설정

이 스크립트에서 LoRA는 트랜스포머 레이어의 셀프 어텐션 메커니즘에서 쿼리 및 값 프로젝션 레이어(`q_proj` 및 `v_proj`)에 적용됩니다. 모델에 MoE(Mixture-of-Experts) 블록이 포함된 경우 게이트 레이어(`gate`)도 LoRA로 파인튜닝됩니다.

`--lora-layers` 인수는 모델의 마지막 몇 개 레이어를 적응시킬지를 지정합니다. 예를 들어, `--lora-layers 16`으로 설정하면 마지막 16개 레이어의 어텐션 및/또는 MoE 레이어가 LoRA로 적응됩니다.

## Usage

To use this script, you will typically follow these steps:

1.  **Prepare your data:** Ensure you have `train.jsonl`, `valid.jsonl`, and optionally `test.jsonl` files in a directory (default: `lora_finetune/data`). Each line in these files should be a JSON object containing a `text` key with your training data.
2.  **Run the script:** Execute the script with appropriate arguments for your task.

### Command-line Arguments

Here's a breakdown of the available command-line arguments:

*   `--model`: Path to the local model directory or Hugging Face repository (e.g., `GuminiResearch/Gumini-1B-Base`).
*   `--max-tokens` (`-m`): Maximum number of tokens to generate (default: `100`).
*   `--temp`: Sampling temperature for generation (default: `0.8`).
*   `--prompt` (`-p`): The prompt for text generation. If provided, the script will generate text after training/loading adapters.
*   `--train`: Flag to enable training mode.
*   `--add-eos-token`: Enable `add_eos_token` for the tokenizer (default: `1`).
*   `--data`: Directory containing `train.jsonl`, `valid.jsonl`, and `test.jsonl` files (default: `lora_finetune/data`).
*   `--lora-layers`: Number of layers to fine-tune using LoRA (default: `16`).
*   `--batch-size`: Mini-batch size for training (default: `10`).
*   `--epochs`: Number of epochs to train for. Overrides `--iters` if both are specified (default: `1`).
*   `--iters`: Number of iterations to train for. If specified, it overrides `--epochs`.
*   `--start-iter`: Iteration to start training from (useful for resuming, default: `0`).
*   `--val-batches`: Number of validation batches, `-1` uses the entire validation set (default: `25`).
*   `--learning-rate`: Adam optimizer learning rate (default: `2e-5`).
*   `--steps-per-report`: Number of training steps between loss reporting (default: `10`).
*   `--steps-per-eval`: Number of training steps between validations (default: `200`).
*   `--resume-adapter-file`: Path to load adapter weights to resume training.
*   `--adapter-file`: Path to save/load trained adapter weights (default: `lora_finetune/lora_adapters.npz`).
*   `--save-every`: Save the model every N iterations (default: `100`).
*   `--test`: Flag to evaluate on the test set after training or loading adapters.
*   `--test-batches`: Number of test set batches, `-1` uses the entire test set (default: `500`).
*   `--seed`: The PRNG seed (default: `0`).

### Example Training Run

```bash
python lora_finetune/train_lora.py \
    --model "mlx-community/Phi-3-mini-4k-instruct-8bit" \
    --train \
    --data "lora_finetune/data" \
    --epochs 3 \
    --batch-size 4 \
    --learning-rate 1e-5 \
    --lora-layers 8 \
    --adapter-file "lora_finetune/my_model_adapters.npz" \
    --test
```

### Example Generation Run (after training)

```bash
python lora_finetune/train_lora.py \
    --model "mlx-community/Phi-3-mini-4k-instruct-8bit" \
    --adapter-file "lora_finetune/my_model_adapters.npz" \
    --prompt "What is the capital of France?" \
    --max-tokens 50
```

## Data Format

Your training, validation, and test data files (e.g., `train.jsonl`) should be in JSON Lines format, where each line is a JSON object with at least a `"text"` key:

```json
{"text": "This is a training example."}
{"text": "Another example for fine-tuning."}
{"text": "The quick brown fox jumps over the lazy dog."}
```