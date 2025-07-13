from transformers import AutoTokenizer, AutoModelForCausalLM
import os

model_name = "beomi/gemma-ko-2b"
# 모델 파일을 저장할 로컬 디렉토리입니다.
# 이 디렉토리에 모델의 가중치와 토크나이저 파일이 저장됩니다.
local_dir = "./gemma_2B_kor_local"

# 디렉토리가 없으면 생성합니다.
os.makedirs(local_dir, exist_ok=True)

print(f"Downloading tokenizer for {model_name} to {local_dir}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(local_dir)
print("Tokenizer download complete.")

print(f"Downloading model for {model_name} to {local_dir}...")
model = AutoModelForCausalLM.from_pretrained(model_name)
model.save_pretrained(local_dir)
print("Model download complete.")

print(f"Model and tokenizer saved to: {os.path.abspath(local_dir)}")
print("\n--- 다음 단계 ---")
print("1. 다운로드된 모델을 GGUF 형식으로 변환해야 합니다. 이는 일반적으로 llama.cpp 프로젝트의 'convert.py' 스크립트를 사용합니다.")
print("   예시: python llama.cpp/convert.py --outfile llama3.2_kor_local.gguf --model_type llama <다운로드된 모델 디렉토리>")
print("2. GGUF 파일이 준비되면, ollama에서 사용할 Modelfile을 생성해야 합니다.")
print("   Modelfile 예시:")
print("   ```")
print("   FROM ./llama3.2_kor_local.gguf")
print("   PARAMETER stop \"<|eot_id|>\"")
print("   PARAMETER stop \"<|end_of_text|>\"")
print("   # 필요에 따라 다른 파라미터 추가 (예: temperature, top_k, top_p)")
print("   ```")
print("3. Modelfile을 저장한 후, 다음 ollama 명령어로 모델을 생성합니다:")
print("   ollama create llama3.2_kor_local -f ./Modelfile")
print("4. 모델이 성공적으로 생성되면, ollama run llama3.2_kor_local 명령어로 모델을 실행할 수 있습니다.")