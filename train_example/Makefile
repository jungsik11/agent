.PHONY: all convert-gguf create-modelfile deploy-ollama clean

MODEL_NAME := llama3_8b_kor_local
HF_MODEL_ID := VIRNECT/llama-3-Korean-8B-r-v1
HF_MODEL_DIR := ./$(MODEL_NAME)
GGUF_MODEL_PATH := ./$(MODEL_NAME).gguf
MODELFILE_PATH := ./Modelfile_$(MODEL_NAME)

# --- 사용자 설정 ---
LLAMA_CPP_DIR := llama.cpp # llama.cpp 경로가 다르면 이 경로를 조정하세요

all: deploy-ollama

convert-gguf: $(GGUF_MODEL_PATH)

$(GGUF_MODEL_PATH):
	@echo "--- Hugging Face 모델을 GGUF 형식으로 변환 중 ---"
	@echo "이 단계는 llama.cpp와 해당 Python 의존성을 필요로 합니다."
	@echo "llama.cpp가 클론되고 빌드되었으며, Python 환경이 설정되었는지 확인하세요."
	cd $(LLAMA_CPP_DIR) && \
	pip3 install transformers && \
	pip3 install torch && \
	python3 convert_hf_to_gguf.py --outfile ../$(GGUF_MODEL_PATH) ../$(HF_MODEL_DIR)

create-modelfile: $(MODELFILE_PATH)

$(MODELFILE_PATH):
	@echo "--- Ollama Modelfile 생성 중 ---"
	@echo "FROM $(GGUF_MODEL_PATH)" > $@
	@echo "PARAMETER stop \"<|eot_id|>\"" >> $@
	@echo "PARAMETER stop \"<|end_of_text|>\"" >> $@
	@echo "# 필요에 따라 다른 파라미터를 여기에 추가하세요 (예: PARAMETER temperature 0.7)" >> $@
	@echo "Modelfile이 $@에 생성되었습니다."

deploy-ollama: convert-gguf create-modelfile
	@echo "--- Ollama에 모델 배포 중 ---"
	@echo "이 단계는 Ollama가 설치되어 있고 실행 중이어야 합니다."
	@echo "실행 중: ollama create $(MODEL_NAME) -f $(MODELFILE_PATH)"
	ollama create $(MODEL_NAME) -f $(MODELFILE_PATH)
	@echo "배포 시도 완료. 이제 다음을 시도할 수 있습니다: ollama run $(MODEL_NAME)"

clean:
	@echo "--- 생성된 파일 정리 중 ---"
	rm -f $(GGUF_MODEL_PATH) $(MODELFILE_PATH)
	rm -rf $(HF_MODEL_DIR) # 선택 사항: Hugging Face 모델 디렉토리도 제거
	@echo "정리 완료."