# mlx_lm을 이용한 LLM 사전 훈련 예제 (최신 버전)

이 프로젝트는 `mlx-lm` 라이브러리의 최신 버전을 사용하여 소규모 언어 모델을 처음부터 사전 훈련(pre-training)하는 방법을 보여줍니다.

## 1. 프로젝트 설정 및 의존성 설치

먼저, 가상 환경을 만들고 필요한 라이브러리를 설치합니다.

```bash
# 1. 이 디렉토리로 이동합니다.
cd mlx_pretrain_example

# 2. 파이썬 가상 환경을 생성하고 활성화합니다.
python3 -m venv venv
source venv/bin/activate

# 3. uv 또는 pip을 사용하여 mlx-lm을 설치합니다.
# (uv 사용 권장)
uv add mlx-lm
# 또는
pip install mlx-lm
```

## 2. 데이터 준비

훈련 데이터는 `data/` 디렉토리 안에 `.jsonl` 형식으로 준비해야 합니다. 각 줄은 `{"text": "..."}` 형태의 JSON 객체여야 합니다.

- `train.jsonl`: 훈련용 데이터
- `valid.jsonl`: 검증용 데이터 (없을 경우 빈 파일)
- `test.jsonl`: 테스트용 데이터 (없을 경우 빈 파일)

이 예제에서는 `train.jsonl`만 사용합니다.

## 3. 모델 사전 훈련

이제 모든 준비가 끝났습니다. 아래 명령어를 실행하여 모델 사전 훈련을 시작합니다. `mlx-lm`은 `config.yaml` 파일을 읽어 모든 설정을 가져옵니다.

최신 버전에서는 `lora` 명령어를 사용하며, `--fine-tune-type full` 옵션을 통해 사전 훈련을 수행합니다.

```bash
python -m mlx_lm.lora --config config.yaml
```

훈련이 진행되면서 로그가 터미널에 출력되고, `config.yaml`에 설정된 `adapter_path`(`models/pretrained_model/`)에 훈련된 모델이 저장됩니다.

## 4. 훈련된 모델로 텍스트 생성하기

훈련이 완료되면, `generate` 명령어를 사용하여 훈련된 모델로 텍스트를 생성해볼 수 있습니다.

```bash
python -m mlx_lm.generate \
  --model models/pretrained_model \
  --prompt "인공지능은" \
  --max-tokens 100
```

`--prompt` 인자에 원하는 시작 문구를 넣어 테스트해 보세요.
