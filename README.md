# FM_KD 학습 방식 상세 문서

이 문서는 현재 저장소의 **실제 코드 기준 학습 방식**을 정리한 문서입니다.  
기준 코드:

- `script/train.bash`
- `train_fsdp.py`
- `train.py` (`compute_losses_skeleton` 포함)
- `dataset/hidden_state_dataset.py`
- `student_model/model.py`
- `dataset/collect_data.py`
- `init_student_from_teacher.py`

---

## 1. 현재 학습의 핵심 개념

이 프로젝트는 Teacher LLM의 은닉상태(hidden states)를 이용해 Student 모델이 **조건부 연속 흐름(Flow Matching)** 을 학습하도록 구성되어 있습니다.

- 조건부 입력(`X`): 시스템/유저 프롬프트 구간
- 생성 대상(`Y`): 어시스턴트 응답 구간
- Teacher에서 저장한 여러 레이어의 `Y` hidden을 앵커로 사용
- Student는 시간 `t`에서의 `Y` latent(`z_t`)를 받아 **velocity**를 예측
- 손실은 기본적으로:
1. `L_fm` (flow-matching MSE)
2. `L_anchor` (구간 끝점 일치 MSE)

옵션으로 `L_ce`, `L_kl`이 있지만, 현재 설정/데이터 상태에서는 기본적으로 비활성입니다.

---

## 2. 전체 파이프라인

학습은 아래 3단계로 구성됩니다.

1. 데이터 수집 (`dataset/collect_data.py`)
2. 학생 초기화 체크포인트 생성 (`init_student_from_teacher.py`)
3. FSDP 학습 (`script/train.bash` -> `train_fsdp.py`)

---

## 3. 데이터 수집 단계

### 3.1 소스 데이터와 프롬프트 포맷

- 모델: `meta-llama/Meta-Llama-3.1-8B-Instruct`
- 데이터셋: `meta-math/MetaMathQA`
- 각 샘플은 system/user/assistant 템플릿으로 합쳐 토크나이즈됨

### 3.2 길이 기반 샘플 선택

`collect_data.py`는 전체 데이터셋에 대해 토큰 길이를 계산한 뒤:

- 최대 길이(`MAX_LENGTH`) 이하 샘플만 후보로 사용
- 길이 버킷 균형 샘플링
- 목표 샘플 수(`TARGET_SAMPLE_COUNT`)만 선택

현재 메타(`data/collected_data/selection_meta.json`) 기준:

- `target_sample_count = 50000`
- `max_len_used = 1024`
- `selection_strategy = bucketed_by_token_length`

### 3.3 저장 포맷

샤드별 safetensors 파일(`data/collected_data/shard_*/chunk_*.safetensors`)에 샘플 단위로 저장:

- `sample_{id}_hidden`: `[L_total, seq_len, D]`
- `sample_{id}_qlen`: `[1]`
- `sample_{id}_input_ids`: `[seq_len]`

코드상 teacher logits 저장 기능은 있으나, 현재 데이터 파일을 스캔한 결과 logits key는 존재하지 않습니다.

---

## 4. Student 초기화 단계

`init_student_from_teacher.py`가 수행:

1. Teacher config 기반으로 Student Llama 구성 (레이어 수 축소)
2. Teacher 가중치를 Student에 복사
3. 결과를 `student_init_ckpt`로 저장

기본 레이어 매핑(32 -> 16, `odd`)은:

- `[1, 3, 5, ..., 31]` teacher layer를 student layer에 복사

학습 스크립트(`train_fsdp.py`)는 이 초기화 체크포인트를 **필수**로 요구합니다.

---

## 5. 학습 데이터 로더 상세

`FlowHiddenStateDataset` / `FlowBatchCollator`가 배치를 구성합니다.

### 5.1 샘플 분해

샘플의 전체 시퀀스를 `qlen`으로 분리:

- `X`: `[0:qlen)` 구간
- `Y`: `[qlen:seq_len)` 구간

또한 fixed-length 정책으로 `fixed_total_length`(기본 1024)에 맞춰 패딩합니다.

### 5.2 핵심 텐서

배치에서 학습에 직접 쓰이는 핵심 텐서:

- `x_mask`: `[B, x_len]`
- `y_mask`: `[B, y_len]`
- `teacher_anchors`: `[B, M, y_len, D]`
- `teacher_x_layers`: `[B, M+1, x_len, D]`
- `times`: `[M+1]` (기본 균등분할 `linspace(0,1,M+1)`)

여기서 `M = num_intervals`이며, 현재 설정은 `M=12`입니다.

---

## 6. Student 모델 구조 (`student_model/model.py`)

### 6.1 Backbone

- `LlamaForCausalLM` backbone 사용
- hidden size는 teacher와 동일 (`D=4096`)
- student layer 수는 기본 16

### 6.2 Attention 구조

비대칭 bidirectional mask를 사용:

- `X -> Y` attention 금지
- `Y -> X`, `Y -> Y`는 허용
- 패딩 키는 항상 mask

### 6.3 시간 조건 주입

- `t`를 sinusoidal + MLP로 임베딩
- 각 레이어마다 AdaLN 방식 shift/scale/gate 생성
- 중요한 설계: 레이어 내부에서 **X branch는 고정**, Y branch만 modulation/gating 적용

### 6.4 출력

- 최종 `h_y`에 `velocity_head` 적용 -> `v_y`
- 즉 Student는 토큰 확률이 아닌 **Y hidden의 시간 미분(velocity)** 를 직접 예측

---

## 7. 손실 함수 상세 (`compute_losses_skeleton` in `train.py`)

학습의 핵심입니다.

### 7.1 구간/시간 샘플링

배치별로:

- `k ~ Uniform({0,...,M-1})`
- `s ~ Uniform([0,1])`
- `t = (1-s)t0 + s t1`, where `t0 = times[k]`, `t1 = times[k+1]`

### 7.2 구간 끝점 `(a,b)` 정의

- `k=0`: `a=z0`, `b=teacher_anchor[0]`
- `k>0`: `a=teacher_anchor[k-1]`, `b=teacher_anchor[k]`

여기서 `z0`는 learnable mask embedding + 가우시안 노이즈(`z0_noise_std`)로 초기화됩니다.

### 7.3 중간 상태 구성

- `z_t = (1-s)a + s b`
- `x_t = teacher_x_layers[k]`

Student 입력:

- `z_y = z_t`
- `t`
- `x_states = x_t`
- `x_mask`, `y_mask`

### 7.4 목표 velocity와 FM 손실

- `u* = (b-a)/(t1-t0)`
- `L_fm = masked_mse(v_theta, u*)`

마스킹은 `y_mask`(+ 필요 시 `y_len_real`) 기준으로 PAD를 제외합니다.

### 7.5 Anchor 손실

Euler 1-step으로 구간 끝점 복원:

- `b_hat = z_t + (t1-t) * v_theta`
- `L_anchor = masked_mse(b_hat, b)`

### 7.6 CE/KL (옵션)

- `k == M-1`인 샘플에 대해서만 계산
- `lambda_ce`, `lambda_kl`이 0이 아니면 활성
- KL은 dense teacher logits가 필요

현재 실행 스크립트는 `--lambda-kl 0.0`이며, 현재 데이터에도 teacher logits가 없어 KL은 사실상 비활성입니다.

### 7.7 총손실

`L_total = lambda_fm * L_fm + lambda_anchor * L_anchor + lambda_ce * L_ce + lambda_kl * L_kl`

---

## 8. FSDP 학습 루프 (`train_fsdp.py`)

### 8.1 분산 실행

`torchrun` 기반 다중 GPU:

- `RANK`, `WORLD_SIZE`, `LOCAL_RANK` 환경변수 사용
- `DistributedSampler` + `drop_last=True`

### 8.2 FSDP 설정

현재 코드 기본:

- `ShardingStrategy.FULL_SHARD`
- `BackwardPrefetch.BACKWARD_PRE`
- `limit_all_gathers=True`
- `use_orig_params=False`
- BF16 mixed precision(옵션)
- activation checkpointing(옵션)
- auto-wrap 기준: `LlamaDecoderLayer`

### 8.3 학습 루프

각 step:

1. `losses = fsdp_model(batch)`
2. `loss_total / grad_accum_steps` backward
3. grad clip (옵션)
4. optimizer step / zero_grad
5. 주기적 로그(all-reduce 평균)
6. 주기적 체크포인트 저장

### 8.4 체크포인트

FSDP full-state 방식으로 rank0에 저장:

- `step`
- `model` (full state dict)
- `optimizer`
- `cfg`

파일명: `{output_dir}/{run_name}_step{step}.pt`

---

## 9. 현재 `script/train.bash` 기준 실행값

학습 엔트리:

```bash
torchrun --nnodes=1 --nproc_per_node=2 train_fsdp.py ...
```

명시된 인자:

- `--student-init-ckpt /home/minhae/diffusion/FM_KD/checkpoints/student_init.pt`
- `--data-root /home/minhae/diffusion/FM_KD/data/collected_data`
- `--lambda-kl 0.0`
- `--batch-size 1` (per-rank)
- `--num-workers 2`
- `--bf16`
- `--activation-checkpointing`
- `--output-dir /home/minhae/diffusion/FM_KD/checkpoints_cfd_fsdp`
- `--run-name cfd_fsdp_2gpu`

명시 안 된 값은 `train_fsdp.py` 기본값 사용:

- `max_steps=2000`
- `log_every=20`
- `save_every=200`
- `lr=2e-4`
- `weight_decay=0.01`
- `betas=(0.9,0.95)`
- `grad_clip=1.0`
- `num_intervals=12`
- `lambda_fm=1.0`
- `lambda_anchor=1.0`
- `lambda_ce=0.0`

실효 전역 배치크기:

- `global_batch = batch_size(per-rank) * world_size = 1 * 2 = 2`

---

## 10. 단일 GPU 경로와의 관계

`train.py`는 단일 프로세스 학습 경로이며, 손실 정의는 동일한 `compute_losses_skeleton`를 사용합니다.  
현재 메인 실행 스크립트는 `train_fsdp.py`이므로, 실제 운영 학습 경로는 FSDP 버전입니다.

---

## 11. 현재 상태에서의 중요한 제약

1. KL distillation 비활성  
   현재 데이터에 teacher logits가 없어 `lambda_kl > 0` 실효를 기대하기 어렵습니다.

2. `num_intervals` 정합 필수  
   `cfg.num_intervals`와 데이터 앵커 개수(`num_anchor_layers`)가 다르면 즉시 에러가 발생합니다.

3. fixed length 전제  
   데이터셋이 `fixed_total_length` 기반으로 `Y`를 패딩하므로, `y_mask` 처리가 손실 안정성에 중요합니다.

---

## 12. 재현 실행 순서 (권장)

1. 데이터 준비

```bash
python dataset/collect_data.py
```

2. Student 초기화 ckpt 생성

```bash
python init_student_from_teacher.py \
  --output /home/minhae/diffusion/FM_KD/checkpoints/student_init.pt \
  --device cuda \
  --teacher-model-id meta-llama/Meta-Llama-3.1-8B-Instruct \
  --student-num-layers 16 \
  --layer-copy-mode odd
```

3. FSDP 학습 실행

```bash
bash script/train.bash
```

또는 직접:

```bash
torchrun --nnodes=1 --nproc_per_node=2 \
  --master_addr=127.0.0.1 --master_port=29500 \
  train_fsdp.py \
  --student-init-ckpt /home/minhae/diffusion/FM_KD/checkpoints/student_init.pt \
  --data-root /home/minhae/diffusion/FM_KD/data/collected_data \
  --lambda-kl 0.0 \
  --batch-size 1 \
  --num-workers 2 \
  --bf16 \
  --activation-checkpointing \
  --output-dir /home/minhae/diffusion/FM_KD/checkpoints_cfd_fsdp \
  --run-name cfd_fsdp_2gpu
```

---

필요하면 다음 단계로, 이 문서에 `손실별 튜닝 가이드(예: lambda, z0_noise_std, interval 스케줄링)`나 `메모리 최적화 체크리스트(FSDP/activation checkpointing/optimizer state)`를 추가할 수 있습니다.
