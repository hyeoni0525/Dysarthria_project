# Dysarthria_project

구음장애인의 의사소통 지원을 위한 발화 재구성 시스템용 언어 모델 선정 평가

Whisper가 변환한 STT 문장을 자연스럽게 재구성할 한국어 생성 모델을 비교한 연구입니다. 원문의 의미와 뉘앙스를 보존하는 것을 목표로 Ko-BART와 Ko-T5를 정량 지표와 정성 분석으로 평가했습니다.

| 항목 | 내용 |
| --- | --- |
| 연구 시점 | 2025.05 |
| 개인 역할 | Ko-T5 파인튜닝 담당 |
| 공동 수행 | 음성 전처리, CER와 WER 기반 데이터 선별, 비교 실험 설계와 평가 |
| 연구 결과 | Ko-BART를 후속 시스템용 모델로 제안 |
| 학술 성과 | 한국인공지능융합기술학회 2025 춘계학술대회 우수상 |

## 연구 과정

```text
AI Hub 구음장애 음성
  → VAD와 Whisper 전처리
  → STT 문장과 레이블 비교 및 선별
  → Ko-BART, Ko-T5 문장 재구성
  → ROUGE-L, BLEURT, BERTScore와 정성 분석
  → 후속 시스템용 모델 선정
```

데이터는 [AI Hub 구음장애 음성 인식 데이터셋](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=608)을 활용했습니다.

## 실험 결과와 모델 선정

연구에서 보고한 모델 비교 결과입니다.

| 지표 | Ko-BART | Ko-T5 |
| --- | ---: | ---: |
| ROUGE-L | 0.855 | 0.886 |
| BLEURT | 0.787 | 0.765 |
| BERTScore | 0.9358 | 0.963 |

Ko-T5는 ROUGE-L과 BERTScore에서 높았고 Ko-BART는 BLEURT에서 높았습니다. 연구진은 지표별 차이와 출력 문장의 자연스러움, 전반적인 품질에 대한 정성 분석을 함께 고려해 Ko-BART를 후속 시스템용 모델로 제안했습니다.

## 공개 코드

소스 파일은 [Dysarthria_project/](Dysarthria_project/)에 있습니다.

| 파일 | 내용 |
| --- | --- |
| [main.py](Dysarthria_project/main.py) | 학습과 검증 CSV 로드, 문장 분할, Ko-BART 학습 호출 |
| [dataset.py](Dysarthria_project/dataset.py) | source와 target 토큰화, label padding 처리 |
| [model.py](Dysarthria_project/model.py) | Ko-BART 모델과 토크나이저 로드 |
| [T5_model.py](Dysarthria_project/T5_model.py) | Ko-T5 모델과 토크나이저 로드 |
| [train.py](Dysarthria_project/train.py) | Seq2SeqTrainer 학습과 체크포인트 저장 |
| [util.py](Dysarthria_project/util.py) | 생성 결과의 ROUGE 계산 |
| [평가 예시](Dysarthria_project/BertScore%2Brouge%2Bbleurt.py) | CSV 문장 쌍에 대한 BERTScore, ROUGE와 BLEURT 계산 |

공개 코드는 Ko-BART 학습과 평가 예시가 중심입니다. Ko-T5 파인튜닝 전체 과정과 연구의 모든 실험 조건이 포함된 재현 패키지는 아닙니다.

## 실행 전 준비와 확인 범위

- `main.py`의 학습, 검증 CSV 경로를 실행 환경에 맞게 조정해야 합니다. CSV에는 `source`와 `target` 열이 필요합니다.
- 원천 음성, 전처리 CSV와 학습 체크포인트는 저장소에 포함되어 있지 않습니다.
- `T5_model.py`의 접근 토큰 자리표시자는 본인의 모델 접근 환경에 맞게 처리해야 합니다.
- 평가 예시는 CSV 문장 쌍을 비교합니다. 논문 결과를 재현하려면 모델이 생성한 문장과 정답, 실제 평가 데이터와 조건을 별도로 맞춰야 합니다.

## 참고자료

- [기존 연구 개요 그림](https://github.com/user-attachments/assets/0d437db6-9e0f-4e05-81bd-24e334998b2b)
- [기존 결과 그림](https://github.com/user-attachments/assets/39077229-6aa4-4900-bce6-29dec7373cf7)
- [ROUGE 논문](https://aclanthology.org/W04-1013/)
- [BERTScore 논문](https://arxiv.org/abs/1904.09675)
- [BLEURT 논문](https://arxiv.org/abs/2004.04696)
