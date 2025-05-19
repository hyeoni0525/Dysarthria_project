# Dysarthria_project
**구음장애인의 의사소통 지원을 위한 발화 재구성 시스템용 언어 모델 선정 평가**

<p align=center><img src ="https://github.com/user-attachments/assets/0d437db6-9e0f-4e05-81bd-24e334998b2b" width="500" height="500"/></p>

------------
# Contents
1. Overview

2. Main Function

3. Dataset

4. Model

5. Results

6. Expected Impact

7. Future Plans

8. Reference

----------
# Overview

구음장애 :arrow_forward: 운동신경의 이상으로 조음 기관을 제대로 조절하지 못하여 발음하는 데 어려움을 겪는 언어 장애

의사소통 단절 :arrow_forward: 사회적 고립, 우울감, 의사소통 문제로 인한 정보파악의 어려움

대안 필요성: 기존 STT 시스템은 구음장애인의 발화를 제대로 인식하지 못함 

:arrow_forward: 후처리/보정을 통해 구음장애인들의 발음 인식률을 개선하여 삶의 질을 높이고자 함

연구 목적:

- Whisper로 변환된 STT 문장을 자연스럽고 정형화된 문장으로 복원
- 이를 위한 한국어 LLM(KoBART, KoT5)의 성능 비교 및 적합 모델 선정

-----------
# Main Function

전처리 절차
-----------

1. VAD (Voice Activity Detection): 비음성 구간 제거 :sound:

2. STT 처리: Whisper 모델로 음성을 텍스트로 변환 :sound:

3. 정제 자동화:

   불필요한 반복어, 외래어, 특수문자 제거

   1:1 Mapping 구조 정립 (STT → 정답 문장)

   CER > 0.5 또는 WER > 0.7 샘플 제거

4. 토큰 수 기준 정렬:

   KoBART: 256 토큰

   Ko-T5: 512 토큰 기준으로 문장 구성

----------
# Dataset

AI Hub 구음장애 음성 인식 데이터셋 활용 - https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=608

총 486개 샘플 확보:

- 학습: 339개

- 검증: 98개

- 테스트: 49개

---------
# Model

## 📚 모델 비교 | Model Comparison

| 항목․Feature              | **KoBART**                                                            | **KoT5**                                                                                      |
|--------------------------|------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| 출처․Origin              | Facebook BART 구조 기반, SK텔레콤 한국어 특화                                         | Google T5 구조 기반, KETI 한국어 확장                                                        |
| 구조․Architecture        | Encoder–Decoder + Denoising                                             | Text-to-Text (Prefix 방식)                                                                    |
| 주용도․Primary Use       | 텍스트 생성·요약·번역 등 다양한 문장 처리                                           | 텍스트 생성·문서 요약·QA 등 광범위한 자연어 처리                                               |
| 학습 방식․Training       | 마스크된 토큰 예측(denoising)                                              | 프리픽스(prefix)로 작업 유형 지정, End-to-End 학습                                             |
| 장점․Strengths           | 의미 정밀성·문장 구조 복원에 강함                                               | 경량 모델로 파인튜닝·실시간 처리에 유리, 높은 범용성                                           |
| 활용 예시․Example Apps   | 번역, 요약, 텍스트 생성                                                   | 요약, 질의응답, 챗봇, 다중 태스크                                                              |

------------
# Results

<img src ="https://github.com/user-attachments/assets/39077229-6aa4-4900-bce6-29dec7373cf7" width="400" height="200"/>

- KoT5는 구조 유사성 및 문맥 일치 측면(ROUGE-L, BERTScore)에서 우세
- KoBART는 자연스러움과 의미 정밀성(BLEURT)에서 우세

▶️ 자연스러운 문장 복원을 더 중요시하는 본 과제에서는 KoBART가 보다 적합한 모델로 판단

------------
# Expected Impact

1. 의사소통 만족도 개선 및 대화 참여 확대

2. 일상 속 의사소통 문제를 보조하며 사회적 참여 기회 확대

3. 지속적인 발전 가능성을 지닌 사회적 의사소통 플랫폼으로 확장

--------------
# Future Plans

- 구음장애인의 발화 특성에 최적화된 특화 데이터셋 구성

- 일상 대화 환경을 반영한 잡음 포함 음성 학습

- 실시간 양방향 통신 시스템으로 확장 가능성 확보 (예: TTS 연계 등)

- 향후 온디바이스 AI를 통한 챗봇·앱과의 연동까지 고려

---------------
# Reference

사회적고립 기술해결 완전한 해결책이 될 수 있을까?  https://www.peoplepower21.org/welfarenow/1982302

장애인 사회 고립, 의사소통 해결이 먼저다  https://m.health.chosun.com/svc/news_view.html?contid=2024101802413

OpenAI – Whisper https://openai.com/index/whisper/

BART - Lewis, M., Liu, Y., Goyal, N., Ghazvininejad, M., Mohamed, A., Levy, O., Stoyanov, V., & Zettlemoyer, L. (2019). BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension. arXiv preprint arXiv:1910.13461

T5 - Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., & Liu, P. J. (2020). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. Journal of Machine Learning Research, 21(140), 1–67.

성능지표 -  T. Zhang, V. Kishore, F. Wu, K. Q. Weinberger, Y. Artzi, "BERTScore: Evaluating Text Generation with BERT", arXiv:1904.09675, Apr. 2019. 

성능지표 - T. Sellam, D. Das, A. P. Parikh, "BLEURT: Learning Robust Metrics for Text Generation", arXiv:2004.04696, Apr. 2020.

성능지표 - Chin-Yew Lin. 2004. ROUGE: A Package for Automatic Evaluation of Summaries. In Text Summarization Branches Out, pages 74–81, Barcelona, Spain. Association for Computational Linguistics. https://aclanthology.org/W04-1013/



