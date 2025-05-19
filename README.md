# Dysarthria_project
**구음장애인의 의사소통 지원을 위한 발화 재구성 시스템용 언어 모델 선정 평가**

<p align=center><img src ="https://github.com/user-attachments/assets/0d437db6-9e0f-4e05-81bd-24e334998b2b" width="500" height="500"/></p>

------------
# Contents
1. Overview

2. Main Function

3. Dataset

4. Experiment

5. Results

6. Conclusion

7. Reference

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
