# COMMIT TO FITNESS

## 팀 소개 👨‍👨‍👧‍👧

          조원창          |          이영현          |          조명환          |          김세현          
## 프로젝트 소개 🖋
본 프로젝트는 사용자의 운동 자세를 
교정하고, 운동을 보조하는 스마트 시스템을 개발하는 것을 목표로 합니다. 
크게 세 가지 주요 기능으로 자세 교정 그리고 운동 보조(카운팅) 기능입니다.

---
### 주요 기능 및 특징:
- 실시간 모니터링: 웹캠을 통해 실시간으로 스트리밍하여, 운동 자세를 즉시 확인하고 조정할 수 있음.
- 시각적 피드백: 올바른 자세일 때 관련 랜드마크들이 녹색으로 변하여 올바른 자세를 취하고 있는지 시각적으로 확인 가능.
- 안내 문구 표시: 팔을 올리거나 내릴 필요가 있을 때, 화면에 안내 문구를 띄워 사용자가 운동을 올바르게 수행하도록 유도.
- 카운트 리셋 기능: 사용자가 원할 경우 스쿼트 카운터를 리셋할 수 있는 버튼 제공.

### 구현 상세:
- HTML 파일에서 Semantic UI를 사용하여 UI 디자인을 구현함.
- Python 스크립트에서 Mediapipe를 통해 포즈 감지 및 관절 각도 계산을 수행함.
- 사용자가 "Reset Count Button"을 클릭하면 해당 URL로 이동하여 스쿼트 카운터를 리셋함.
- 운동 자세를 평가하기 위해 각 관절의 각도를 계산하고, 이를 시각적으로 보여줌.
- 감지된 운동 자세에 따라 올바른 자세인지 여부를 판단하여 카운트를 증가시킴.
