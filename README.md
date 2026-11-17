# ML-imu
# IMU 진동 신호 기반 로봇청소기 바닥 재질 인식 및 적응형 청소 모드 전환

기계학습 팀 프로젝트 (2026-2학기)
IMU 진동 신호만으로 바닥 재질을 인식하고, 학습에 없던 바닥은 "미등록"으로 판별해 재학습 없이 새 재질로 등록한 뒤 청소 모드를 승계한다.

## 데이터

**공개 데이터**: [Lomio et al., 2019, Zenodo](https://zenodo.org/records/2653918) (CC BY 4.0)

| 파일 | 형태 | 내용 |
|---|---|---|
| `X_data.npy` | (7626, 128, 10) | 구간 × 시간 스텝 × 채널 |
| `label.npy` | (7626,) | 재질 라벨 9종 |
| `groups.npy` | (7626,) | 그룹 번호 0~79 |
| `details.csv` | 7626행 | id, 그룹, 재질 |

- **채널**: 0~3 방향(쿼터니언), 4~6 각속도, 7~9 선가속도. 모델 입력은 방향을 뺀 6채널(4~9)
- **녹화 복원**: 80개 그룹은 연속 녹화 21개를 최대 128구간씩 자른 것
  - 같은 재질로 이어지는 그룹 경계 59곳: 쿼터니언 차이 0.0014 이하 (구간 안 한 스텝 변화 중앙값 0.0002)
  - 재질이 바뀌는 경계 20곳: 0.10~2.00
- **8종 통합**: `hard_tiles`와 `hard_tiles_large_space`는 녹화가 각 1개뿐이라 `hard_tiles`로 합침

| 재질(8종) | 구간 수 | 녹화 수 | 청소 모드 |
|---|---|---|---|
| carpet | 869 | 3 | 카펫 |
| soft_pvc | 1,176 | 4 | 부드러운 바닥 |
| soft_tiles | 522 | 2 | 부드러운 바닥 |
| concrete | 1,057 | 3 | 딱딱한 바닥 |
| fine_concrete | 821 | 2 | 딱딱한 바닥 |
| tiled | 1,267 | 3 | 딱딱한 바닥 |
| hard_tiles (통합) | 879 | 2 | 딱딱한 바닥 |
| wood | 1,035 | 2 | 딱딱한 바닥 |
| **합계** | **7,626** | **21** | |

- "전부 딱딱한 바닥" 모드 기준선: 66.3%
- wood는 진동 세기가 조용한 그룹(carpet·soft_pvc)과 비슷 → 모드 오류 별도 분석

**직접 수집 데이터**: 스마트폰을 바퀴 달린 카트에 고정, 아는 바닥 3종 + 새 바닥 2종, 바닥당 장소를 달리해 녹화 3회(총 15회). 128스텝 구간으로 분할. 구글 드라이브 공유 폴더에 보관(레포에 올리지 않음).

## 시작하기 (Colab)

```python
!git clone https://github.com/<팀계정>/imu-floor.git
%cd imu-floor
!pip install -q -r requirements.txt
!python data/download.py          # Zenodo에서 data/ 로 다운로드

import sys; sys.path.append("src")
from recordings import load_dataset      # X(6채널), y(8종), recording(0~20)
from splits import load_folds            # splits/folds.json
from evaluate import evaluate
```

## 폴더 구조

```
ML-imu/
├── README.md
├── requirements.txt
├── data/                  # 데이터 파일은 .gitignore
│   └── download.py
├── src/                   # 공통 코드 (A 관리)
│   ├── recordings.py      # 녹화 21개 복원, 8종 통합, 6채널 선택
│   ├── splits.py          # 녹화 하나 빼기 21-fold, 재질 하나 빼기 분할
│   ├── preprocess.py      # 학습 녹화 기준 정규화, 증강
│   ├── features.py        # 특성 A(진동 세기)·B(분포 형태)·C(주파수)
│   └── evaluate.py        # macro-F1, 정확도, 모드 macro-F1, AUROC, FPR@95%TPR
├── splits/
│   └── folds.json         # 고정된 분할 정의 (모두 같은 파일 사용)
├── notebooks/
│   ├── A_data/
│   ├── B_classic/
│   ├── C_deep/
│   ├── D_ood/
│   └── E_register/
├── embeddings/            # .gitignore, 드라이브에 공유
└── results/
    └── results.csv
```

## 역할 분담

| 역할 | 담당 | 담당 작업 | 산출물 |
|---|---|---|---|
| A. 데이터·평가 기반 | 박태연 | 녹화 복원, 8종 통합, 21-fold 분할, 전처리, 증강, 특성 A~C, 공통 평가 함수, 분할 방식 3단계 비교(무작위 → 80그룹 → 녹화, 방향 채널 포함 여부), 증강 범위·녹화 앞뒤 구간·구간 길이 ablation | `src/`, `splits/folds.json`, 분할별 성능 하락 표 |
| B. 고전 모델 | 주민규 | 진동 크기 규칙 기준선, 로지스틱 회귀(L1·L2), KNN, 랜덤포레스트, 특성 MLP, 특성 ablation(A → A+B → A+B+C), 혼동행렬 오류 분석, wood 모드 오류 분석 | 모델별 성능 표, 혼동행렬, 오류 사례 그림 |
| C. 시계열 딥러닝 | 진주영 | MiniRocket, 1D-CNN(FCN), LSTM, center loss(유무 비교), 임베딩 추출 | 모델별 성능 표, 임베딩 파일 |
| D. 미등록 판별 | 이다혜 | 재질 하나 빼기 8회, τ 설정(검증 녹화 기준), OOD 4종 비교(softmax·Energy·Mahalanobis·KNN 거리), near/far 구분 보고, 확신하며 틀린 사례 검토 | AUROC·FPR@95%TPR 표 |
| E. 신규 등록·모드·스마트폰 | 강섬희 | k개 등록(1·5·10·20), 다른 녹화로 인식률, 기존 재질 성능 유지(전체 재학습과 비교), 모드 승계 평가, 스마트폰 실험, t-SNE | k별 정확도 그래프, 폰 실험 결과, 임베딩 시각화 |

- 공동: 바닥 수집(11/27까지 녹화 15회), 발표 준비(각자 결과 슬라이드 1~2장, 12/5부터 리허설)
- C의 임베딩이 나오기 전까지 D·E는 B의 특성 MLP 임베딩으로 파이프라인을 먼저 구성

## 공통 

**평가**
- 모든 실험은 `splits/folds.json`의 녹화 단위 분할만 사용한다. 각자 따로 나누지 않는다.
- 정규화 통계, τ, 하이퍼파라미터는 학습 녹화(또는 학습 녹화에서 떼어낸 검증 녹화)에서만 계산한다.
- 랜덤 시드는 `seed=42`로 고정한다.

**결과 기록**: `results/results.csv`에 한 줄씩 추가한다.

```
date,owner,model,setting,fold,macro_f1,accuracy,mode_macro_f1,auroc,fpr95,note
2026-11-20,B,rule_baseline,8class,all,0.xx,0.xx,0.xx,,,진동 크기 임계값
```

**임베딩 파일**: `emb_{모델}_{빠진재질}_{fold}.npy` (C가 생성, D·E가 사용). 빠진 재질이 없으면 `none`. 드라이브 `embeddings/` 폴더에 올린다.

## Git 규칙

- `src/`, `splits/` 수정은 브랜치를 만들어 PR로 올리고 A가 확인 후 병합한다.
- `notebooks/본인폴더/`는 main에 바로 push해도 된다. 다른 사람 폴더의 노트북은 수정하지 않는다.
- 노트북에서 반복해서 쓰는 코드는 `src/`로 옮긴다.
- 데이터(`*.npy`, 폰 녹화), 임베딩, 모델 가중치는 커밋하지 않는다.
- 커밋 메시지 예: `[B] 랜덤포레스트 21-fold 결과 추가`

## 참고문헌

- Lomio, F. et al. (2019). Surface Type Classification for Autonomous Robot Indoor Navigation. arXiv:1905.00252.
- Belyaev, A., & Kushnarev, O. (2025). Energy-Based Surface Classification for Mobile Robots in Known and Unexplored Terrains. *Robotics*, 14(9), 130.
- Hendrycks, D., & Gimpel, K. (2017). A Baseline for Detecting Misclassified and Out-of-Distribution Examples. ICLR.
- Lee, K. et al. (2018). A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks. NeurIPS.
- Liu, W. et al. (2020). Energy-based Out-of-distribution Detection. NeurIPS.
- Sun, Y. et al. (2022). Out-of-Distribution Detection with Deep Nearest Neighbors. ICML.
- Dempster, A. et al. (2021). MiniRocket: A Very Fast (Almost) Deterministic Transform for Time Series Classification. KDD.
- Wang, Z. et al. (2017). Time Series Classification from Scratch with Deep Neural Networks: A Strong Baseline. IJCNN.
- Snell, J. et al. (2017). Prototypical Networks for Few-shot Learning. NeurIPS.
- Wen, Y. et al. (2016). A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV.
