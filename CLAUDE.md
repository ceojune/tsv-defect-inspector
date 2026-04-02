# CLAUDE.md

## 대화 스타일
- 한국어로 대화
- 해요체 사용
- 비유를 포함해서 설명

## 작업 방식
- 코드 수정 전에 먼저 설명
- 파일 변경 후 테스트 실행
# Agent Reliability Rules (Source: 2026-03-31 Analysis)

## Memory & Context Management
- Memory is a Hint, Not Truth: Do not trust internal memory for file contents. Always re-verify.
- Context Decay Awareness: After 10 messages, re-read the target file before any edit.
- 3-Layer Design: Do not read entire files. Use `sed` or `grep` for specific lines to save bandwidth.

## Write Discipline
- Edit Integrity: Read file -> Edit -> Read again to verify.
- Max Batch: Never modify more than 3 sections of the same file in one turn.
- Append Policy: Use `echo >>` or direct append for adding notes to avoid string-match failures.

## Verification
- Evidence-Based Reporting: Use `git diff` to show actual changes. Do not just say "Done".
- Phased Execution: Max 5 files per phase. Complete and verify before starting the next.

# SEM/FIB 이미지 결함 탐지 설계 교훈 (Source: 2026-04-02)

## FIB-SEM 이미지의 결함 밝기 양면성
- **Incomplete Fill (TSV) 탐지는 어두운 영역만 보면 안 돼요.**
  FIB로 단면을 자르면 세엄(seam)/균열 표면에서 2차 전자 방출이 증가해,
  Cu 평균보다 오히려 **밝게** 보이는 경우가 흔해요.
  → 탐지 로직은 반드시 dark pass + bright pass 두 가지를 모두 실행해야 해요.

## TSV 마스크 범위 한계
- Otsu 임계로 만든 TSV 마스크는 "구리처럼 밝은 영역"만 포함해요.
  Incomplete fill seam은 밝기가 달라서 마스크 경계에서 잘려 나올 수 있어요.
  → `detect_incomplete_fill`은 tsv_mask를 약간 dilate한 `tsv_mask_expanded`로 검사해야 해요.

## 임계값·비율 초기 설정 시 주의
- 첫 구현에서 너무 엄격한 값을 쓰면 탐지 자체가 안 돼요:
  - `dark_threshold = tsv_mean * 0.68` → 0.75로 완화 필요
  - `aspect ≥ 2.5` → 2.0으로 완화 필요
  - `darkness_ratio > 0.78` → 0.85로 완화 필요
  → 새 결함 유형 추가 시 먼저 느슨하게 잡고, 오탐 줄이는 방향으로 조이세요.

## 다중 패스 중복 탐지 방지
- dark/bright 두 패스가 같은 위치를 중복 탐지할 수 있어요.
  IoU 기반 중복 제거(`seen_boxes`) 로직을 반드시 포함하세요.