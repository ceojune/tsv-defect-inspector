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