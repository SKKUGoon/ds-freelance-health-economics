# Patch Plan — PAPER_KOR 문체 자연화

## Objective

`notebooks/PAPER_KOR.md`의 내용을 번역체/경직된 표현에서 자연스러운 한국어 기술 문서 톤으로 다듬는다.

## Scope

- 수치, 표, 결론, 파일 경로, 실험 결과는 유지
- 문장 흐름, 어휘 선택, 단락 연결만 개선
- 구조(섹션/그림/산출물 목록)는 유지

## Files

- Update: `notebooks/PAPER_KOR.md`

## Implementation Steps

1. 요약(Executive Summary) 문단을 자연스러운 서술형으로 재작성
2. Stage별 해석 문장을 직역투 표현에서 실무형 문체로 조정
3. 결론/배포 권장 사항을 간결하고 읽기 쉬운 표현으로 정리
4. 과도하게 딱딱한 용어(예: 강건성, 꼬리 위험 압축)를 맥락에 맞는 표현으로 완화

## Validation Criteria

- 핵심 수치와 실험 결론이 변경되지 않을 것
- 문단을 소리 내어 읽었을 때 번역체 어색함이 줄어들 것
- 기존 표/그림 참조 및 경로가 손상되지 않을 것
