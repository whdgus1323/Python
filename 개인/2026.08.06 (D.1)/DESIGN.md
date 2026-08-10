# 키워드 서칭 프로 디자인 시스템

## 0. Research Log

- Embedded refs: Supabase, Linear, Vercel 후보 중 Supabase를 선택했다. 데이터 작업 도구에 맞는 어두운 표면과 절제된 에메랄드 신호 체계가 적합하다.
- Skipped lanes: 이미지 생성과 웹 화면 연구는 네이티브 Qt 데스크톱 앱이며 외부 이미지 없이 정보 밀도를 우선한다.

## 1. Atmosphere & Identity

조용한 소싱 지휘실. 짙은 청록빛 배경 위에 데이터가 정밀하게 떠 있고, 에메랄드색은 다음 행동과 검증된 후보만 알려준다.

## 2. Color

| Token | Value | Usage |
|---|---|---|
| Canvas | #07100F | 앱 배경 |
| Surface | #0B1916 | 주요 패널 |
| Elevated | #10251F | 입력·표 |
| Text | #F3FBF7 | 주 텍스트 |
| Muted | #92AAA0 | 보조 텍스트 |
| Accent | #39D996 | 실행·선택 |
| Line | #203F36 | 패널 경계 |
| Warning | #F6C56D | 해외배송 |

## 3. Typography

Plus Jakarta Sans, Pretendard, Segoe UI, sans-serif를 사용한다. 제목 31px/600, 빈 상태 제목 22px/600, 본문 13px/400, 메타 10~11px/600이다.

## 4. Spacing & Layout

4px 기반 간격을 사용한다. 헤더 24px, 패널 20px, 컨트롤 12px, 표 행 36px이다. 왼쪽 조건 패널은 320px, 결과 표는 남은 공간을 소유한다.

## 5. Components

### Action button
- States: 기본, hover, pressed, disabled, focus
- Primary는 Accent, 보조는 Surface를 사용한다.
- disabled Primary는 Accent를 절대 유지하지 않는다. 어두운 Surface와 저대비 텍스트로 비활성 상태를 명확히 구분한다.

### Analysis panel
- States: empty, loaded, analyzed
- 빈 상태는 현재 단계, 다음 행동, 결과가 나타날 위치를 한 화면에서 안내하고 분석 후 표가 그 자리를 대체한다.

### Dashboard metrics
- 상품 수, 진입 기준, 후보 수는 정적 설명이 아니라 현재 데이터 상태를 보여준다.
- 파일을 불러오면 상품 수가 갱신되고, 분석 후 후보 수가 갱신된다.

### Result table
- States: default, selected, overseas, recommended
- 추천은 에메랄드, 해외배송은 황금색 톤으로만 구분한다.

## 6. Motion & Interaction

헤더는 시작 시 짧은 opacity 등장 애니메이션을 사용한다. 버튼과 탭은 hover·pressed·disabled 상태를 색과 테두리로 분명히 구분하며, 키보드 포커스는 에메랄드 외곽선으로 보인다.

## 7. Depth & Surface

Tonal-shift와 얇은 반투명 라인을 병행한다. 그림자는 최소화하고, 표면 명도 차이와 테두리로 깊이를 만든다.

## 8. Accessibility Constraints & Accepted Debt

본문 대비는 4.5:1 이상, 모든 버튼은 키보드 포커스를 가진다. 창 폭이 매우 좁을 때 표의 가로 스크롤은 허용한다.
