::: mermaid
flowchart TD
    Start([시작]) --> A[메인 페이지 진입]

    A --> B{로그인 여부 확인} 
    B -->|로그인 성공| D[관리자 페이지 이동]
    B -->|비로그인 상태| C[회원가입/로그인]
    B -->|비밀번호 찾기| E[비밀번호 재설정]

    D --> G[관리자 메뉴]
    G --> G1[게시글 관리]
    G --> G2[공지사항 관리]
    G --> G3[뉴스 관리]
    G --> G7[상담 신청 처리]

    G7 -->|신청 접수| H{상담 처리 확인}
    H -->|처리 완료| I[사용자 알림]
    H -->|처리 대기| G7[상담 처리 재시작]

    A --> F[주요 기능]
    F --> J[공지사항/뉴스 보기]
    F --> K[주요 제품 확인]
    F --> L[회사소개]
    F --> M[사업안내]
    F --> N[연구소 소개]
    F --> O[고객센터]
    F --> P[검색 기능 사용]

    P -->|검색 결과 확인| F

    C --> A
    E --> A
    I --> End([종료])

    %%linkStyle default interpolate step
    %%classDef box fill:#f9f,stroke:#333,stroke-width:2px;
    class Start,End box;
    class B,H,P box;



:::
