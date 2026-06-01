# SUMO Synthetic net.xml Generator

이 프로젝트는 기준 SUMO `map.net.xml` 파일을 XML 구조로 분석한 뒤, 비슷한 규모와 밀도를 가진 새로운 `generated_map.net.xml`을 생성하는 Python 프로그램입니다.

## 목적

`generate_sumo_net.py`는 실제 OSM 도로망을 그대로 복제하지 않습니다. 기준 파일의 `<location>`, `<type>`, edge/junction/connection 통계를 읽고, 약간 불규칙한 격자형 synthetic road network를 새로 생성합니다.

최종 출력은 SUMO의 `netconvert`로 한 번 재빌드합니다. 이 단계는 SUMO가 요구하는 junction logic과 internal link index를 정규화해서 `randomTrips.py`, `duarouter`, `sumo` 실행이 가능하도록 만들기 위한 단계입니다.

현재 GUI 실행 파일은 더 자연스러운 지도를 만들기 위해 `generate_sumo_net_blocks.py`의 baseline 블록 재조합 방식을 사용합니다. baseline을 4x4 공간 블록으로 나누고, seed별로 블록을 재배치한 뒤 경계부를 연결해서 새로운 `map.net.xml`을 만듭니다.

## 입력 파일

기본 입력 파일은 다음 경로입니다.

```powershell
C:\Users\Choe JongHyeon\Desktop\OSM_project\baseline\map.net.xml
```

프로그램은 이 파일에서 다음 정보를 파싱합니다.

- `<location>`의 `netOffset`, `convBoundary`, `origBoundary`, `projParameter`
- 전체 `<type>` 정의
- 일반 edge/internal edge 수
- lane 수
- priority/internal junction 수
- connection 수

## 출력 파일

기본 출력 파일은 현재 작업 폴더의 `generated_map.net.xml`입니다.

출력 XML은 다음 요소를 포함합니다.

- XML declaration: `<?xml version="1.0" encoding="UTF-8"?>`
- 최상위 `<net>` 태그
- `<location>`
- `<type>`
- 일반 `<edge>`와 internal `<edge function="internal">`
- `<lane>`
- priority/internal `<junction>`
- `<connection>`

## 실행 예시

Windows PowerShell:

```powershell
python .\generate_sumo_net.py `
  --input "C:\Users\Choe JongHyeon\Desktop\OSM_project\baseline\map.net.xml" `
  --output "C:\Users\Choe JongHyeon\Desktop\OSM_project\test_3\map.net.xml" `
  --junction-target 148 `
  --internal-junction-target 216 `
  --width-m 412.65 `
  --height-m 387.65 `
  --seed 42
```

생성 후 route 생성과 SUMO 실행까지 한 번에 처리하려면 `--run-sumo`를 추가합니다.

```powershell
python .\generate_sumo_net.py `
  --input "C:\Users\Choe JongHyeon\Desktop\OSM_project\baseline\map.net.xml" `
  --output "C:\Users\Choe JongHyeon\Desktop\OSM_project\test_3\map.net.xml" `
  --run-sumo
```

Linux/macOS 또는 WSL:

```bash
python generate_sumo_net.py \
  --input /mnt/data/map.net.xml \
  --output generated_map.net.xml \
  --junction-target 148 \
  --internal-junction-target 216 \
  --width-m 412.65 \
  --height-m 387.65 \
  --seed 42
```

## 주요 파라미터

- `--width-m`: 생성 맵의 x축 길이입니다. 기본값은 `412.65`입니다.
- `--height-m`: 생성 맵의 y축 길이입니다. 기본값은 `387.65`입니다.
- `--rows`: y축 방향 priority junction 격자 수입니다. 기본값은 `12`입니다.
- `--cols`: x축 방향 priority junction 격자 수입니다. 기본값은 `13`입니다.
- `--junction-target`: 목표 priority junction 수입니다. 기본값은 `148`입니다.
- `--internal-junction-target`: 목표 internal junction 수입니다. 기본값은 `216`입니다.
- `--road-missing-prob`: 인접 노드 사이 도로를 제거할 확률입니다. 기본값은 `0.08`입니다.
  단, 제거 후에도 전체 도로망이 연결되어 있고 막다른 도로가 생기지 않는 경우에만 제거합니다.
- `--jitter-m`: junction 좌표에 적용할 random jitter 크기입니다. 기본값은 `6.0` meter입니다.
- `--seed`: 난수 seed입니다. 같은 seed는 같은 결과를 생성합니다.
- `--output`: 출력 파일 경로입니다. 기본값은 `generated_map.net.xml`입니다.

## 검증 방법

프로그램 실행 중 `validate_network()`가 자동으로 호출됩니다. 검증 항목은 다음과 같습니다.

- XML 파싱 가능 여부
- edge id 중복 여부
- lane id 중복 여부
- junction id 중복 여부
- connection의 `from`/`to` edge 존재 여부
- `via` lane 존재 여부
- lane length가 0보다 큰지 여부
- lane 좌표가 `convBoundary` 안에 포함되는지 여부
- priority junction/internal junction/edge/lane/connection 수 출력

생성 후 다시 검증하려면 Python에서 다음처럼 호출할 수 있습니다.

```python
from generate_sumo_net import validate_network

validate_network("generated_map.net.xml")
```

## SUMO에서 불러오기

SUMO GUI에서 직접 확인하려면 다음처럼 실행합니다.

```powershell
sumo-gui -n generated_map.net.xml
```

route까지 생성해 실행하려면 다음 흐름을 사용할 수 있습니다.

```powershell
randomTrips.py -n generated_map.net.xml -o map.trips.xml -r map.rou.xml -p 0.3 -e 200 --seed 1
duarouter -n generated_map.net.xml --route-files map.trips.xml -o map.rou.xml --ignore-errors
sumo -n generated_map.net.xml -r map.rou.xml --fcd-output map.xml --end 200
```

GUI 실행 파일 `SUMO_Net_Generator.exe`에서는 기본 출력 경로가 `test_3\map.net.xml`이며, `randomTrips.py`, `duarouter`, `sumo` 자동 실행 옵션이 기본으로 켜져 있습니다.

블록 조합 방식을 CLI에서 직접 실행하려면 다음을 사용합니다.

```powershell
python .\generate_sumo_net_blocks.py `
  --input "C:\Users\Choe JongHyeon\Desktop\OSM_project\baseline\map.net.xml" `
  --output-dir "C:\Users\Choe JongHyeon\Desktop\OSM_project\block_map_1" `
  --seed 1 `
  --run-sumo
```

여러 개를 만들려면:

```powershell
python .\generate_sumo_net_blocks.py `
  --input "C:\Users\Choe JongHyeon\Desktop\OSM_project\baseline\map.net.xml" `
  --output-dir "C:\Users\Choe JongHyeon\Desktop\OSM_project\block_batch" `
  --batch-count 10 `
  --batch-start-index 1 `
  --run-sumo
```

여러 맵을 한 번에 만들려면 batch 옵션을 사용합니다. 아래 예시는 지정 폴더 아래에 `map_1`부터 `map_10`까지 만들고, 각 폴더 번호를 seed로 사용합니다. 예를 들어 `map_5`는 seed `5`로 생성됩니다.

```powershell
python .\generate_sumo_net.py `
  --input "C:\Users\Choe JongHyeon\Desktop\OSM_project\baseline\map.net.xml" `
  --batch-dir "C:\Users\Choe JongHyeon\Desktop\OSM_project\batch_maps" `
  --batch-count 10 `
  --batch-start-index 1 `
  --run-sumo
```

batch 생성에서는 기본적으로 seed별 레이아웃 변화가 적용됩니다. 즉 `map_1`, `map_2`, `map_3`은 단순히 좌표 jitter만 다른 맵이 아니라 행/열 비율, 블록 간격, jitter 강도가 함께 달라집니다. 다만 너무 큰 빈 공간이 생기지 않도록 행/열 비율과 블록 간격 변화폭은 제한합니다. CLI 단일 생성에서도 같은 변화를 원하면 `--vary-layout`을 추가합니다.

```powershell
python .\generate_sumo_net.py `
  --input "C:\Users\Choe JongHyeon\Desktop\OSM_project\baseline\map.net.xml" `
  --output "C:\Users\Choe JongHyeon\Desktop\OSM_project\test_varied\map.net.xml" `
  --seed 7 `
  --vary-layout `
  --run-sumo
```

각 하위 폴더에는 다음 파일이 생성됩니다.

- `map.net.xml`
- `map.sumo.cfg`
- `map.trips.xml`
- `map.rou.xml`
- `map.xml`

생성기는 최종 `netconvert` 이후에도 검사를 수행합니다. 일반 edge는 방향별 1 lane인지, priority junction 그래프가 연결되어 있는지, degree 1 끝점이나 직선 degree 2 통과점처럼 SUMO GUI에서 빨간 동그라미로 보이는 비교차 junction이 없는지 확인합니다. 또한 4x4 구역별 교차로 밀도를 검사해서 큰 공백 구역이 생긴 맵은 실패 처리하고 자동으로 더 보수적인 레이아웃으로 재시도합니다.

또는 SUMO 설정 파일의 network 입력으로 사용할 수 있습니다.

```xml
<input>
    <net-file value="generated_map.net.xml"/>
</input>
```

## 주의사항

이 프로그램은 실제 OSM 기반 도로망 생성기가 아닙니다. 기준 `map.net.xml`의 구조와 규모를 참고해 SUMO 형식에 가까운 synthetic road network를 만드는 도구입니다. 실제 교통 시뮬레이션 품질이 중요한 경우에는 OSM 데이터와 SUMO `netconvert`를 사용해 실제 네트워크를 생성하는 방식을 권장합니다.

현재 생성 규칙은 baseline과 유사하게 priority junction 약 148개를 만들고, 일반 도로는 왕복 2차선이 되도록 방향별 edge에 1개 lane을 부여합니다. 또한 생성 단계에서 연결 그래프를 검사해 끊어진 도로와 degree 1 막다른 교차로가 생기지 않도록 합니다.
