# platoon_traci.py
# - 작업폴더 자동 고정
# - sumo-gui 경로 자동 탐색(확장자/여러 후보/비GUI 폴백)
# - cfg가 안 열릴 땐 -n/-r 우회
# - 통신 없는 CTH + PD(+I) 제어

import os, sys, math, traceback

# ==== 0) 작업 폴더 고정 ====
BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)

# ==== 1) SUMO tools 경로 등록 ====
SUMO_HOME = os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\Eclipse\Sumo")
TOOLS = os.path.join(SUMO_HOME, "tools")
if os.path.isdir(TOOLS) and TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

try:
    import traci, sumolib
except Exception as e:
    print("[ERR] import 실패:", repr(e))
    print("SUMO_HOME =", SUMO_HOME)
    print("tools     =", TOOLS, "(exists:", os.path.isdir(TOOLS), ")")
    sys.exit(1)

# ==== 2) 파일 경로 ====
CFG = os.path.join(BASE, "platoon.sumocfg")
NET = os.path.join(BASE, "net.net.xml")
ROU = os.path.join(BASE, "platoon.rou.xml")

def _is_file(p):
    try:
        return p and os.path.isfile(p)
    except Exception:
        return False

def find_sumo_binary():
    # 1) checkBinary 시도
    try:
        p = sumolib.checkBinary("sumo-gui")
        # 어떤 환경에선 확장자 없이 리턴되기도 함 → 그대로 사용 가능하게 두고,
        # 파일 검사에서 False가 나와도 실행을 시도할 수 있도록 폴백 로직 추가
        if _is_file(p):
            return p
        # .exe 붙여 재시도
        if _is_file(p + ".exe"):
            return p + ".exe"
        # 2) 일반적인 설치 후보 경로들
    except Exception:
        p = None

    candidates = [
        os.path.join(SUMO_HOME, "bin", "sumo-gui.exe"),
        os.path.join(SUMO_HOME, "bin", "sumo-gui"),
        os.path.join(SUMO_HOME, "bin", "sumo.exe"),
        os.path.join(SUMO_HOME, "bin", "sumo"),
        r"C:\Program Files\Eclipse\Sumo\bin\sumo-gui.exe",
        r"C:\Program Files\Eclipse\Sumo\bin\sumo.exe",
        r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo-gui.exe",
        r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe",
    ]
    for c in candidates:
        if _is_file(c):
            return c

    # 3) 마지막 폴백: checkBinary로 'sumo' 비GUI
    try:
        p2 = sumolib.checkBinary("sumo")
        if _is_file(p2):
            return p2
        if _is_file(p2 + ".exe"):
            return p2 + ".exe"
        return p2  # 존재 검증이 False여도 실행을 시도하게 반환
    except Exception:
        return None

SUMO_BIN = find_sumo_binary()

# ==== 3) PRECHECK ====
print("=== PRECHECK ===")
print("BASE     =", BASE)
print("SUMO_BIN =", SUMO_BIN)
print("CFG      =", CFG, "exists:", os.path.isfile(CFG))
print("NET      =", NET, "exists:", os.path.isfile(NET))
print("ROU      =", ROU, "exists:", os.path.isfile(ROU))

if not SUMO_BIN:
    print("[ERR] sumo 실행 파일을 찾지 못했습니다. SUMO_HOME 또는 PATH를 확인하세요.")
    print("      예) setx SUMO_HOME \"C:\\Program Files (x86)\\Eclipse\\Sumo\"")
    sys.exit(1)

# ==== 4) 실행 커맨드 구성 (cfg 또는 -n/-r 우회) ====
DT = 0.05
use_cfg = os.path.isfile(CFG)
if use_cfg:
    cmd = [SUMO_BIN, "-c", CFG, "--step-length", str(DT)]
else:
    cmd = [SUMO_BIN, "-n", NET, "-r", ROU, "--step-length", str(DT)]
print("[INFO] starting SUMO with:", cmd)

# ==== 5) 제어 파라미터 ====
leader = "leader"
followers = ["f1","f2","f3","f4"]

D0, H = 5.0, 0.9
Kp, Kd, Ki = 0.45, 0.75, 0.00
A_MAX, D_MAX, V_MAX = 2.5, -4.5, 40.0

def clamp(x, lo, hi): return lo if x < lo else hi if x > hi else x
def leader_profile(t):
    if t < 30:   return 10.0*(t/30.0)
    if t < 60:   return 10.0
    if t < 90:   return 10.0 - 7.0*((t-60)/30.0)
    return 3.0

def set_ext(vid):
    traci.vehicle.setSpeedMode(vid, 0)
    traci.vehicle.setLaneChangeMode(vid, 0)

def safe_get_leader_gap(my_id, rng=300.0):
    info = traci.vehicle.getLeader(my_id, rng)
    if not info:
        return None, None
    if isinstance(info, (list, tuple)) and len(info) == 2:
        return info[0], float(info[1])
    if isinstance(info, str):
        lead = info
        try:
            gap = (traci.vehicle.getDistance(lead)
                   - traci.vehicle.getDistance(my_id)
                   - traci.vehicle.getLength(lead))
            return lead, float(gap)
        except Exception:
            return lead, None
    return None, None

# ==== 6) 실행 ====
try:
    traci.start(cmd)
    print("[OK] TraCI connected")
except Exception as e:
    print("[ERR] traci.start 실패:", repr(e))
    traceback.print_exc()
    sys.exit(1)

t = 0.0
integ = {vid: 0.0 for vid in followers}

try:
    print("[INFO] 초기 vehicle IDs:", list(traci.vehicle.getIDList()))

    if leader in traci.vehicle.getIDList(): set_ext(leader)
    for vid in followers:
        if vid in traci.vehicle.getIDList(): set_ext(vid)

    steps = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        if leader in traci.vehicle.getIDList():
            v_cmd = leader_profile(t)
            v_now = traci.vehicle.getSpeed(leader)
            a = clamp((v_cmd - v_now)*1.5, D_MAX, A_MAX)
            traci.vehicle.setSpeed(leader, clamp(v_now + a*DT, 0.0, V_MAX))

        for i, vid in enumerate(followers):
            if vid not in traci.vehicle.getIDList():
                continue
            lead_id = leader if i == 0 else followers[i-1]
            if lead_id not in traci.vehicle.getIDList():
                continue

            v_f = traci.vehicle.getSpeed(vid)
            v_l = traci.vehicle.getSpeed(lead_id)
            _, gap = safe_get_leader_gap(vid, 300.0)
            if gap is None:
                traci.vehicle.setSpeed(vid, min(max(v_f,0.0), V_MAX))
                continue

            e = gap - (D0 + H*v_f)
            rel_v = v_l - v_f
            integ[vid] += e*DT
            a_cmd = clamp(Kp*e + Kd*rel_v + Ki*integ[vid], D_MAX, A_MAX)
            traci.vehicle.setSpeed(vid, clamp(v_f + a_cmd*DT, 0.0, V_MAX))

        if steps in (0, 1, 2, 10, 50):
            print(f"[STEP {steps}] IDs:", list(traci.vehicle.getIDList()))
        traci.simulationStep()
        t += DT
        steps += 1

    print("[DONE] simulation finished. remaining:", traci.simulation.getMinExpectedNumber())
except Exception as e:
    print("[ERR] 루프 중 예외:", repr(e))
    traceback.print_exc()
finally:
    try:
        traci.close()
    except Exception:
        pass
