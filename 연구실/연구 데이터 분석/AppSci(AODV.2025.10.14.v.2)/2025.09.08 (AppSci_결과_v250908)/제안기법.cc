
/* ===== ANALYSIS-ONLY SHIM (VSCode squiggle killer) =====
   목적: 현재 파일에서만 IntelliSense 오류 제거 (실행/빌드용 아님)
   제거 시점: 실제 빌드할 때는 이 블록 전체 삭제
*/
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <utility>

// --- MSVC에서 M_PI 미정의 문제 회피 ---
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using std::string;
using std::endl;

// SimTime 비교 에러 제거(함수형 캐스트 허용)
using simtime_t = double;
using SimTime   = double;
inline simtime_t simTime() { return 0.0; }

#define EV_WARN std::cerr

// ----- inet namespace 최소 더미 -----
namespace inet {
struct Coord {
    double x=0, y=0, z=0;
    double distance(const Coord& o) const {
        double dx=x-o.x, dy=y-o.y, dz=z-o.z;
        return std::sqrt(dx*dx+dy*dy+dz*dz);
    }
};
struct IMobility {
    struct Vec { double length() const { return 0.0; } };
    virtual Vec  getCurrentVelocity()  const { return {}; }
    virtual Coord getCurrentPosition() const { return {}; }
    virtual ~IMobility() = default;
};
}

// ----- OMNeT++ 더미 -----
struct cModule {
    const char* getFullName() const { return "node[0]"; }
    cModule* getParentModule() const { return nullptr; }
    cModule* getParent() const { return getParentModule(); } // 일부 코드 호환
    cModule* getSubmodule(const char*) const { return nullptr; }
    struct SubmoduleIterator {
        explicit SubmoduleIterator(cModule*) {}
        bool end() const { return true; } // 빈 반복자(스캔 스킵)
        cModule* operator*() const { return nullptr; }
        void operator++() {}
    };
};

// ----- AODV/INET 타입 더미 -----
template<typename T> using Ptr = T*;

struct L3Address {
    bool operator==(const L3Address&) const { return true; }
};

struct Rrep {
    L3Address getDestAddr() const { return {}; }
    unsigned int getDestSeqNum() const { return 0; }
};

struct AodvRouteData {
    simtime_t getLifeTime() const { return 0.0; }
    void* getProtocolData() { return nullptr; }
};

struct IRoute {
    void* getProtocolData() { static AodvRouteData d; return &d; }
    const void* getSource() const { return (const void*)1; } // 임의 포인터
};

struct RoutingTableLike {
    IRoute* findBestMatchingRoute(const L3Address&) { return nullptr; }
};
static RoutingTableLike __routingTableObj;      // null 역참조 경고 회피
static RoutingTableLike* routingTable = &__routingTableObj;

// ----- 외부(멤버)처럼 보이는 심볼 더미 -----
static bool pFlag = true;
static std::string pwd = "C:/tmp";
static double allowedHelloLoss = 2.0;
static double helloInterval = 1.0;
static double activeRouteTimeout = 0.0;
static double deletePeriod = 0.0;
static std::map<cModule*, double> nodeSpeeds;
static std::map<cModule*, double> nodeArt;
static std::map<cModule*, double> nodeDpc;
static double totalDensity = 0.0; static int densityCount = 0;
static double totalSpeed   = 0.0; static int speedCount   = 0;

// check_and_cast 더미(아무 포인터나 그대로 캐스팅)
template<typename T, typename U>
T check_and_cast(U p){ return (T)p; }

// this 포인터 비교/멤버 함수 소속인 것처럼 보이게
struct Aodv {
    cModule* getParentModule() { static cModule m; return &m; }
    void createRoute(const L3Address&, const L3Address&, int, bool,
                     unsigned int, bool, simtime_t) {}
    void updateRoutingTable(IRoute*, const L3Address&, int, bool,
                            unsigned int, bool, simtime_t) {}
    void handleHelloMessage(const Ptr<Rrep> &helloMessage) {}
};
/* ===== END OF SHIM ===== */






void Aodv::handleHelloMessage(const Ptr<Rrep> &helloMessage)
{
    // ===== 0) 수신 노드 속도 갱신 (초기 130초까지만) =====
    double receiverSpeed = 0.0;
    if (simTime() <= SimTime(130)) {
        if (cModule *mobilityModule = getParentModule()->getSubmodule("mobility")) {
            inet::IMobility *mobility = check_and_cast<inet::IMobility*>(mobilityModule);
            receiverSpeed = mobility->getCurrentVelocity().length();
        }
        // 수신 노드 기준 속도 캐시
        nodeSpeeds[getParentModule()] = receiverSpeed;
    }

    // ===== A) Hello 기반 라우팅 테이블 생성/갱신 =====
    const L3Address &helloOriginatorAddr = helloMessage->getDestAddr();
    IRoute *routeHelloOriginator = routingTable->findBestMatchingRoute(helloOriginatorAddr);
    unsigned int latestDestSeqNum = helloMessage->getDestSeqNum();
    simtime_t newLifeTime = simTime() + allowedHelloLoss * helloInterval;

    if (!routeHelloOriginator || routeHelloOriginator->getSource() != this) {
        createRoute(helloOriginatorAddr, helloOriginatorAddr,
                    /*hop=*/1, /*isActive=*/true,
                    latestDestSeqNum, /*hasValidDestSeqNum=*/true,
                    newLifeTime);
    }
    else {
        AodvRouteData *routeData = check_and_cast<AodvRouteData*>(routeHelloOriginator->getProtocolData());
        simtime_t lifeTime = routeData->getLifeTime();
        updateRoutingTable(routeHelloOriginator, helloOriginatorAddr,
                           /*hop=*/1, /*isActive=*/true,
                           latestDestSeqNum, /*hasValidDestSeqNum=*/true,
                           std::max(lifeTime, newLifeTime));
    }


    

    // ===== B) Hello 시점 상태 기반 ART/DPC 동적 조정 =====
    // - 기본 구조: 최종값 = base_eff × F_adjust
    // - base_eff는 N_t(반경 110m 이웃 수)로 선형 보간
    if (pFlag) {
        cModule *host = getParentModule();
        const char *currentNodeName = host->getFullName();

        // ── B-1) 자기 속도/위치, 이웃 수(N_t) 산출 ───────────────────────────
        double radius = 110.0;   // 요청 반영
        double S_cur = 0.0;      // 현재 속력
        double S_prev = 0.0;     // 직전 속력 (캐시)
        double deltaS = 0.0;     // 상대 변화율
        int    N_t = 0;          // 반경 내 이웃 수

        inet::Coord myPosition;
        if (cModule *mobilityModule = host->getSubmodule("mobility")) {
            inet::IMobility *mobility = check_and_cast<inet::IMobility*>(mobilityModule);
            S_cur = mobility->getCurrentVelocity().length();
            myPosition = mobility->getCurrentPosition();
        }

        // 이전 속력 로드 (없으면 현재값 저장)
        auto it = nodeSpeeds.find(host);
        if (it != nodeSpeeds.end()) S_prev = it->second;
        else { nodeSpeeds[host] = S_cur; S_prev = S_cur; }

        if (S_prev <= 1e-9) S_prev = std::max(S_prev, 1e-3);
        deltaS = std::abs(S_cur - S_prev) / S_prev;

        // 반경 110m 이웃 수 스캔
        cModule *network = host->getParentModule();
        for (cModule::SubmoduleIterator iter(network); !iter.end(); ++iter) {
            cModule *otherVehicle = *iter;
            if (otherVehicle == host) continue;
            if (strncmp(otherVehicle->getFullName(), "node[", 5) != 0) continue;

            cModule *otherMobilityModule = otherVehicle->getSubmodule("mobility");
            if (!otherMobilityModule) continue;

            inet::IMobility *otherMobility = check_and_cast<inet::IMobility*>(otherMobilityModule);
            inet::Coord otherPosition = otherMobility->getCurrentPosition();

            if (myPosition.distance(otherPosition) <= radius) N_t++;
        }

        // 밀도 (m^-2) – RREQ와 동일 스케일 사용
        double rho_t = (N_t / (M_PI * radius * radius)) * 10.0;

        // 전역 평균(모든 노드/시점 공유) 갱신
        totalDensity += rho_t;  densityCount++;
        double rho_t_avg = totalDensity / std::max(1, densityCount);
        totalSpeed   += S_cur;  speedCount++;
        double S_avg   = totalSpeed / std::max(1, speedCount);

        // 공통 clip
        auto clip = [](double x, double lo, double hi){ return std::max(lo, std::min(x, hi)); };

        // ── B-2) 가중/보정식(F_adjust) 계산 ────────────────────────────────
        const double W_base_rho  = 0.027;
        const double W_base_S    = 0.037;
        const double S_change    = -0.003;
        const double F_change    = -0.03;
        const double F_rho       = 10.0;
        const double W_rho_min   = 0.01, W_rho_max = 0.10;
        const double W_min       = 0.01, W_max     = 0.10;
        const double inter_scale = 0.01;

        double S_cur_safe = (S_cur <= 1e-9 ? 1e-3 : S_cur);

        double W_S = clip(
            W_base_S
          + clip(std::sqrt(std::abs(S_cur - S_avg)) * S_change, -0.03, 0.03)
          + clip(deltaS * F_change, -0.02, 0.02),
            W_min, W_max
        );

        double W_rho_t = clip(
            W_base_rho + clip((rho_t - rho_t_avg) * F_rho, -0.5, 0.5),
            W_rho_min, W_rho_max
        );

        double F_rho_t = clip(rho_t / 0.05, 0.08, 10.2);
        double F_S     = clip(35.0 / S_cur_safe, 0.9, 1.1);
        double F_inter = 1.0 + std::abs(W_rho_t - W_S) * inter_scale;

        double F_adjust = (W_S * F_S + W_rho_t * F_rho_t) * F_inter;

        // ── B-2.5) N_t 고정 임계 기반 base 선형보간 ───────────────────────
        //  * N_t=15 → (60,160), N_t=20 → (40,140)
        //  * 사이값 선형보간, 범위 밖은 양끝 고정
        double art_base_eff;
        double dpc_base_eff;
        {
            const double ntLow  = 15.0;
            const double ntHigh = 20.0;

            double t;
            if (N_t <= ntLow)       t = 0.0;
            else if (N_t >= ntHigh) t = 1.0;
            else                    t = (static_cast<double>(N_t) - ntLow) / (ntHigh - ntLow);
            if (t < 0.0) t = 0.0; else if (t > 1.0) t = 1.0;

            art_base_eff = 60.0 * (1.0 - t) + 40.0 * t;
            dpc_base_eff = 160.0 * (1.0 - t) + 140.0 * t;
        }

        // ── B-3) 최종 적용: base_eff × F_adjust ───────────────────────────
        double art = clip(art_base_eff * F_adjust, 2.0, 100.0);
        double dpc = clip(dpc_base_eff * F_adjust, 5.0, 500.0);

        activeRouteTimeout = art;
        deletePeriod       = dpc;
        nodeArt[host] = art;
        nodeDpc[host] = dpc;

    }
}