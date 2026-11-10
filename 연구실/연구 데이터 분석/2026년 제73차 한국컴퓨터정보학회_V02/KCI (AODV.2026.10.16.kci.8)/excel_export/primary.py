from pathlib import Path
import hashlib
import re
import duckdb
import numpy as np
from .model import Config, Dataset, ExportProblem, GRID, Result, ResultSet, RunKey, Sample, Settings, configurations, folder_for

FINAL_RESULT = re.compile(r'sent:\s*(\d+)\s+received:\s*(\d+)\s+PDR\s*\(%\):\s*(\S+)')


def read_result(path: Path) -> Result:
    before = path.stat()
    text = path.read_text(encoding='utf-8-sig')
    after = path.stat()
    matches = FINAL_RESULT.findall(text)
    rtt = re.search(r'round-trip min/avg/max \(ms\):\s*([^\r\n]+)',text)
    if len(matches)!=1 or rtt is None or (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
        raise ExportProblem(f'완료 결과를 확인할 수 없습니다: {path}')
    sent,received,recorded = int(matches[0][0]),int(matches[0][1]),float(matches[0][2])
    if sent!=7000 or not 0<=received<=sent or not np.isclose(100*received/sent,recorded,atol=.001,rtol=0):
        raise ExportProblem(f'송수신 수와 PDR이 일치하지 않습니다: {path}')
    return Result(sent,received,100*received/sent,float(rtt.group(1).split('/')[1]),str(path))


def load_results(settings: Settings) -> ResultSet:
    configs = configurations(settings.data_root)
    records = {}
    for config in configs:
        for key in GRID:
            records[config.name,key] = read_result(folder_for(config,key)/'result.txt')
        print(config.name,': result.txt 450개 확인')
    return ResultSet(configs,records)


def pdr_datasets(results: ResultSet) -> tuple[Dataset,...]:
    pdr,delta = [],[]
    for (mode,key),record in results.records.items():
        baseline = results.records['Legacy',key]
        pdr.append(Sample(mode,key,record.pdr,record.received,record.sent,record.source))
        delta.append(Sample(mode,key,record.pdr-baseline.pdr,record.received-baseline.received,record.sent,record.source))
    random_delta = tuple(Sample(mode,key,record.pdr-results.records['Random(noCBP)',key].pdr,record.received-results.records['Random(noCBP)',key].received,record.sent,record.source) for (mode,key),record in results.records.items() if mode in ('Legacy','Random(noCBP)','LLC(RREQ+RREP,q_1)','LLC(RREQ+RREP,q_2)','LLC(RREQ+RREP,q_3)','LLC(RREQ+RREP,q_4)'))
    return (Dataset('26_PDR_Random대비차이','같은 시드 Random 대비 PDR 차이','%p','표 6의 같은 조건 450개 실행','mean',random_delta),Dataset('01_PDR','Ping 왕복 패킷 전달률','%','설정별 동일한 450개 실행','mean',tuple(pdr)),Dataset('02_PDR_Legacy대비차이','같은 시드 Legacy 대비 PDR 차이','%p','설정별 동일한 450개 실행','mean',tuple(delta)))


class CsvCache:
    def __init__(self, folder: Path) -> None:
        self.folder = folder
        folder.mkdir(exist_ok=True,parents=True)

    def read(self, path: Path, query: str) -> tuple[float,...]:
        before = path.stat()
        signature = f'{path.resolve().as_posix()}|{before.st_size}|{before.st_mtime_ns}|{query}'
        cache = self.folder/(hashlib.sha256(signature.encode()).hexdigest()+'.npz')
        if cache.exists():
            with np.load(cache,allow_pickle=False) as saved:
                return tuple(float(x) for x in saved['values'])
        with duckdb.connect(config={'threads':1,'preserve_insertion_order':True}) as db:
            row = db.execute(query,[str(path)]).fetchone()
        if row is None or any(x is None for x in row):
            raise ExportProblem(f'필수 집계값이 없습니다: {path}')
        values = tuple(float(x) for x in row)
        after = path.stat()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
            raise ExportProblem(f'읽는 중 변경된 파일: {path}')
        np.savez_compressed(cache,values=np.asarray(values,dtype=float))
        return values


def q4_datasets(settings: Settings, results: ResultSet) -> tuple[Dataset,...]:
    selected = tuple(config for config in results.configs if config.name in ('Legacy','LLC(RREQ,q_4)','LLC(RREP,q_4)','LLC(RREQ+RREP,q_4)'))
    cache = CsvCache(settings.output/'_cache')
    with np.load(settings.previous/'v8/route_ping_outputs/review28_all_metrics.npz',allow_pickle=False) as saved:
        control_keys = {RunKey(*map(int,key)) for key in saved['keys']}
    if len(control_keys)!=448:
        raise ExportProblem('현재 논문은 제어 지표의 공통 448개 실행을 사용합니다.')
    rtt_keys = {key for key in GRID if all(np.isfinite(results.records[c.name,key].rtt) and results.records[c.name,key].rtt>0 for c in selected)}
    if len(rtt_keys)!=441:
        raise ExportProblem('현재 논문의 RTT 공통 441개 실행과 다릅니다.')
    phy,tx,tx_request,rtt,coverage,subset_pdr = [],[],[],[],[],[]
    phy_sql = "SELECT sum(CAST(receptionAttempts AS DOUBLE)),sum(CAST(failedReceptionsWithInterference AS DOUBLE)),sum(CAST(successfulReceptions AS DOUBLE)+CAST(failedReceptionsWithInterference AS DOUBLE)+CAST(failedReceptionsWithoutInterference AS DOUBLE)) FROM read_csv(?,header=true,all_varchar=true) WHERE lower(trim(pingType)) IN ('request','reply')"
    tx_sql = "SELECT count(*) FILTER(WHERE event IN ('RREQ_SEND','RREQ_FORWARD','RREP_SEND','RREP_FORWARD')) FROM read_csv(?,header=true,all_varchar=true)"
    rtt_sql = "WITH e AS (SELECT node,event,CAST(seqNo AS BIGINT) AS seq,CAST(time AS DOUBLE) AS t,CAST(rttMs AS DOUBLE) AS rtt,CAST(isDuplicate AS INTEGER) AS dup FROM read_csv(?,header=true,all_varchar=true)), senders AS (SELECT DISTINCT node FROM e WHERE event='send'), replies AS (SELECT e.* FROM e JOIN senders USING(node) WHERE event='reply' AND dup=0 QUALIFY row_number() OVER(PARTITION BY node,seq ORDER BY t)=1) SELECT count(*),count(*) FILTER(WHERE rtt>0 AND isfinite(rtt)) FROM replies"
    for config in selected:
        for index,key in enumerate(GRID,1):
            folder = folder_for(config,key)
            record = results.records[config.name,key]
            attempts,failures,total = cache.read(folder/'ping_collision_summary.csv',phy_sql)
            if attempts<=0 or failures<0 or failures>attempts or attempts!=total:
                raise ExportProblem(f'PHY 수신 건수가 일치하지 않습니다: {folder}')
            phy.append(Sample(config.name,key,100*failures/attempts,failures,attempts,str(folder/'ping_collision_summary.csv')))
            replies,valid = cache.read(folder/'ping_trace_log.csv',rtt_sql)
            if replies!=record.received or valid>replies:
                raise ExportProblem(f'Ping 최종 응답 수가 다릅니다: {folder}')
            if replies>0:
                coverage.append(Sample(config.name,key,100*valid/replies,valid,replies,str(folder/'ping_trace_log.csv')))
            if key in rtt_keys:
                rtt.append(Sample(config.name,key,record.rtt,source=record.source))
            if key in control_keys:
                count = cache.read(folder/'aodv_control_log.csv',tx_sql)[0]
                tx.append(Sample(config.name,key,count,source=str(folder/'aodv_control_log.csv')))
                tx_request.append(Sample(config.name,key,count/record.sent,count,record.sent,str(folder/'aodv_control_log.csv')))
                subset_pdr.append(Sample(config.name,key,record.pdr,record.received,record.sent,record.source))
            if index%100==0:
                print(config.name,': PHY·RTT·제어 집계',index,'/450')
    return (Dataset('03_PHY수신실패율_Q4','간섭 동반 Ping 수신 실패율','%','Q4 네 방식의 공통 450개 실행','mean',tuple(phy)),Dataset('04_제어송신량','RREQ·RREP 생성 및 전달 송신 수','회/실행','Q4 네 방식의 공통 448개 실행','mean',tuple(tx)),Dataset('05_요청당제어송신량','Ping 요청 한 건당 제어 송신 수','회/요청','Q4 네 방식의 공통 448개 실행','mean',tuple(tx_request),scale=1),Dataset('06_평균RTT','송신 시각이 확인되는 응답의 평균 RTT','ms','네 방식 모두 유효 RTT가 있는 공통 441개 실행','mean',tuple(rtt)),Dataset('07_RTT표본확인비율','전체 응답 중 유효 RTT 표본의 비율','%','전체 450개 실행; 응답 0인 실행은 빈칸','pooled',tuple(coverage)),Dataset('08_PDR_제어비교448','제어 지표와 같은 실행 집합의 PDR','%','공통 448개 실행','mean',tuple(subset_pdr)))
