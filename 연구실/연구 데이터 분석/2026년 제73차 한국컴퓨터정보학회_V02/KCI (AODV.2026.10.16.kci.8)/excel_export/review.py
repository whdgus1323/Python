from dataclasses import replace
import duckdb
import numpy as np
from .model import Aggregation, Dataset, ExportProblem, GRID, RawTable, RunKey, Sample, Settings

SAVED_MODES = ('Legacy','LLC(RREQ+RREP,q_1)','LLC(RREQ+RREP,q_2)','LLC(RREQ+RREP,q_3)','LLC(RREQ+RREP,q_4)','Random(noCBP)')


def route_datasets(settings: Settings) -> tuple[Dataset,...]:
    path = settings.previous/'v8/route_ping_outputs/review28_all_metrics.npz'
    with np.load(path,allow_pickle=False) as saved:
        keys = tuple(RunKey(*map(int,key)) for key in saved['keys'])
        values,names = saved['metrics'].copy(),saved['metric_names'].tolist()
    if values.shape!=(448,6,len(names)) or len(set(keys))!=448:
        raise ExportProblem('리뷰어2-8 저장 결과의 비교 단위가 다릅니다.')
    specs: tuple[tuple[str,str,str,str,str | None,str | None,Aggregation,float],...] = (('04_제어송신량','RREQ·RREP 생성 및 전달 송신 수','회/실행','tx',None,None,'mean',1),('05_요청당제어송신량','Ping 요청 한 건당 제어 송신 수','회/요청','tx','tx',None,'mean',1),('08_PDR_제어비교448','제어 비교 집합의 PDR','%','pdr',None,None,'mean',100),('09_RREQ시작차단','중계 데이터에 대한 RREQ 시작 차단 횟수','회/실행','rreq_blocks',None,None,'mean',1),('10_RREP허용률','중간 RREP 생성 허용률','%','rrep_allow_pct','rrep_allowed','rrep_candidates','pooled',100),('11_활성경로보유비율','Ping 전송 시 사용 가능한 출발지 경로 정보의 비율','%','active_share_pct','active_sends',None,'mean',100),('12_경로보유시응답성공률','사용 가능한 출발지 경로 정보가 있을 때 응답 도착 비율','%','reply_active_pct',None,'active_sends','pooled',100))
    datasets = []
    for filename,title,unit,column,numerator,denominator,method,scale in specs:
        samples = []
        for i,key in enumerate(keys):
            for mode,name in enumerate(SAVED_MODES):
                row = values[i,mode]
                value = float(row[names.index(column)])
                if not np.isfinite(value):
                    continue
                num = float(row[names.index(numerator)]) if numerator is not None else None
                den = float(row[names.index(denominator)]) if denominator is not None else None
                if filename.startswith('05_'):
                    value,den = value/7000,7000.
                if filename.startswith('11_'):
                    den = float(row[names.index('active_sends')]+row[names.index('absent_sends')])
                if filename.startswith('12_'):
                    num = float(np.rint(value*den/100)) if den is not None else None
                samples.append(Sample(name,key,value,num,den,str(path)))
        datasets.append(Dataset(filename,title,unit,'Legacy·결합 Q1~Q4·Random의 공통 448개 실행',method,tuple(samples),scale=scale))
    context_path = settings.previous/'v8/route_ping_outputs/review28_context_evidence.jsonl'
    with duckdb.connect() as db:
        rows = db.execute("SELECT key,count(*) FROM read_ndjson_auto(?) WHERE mode='Q4' AND context_id>0 AND chain_status='complete' AND onchain_generators=1 AND inside_interval=1 AND window_replies>0 GROUP BY key",[str(context_path)]).fetchall()
    counts = {RunKey(*map(int,key)):int(count) for key,count in rows}
    if sum(counts.values())!=1088 or len(counts)!=347:
        raise ExportProblem('논문의 경로 선택·Ping 연결 347개 실행/1088회와 다릅니다.')
    samples = tuple(Sample('LLC(RREQ+RREP,q_4)',key,float(counts.get(key,0)),source=str(context_path)) for key in keys)
    datasets.append(Dataset('13_Q4_경로선택_Ping연결횟수','높은 CBP 중간 노드의 경로 선택과 Ping 응답 연결','회','공통 448개 실행; 연결 관측 347개 실행·1088회','sum',samples))
    return tuple(datasets)


def merge_datasets(datasets: tuple[Dataset,...]) -> tuple[Dataset,...]:
    merged: dict[str,Dataset] = {}
    for dataset in datasets:
        if dataset.filename not in merged:
            merged[dataset.filename] = dataset
            continue
        old = merged[dataset.filename]
        rows = {(r.setting,r.key):r for r in old.samples}
        for row in dataset.samples:
            identity = (row.setting,row.key)
            if identity in rows and not np.isclose(rows[identity].value,row.value,atol=1e-8,rtol=0):
                raise ExportProblem(f'같은 실행의 저장 결과가 다릅니다: {dataset.filename}, {identity}')
            rows.setdefault(identity,row)
        merged[dataset.filename] = replace(old,samples=tuple(rows.values()),population='논문 표 5·6에서 사용한 설정별 동일한 448개 실행')
    return tuple(merged.values())


def decreased_datasets(settings: Settings) -> tuple[Dataset,...]:
    path = settings.previous/'v9/cause_outputs/review29_summary.npz'
    with np.load(path,allow_pickle=False) as saved:
        keys = tuple(RunKey(*map(int,key)) for key in saved['all_keys'][saved['analyzed_indices']])
        summary,names,phy = saved['summary'].copy(),saved['summary_names'].tolist(),saved['phy'].copy()
    if len(keys)!=41:
        raise ExportProblem('논문의 PDR 감소 41개 실행과 다릅니다.')
    datasets = []
    definitions: tuple[tuple[str,str,str,Aggregation],...] = (('14_PDR_감소41','PDR 감소 실행의 왕복 전달률','pdr','mean'),('15_PHY수신실패율_감소41','PDR 감소 실행의 PHY 수신 실패율','phy','mean'),('16_경로보유비율_감소41','PDR 감소 실행의 출발지 경로 보유 비율','active','pooled'),('17_경로보유시응답_감소41','PDR 감소 실행의 경로 보유 시 응답 성공률','reply','pooled'))
    for filename,title,kind,method in definitions:
        samples = []
        for i,key in enumerate(keys):
            for mode,label in enumerate(('Legacy','LLC(RREQ+RREP,q_4)')):
                prefix = ('legacy','q4')[mode]
                fields = {'pdr':(summary[i,names.index(prefix+'_replies')],7000.),'phy':(phy[i,mode,1],phy[i,mode,0]),'active':(summary[i,names.index(prefix+'_active')],7000.),'reply':(summary[i,names.index(prefix+'_active_replies')],summary[i,names.index(prefix+'_active')])}
                num,den = map(float,fields[kind])
                if den>0:
                    samples.append(Sample(label,key,100*num/den,num,den,str(path)))
        datasets.append(Dataset(filename,title,'%','결합 Q4의 PDR이 감소한 동일한 41개 실행',method,tuple(samples)))
    counts = (('18_Legacy만응답한요청수','Legacy에서만 응답이 도착한 요청 수','legacy_only'),('19_Q4만응답한요청수','결합 Q4에서만 응답이 도착한 요청 수','q4_only'),('20_응답수_순감소','Legacy 대비 결합 Q4의 응답 수 감소','net_decrease'),('21_종료까지출발지경로부재','종료까지 출발지 경로 정보가 복구되지 않은 Legacy만 응답 요청','lost_terminal_source_gap'),('22_99초이후_미응답요청','99초 이후 전송한 Legacy만 응답 요청','lost_after99s'))
    for filename,title,column in counts:
        samples = tuple(Sample('LLC(RREQ+RREP,q_4)',key,float(summary[i,names.index(column)]),source=str(path)) for i,key in enumerate(keys))
        datasets.append(Dataset(filename,title,'건','PDR 감소 41개 실행의 같은 번호·송신 시각 요청 비교','sum',samples))
    packet_path = settings.previous/'v9/cause_outputs/review29_packet_evidence.jsonl'
    with duckdb.connect() as db:
        rows = db.execute('SELECT key,evidence_group,count(*) FROM read_ndjson_auto(?) WHERE outcome=-1 GROUP BY ALL ORDER BY evidence_group,key',[str(packet_path)]).fetchall()
        exposure_rows = db.execute('SELECT key,outcome,count(*) FROM read_ndjson_auto(?) WHERE blocked_only_context=1 GROUP BY ALL',[str(packet_path)]).fetchall()
    for filename,title,outcome in (('34_차단탐색구간_Legacy만응답','중간 응답이 차단된 탐색 구간의 Legacy만 응답 요청 수',-1),('35_차단탐색구간_Q4만응답','중간 응답이 차단된 탐색 구간의 Q4만 응답 요청 수',1)):
        exposure = {RunKey(*map(int,key)):int(count) for key,kind,count in exposure_rows if kind==outcome}
        samples = tuple(Sample('LLC(RREQ+RREP,q_4)',key,float(exposure.get(key,0)),source=str(packet_path)) for key in keys)
        datasets.append(Dataset(filename,title,'건','감소 41개 실행의 RSU 출발 탐색 구간; 같은 요청 번호·송신 시각 비교','sum',samples))
    groups: dict[str,dict[RunKey,int]] = {}
    for key,group,count in rows:
        groups.setdefault(str(group),{})[RunKey(*map(int,key))] = int(count)
    labels = {'Source: local link break':'출발지_링크단절','Source: received RERR':'출발지_RERR수신','Source: unavailable':'출발지_기타경로부재','Source: expired route':'출발지_경로정보만료','Source: removed route':'출발지_경로정보삭제','Downstream: inactive entry':'중간노드_경로정보부재','Route evidence incomplete':'경로기록불충분','Forward-table loop':'경로정보_반복','Forward-table chain complete':'목적지까지_경로정보연결'}
    for index,(group,counts_by_key) in enumerate(groups.items(),27):
        label = labels.get(group,group.replace(':','_').replace('/','_'))
        samples = tuple(Sample('LLC(RREQ+RREP,q_4)',key,float(counts_by_key.get(key,0)),source=str(packet_path),detail=group) for key in keys)
        datasets.append(Dataset(f'{index:02d}_미응답요청_{label}',label.replace('_',' ')+' 상태에서 전송한 Legacy만 응답 요청 수','건','감소 41개 실행의 요청 송신 시점; 실제 무선 drop 원인과 구분','sum',samples))
        legacy_only = {key:float(summary[i,names.index('legacy_only')]) for i,key in enumerate(keys)}
        rate_samples = tuple(Sample('LLC(RREQ+RREP,q_4)',key,100*counts_by_key.get(key,0)/legacy_only[key],float(counts_by_key.get(key,0)),legacy_only[key],str(packet_path),group) for key in keys)
        datasets.append(Dataset(f'{index+10:02d}_경로상태비율_{label}',label.replace('_',' ')+' 상태의 Legacy만 응답 요청 비율','%','분모는 감소 41개 실행의 Legacy만 응답한 전체 요청','pooled',rate_samples))
    return tuple(datasets)


def recovery_datasets(settings: Settings) -> tuple[Dataset,...]:
    path = settings.previous/'v9/recovery_outputs/review29_recovery_evidence.jsonl'
    with duckdb.connect() as db:
        rows = db.execute('SELECT key,context,origin,destination,start,selected,invalidated,recovery_wait_ms,active_entry_ms,is_source,is_return FROM read_ndjson_auto(?) WHERE is_source OR is_return ORDER BY key,start',[str(path)]).fetchall()
    datasets = []
    for filename,title,column in (('23_재탐색_경로선택대기시간','차단 탐색 요청부터 이후 경로 선택까지 걸린 시간',7),('24_선택후_경로정보유지시간','선택한 경로 정보가 다시 사용 불가로 바뀔 때까지의 시간',8)):
        samples = []
        for row in rows:
            if row[column] is None:
                continue
            label = 'LLC(RREQ+RREP,q_4)-요청' if row[9] else 'LLC(RREQ+RREP,q_4)-응답'
            detail = f'RREQ {int(row[1])}; {row[2]} → {row[3]}; 시작 {row[4]}; 선택 {row[5]}; 사용불가 {row[6]}'
            samples.append(Sample(label,RunKey(*map(int,row[0])),float(row[column]),source=str(path),detail=detail))
        datasets.append(Dataset(filename,title,'ms','PDR 감소 41개 실행에서 관측한 차단 탐색 요청; 요청=RSU→목적지, 응답=목적지→RSU; 표의 한 실행 값은 해당 요청들의 중앙값, 논문 집계는 요청 전체의 중앙값','median',tuple(samples),grid_median=True))
    samples = tuple(Sample('LLC(RREQ+RREP,q_4)-요청' if row[9] else 'LLC(RREQ+RREP,q_4)-응답',RunKey(*map(int,row[0])),1.,source=str(path),detail=f'RREQ {int(row[1])}') for row in rows)
    datasets.append(Dataset('25_중간응답모두차단_탐색횟수','기록된 중간 RREP 후보가 모두 차단된 탐색 횟수','회','PDR 감소 41개 실행; 요청=RSU→목적지, 응답=목적지→RSU','sum',samples))
    return tuple(datasets)
