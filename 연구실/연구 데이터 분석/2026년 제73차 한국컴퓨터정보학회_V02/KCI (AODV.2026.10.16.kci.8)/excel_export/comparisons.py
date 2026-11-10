from itertools import product
import numpy as np
from .model import RunKey, ExportView


def add_q_comparisons(writer: ExportView) -> None:
    for title,seed_groups in (('Q최고빈도_시드별',tuple((s,) for s in range(1,6))),('Q최고빈도_시나리오',((1,2,3,4,5),))):
        sheet = writer.book.add_worksheet(title)
        sheet.write_row(0,0,('제어 방식','차량 수','최대 속력','시드','Q1 AVG','Q2 AVG','Q3 AVG','Q4 AVG','Q1 최고','Q2 최고','Q3 최고','Q4 최고'),writer.styles['header'])
        totals = np.zeros((3,4),dtype=int)
        for i,(family,vehicles,speed,seeds) in enumerate(product(('RREQ','RREP','RREQ+RREP'),(70,100,120),(10,20,30),seed_groups),1):
            sheet.write_row(i,0,(family,vehicles,speed,','.join(map(str,seeds))))
            averages = []
            for q in range(1,5):
                mode = f'LLC({family},q_{q})'
                keys = [RunKey(vehicles,speed,node,seed) for node in range(10) for seed in seeds]
                values = [writer.values[mode,key][0] for key in keys]
                refs = ','.join(f"'원자료'!F{writer.index[mode,key][0]}" for key in keys)
                value = float(np.mean(values))
                averages.append(value)
                sheet.write(i,3+q,f'=AVERAGE({refs})',writer.styles['raw'],value)
            for q,value in enumerate(averages,1):
                high = int(np.isclose(value,max(averages),atol=1e-9,rtol=0))
                column = ('E','F','G','H')[q-1]
                sheet.write(i,7+q,f'=IF(ABS({column}{i+1}-MAX(E{i+1}:H{i+1}))<=0.000000001,1,0)',None,high)
                totals[('RREQ','RREP','RREQ+RREP').index(family),q-1] += high
        start = 3+3*9*len(seed_groups)
        sheet.write_row(start,0,('제어 방식','Q1 최고 횟수','Q2 최고 횟수','Q3 최고 횟수','Q4 최고 횟수','조건 수'),writer.styles['header'])
        for family_index,family in enumerate(('RREQ','RREP','RREQ+RREP')):
            r = start+1+family_index
            sheet.write(r,0,family)
            for q in range(1,5):
                col = ('I','J','K','L')[q-1]
                formula = f'=SUMIF(A2:A{start-2},A{r+1},{col}2:{col}{start-2})'
                sheet.write(r,q,formula,None,int(totals[family_index,q-1]))
            sheet.write(r,5,9*len(seed_groups))
        sheet.write(start+6,0,'동률이면 최고인 Q를 각각 집계합니다. 시드별 45개 조건과 시나리오 평균 9개 조건을 구분합니다.')
        sheet.set_column(0,0,22)
        sheet.set_column(1,11,16)
        sheet.freeze_panes(1,4)


def add_relative_reductions(writer: ExportView) -> None:
    if writer.dataset.filename not in ('04_제어송신량','05_요청당제어송신량'):
        return
    sheet = writer.summary_sheet
    sheet.write_row(0,18,('Legacy 대비 감소(%)','Random 대비 감소(%)'),writer.styles['header'])
    for column,baseline in ((18,'Legacy'),(19,'Random(noCBP)')):
        if baseline not in writer.mode_names:
            continue
        base_row = writer.mode_names.index(baseline)+2
        base = float(np.mean([r.value for r in writer.rows if r.setting==baseline]))
        for i,mode in enumerate(writer.mode_names,1):
            value = float(np.mean([r.value for r in writer.rows if r.setting==mode]))
            formula = f'=IF(I{base_row}>0,100*(I{base_row}-I{i+1})/I{base_row},"")'
            sheet.write(i,column,formula,writer.styles['raw'],100*(base-value)/base if base>0 else '')


def add_linked_summary(writer: ExportView) -> None:
    if writer.dataset.filename!='13_Q4_경로선택_Ping연결횟수':
        return
    sheet = writer.summary_sheet
    sheet.write(0,18,'연결 관측이 있는 실행 수',writer.styles['header'])
    for i,mode in enumerate(writer.mode_names,1):
        rows = [row for row in writer.rows if row.setting==mode]
        first,last = writer.ranges[mode][0],writer.ranges[mode][-1]
        formula = f"=COUNTIF('원자료'!F{first}:F{last},\">0\")"
        sheet.write(i,18,formula,None,sum(row.value>0 for row in rows))


def add_pdr_statistics(writer: ExportView) -> None:
    if writer.dataset.filename not in ('01_PDR','02_PDR_Legacy대비차이'):
        return
    summary = writer.summary_sheet
    summary.write_row(0,18,('시드 평균의 표준편차','가장 작은 시드 평균','가장 큰 시드 평균'),writer.styles['header'])
    for i,mode in enumerate(writer.mode_names,1):
        means = [float(np.mean([r.value for r in writer.rows if r.setting==mode and r.key.seed==seed])) for seed in range(1,6)]
        for col,operation,value in ((18,'STDEV',float(np.std(means,ddof=1))),(19,'MIN',min(means)),(20,'MAX',max(means))):
            summary.write(i,col,f'={operation}(J{i+1}:N{i+1})',writer.styles['raw'],value)
    if writer.dataset.filename!='02_PDR_Legacy대비차이':
        return
    sheet = writer.book.add_worksheet('시나리오별_개선감소빈도')
    sheet.write_row(0,0,('설정','차량 수','최대 속력','실행 수','평균 차이(%p)','중앙값 차이(%p)','개선','동률','감소','감소 비율(%)','40%p 이상 개선'),writer.styles['header'])
    output_row = 1
    for mode,vehicles,speed in product(writer.mode_names,(70,100,120),(10,20,30)):
        selected = [r for r in writer.rows if r.setting==mode and r.key.vehicles==vehicles and r.key.speed==speed]
        if not selected:
            continue
        indices = [writer.index[mode,r.key][0] for r in selected]
        ref = f"'원자료'!F{min(indices)}:F{max(indices)}"
        values = np.array([r.value for r in selected])
        sheet.write_row(output_row,0,(mode,vehicles,speed,len(selected)))
        sheet.write(output_row,4,f'=AVERAGE({ref})',writer.styles['raw'],float(values.mean()))
        sheet.write(output_row,5,f'=MEDIAN({ref})',writer.styles['raw'],float(np.median(values)))
        for col,criterion,count in ((6,'>0',int(np.sum(values>0))),(7,'=0',int(np.sum(values==0))),(8,'<0',int(np.sum(values<0))),(10,'>=40',int(np.sum(values>=40)))):
            sheet.write(output_row,col,f'=COUNTIF({ref},"{criterion}")',None,count)
        sheet.write(output_row,9,f'=100*I{output_row+1}/D{output_row+1}',writer.styles['raw'],100*float(np.sum(values<0))/len(values))
        output_row += 1
    sheet.set_column(0,0,28)
    sheet.set_column(1,10,18)
    sheet.freeze_panes(1,3)
