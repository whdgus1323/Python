from collections import defaultdict
from pathlib import Path
from typing import assert_never
import numpy as np
import xlsxwriter
from xlsxwriter.utility import xl_col_to_name
from xlsxwriter.worksheet import Worksheet
from .model import Dataset, ExportProblem, GRID, RunKey, Sample, Settings, finite_samples


class WorkbookWriter:
    def __init__(self, dataset: Dataset, settings: Settings) -> None:
        self.dataset,self.settings = dataset,settings
        self.path = settings.output/(dataset.filename+'.xlsx')
        self.book = xlsxwriter.Workbook(str(self.path))
        self.book.set_calc_mode('auto')
        self.summary_sheet = self.book.add_worksheet('논문집계')
        common = {'font_name':'Arial','font_size':10,'valign':'vcenter'}
        self.styles = {name:self.book.add_format({**common,**options}) for name,options in {'title':{'bold':True,'font_size':13,'font_color':'#17365D'},'scenario':{'bold':True,'bg_color':'#D9E5F2','border':1,'align':'center'},'header':{'bold':True,'bg_color':'#EDF2F7','border':1,'align':'center','text_wrap':True},'cell':{'border':1,'align':'center'},'number':{'border':1,'num_format':'0.0000','align':'center'},'avg':{'bold':True,'bg_color':'#E2F0D9','border':1,'num_format':'0.0000','align':'center'},'note':{'font_color':'#555555','text_wrap':True},'raw':{'num_format':'0.000000','align':'right'}}.items()}
        self.rows = sorted(finite_samples(dataset),key=lambda r:(r.setting,r.key,r.detail))
        if not self.rows:
            raise ExportProblem(f'출력할 수치가 없습니다: {dataset.filename}')
        self.index: dict[tuple[str,RunKey],list[int]] = defaultdict(list)
        self.values: dict[tuple[str,RunKey],list[float]] = defaultdict(list)
        self.ranges: dict[str,list[int]] = defaultdict(list)
        self.mode_names = list(dict.fromkeys(row.setting for row in self.rows))

    def raw(self) -> None:
        sheet = self.book.add_worksheet('원자료')
        headers = ('설정','차량 수','최대 속력(km/h)','목적지 nodeId','시드','값','분자','분모','원본 파일','계산 및 연결 정보')
        sheet.write_row(0,0,headers,self.styles['header'])
        for i,row in enumerate(self.rows,1):
            sheet.write_row(i,0,(row.setting,*row.key),self.styles['cell'])
            sheet.write_number(i,5,row.value,self.styles['raw'])
            if row.numerator is not None and row.denominator is not None and row.denominator>0:
                expected = self.dataset.scale*row.numerator/row.denominator
                if not np.isclose(expected,row.value,atol=1e-4,rtol=1e-7):
                    raise ExportProblem(f'분자·분모와 값이 일치하지 않습니다: {row}')
                sheet.write_number(i,6,row.numerator,self.styles['raw'])
                sheet.write_number(i,7,row.denominator,self.styles['raw'])
                sheet.write(i,5,f'={self.dataset.scale}*G{i+1}/H{i+1}',self.styles['raw'],row.value)
            sheet.write_string(i,8,row.source)
            sheet.write_string(i,9,row.detail)
            self.index[row.setting,row.key].append(i+1)
            self.values[row.setting,row.key].append(row.value)
            self.ranges[row.setting].append(i+1)
        sheet.freeze_panes(1,5)
        sheet.autofilter(0,0,len(self.rows),9)
        sheet.set_column(0,0,27)
        sheet.set_column(1,4,12)
        sheet.set_column(5,7,17)
        sheet.set_column(8,8,52)
        sheet.set_column(9,9,75)

    def sample_cell(self, sheet: Worksheet, position: tuple[int,int], identity: tuple[str,RunKey]) -> float | None:
        r,c = position
        indices,values = self.index.get(identity,[]),self.values.get(identity,[])
        if not indices:
            sheet.write_blank(r,c,None,self.styles['number'])
            return None
        reference = f"'원자료'!F{indices[0]}:F{indices[-1]}"
        if len(indices)==1:
            formula,value = f"='원자료'!F{indices[0]}",values[0]
        else:
            operation = 'MEDIAN' if self.dataset.grid_median else 'SUM'
            value = float(np.median(values)) if self.dataset.grid_median else float(np.sum(values))
            formula = f'={operation}({reference})'
        sheet.write(r,c,formula,self.styles['number'],value)
        return value

    def panels(self) -> None:
        for mode in self.mode_names:
            if mode=='Legacy':
                continue
            sheet = self.book.add_worksheet(mode)
            sheet.merge_range(0,0,0,39,self.dataset.title+' ('+self.dataset.unit+')',self.styles['title'])
            sheet.merge_range(1,0,1,39,self.dataset.population,self.styles['note'])
            for row_index,vehicles in enumerate((70,100,120)):
                for col_index,speed in enumerate((10,20,30)):
                    top,left = 3+18*row_index,14*col_index
                    sheet.merge_range(top,left,top,left+11,f'차량대수 {vehicles}, 최고 속력 {speed} km/h',self.styles['scenario'])
                    sheet.merge_range(top+1,left,top+2,left,'No',self.styles['header'])
                    sheet.merge_range(top+1,left+1,top+2,left+1,'nodeId',self.styles['header'])
                    sheet.merge_range(top+1,left+2,top+1,left+6,'Legacy',self.styles['header'])
                    sheet.merge_range(top+1,left+7,top+1,left+11,mode,self.styles['header'])
                    sheet.write_row(top+2,left+2,[f'Seed{i}' for i in range(1,6)]*2,self.styles['header'])
                    columns: list[list[float]] = [[] for _ in range(10)]
                    for node in range(10):
                        sheet.write(top+3+node,left,node+1,self.styles['cell'])
                        sheet.write(top+3+node,left+1,f'node[{node}]',self.styles['cell'])
                        for group,setting in enumerate(('Legacy',mode)):
                            for seed in range(1,6):
                                offset = group*5+seed-1
                                value = self.sample_cell(sheet,(top+3+node,left+2+offset),(setting,RunKey(vehicles,speed,node,seed)))
                                if value is not None:
                                    columns[offset].append(value)
                    sheet.merge_range(top+13,left,top+13,left+1,'AVG',self.styles['avg'])
                    for offset,values in enumerate(columns):
                        column = xl_col_to_name(left+2+offset)
                        formula = f'=IF(COUNT({column}{top+4}:{column}{top+13})>0,AVERAGE({column}{top+4}:{column}{top+13}),"")'
                        sheet.write(top+13,left+2+offset,formula,self.styles['avg'],float(np.mean(values)) if values else '')
            sheet.merge_range(56,0,57,39,'AVG는 해당 열의 숫자가 있는 목적지 차량들의 산술평균입니다. 빈칸은 분석 대상 아님·분모 0·자료 미포함을 뜻하며 실제 0과 구분합니다. 논문 집계는 별도 시트를 확인하십시오.',self.styles['note'])
            if self.dataset.grid_median:
                sheet.merge_range(59,0,60,39,'한 실행에 여러 탐색 요청이 있으면 표에는 그 요청들의 중앙값을 표시합니다. 하단 AVG는 이 중앙값들의 평균이며 논문 전체 요청의 중앙값과 다릅니다.',self.styles['note'])
            sheet.set_column(0,39,12)
            for left in (0,14,28):
                sheet.set_column(left,left,5)
                sheet.set_column(left+1,left+1,11)
                sheet.set_column(left+12,left+13,3)
            sheet.freeze_panes(6,2)
            sheet.hide_gridlines(2)
            sheet.set_landscape()
            sheet.set_paper(8)
            sheet.fit_to_pages(1,1)
            sheet.print_area(0,0,60,39)
            sheet.set_zoom(70)

    def summary(self) -> None:
        sheet = self.summary_sheet
        headers = ('설정','유효 표본 수','실행 수','AVG','MEDIAN','SUM','합산 분자','합산 분모','논문 집계','Seed1 AVG','Seed2 AVG','Seed3 AVG','Seed4 AVG','Seed5 AVG','증가','동일','감소','40%p 이상 증가')
        sheet.write_row(0,0,headers,self.styles['header'])
        for i,mode in enumerate(self.mode_names,1):
            rows = [row for row in self.rows if row.setting==mode]
            first,last = self.ranges[mode][0],self.ranges[mode][-1]
            values = np.array([row.value for row in rows],dtype=float)
            ref = f"'원자료'!F{first}:F{last}"
            nums,dens = sum(row.numerator or 0 for row in rows),sum(row.denominator or 0 for row in rows)
            sheet.write(i,0,mode)
            sheet.write(i,1,f'=COUNT({ref})',None,len(rows))
            sheet.write_number(i,2,len({row.key for row in rows}))
            for column,operation,value in ((3,'AVERAGE',float(values.mean())),(4,'MEDIAN',float(np.median(values))),(5,'SUM',float(values.sum()))):
                sheet.write(i,column,f'={operation}({ref})',self.styles['raw'],value)
            sheet.write(i,6,f"=SUM('원자료'!G{first}:G{last})",self.styles['raw'],nums)
            sheet.write(i,7,f"=SUM('원자료'!H{first}:H{last})",self.styles['raw'],dens)
            match self.dataset.method:
                case 'mean': formula,value = f'=D{i+1}',float(values.mean())
                case 'median': formula,value = f'=E{i+1}',float(np.median(values))
                case 'sum': formula,value = f'=F{i+1}',float(values.sum())
                case 'pooled': formula,value = f'=IF(H{i+1}>0,{self.dataset.scale}*G{i+1}/H{i+1},"")',self.dataset.scale*nums/dens if dens>0 else ''
                case unreachable: assert_never(unreachable)
            sheet.write(i,8,formula,self.styles['raw'],value)
            for seed in range(1,6):
                subset = [row.value for row in rows if row.key.seed==seed]
                formula = f"=IF(COUNTIFS('원자료'!E{first}:E{last},{seed})>0,AVERAGEIF('원자료'!E{first}:E{last},{seed},{ref}),\"\")"
                sheet.write(i,8+seed,formula,self.styles['raw'],float(np.mean(subset)) if subset else '')
            if self.dataset.filename.startswith('02_'):
                for col,criterion in enumerate(('>0','=0','<0','>=40'),14):
                    count = int({'>0':np.sum(values>0),'=0':np.sum(values==0),'<0':np.sum(values<0),'>=40':np.sum(values>=40)}.get(criterion,np.sum(values>0)))
                    sheet.write(i,col,f'=COUNTIF({ref},"{criterion}")',None,count)
        sheet.write(len(self.mode_names)+3,0,'집계 방법: '+self.dataset.method+'; 단위: '+self.dataset.unit)
        sheet.write(len(self.mode_names)+4,0,self.dataset.population)
        sheet.write(len(self.mode_names)+5,0,'AVG와 논문 집계는 합산 비율·요청 단위 중앙값에서 서로 다를 수 있습니다.')
        sheet.set_column(0,0,31)
        sheet.set_column(1,17,18)
        sheet.freeze_panes(1,1)

    def save(self) -> Path:
        self.raw()
        self.panels()
        self.summary()
        from .comparisons import add_relative_reductions, add_linked_summary, add_pdr_statistics
        add_relative_reductions(self)
        add_linked_summary(self)
        add_pdr_statistics(self)
        if self.dataset.filename=='01_PDR':
            from .comparisons import add_q_comparisons
            add_q_comparisons(self)
        self.book.close()
        return self.path
