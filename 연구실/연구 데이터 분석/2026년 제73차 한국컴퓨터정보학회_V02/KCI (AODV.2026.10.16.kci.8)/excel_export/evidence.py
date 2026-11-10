from html import unescape
from pathlib import Path
import re
import duckdb
import numpy as np
import xlsxwriter
from .model import RawTable, Settings


def write_tables(path: Path, tables: tuple[RawTable,...]) -> Path:
    with xlsxwriter.Workbook(str(path)) as book:
        header = book.add_format({'font_name':'Arial','font_size':10,'bold':True,'bg_color':'#D9E5F2','border':1,'text_wrap':True})
        body = book.add_format({'font_name':'Arial','font_size':10,'valign':'top','text_wrap':True,'num_format':'0.000000'})
        for table in tables:
            sheet = book.add_worksheet(table.title[:31])
            sheet.write_row(0,0,table.columns,header)
            for i,row in enumerate(table.rows,1):
                sheet.write_row(i,0,row,body)
            sheet.set_column(0,len(table.columns)-1,22)
            sheet.freeze_panes(1,1)
            sheet.autofilter(0,0,len(table.rows),len(table.columns)-1)
    return path


def json_table(path: Path, title: str) -> RawTable:
    with duckdb.connect() as db:
        query = 'SELECT key[1] AS vehicles,key[2] AS speed,key[3] AS nodeId,key[4] AS seed,* EXCLUDE(key) FROM read_json_auto(?)'
        relation = db.execute(query,[str(path)])
        columns = tuple(str(column[0]) for column in relation.description)
        rows = tuple(tuple(row) for row in relation.fetchall())
    return RawTable(title,columns,rows)


def manuscript_tables(path: Path) -> tuple[RawTable,...]:
    text = path.read_text(encoding='utf-8')
    blocks = re.findall(r'(?:^\|.*\n?)+',text,flags=re.MULTILINE)
    tables = []
    for block in blocks:
        if '![](' in block:
            continue
        lines = block.strip().splitlines()
        clean = lambda value:unescape(re.sub(r'<[^>]*>','',re.sub(r'<br\s*/?>','\n',value))).strip()
        cells = tuple(tuple(clean(value) for value in line.strip('|').split('|')) for line in lines if not re.match(r'^\|\s*[-:]',line))
        tables.append(RawTable(f'Table{len(tables)+1}',cells[0],cells[1:]))
    return tuple(tables)


def evidence_workbooks(settings: Settings, manuscript: Path) -> tuple[Path,...]:
    paths = [write_tables(settings.output/'90_논문에사용한표.xlsx',manuscript_tables(manuscript))]
    p = settings.previous
    paths.append(write_tables(settings.output/'91_경로선택_Ping연결_상세기록.xlsx',(json_table(p/'v8/route_ping_outputs/review28_context_evidence.jsonl','경로선택과Ping연결'),)))
    paths.append(write_tables(settings.output/'92_PDR감소_요청별상세기록.xlsx',(json_table(p/'v9/cause_outputs/review29_packet_evidence.jsonl','같은요청의응답과경로'),json_table(p/'v9/cause_outputs/review29_attempt_evidence.jsonl','각경로탐색의응답차단'))))
    paths.append(write_tables(settings.output/'93_재탐색복구_상세기록.xlsx',(json_table(p/'v9/recovery_outputs/review29_recovery_evidence.jsonl','경로선택과다시사용불가'),)))
    return tuple(paths)


def saved_audit(settings: Settings) -> RawTable:
    audit = []
    for rel in ('v8/route_ping_outputs/review28_all_metrics.npz','v9/cause_outputs/review29_summary.npz'):
        path = settings.previous/rel
        with np.load(path,allow_pickle=False) as data:
            for field in data.files:
                audit.append((str(path),field,str(data[field].shape)))
    return RawTable('출처파일',('파일','저장 배열','배열 모양'),tuple(audit))


def input_manifest(settings: Settings) -> RawTable:
    with duckdb.connect() as db:
        rows = db.execute('SELECT original,relative,bytes,sha256 FROM read_json_auto(?)',[str(settings.previous/'source_manifest.json')]).fetchall()
    return RawTable('원본과보존사본',('원본 파일','보존한 파일','바이트 수','SHA256'),tuple(tuple(row) for row in rows))
