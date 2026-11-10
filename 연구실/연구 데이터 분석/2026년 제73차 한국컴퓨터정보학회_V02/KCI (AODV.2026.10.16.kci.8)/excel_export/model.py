from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Final, Literal, NamedTuple, Protocol
from collections.abc import Mapping, Sequence
from xlsxwriter import Workbook
from xlsxwriter.format import Format
from xlsxwriter.worksheet import Worksheet
import numpy as np

Aggregation = Literal['mean','pooled','median','sum']
Scalar = str | int | float | None

class ExportProblem(RuntimeError):
    pass

class RunKey(NamedTuple):
    vehicles: int
    speed: int
    node: int
    seed: int

GRID: Final = tuple(RunKey(*key) for key in product((70,100,120),(10,20,30),range(10),range(1,6)))
BOUNDS: Final = ((0,21,0,19),(21,33,19,32),(33,44,32,43),(44,101,43,101))

@dataclass(frozen=True,slots=True)
class Config:
    name: str
    folder: Path

@dataclass(frozen=True,slots=True)
class Settings:
    data_root: Path
    previous: Path
    output: Path

@dataclass(frozen=True,slots=True)
class Sample:
    setting: str
    key: RunKey
    value: float
    numerator: float | None = None
    denominator: float | None = None
    source: str = ''
    detail: str = ''

@dataclass(frozen=True,slots=True)
class Dataset:
    filename: str
    title: str
    unit: str
    population: str
    method: Aggregation
    samples: tuple[Sample,...]
    grid_median: bool = False
    scale: float = 100.0

@dataclass(frozen=True,slots=True)
class RawTable:
    title: str
    columns: tuple[str,...]
    rows: tuple[tuple[Scalar,...],...]

@dataclass(frozen=True,slots=True)
class Result:
    sent: int
    received: int
    pdr: float
    rtt: float
    source: str

@dataclass(frozen=True,slots=True)
class ResultSet:
    configs: tuple[Config,...]
    records: dict[tuple[str,RunKey],Result]


def configurations(root: Path) -> tuple[Config,...]:
    groups = [Config('Legacy',root/'legacy')]
    for mode in ('RREQ','RREP','RREQ+RREP'):
        for q,(rl,rh,ql,qh) in enumerate(BOUNDS,1):
            folders = {'RREQ':root/'rreq'/f'rreq#{ql}_{qh}_result','RREP':root/'rrep'/f'rrep#{rl}_{rh}_result','RREQ+RREP':root/'full'/f'full#{rl}, {rh}_{ql}, {qh}_result'/'q_sweep'}
            groups.append(Config(f'LLC({mode},q_{q})',folders[mode]))
    groups.append(Config('Random(noCBP)',root/'random_no_cbp'))
    return tuple(groups)


def folder_for(config: Config, key: RunKey) -> Path:
    return config.folder/f'd#{key.vehicles}'/f'v#{key.speed}'/str(key.node)/f'seed#{key.seed}'


def finite_samples(dataset: Dataset) -> tuple[Sample,...]:
    return tuple(row for row in dataset.samples if np.isfinite(row.value))


class ExportView(Protocol):
    @property
    def dataset(self) -> Dataset: ...
    @property
    def book(self) -> Workbook: ...
    @property
    def styles(self) -> Mapping[str,Format]: ...
    @property
    def rows(self) -> Sequence[Sample]: ...
    @property
    def mode_names(self) -> Sequence[str]: ...
    @property
    def ranges(self) -> Mapping[str,Sequence[int]]: ...
    @property
    def values(self) -> Mapping[tuple[str,RunKey],Sequence[float]]: ...
    @property
    def index(self) -> Mapping[tuple[str,RunKey],Sequence[int]]: ...
    @property
    def summary_sheet(self) -> Worksheet: ...
