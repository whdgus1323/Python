from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import final

import polars as pl
from PySide6.QtCore import QEasingCurve, QPropertyAnimation, Qt, QUrl
from PySide6.QtGui import QAction, QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QTableView,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from keyword_sourcing.analysis import (
    AnalysisResult,
    AnalysisSettings,
    InputSchemaError,
    analyze_products,
)
from keyword_sourcing.files import (
    EmptyInputDirectoryError,
    UnsupportedInputError,
    export_recommendations,
    load_product_folder,
    load_products,
)
from keyword_sourcing.table_model import ProductTableModel

DASHBOARD_STYLE = """
QMainWindow { background: #07100f; color: #edf8f4; font-family: 'Plus Jakarta Sans', Pretendard, 'Segoe UI'; }
QWidget { color: #d8e8e2; }
QLabel { color: #d8e8e2; }
QMenuBar { background: #07100f; color: #8ea69d; padding: 8px 16px; border: none; }
QMenuBar::item:selected, QMenu { background: #142926; color: #eefaf5; }
QStatusBar { background: #091513; color: #8ea69d; border-top: 1px solid #203d35;
padding: 4px 14px; }
QLabel#eyebrow { color: #5de5ac; font-size: 11px; font-weight: 600; letter-spacing: 1.4px; }
QLabel#title { color: #f3fbf7; font-size: 31px; font-weight: 600; letter-spacing: -0.6px; }
QLabel#subtitle { color: #96aea4; font-size: 13px; }
QFrame#sidebar { background: #050c0b; border-right: 1px solid #17342d; }
QLabel#brand { color: #dfffee; background: qradialgradient(cx:0.2, cy:0.2, radius:1.2, stop:0 #38d996, stop:0.45 #167853, stop:1 #0b2c21); border: 1px solid #58eaaa; border-radius: 13px; padding: 10px; font-size: 15px; font-weight: 700; }
QLabel#brandName { color: #effbf6; font-size: 15px; font-weight: 600; letter-spacing: 1px; }
QPushButton#nav, QPushButton#navActive { text-align: left; padding: 12px 13px; border: 1px solid transparent; border-radius: 10px; }
QPushButton#nav { color: #829d92; background: transparent; }
QPushButton#nav:hover { color: #e9f8f1; background: #10251f; border-color: #1c493a; }
QPushButton#navActive { color: #a5f5ca; background: #11382b; border-color: #237a55; }
QLabel#sidebarNote, QLabel#metricLabel { color: #718f83; font-size: 10px; font-weight: 600; letter-spacing: 1.2px; }
QFrame#metric { background: #0c1a17; border: 1px solid #203f36; border-radius: 16px; min-height: 96px; }
QFrame#metric:hover { background: #10231e; border-color: #2f765a; }
QLabel#metricValue { color: #f3fbf7; font-size: 23px; font-weight: 600; }
QLabel#metricDetail { color: #92aaa0; font-size: 11px; }
QWidget#header { background: qradialgradient(cx:0.94, cy:0.08, radius:0.65, stop:0 #164c37, stop:0.35 #0e251d, stop:1 #0b1916); border: 1px solid #245342; border-radius: 19px; }
QWidget#settings { background: #0b1916; border: 1px solid #203f36; border-radius: 16px; }
QPushButton { background: #122822; color: #c9dbd3; border: 1px solid #23493d;
border-radius: 10px; padding: 10px 15px; font-weight: 600; }
QPushButton:hover { background: #19382f; border-color: #47d38e; color: #f2fcf7; }
QPushButton:pressed { background: #0b1815; }
QPushButton:disabled { background: #0a1512; color: #51675e; border-color: #183027; }
QPushButton#primary { background: #39d996; color: #052116; border-color: #5aefae; }
QPushButton#primary:hover { background: #6bedb7; border-color: #9dffcd; }
QPushButton#primary:disabled { background: #0d211a; color: #547166; border-color: #1e4c39; }
QDoubleSpinBox, QSpinBox { background: #122822; color: #edf9f3; border: 1px solid #23493d;
border-radius: 8px; padding: 7px; min-height: 20px; }
QTabWidget::pane { background: #0b1916; border: 1px solid #203f36; border-radius: 16px; top: -1px; }
QTabBar::tab { background: transparent; color: #8ba399; padding: 11px 18px; margin-right: 5px; }
QTabBar::tab:hover { color: #d5eee2; }
QTabBar::tab:selected { color: #69ebb1; border-bottom: 2px solid #52e3a3; }
QFrame#emptyState { background: qradialgradient(cx:0.88, cy:0.13, radius:0.7, stop:0 #133e2e, stop:0.35 #0d211b, stop:1 #0a1714); border: 1px solid #265643; border-radius: 14px; }
QLabel#emptyStep { color: #6deeb4; background: #12382b; border: 1px solid #2a7b59; border-radius: 11px; padding: 6px 10px; font-size: 10px; font-weight: 600; letter-spacing: 1.1px; }
QLabel#emptyTitle { color: #effbf5; font-size: 22px; font-weight: 600; letter-spacing: -0.3px; }
QLabel#emptyDescription { color: #91aca0; font-size: 13px; line-height: 1.5; }
QLabel#emptyHint { color: #c2dbd1; background: #10241e; border: 1px solid #1f513e; border-radius: 10px; padding: 12px; font-size: 12px; }
QTableView { color: #dcebe5; gridline-color: #1b3830; border: none;
selection-background-color: #1d6048; selection-color: #f3fff8; }
QHeaderView::section { background: #10251f; color: #91aaa0; border: none;
border-bottom: 1px solid #203f36; padding: 10px; font-weight: 600; }
QTableCornerButton::section { background: #10251f; border: none; border-bottom: 1px solid #203f36; }
QSplitter::handle { background: #16382d; }
QSplitter::handle:hover { background: #3bbf7d; }
QScrollBar:vertical { background: #0b1916; width: 10px; }
QScrollBar::handle:vertical { background: #2f5e4d; border-radius: 5px; min-height: 28px; }
"""


@final
class AnalysisState:
    def __init__(self) -> None:
        self.products: pl.DataFrame | None = None
        self.result: AnalysisResult | None = None


@final
class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self._state: AnalysisState = AnalysisState()
        self._ratio: QDoubleSpinBox = QDoubleSpinBox()
        self._ratio_label: QLabel = QLabel()
        self._review_floor: QSpinBox = QSpinBox()
        self._review_ceiling: QSpinBox = QSpinBox()
        self._recommendation_limit: QSpinBox = QSpinBox()
        self._keyword_table: QTableView = QTableView()
        self._recommendation_table: QTableView = QTableView()
        self._product_table: QTableView = QTableView()
        self._keyword_table.setModel(ProductTableModel(pl.DataFrame()))
        self._recommendation_table.setModel(ProductTableModel(pl.DataFrame()))
        self._product_table.setModel(ProductTableModel(pl.DataFrame()))
        self._tabs: QTabWidget = QTabWidget()
        self._keyword_stack: QStackedWidget = QStackedWidget()
        self._recommendation_stack: QStackedWidget = QStackedWidget()
        self._product_stack: QStackedWidget = QStackedWidget()
        self._navigation_buttons: list[QPushButton] = []
        self._analyze_button: QPushButton = QPushButton("분석 실행")
        self._export_button: QPushButton = QPushButton("추천 결과 저장")
        self._link_button: QPushButton = QPushButton("선택 상품 링크 열기")
        self._export_action: QAction | None = None
        self._data_metric_value: QLabel = QLabel("0")
        self._data_metric_detail: QLabel = QLabel("분석 파일을 불러오세요")
        self._ratio_metric_value: QLabel = QLabel("40%")
        self._candidate_metric_value: QLabel = QLabel("—")
        self._candidate_metric_detail: QLabel = QLabel("분석을 기다리는 중")
        self._intro_animation: QPropertyAnimation | None = None
        self._setup_window()

    def _setup_window(self) -> None:
        self.setWindowTitle("키워드 서칭 프로")
        self.resize(1440, 860)
        self.setStyleSheet(DASHBOARD_STYLE)
        self.setCentralWidget(self._create_content())
        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("상품 검색 결과 CSV 또는 Excel 파일을 불러오세요.")
        self._create_actions()

    def _create_content(self) -> QWidget:
        container = QWidget(self)
        root = QHBoxLayout(container)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._create_sidebar())
        content = QWidget()
        root.addWidget(content, 1)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(16)
        header = QWidget()
        header.setObjectName("header")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(24, 18, 24, 18)
        eyebrow = QLabel("MARKET INTELLIGENCE · 01")
        eyebrow.setObjectName("eyebrow")
        header_layout.addWidget(eyebrow)
        title = QLabel("키워드 서칭 프로")
        title.setObjectName("title")
        header_layout.addWidget(title)
        description = "해외배송 비율과 리뷰 경쟁도를 기준으로 구매대행 벤치마킹 후보를 찾습니다."
        subtitle = QLabel(description)
        subtitle.setObjectName("subtitle")
        header_layout.addWidget(subtitle)
        layout.addWidget(header)
        self._animate_intro(header)
        metrics = QHBoxLayout()
        metrics.setSpacing(12)
        data_card, self._data_metric_value, self._data_metric_detail = self._metric_card(
            "분석 데이터", "0", "분석 파일을 불러오세요"
        )
        ratio_card, self._ratio_metric_value, _ = self._metric_card(
            "진입 기준", "40%", "해외배송 비율"
        )
        candidate_card, self._candidate_metric_value, self._candidate_metric_detail = (
            self._metric_card("벤치마킹 후보", "—", "분석을 기다리는 중")
        )
        for card in (data_card, ratio_card, candidate_card):
            metrics.addWidget(card)
        layout.addLayout(metrics)
        controls = QHBoxLayout()
        file_button = QPushButton("분석 파일 열기")
        file_button.clicked.connect(self._open_file)
        folder_button = QPushButton("카테고리 폴더 열기")
        folder_button.clicked.connect(self._open_folder)
        self._analyze_button.setObjectName("primary")
        self._analyze_button.setEnabled(False)
        self._analyze_button.clicked.connect(self._run_analysis)
        self._export_button.setEnabled(False)
        self._export_button.clicked.connect(self._export_recommendations)
        self._link_button.setEnabled(False)
        self._link_button.clicked.connect(self._open_selected_link)
        for button in (
            file_button,
            folder_button,
            self._analyze_button,
            self._export_button,
            self._link_button,
        ):
            controls.addWidget(button)
        controls.addStretch()
        layout.addLayout(controls)
        splitter = QSplitter()
        splitter.addWidget(self._create_settings_panel())
        splitter.addWidget(self._create_results_tabs())
        splitter.setSizes([300, 1140])
        layout.addWidget(splitter, 1)
        return container

    def _animate_intro(self, header: QWidget) -> None:
        effect = QGraphicsOpacityEffect(header)
        effect.setOpacity(0.0)
        header.setGraphicsEffect(effect)
        animation = QPropertyAnimation(effect, b"opacity", self)
        animation.setDuration(520)
        animation.setStartValue(0.0)
        animation.setEndValue(1.0)
        animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        animation.start()
        self._intro_animation = animation

    def _create_sidebar(self) -> QFrame:
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(236)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(20, 28, 20, 24)
        layout.setSpacing(8)
        brand = QLabel("KSP")
        brand.setObjectName("brand")
        layout.addWidget(brand)
        name = QLabel("KEYWORD\nSOURCING")
        name.setObjectName("brandName")
        layout.addWidget(name)
        layout.addSpacing(36)
        for index, item in enumerate(("키워드 분석", "벤치마킹 후보", "상품 데이터")):
            button = QPushButton(item)
            button.setObjectName("navActive" if index == 0 else "nav")
            button.setFlat(True)
            button.clicked.connect(partial(self._select_result_tab, index))
            self._navigation_buttons.append(button)
            layout.addWidget(button)
        layout.addStretch()
        note = QLabel("LOCAL ANALYSIS\n데이터는 이 PC에서만 처리됩니다")
        note.setObjectName("sidebarNote")
        layout.addWidget(note)
        return sidebar

    def _metric_card(self, label: str, value: str, detail: str) -> tuple[QFrame, QLabel, QLabel]:
        card = QFrame()
        card.setObjectName("metric")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 15, 18, 15)
        label_widget = QLabel(label.upper())
        label_widget.setObjectName("metricLabel")
        value_widget = QLabel(value)
        value_widget.setObjectName("metricValue")
        detail_widget = QLabel(detail)
        detail_widget.setObjectName("metricDetail")
        layout.addWidget(label_widget)
        layout.addWidget(value_widget)
        layout.addWidget(detail_widget)
        return card, value_widget, detail_widget

    def _create_settings_panel(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("settings")
        form = QFormLayout(panel)
        form.setContentsMargins(20, 20, 20, 20)
        form.setVerticalSpacing(14)
        self._ratio.setRange(0.0, 1.0)
        self._ratio.setSingleStep(0.05)
        self._ratio.setValue(0.4)
        self._ratio.valueChanged.connect(self._update_ratio_label)
        self._review_floor.setRange(0, 1000000)
        self._review_floor.setValue(10)
        self._review_ceiling.setRange(0, 1000000)
        self._review_ceiling.setValue(300)
        self._recommendation_limit.setRange(1, 20)
        self._recommendation_limit.setValue(3)
        self._update_ratio_label(self._ratio.value())
        form.addRow(self._ratio_label, self._ratio)
        form.addRow("최소 리뷰 수", self._review_floor)
        form.addRow("최대 리뷰 수", self._review_ceiling)
        form.addRow("키워드별 추천 수", self._recommendation_limit)
        form.addRow(QLabel("노랑: 해외배송\n초록: 벤치마킹 후보"))
        return panel

    def _update_ratio_label(self, value: float) -> None:
        self._ratio_label.setText(f"최소 해외배송 비율 ({value:.0%})")
        self._ratio_metric_value.setText(f"{value:.0%}")

    def _create_results_tabs(self) -> QTabWidget:
        self._keyword_stack.addWidget(
            self._create_empty_state(
                "01 / PRODUCT DATA",
                "먼저 상품 데이터를 불러오세요.",
                "CSV 또는 Excel 파일을 가져오면 전체 상품 목록과 현재 분석 상태가 표시됩니다.",
                "상단의 분석 파일 열기 또는 카테고리 폴더 열기를 선택하세요.",
            )
        )
        self._keyword_stack.addWidget(
            self._create_empty_state(
                "02 / KEYWORD ANALYSIS",
                "상품 데이터를 분석할 준비가 됐습니다.",
                "분석 실행을 누르면 키워드별 해외배송 비율과 리뷰 경쟁도가 나타납니다.",
                "01  파일 불러오기 완료  →  02  분석 실행",
            )
        )
        self._keyword_stack.addWidget(self._keyword_table)
        self._recommendation_stack.addWidget(
            self._create_empty_state(
                "03 / BENCHMARKING",
                "아직 후보가 없습니다.",
                "분석이 끝나면 기준을 충족한 상품을 벤치마킹 후보로 정리합니다.",
                "키워드 분석 완료 후 자동으로 후보가 생성됩니다.",
            )
        )
        self._recommendation_stack.addWidget(self._recommendation_table)
        self._product_stack.addWidget(
            self._create_empty_state(
                "01 / PRODUCT DATA",
                "먼저 상품 데이터를 불러오세요.",
                "CSV 또는 Excel 파일을 가져오면 전체 상품 목록과 현재 분석 상태가 이곳에 표시됩니다.",
                "상단의 분석 파일 열기 또는 카테고리 폴더 열기를 선택하세요.",
            )
        )
        self._product_stack.addWidget(self._product_table)
        self._tabs.addTab(self._keyword_stack, "키워드 분석")
        self._tabs.addTab(self._recommendation_stack, "벤치마킹 후보")
        self._tabs.addTab(self._product_stack, "전체 상품")
        self._tabs.currentChanged.connect(self._sync_sidebar_selection)
        self._tabs.currentChanged.connect(self._update_link_action_state)
        return self._tabs

    def _create_empty_state(
        self, step: str, title: str, description: str, hint: str
    ) -> QFrame:
        panel = QFrame()
        panel.setObjectName("emptyState")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(38, 34, 38, 34)
        layout.setSpacing(14)
        step_label = QLabel(step)
        step_label.setObjectName("emptyStep")
        step_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        title_label = QLabel(title)
        title_label.setObjectName("emptyTitle")
        description_label = QLabel(description)
        description_label.setObjectName("emptyDescription")
        description_label.setWordWrap(True)
        hint_label = QLabel(hint)
        hint_label.setObjectName("emptyHint")
        hint_label.setWordWrap(True)
        layout.addWidget(step_label)
        layout.addSpacing(8)
        layout.addWidget(title_label)
        layout.addWidget(description_label)
        layout.addSpacing(12)
        layout.addWidget(hint_label)
        layout.addStretch()
        return panel

    def _select_result_tab(self, index: int) -> None:
        self._tabs.setCurrentIndex(index)

    @property
    def selected_result_tab(self) -> int:
        return self._tabs.currentIndex()

    def _sync_sidebar_selection(self, selected_index: int) -> None:
        for index, button in enumerate(self._navigation_buttons):
            object_name = "navActive" if index == selected_index else "nav"
            if button.objectName() != object_name:
                button.setObjectName(object_name)
                button.style().unpolish(button)
                button.style().polish(button)

    def _create_actions(self) -> None:
        menu = self.menuBar().addMenu("파일")
        open_file_action = QAction("분석 파일 열기", self)
        open_file_action.triggered.connect(self._open_file)
        open_folder_action = QAction("카테고리 폴더 열기", self)
        open_folder_action.triggered.connect(self._open_folder)
        self._export_action = QAction("추천 결과 저장", self)
        self._export_action.setEnabled(False)
        self._export_action.triggered.connect(self._export_recommendations)
        menu.addActions((open_file_action, open_folder_action, self._export_action))

    def _open_file(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self, "분석 파일 열기", "", "데이터 파일 (*.csv *.xlsx *.xls)"
        )
        if filename:
            self.load_data(Path(filename))

    def _open_folder(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "카테고리 폴더 열기")
        if directory:
            self.load_data(Path(directory), from_folder=True)

    def load_data(self, path: Path, from_folder: bool = False) -> None:
        try:
            products = load_product_folder(path) if from_folder else load_products(path)
        except (EmptyInputDirectoryError, OSError, UnsupportedInputError) as error:
            self._show_error(str(error))
            return
        self._state.products = products
        self._state.result = None
        self._product_table.setModel(ProductTableModel(products))
        self._resize_tables()
        self._data_metric_value.setText(f"{products.height:,}")
        self._data_metric_detail.setText("상품 불러오기 완료")
        self._candidate_metric_value.setText("—")
        self._candidate_metric_detail.setText("분석 실행을 기다리는 중")
        self._analyze_button.setEnabled(True)
        self._export_button.setEnabled(False)
        self._set_export_enabled(False)
        self._link_button.setEnabled(False)
        self._keyword_stack.setCurrentIndex(1)
        self._recommendation_stack.setCurrentIndex(0)
        self._product_stack.setCurrentIndex(1)
        self._connect_selection_tracking()
        self._tabs.setCurrentIndex(2)
        self.statusBar().showMessage(
            f"{products.height:,}개 상품을 불러왔습니다. 분석 실행을 누르세요."
        )

    def _run_analysis(self) -> None:
        if self._state.products is None:
            self._show_error("먼저 분석 파일 또는 카테고리 폴더를 선택하세요.")
            return
        try:
            result = analyze_products(self._state.products, self._settings())
        except (InputSchemaError, pl.exceptions.PolarsError) as error:
            self._show_error(str(error))
            return
        self._state.result = result
        self._keyword_table.setModel(ProductTableModel(result.keywords))
        candidate_model = ProductTableModel(result.recommendations, mark_candidates=True)
        self._recommendation_table.setModel(candidate_model)
        self._product_table.setModel(ProductTableModel(result.products))
        self._resize_tables()
        self._candidate_metric_value.setText(f"{result.recommendations.height:,}")
        self._candidate_metric_detail.setText("추천 후보 생성 완료")
        self._set_export_enabled(True)
        self._keyword_stack.setCurrentIndex(2)
        self._recommendation_stack.setCurrentIndex(1)
        self._product_stack.setCurrentIndex(1)
        self._connect_selection_tracking()
        self._tabs.setCurrentIndex(0)
        message = (
            f"{result.keywords.height:,}개 키워드 분석 완료, "
            + f"{result.recommendations.height:,}개 후보를 추천합니다."
        )
        self.statusBar().showMessage(message)

    def _settings(self) -> AnalysisSettings:
        return AnalysisSettings(
            overseas_ratio_threshold=self._ratio.value(),
            review_floor=self._review_floor.value(),
            review_ceiling=self._review_ceiling.value(),
            recommendation_limit=self._recommendation_limit.value(),
        )

    @property
    def dashboard_data_count(self) -> str:
        return self._data_metric_value.text()

    def _export_recommendations(self) -> None:
        if self._state.result is None:
            self._show_error("분석 실행 후 추천 결과를 저장할 수 있습니다.")
            return
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "추천 결과 저장",
            "benchmark_candidates.xlsx",
            "Excel 파일 (*.xlsx);;CSV 파일 (*.csv)",
        )
        if not filename:
            return
        path = Path(filename)
        if not path.suffix:
            path = path.with_suffix(".xlsx")
        try:
            export_recommendations(path, self._state.result)
        except (OSError, UnsupportedInputError) as error:
            self._show_error(str(error))
            return
        self.statusBar().showMessage(f"추천 결과를 저장했습니다: {path.name}")

    def _open_selected_link(self) -> None:
        table = self._selected_product_table()
        if table is None:
            self._show_error("상품 데이터 또는 벤치마킹 후보 탭에서 상품 행을 선택하세요.")
            return
        selected_rows = table.selectionModel().selectedRows()
        if not selected_rows:
            self._show_error("링크를 열 상품 행을 먼저 선택하세요.")
            return
        model = table.model()
        match model:
            case ProductTableModel() as product_model:
                url_column = self._url_column(product_model)
            case _:
                self._show_error("선택한 상품의 링크를 확인할 수 없습니다.")
                return
        if url_column is None:
            self._show_error("상품 링크 열이 없습니다.")
            return
        product_url = product_model.data(product_model.index(selected_rows[0].row(), url_column))
        if isinstance(product_url, str):
            QDesktopServices.openUrl(QUrl(product_url))

    def _selected_product_table(self) -> QTableView | None:
        current_index = self._tabs.currentIndex()
        if current_index == 1:
            return self._recommendation_table
        if current_index == 2:
            return self._product_table
        return None

    def _set_export_enabled(self, enabled: bool) -> None:
        self._export_button.setEnabled(enabled)
        if self._export_action is not None:
            self._export_action.setEnabled(enabled)

    def _connect_selection_tracking(self) -> None:
        for table in (self._recommendation_table, self._product_table):
            table.selectionModel().selectionChanged.connect(self._update_link_action_state)
        self._update_link_action_state()

    def _update_link_action_state(self) -> None:
        table = self._selected_product_table()
        if table is None:
            self._link_button.setEnabled(False)
            return
        selection_model = table.selectionModel()
        model = table.model()
        has_link = isinstance(model, ProductTableModel) and self._url_column(model) is not None
        self._link_button.setEnabled(selection_model.hasSelection() and has_link)

    def _url_column(self, model: ProductTableModel) -> int | None:
        for column in range(model.columnCount()):
            if model.column_name(column) == "product_url":
                return column
        return None

    def _resize_tables(self) -> None:
        for table in (self._keyword_table, self._recommendation_table, self._product_table):
            table.resizeColumnsToContents()
            table.setAlternatingRowColors(True)
            table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
            table.horizontalHeader().setStretchLastSection(True)
            model = table.model()
            if isinstance(model, ProductTableModel):
                url_column = self._url_column(model)
                if url_column is not None:
                    table.setColumnHidden(url_column, True)

    def _show_error(self, message: str) -> None:
        QMessageBox.warning(self, "키워드 서칭 프로", message)


def run_desktop(smoke_test: bool) -> int:
    application = QApplication.instance() or QApplication([])
    window = MainWindow()
    if smoke_test:
        window.close()
        return 0
    window.show()
    return application.exec()
