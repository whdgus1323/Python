import sys
from pathlib import Path
from subprocess import CompletedProcess, run

from PySide6.QtGui import QAction
from PySide6.QtWidgets import QApplication, QPushButton

from keyword_sourcing.desktop import MainWindow


def test_desktop_app_constructs_its_main_window_when_smoke_tested() -> None:
    # Given: the package entry point and its isolated smoke-test mode.
    entry_point = Path(__file__).parents[1] / "desktop_app.py"

    # When: the real desktop entry point is started without opening a browser or server.
    completed: CompletedProcess[str] = run(
        [sys.executable, str(entry_point), "--smoke-test"],
        check=False,
        capture_output=True,
        text=True,
    )

    # Then: Qt constructs the native window successfully.
    assert completed.returncode == 0, completed.stderr


def test_desktop_app_finds_project_source_when_started_from_parent_venv() -> None:
    # Given: the parent workspace interpreter used by the reported failing command.
    entry_point = Path(__file__).parents[1] / "desktop_app.py"
    parent_python = Path(__file__).parents[3] / ".venv" / "Scripts" / "python.exe"

    # When: the application starts directly from that parent interpreter.
    completed: CompletedProcess[str] = run(
        [str(parent_python), str(entry_point), "--smoke-test"],
        check=False,
        capture_output=True,
        text=True,
    )

    # Then: the local src package is importable without an editable installation.
    assert completed.returncode == 0, completed.stderr


def test_sidebar_menu_selects_its_matching_result_tab() -> None:
    # Given: a real native application window.
    application = QApplication.instance() or QApplication([])
    window = MainWindow()

    # When: the benchmark-candidate item in the sidebar is clicked.
    benchmark_button = next(
        button
        for button in window.findChildren(QPushButton)
        if button.text() == "벤치마킹 후보"
    )
    benchmark_button.click()

    # Then: the matching benchmark-candidate result tab is selected.
    assert window.selected_result_tab == 1
    application.quit()


def test_dashboard_updates_data_metric_when_products_are_loaded() -> None:
    # Given: a local sample product file and the native desktop window.
    application = QApplication.instance() or QApplication([])
    window = MainWindow()
    sample_file = Path(__file__).parents[1] / "sample_products.csv"

    # When: product data enters the dashboard.
    window.load_data(sample_file)

    # Then: the dashboard reports the actual imported product count.
    assert window.dashboard_data_count == "12"
    application.quit()


def test_selection_and_export_actions_enable_only_after_their_required_state() -> None:
    # Given: a native window with imported product data but no selected row or analysis result.
    application = QApplication.instance() or QApplication([])
    window = MainWindow()
    sample_file = Path(__file__).parents[1] / "sample_products.csv"
    link_button = next(
        button
        for button in window.findChildren(QPushButton)
        if button.text() == "선택 상품 링크 열기"
    )
    export_action = next(
        action for action in window.findChildren(QAction) if action.text() == "추천 결과 저장"
    )

    # When: the file is loaded and then analyzed.
    window.load_data(sample_file)

    # Then: no row selection keeps linking disabled, and export remains unavailable.
    assert not link_button.isEnabled()
    assert not export_action.isEnabled()
    application.quit()


def test_export_menu_action_enables_when_analysis_completes() -> None:
    # Given: a window with an imported product file.
    application = QApplication.instance() or QApplication([])
    window = MainWindow()
    sample_file = Path(__file__).parents[1] / "sample_products.csv"
    analyze_button = next(
        button for button in window.findChildren(QPushButton) if button.text() == "분석 실행"
    )
    export_action = next(
        action for action in window.findChildren(QAction) if action.text() == "추천 결과 저장"
    )
    window.load_data(sample_file)

    # When: the enabled analysis action is clicked.
    analyze_button.click()

    # Then: the matching menu action becomes available.
    assert export_action.isEnabled()
    application.quit()
