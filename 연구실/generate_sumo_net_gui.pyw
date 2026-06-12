"""Tkinter GUI launcher for generate_sumo_net.py."""

from __future__ import annotations

import queue
import threading
import traceback
from pathlib import Path
from tkinter import (
    BOTH,
    END,
    LEFT,
    RIGHT,
    W,
    BooleanVar,
    Button,
    Checkbutton,
    Entry,
    Frame,
    Label,
    StringVar,
    Text,
    Tk,
    filedialog,
    messagebox,
)

import generate_sumo_net
import generate_sumo_net_blocks


DEFAULT_BASELINE = r"C:\Users\Choe JongHyeon\Desktop\OSM_project\baseline\map.net.xml"
DEFAULT_OUTPUT = r"C:\Users\Choe JongHyeon\Desktop\OSM_project\test_3\map.net.xml"
DEFAULT_BATCH_DIR = r"C:\Users\Choe JongHyeon\Desktop\OSM_project\batch_maps"


class SumoNetGeneratorApp:
    """Small desktop GUI for generating a synthetic SUMO net.xml file."""

    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title("SUMO net.xml Generator")
        self.root.geometry("820x640")
        self.root.minsize(760, 560)
        self.events: queue.Queue[str] = queue.Queue()
        self.running = False

        self.input_path = StringVar(value=DEFAULT_BASELINE)
        self.output_path = StringVar(value=DEFAULT_OUTPUT)
        self.width_m = StringVar(value="412.65")
        self.height_m = StringVar(value="387.65")
        self.rows = StringVar(value="12")
        self.cols = StringVar(value="13")
        self.junction_target = StringVar(value="148")
        self.internal_junction_target = StringVar(value="216")
        self.road_missing_prob = StringVar(value="0.08")
        self.jitter_m = StringVar(value="6.0")
        self.seed = StringVar(value="42")
        self.run_sumo = BooleanVar(value=True)
        self.batch_enabled = BooleanVar(value=False)
        self.batch_dir = StringVar(value=DEFAULT_BATCH_DIR)
        self.batch_count = StringVar(value="10")
        self.batch_start_index = StringVar(value="1")
        self.vary_layout = BooleanVar(value=True)

        self._build_ui()
        self.root.after(100, self._drain_events)

    def _build_ui(self) -> None:
        outer = Frame(self.root, padx=14, pady=14)
        outer.pack(fill=BOTH, expand=True)

        self._path_row(outer, "기준 net.xml", self.input_path, self._browse_input)
        self._path_row(outer, "출력 net.xml", self.output_path, self._browse_output)

        params = Frame(outer, pady=8)
        params.pack(fill="x")
        self._param_row(params, 0, "width_m", self.width_m, "height_m", self.height_m)
        self._param_row(params, 1, "rows", self.rows, "cols", self.cols)
        self._param_row(params, 2, "junction_target", self.junction_target, "internal_junction_target", self.internal_junction_target)
        self._param_row(params, 3, "road_missing_prob", self.road_missing_prob, "jitter_m", self.jitter_m)
        self._param_row(params, 4, "seed", self.seed, "", None)

        batch = Frame(outer, pady=4)
        batch.pack(fill="x")
        Checkbutton(batch, text="여러 맵 생성", variable=self.batch_enabled).pack(side=LEFT)
        Checkbutton(batch, text="seed별 레이아웃 변화", variable=self.vary_layout).pack(side=LEFT, padx=(12, 0))
        Label(batch, text="개수", width=5, anchor=W).pack(side=LEFT, padx=(14, 2))
        Entry(batch, textvariable=self.batch_count, width=8).pack(side=LEFT)
        Label(batch, text="시작 번호", width=8, anchor=W).pack(side=LEFT, padx=(14, 2))
        Entry(batch, textvariable=self.batch_start_index, width=8).pack(side=LEFT)

        self._path_row(outer, "batch 폴더", self.batch_dir, self._browse_batch_dir)

        actions = Frame(outer, pady=8)
        actions.pack(fill="x")
        self.run_button = Button(actions, text="생성+SUMO 실행", command=self._run_generation, width=18)
        self.run_button.pack(side=LEFT)
        Button(actions, text="로그 지우기", command=self._clear_log, width=12).pack(side=LEFT, padx=8)
        Checkbutton(actions, text="randomTrips/duarouter/sumo 자동 실행", variable=self.run_sumo).pack(side=LEFT, padx=8)

        self.log = Text(outer, height=22, wrap="word")
        self.log.pack(fill=BOTH, expand=True)
        self._append_log("기준 파일을 선택한 뒤 '생성 및 검증 실행'을 누르세요.\n")

    def _path_row(self, parent: Frame, label: str, variable: StringVar, command) -> None:
        row = Frame(parent, pady=4)
        row.pack(fill="x")
        Label(row, text=label, width=14, anchor=W).pack(side=LEFT)
        Entry(row, textvariable=variable).pack(side=LEFT, fill="x", expand=True, padx=6)
        Button(row, text="찾기", command=command, width=8).pack(side=RIGHT)

    def _param_row(
        self,
        parent: Frame,
        row_index: int,
        label1: str,
        var1: StringVar,
        label2: str,
        var2: StringVar | None,
    ) -> None:
        row = Frame(parent, pady=3)
        row.grid(row=row_index, column=0, sticky="ew")
        parent.columnconfigure(0, weight=1)
        Label(row, text=label1, width=22, anchor=W).pack(side=LEFT)
        Entry(row, textvariable=var1, width=12).pack(side=LEFT, padx=(0, 18))
        if label2 and var2 is not None:
            Label(row, text=label2, width=24, anchor=W).pack(side=LEFT)
            Entry(row, textvariable=var2, width=12).pack(side=LEFT)

    def _browse_input(self) -> None:
        path = filedialog.askopenfilename(
            title="기준 map.net.xml 선택",
            filetypes=[("SUMO net.xml", "*.net.xml"), ("XML files", "*.xml"), ("All files", "*.*")],
            initialdir=str(Path(self.input_path.get()).parent),
        )
        if path:
            self.input_path.set(path)

    def _browse_output(self) -> None:
        path = filedialog.asksaveasfilename(
            title="출력 generated_map.net.xml 저장",
            defaultextension=".xml",
            filetypes=[("SUMO net.xml", "*.net.xml"), ("XML files", "*.xml"), ("All files", "*.*")],
            initialfile=Path(self.output_path.get()).name,
            initialdir=str(Path(self.output_path.get()).parent),
        )
        if path:
            self.output_path.set(path)

    def _browse_batch_dir(self) -> None:
        path = filedialog.askdirectory(
            title="batch 출력 폴더 선택",
            initialdir=self.batch_dir.get(),
        )
        if path:
            self.batch_dir.set(path)

    def _clear_log(self) -> None:
        self.log.delete("1.0", END)

    def _append_log(self, message: str) -> None:
        self.log.insert(END, message)
        self.log.see(END)

    def _drain_events(self) -> None:
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            self._append_log(event)
        self.root.after(100, self._drain_events)

    def _run_generation(self) -> None:
        if self.running:
            return
        self.running = True
        self.run_button.configure(state="disabled")
        self._append_log("\n생성을 시작합니다...\n")
        thread = threading.Thread(target=self._generate_worker, daemon=True)
        thread.start()

    def _generate_worker(self) -> None:
        try:
            stats, pipeline_logs = self._generate()
            pipeline_text = ""
            if pipeline_logs:
                pipeline_text = "\nSUMO 자동 실행\n" + "".join(
                    f"  {name}: Success\n" for name, _log in pipeline_logs
                )
            self.events.put(
                "\n완료\n"
                f"  priority junctions : {stats['priority_junction_count']}\n"
                f"  internal junctions : {stats['internal_junction_count']}\n"
                f"  normal edges       : {stats['normal_edge_count']}\n"
                f"  internal edges     : {stats['internal_edge_count']}\n"
                f"  lanes              : {stats['lane_count']}\n"
                f"  connections        : {stats['connection_count']}\n"
                f"  output             : {stats.get('output', self.output_path.get())}\n"
                f"{pipeline_text}"
            )
            done_message = "net.xml 생성과 검증이 완료되었습니다."
            if pipeline_logs:
                done_message = "net.xml 생성, 검증, SUMO 실행이 완료되었습니다."
            self.root.after(0, lambda: messagebox.showinfo("완료", done_message))
        except Exception as exc:
            self.events.put("\n오류 발생\n" + "".join(traceback.format_exception_only(type(exc), exc)))
            self.root.after(0, lambda: messagebox.showerror("오류", str(exc)))
        finally:
            self.running = False
            self.root.after(0, lambda: self.run_button.configure(state="normal"))

    def _generate(self) -> tuple[dict[str, int], list[tuple[str, str]]]:
        input_path = self.input_path.get().strip()
        output_path = self.output_path.get().strip()
        if not input_path:
            raise ValueError("기준 net.xml 경로가 비어 있습니다.")
        if not output_path:
            raise ValueError("출력 net.xml 경로가 비어 있습니다.")

        reference = generate_sumo_net.parse_reference_net(input_path)
        self.events.put(
            "기준 파일 분석 완료\n"
            f"  type definitions   : {len(reference['types'])}\n"
            f"  priority junctions : {reference['stats']['priority_junction_count']}\n"
            f"  internal junctions : {reference['stats']['internal_junction_count']}\n"
        )

        width_m = float(self.width_m.get())
        height_m = float(self.height_m.get())
        rows = int(self.rows.get())
        cols = int(self.cols.get())
        junction_target = int(self.junction_target.get())
        internal_junction_target = int(self.internal_junction_target.get())
        road_missing_prob = float(self.road_missing_prob.get())
        jitter_m = float(self.jitter_m.get())
        seed = int(self.seed.get())

        if self.batch_enabled.get():
            batch_root = Path(self.batch_dir.get().strip())
            count = int(self.batch_count.get())
            start_index = int(self.batch_start_index.get())
            if count < 1:
                raise ValueError("batch 생성 개수는 1 이상이어야 합니다.")
            batch_root.mkdir(parents=True, exist_ok=True)
            last_stats: dict[str, int] | dict[str, object] = {}
            pipeline_logs: list[tuple[str, str]] = []
            for index in range(start_index, start_index + count):
                map_dir = batch_root / f"map_{index}"
                map_dir.mkdir(parents=True, exist_ok=True)
                final_output = map_dir / "map.net.xml"
                self.events.put(f"\nmap_{index} 생성 시작: seed={index}\n")
                stats = generate_sumo_net_blocks.generate_one(Path(input_path), map_dir, index, self.run_sumo.get())
                if self.run_sumo.get():
                    pipeline_logs = [("randomTrips.py", ""), ("duarouter", ""), ("sumo", "")]
                self.events.put(
                    f"map_{index} 완료: priority={stats['priority_junction_count']}, "
                    f"lanes={dict(stats['normal_edge_lane_distribution'])}\n"
                )
                last_stats = dict(stats)
                last_stats["output"] = str(batch_root)
            return last_stats, pipeline_logs

        final_output = Path(output_path)
        stats = generate_sumo_net_blocks.generate_one(Path(input_path), final_output.parent, seed, self.run_sumo.get())
        stats["output"] = str(final_output.parent / "map.net.xml")
        pipeline_logs: list[tuple[str, str]] = []
        if self.run_sumo.get():
            pipeline_logs = [("randomTrips.py", ""), ("duarouter", ""), ("sumo", "")]
        return stats, pipeline_logs


def main() -> None:
    """Start the GUI application."""
    root = Tk()
    SumoNetGeneratorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
