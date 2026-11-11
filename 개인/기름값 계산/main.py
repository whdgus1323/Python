import math
import os
import tkinter as tk
from tkinter import messagebox

try:
    from PIL import Image, ImageTk, ImageEnhance
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


TANK_CAPACITY_L = 54.0

REFERENCE_DISTANCE_KM = 234.0
REFERENCE_FUEL_PERCENT = 35.0

FULL_RANGE_KM = REFERENCE_DISTANCE_KM / (REFERENCE_FUEL_PERCENT / 100)


class SportageFuelApp:
    BG = "#070B14"
    SIDEBAR = "#0C1220"
    CARD = "#101827"
    CARD_LIGHT = "#151F32"
    INPUT_BG = "#0B1220"

    BORDER = "#243148"
    BORDER_LIGHT = "#32425E"

    TEXT = "#F7FAFF"
    MUTED = "#8492A9"
    MUTED_DARK = "#59677C"

    BLUE = "#43B7FF"
    CYAN = "#38E8D2"
    GREEN = "#42E39A"
    YELLOW = "#FFC857"
    RED = "#FF647C"

    def __init__(self, root):
        self.root = root
        self.root.title("Sportage NQ5 Fuel Studio")
        self.root.geometry("1220x790")
        self.root.minsize(1080, 720)
        self.root.configure(bg=self.BG)

        self.last_result = None
        self.animation_id = None
        self.car_photo = None

        self.create_ui()

        self.root.bind("<Return>", lambda event: self.calculate())
        self.root.after(100, self.calculate)

    def create_ui(self):
        self.create_topbar()

        body = tk.Frame(self.root, bg=self.BG)
        body.pack(fill="both", expand=True, padx=22, pady=(0, 22))

        body.grid_columnconfigure(0, weight=0)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)

        self.create_sidebar(body)
        self.create_content(body)

    def create_topbar(self):
        topbar = tk.Frame(self.root, bg=self.BG, height=82)
        topbar.pack(fill="x", padx=26, pady=(18, 12))
        topbar.pack_propagate(False)

        left = tk.Frame(topbar, bg=self.BG)
        left.pack(side="left", fill="y")

        brand_row = tk.Frame(left, bg=self.BG)
        brand_row.pack(anchor="w")

        tk.Label(
            brand_row,
            text="KIA",
            bg=self.BG,
            fg=self.TEXT,
            font=("Arial", 15, "bold")
        ).pack(side="left")

        tk.Label(
            brand_row,
            text="  /  ",
            bg=self.BG,
            fg=self.MUTED_DARK,
            font=("맑은 고딕", 12)
        ).pack(side="left")

        tk.Label(
            brand_row,
            text="SPORTAGE NQ5",
            bg=self.BG,
            fg=self.BLUE,
            font=("맑은 고딕", 11, "bold")
        ).pack(side="left")

        tk.Label(
            left,
            text="Fuel Intelligence",
            bg=self.BG,
            fg=self.TEXT,
            font=("맑은 고딕", 26, "bold")
        ).pack(anchor="w", pady=(4, 0))

        right = tk.Frame(topbar, bg=self.BG)
        right.pack(side="right", fill="y")

        self.reference_badge = tk.Frame(
            right,
            bg=self.CARD,
            highlightthickness=1,
            highlightbackground=self.BORDER
        )
        self.reference_badge.pack(side="right", pady=8, ipadx=16, ipady=7)

        tk.Label(
            self.reference_badge,
            text="CALIBRATION",
            bg=self.CARD,
            fg=self.MUTED,
            font=("Arial", 8, "bold")
        ).pack()

        tk.Label(
            self.reference_badge,
            text="234 km  ·  35%",
            bg=self.CARD,
            fg=self.TEXT,
            font=("맑은 고딕", 13, "bold")
        ).pack(pady=(3, 0))

    def create_sidebar(self, parent):
        sidebar = tk.Frame(
            parent,
            width=295,
            bg=self.SIDEBAR,
            highlightthickness=1,
            highlightbackground=self.BORDER
        )
        sidebar.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        sidebar.grid_propagate(False)

        inner = tk.Frame(sidebar, bg=self.SIDEBAR)
        inner.pack(fill="both", expand=True, padx=20, pady=22)

        tk.Label(
            inner,
            text="주유 정보",
            bg=self.SIDEBAR,
            fg=self.TEXT,
            font=("맑은 고딕", 16, "bold")
        ).pack(anchor="w")

        tk.Label(
            inner,
            text="계기판에 표시된 값을 입력하세요.",
            bg=self.SIDEBAR,
            fg=self.MUTED,
            font=("맑은 고딕", 9)
        ).pack(anchor="w", pady=(5, 23))

        self.distance_entry = self.create_input(
            inner,
            icon="⌁",
            title="현재 주행 가능 거리",
            unit="km",
            default="234"
        )

        self.price_entry = self.create_input(
            inner,
            icon="₩",
            title="휘발유 가격",
            unit="원/L",
            default="1,650"
        )

        self.payment_entry = self.create_input(
            inner,
            icon="＋",
            title="주유 금액",
            unit="원",
            default="50,000"
        )

        self.calculate_button = tk.Button(
            inner,
            text="주유 결과 계산",
            command=self.calculate,
            bg=self.BLUE,
            fg="#03111C",
            activebackground="#70CAFF",
            activeforeground="#03111C",
            relief="flat",
            bd=0,
            cursor="hand2",
            font=("맑은 고딕", 11, "bold"),
            pady=13
        )
        self.calculate_button.pack(fill="x", pady=(13, 12))

        self.reset_button = tk.Button(
            inner,
            text="입력값 초기화",
            command=self.reset_inputs,
            bg=self.CARD_LIGHT,
            fg=self.MUTED,
            activebackground=self.BORDER,
            activeforeground=self.TEXT,
            relief="flat",
            bd=0,
            cursor="hand2",
            font=("맑은 고딕", 9),
            pady=10
        )
        self.reset_button.pack(fill="x")

        separator = tk.Frame(inner, bg=self.BORDER, height=1)
        separator.pack(fill="x", pady=24)

        info = tk.Frame(inner, bg=self.SIDEBAR)
        info.pack(fill="x")

        self.create_sidebar_info(
            info,
            "연료탱크",
            f"{TANK_CAPACITY_L:.0f} L"
        )

        self.create_sidebar_info(
            info,
            "예상 최대 주행거리",
            f"{FULL_RANGE_KM:.0f} km"
        )

        self.create_sidebar_info(
            info,
            "차량",
            "1.6 가솔린 터보"
        )

        tk.Label(
            inner,
            text=(
                "주행 가능 거리는 최근 연비와 주행 환경에 따라\n"
                "계기판에서 계속 변하므로 계산값은 추정치입니다."
            ),
            bg=self.SIDEBAR,
            fg=self.MUTED_DARK,
            justify="left",
            font=("맑은 고딕", 8)
        ).pack(side="bottom", anchor="w")

    def create_input(self, parent, icon, title, unit, default):
        wrapper = tk.Frame(parent, bg=self.SIDEBAR)
        wrapper.pack(fill="x", pady=(0, 17))

        tk.Label(
            wrapper,
            text=title,
            bg=self.SIDEBAR,
            fg=self.MUTED,
            font=("맑은 고딕", 9)
        ).pack(anchor="w", pady=(0, 7))

        box = tk.Frame(
            wrapper,
            bg=self.INPUT_BG,
            highlightthickness=1,
            highlightbackground=self.BORDER
        )
        box.pack(fill="x")

        tk.Label(
            box,
            text=icon,
            bg=self.INPUT_BG,
            fg=self.BLUE,
            width=3,
            font=("맑은 고딕", 13, "bold")
        ).pack(side="left", padx=(5, 0))

        entry = tk.Entry(
            box,
            bg=self.INPUT_BG,
            fg=self.TEXT,
            insertbackground=self.TEXT,
            selectbackground=self.BLUE,
            selectforeground="#03111C",
            relief="flat",
            bd=0,
            justify="right",
            font=("맑은 고딕", 12, "bold")
        )
        entry.pack(side="left", fill="x", expand=True, padx=5, pady=13)
        entry.insert(0, default)

        tk.Label(
            box,
            text=unit,
            bg=self.INPUT_BG,
            fg=self.MUTED,
            font=("맑은 고딕", 9)
        ).pack(side="right", padx=(4, 12))

        entry.bind(
            "<FocusOut>",
            lambda event, widget=entry: self.format_entry(widget)
        )

        return entry

    def create_sidebar_info(self, parent, title, value):
        row = tk.Frame(parent, bg=self.SIDEBAR)
        row.pack(fill="x", pady=6)

        tk.Label(
            row,
            text=title,
            bg=self.SIDEBAR,
            fg=self.MUTED,
            font=("맑은 고딕", 8)
        ).pack(side="left")

        tk.Label(
            row,
            text=value,
            bg=self.SIDEBAR,
            fg=self.TEXT,
            font=("맑은 고딕", 9, "bold")
        ).pack(side="right")

    def create_content(self, parent):
        content = tk.Frame(parent, bg=self.BG)
        content.grid(row=0, column=1, sticky="nsew")

        content.grid_columnconfigure(0, weight=1)
        content.grid_columnconfigure(1, weight=1)
        content.grid_columnconfigure(2, weight=1)

        content.grid_rowconfigure(0, weight=3)
        content.grid_rowconfigure(1, weight=2)

        self.create_vehicle_card(content)
        self.create_metric_cards(content)
        self.create_comparison_card(content)

    def create_vehicle_card(self, parent):
        card = tk.Frame(
            parent,
            bg=self.CARD,
            highlightthickness=1,
            highlightbackground=self.BORDER
        )
        card.grid(
            row=0,
            column=0,
            columnspan=3,
            sticky="nsew",
            pady=(0, 14)
        )

        header = tk.Frame(card, bg=self.CARD)
        header.pack(fill="x", padx=22, pady=(18, 0))

        left = tk.Frame(header, bg=self.CARD)
        left.pack(side="left")

        tk.Label(
            left,
            text="LIVE FUEL STATUS",
            bg=self.CARD,
            fg=self.BLUE,
            font=("Arial", 9, "bold")
        ).pack(anchor="w")

        self.main_title = tk.Label(
            left,
            text="현재 연료 상태 분석",
            bg=self.CARD,
            fg=self.TEXT,
            font=("맑은 고딕", 17, "bold")
        )
        self.main_title.pack(anchor="w", pady=(3, 0))

        self.status_chip = tk.Label(
            header,
            text="READY",
            bg=self.CARD_LIGHT,
            fg=self.MUTED,
            padx=13,
            pady=6,
            font=("Arial", 8, "bold")
        )
        self.status_chip.pack(side="right")

        body = tk.Frame(card, bg=self.CARD)
        body.pack(fill="both", expand=True, padx=18, pady=(4, 14))

        body.grid_columnconfigure(0, weight=6)
        body.grid_columnconfigure(1, weight=4)
        body.grid_rowconfigure(0, weight=1)

        self.vehicle_canvas = tk.Canvas(
            body,
            bg=self.CARD,
            highlightthickness=0,
            height=290
        )
        self.vehicle_canvas.grid(
            row=0,
            column=0,
            sticky="nsew"
        )

        self.vehicle_canvas.bind(
            "<Configure>",
            lambda event: self.draw_vehicle_area()
        )

        gauge_container = tk.Frame(body, bg=self.CARD)
        gauge_container.grid(
            row=0,
            column=1,
            sticky="nsew",
            padx=(12, 8)
        )

        self.gauge_canvas = tk.Canvas(
            gauge_container,
            bg=self.CARD,
            highlightthickness=0,
            width=330,
            height=285
        )
        self.gauge_canvas.pack(fill="both", expand=True)

        self.gauge_canvas.bind(
            "<Configure>",
            lambda event: self.redraw_gauge()
        )

    def draw_vehicle_area(self):
        canvas = self.vehicle_canvas
        canvas.delete("all")

        width = max(canvas.winfo_width(), 500)
        height = max(canvas.winfo_height(), 260)

        # 배경 조명
        for radius, color in [
            (190, "#111E31"),
            (145, "#13243A"),
            (100, "#162B45")
        ]:
            canvas.create_oval(
                width / 2 - radius,
                height / 2 - radius * 0.42,
                width / 2 + radius,
                height / 2 + radius * 0.42,
                fill=color,
                outline=""
            )

        image_path = "sportage.png"

        if PIL_AVAILABLE and os.path.exists(image_path):
            try:
                image = Image.open(image_path).convert("RGBA")

                max_width = int(width * 0.78)
                max_height = int(height * 0.68)

                image.thumbnail(
                    (max_width, max_height),
                    Image.Resampling.LANCZOS
                )

                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1.08)

                self.car_photo = ImageTk.PhotoImage(image)

                canvas.create_image(
                    width / 2,
                    height / 2 + 5,
                    image=self.car_photo
                )

            except Exception:
                self.draw_car_silhouette(canvas, width, height)
        else:
            self.draw_car_silhouette(canvas, width, height)

        canvas.create_text(
            25,
            height - 46,
            text="SPORTAGE",
            anchor="w",
            fill=self.TEXT,
            font=("Arial", 15, "bold")
        )

        canvas.create_text(
            25,
            height - 23,
            text="NQ5 · 1.6 TURBO GASOLINE",
            anchor="w",
            fill=self.MUTED,
            font=("Arial", 8)
        )

        canvas.create_text(
            width - 25,
            height - 34,
            text="2026",
            anchor="e",
            fill=self.MUTED_DARK,
            font=("Arial", 10, "bold")
        )

    def draw_car_silhouette(self, canvas, width, height):
        center_x = width / 2
        base_y = height * 0.60

        scale = min(width / 650, height / 300)

        def sx(value):
            return center_x + value * scale

        def sy(value):
            return base_y + value * scale

        canvas.create_oval(
            sx(-210),
            sy(42),
            sx(210),
            sy(75),
            fill="#05080F",
            outline=""
        )

        body_points = [
            sx(-235), sy(15),
            sx(-205), sy(-20),
            sx(-120), sy(-37),
            sx(-65), sy(-90),
            sx(45), sy(-105),
            sx(132), sy(-78),
            sx(188), sy(-34),
            sx(225), sy(-18),
            sx(244), sy(12),
            sx(228), sy(36),
            sx(-220), sy(36)
        ]

        canvas.create_polygon(
            body_points,
            fill="#D9E1EB",
            outline="#FFFFFF",
            width=2,
            smooth=True
        )

        canvas.create_polygon(
            sx(-224), sy(21),
            sx(228), sy(21),
            sx(214), sy(47),
            sx(-210), sy(47),
            fill="#556174",
            outline=""
        )

        canvas.create_polygon(
            sx(-92), sy(-82),
            sx(-58), sy(-42),
            sx(16), sy(-42),
            sx(10), sy(-91),
            fill="#18283C",
            outline="#6D8099",
            width=1
        )

        canvas.create_polygon(
            sx(24), sy(-91),
            sx(37), sy(-42),
            sx(122), sy(-42),
            sx(82), sy(-78),
            fill="#18283C",
            outline="#6D8099",
            width=1
        )

        canvas.create_line(
            sx(45),
            sy(-85),
            sx(92),
            sy(-48),
            fill="#4DB5E8",
            width=2
        )

        canvas.create_polygon(
            sx(174), sy(-25),
            sx(226), sy(-12),
            sx(216), sy(1),
            sx(174), sy(-3),
            fill=self.CYAN,
            outline=""
        )

        canvas.create_rectangle(
            sx(-225),
            sy(-10),
            sx(-203),
            sy(2),
            fill=self.RED,
            outline=""
        )

        canvas.create_line(
            sx(18),
            sy(-39),
            sx(18),
            sy(25),
            fill="#AAB5C4",
            width=1
        )

        canvas.create_line(
            sx(125),
            sy(-38),
            sx(139),
            sy(24),
            fill="#AAB5C4",
            width=1
        )

        for wheel_x in [-145, 150]:
            canvas.create_oval(
                sx(wheel_x - 37),
                sy(5),
                sx(wheel_x + 37),
                sy(79),
                fill="#070A10",
                outline="#374151",
                width=3
            )

            canvas.create_oval(
                sx(wheel_x - 20),
                sy(22),
                sx(wheel_x + 20),
                sy(62),
                fill="#8894A5",
                outline="#D6DCE5",
                width=2
            )

            canvas.create_oval(
                sx(wheel_x - 7),
                sy(35),
                sx(wheel_x + 7),
                sy(49),
                fill="#202938",
                outline=""
            )

    def create_metric_cards(self, parent):
        self.metric_labels = {}

        cards = [
            (
                "after_distance",
                "주유 후 주행거리",
                "km",
                self.BLUE
            ),
            (
                "after_percent",
                "주유 후 연료",
                "%",
                self.GREEN
            ),
            (
                "added_fuel",
                "실제 주유량",
                "L",
                self.CYAN
            )
        ]

        for column, data in enumerate(cards):
            key, title, unit, accent = data

            card = tk.Frame(
                parent,
                bg=self.CARD,
                highlightthickness=1,
                highlightbackground=self.BORDER
            )
            card.grid(
                row=1,
                column=column,
                sticky="nsew",
                padx=(
                    0 if column == 0 else 7,
                    0 if column == 2 else 7
                )
            )

            accent_line = tk.Frame(
                card,
                height=3,
                bg=accent
            )
            accent_line.pack(fill="x")

            inner = tk.Frame(card, bg=self.CARD)
            inner.pack(
                fill="both",
                expand=True,
                padx=18,
                pady=15
            )

            tk.Label(
                inner,
                text=title,
                bg=self.CARD,
                fg=self.MUTED,
                font=("맑은 고딕", 9)
            ).pack(anchor="w")

            value_row = tk.Frame(inner, bg=self.CARD)
            value_row.pack(anchor="w", pady=(8, 3))

            value_label = tk.Label(
                value_row,
                text="—",
                bg=self.CARD,
                fg=self.TEXT,
                font=("맑은 고딕", 23, "bold")
            )
            value_label.pack(side="left")

            tk.Label(
                value_row,
                text=f" {unit}",
                bg=self.CARD,
                fg=self.MUTED,
                font=("맑은 고딕", 10)
            ).pack(side="left", anchor="s", pady=(0, 4))

            description_label = tk.Label(
                inner,
                text="계산 대기 중",
                bg=self.CARD,
                fg=self.MUTED_DARK,
                font=("맑은 고딕", 8)
            )
            description_label.pack(anchor="w")

            self.metric_labels[key] = (
                value_label,
                description_label
            )

    def create_comparison_card(self, parent):
        self.comparison_bar_before = None
        self.comparison_bar_after = None

    def redraw_gauge(self):
        if self.last_result:
            self.draw_main_gauge(
                self.last_result["fuel_after_percent"],
                self.last_result["distance_after"],
                self.last_result["current_percent"]
            )
        else:
            self.draw_main_gauge(0, 0, 0)

    def draw_main_gauge(
        self,
        after_percent,
        distance_after,
        before_percent
    ):
        canvas = self.gauge_canvas
        canvas.delete("all")

        width = max(canvas.winfo_width(), 300)
        height = max(canvas.winfo_height(), 260)

        center_x = width / 2
        center_y = height * 0.54
        radius = min(width, height) * 0.34

        start_angle = 140
        total_angle = 260

        canvas.create_oval(
            center_x - radius - 22,
            center_y - radius - 22,
            center_x + radius + 22,
            center_y + radius + 22,
            outline=self.BORDER,
            width=1
        )

        canvas.create_arc(
            center_x - radius,
            center_y - radius,
            center_x + radius,
            center_y + radius,
            start=start_angle,
            extent=total_angle,
            style="arc",
            outline=self.BORDER_LIGHT,
            width=17
        )

        percent = max(0, min(after_percent, 100))
        active_angle = total_angle * percent / 100

        gauge_color = self.get_fuel_color(percent)

        canvas.create_arc(
            center_x - radius,
            center_y - radius,
            center_x + radius,
            center_y + radius,
            start=start_angle,
            extent=active_angle,
            style="arc",
            outline=gauge_color,
            width=17
        )

        for index in range(21):
            ratio = index / 20
            angle = math.radians(start_angle + total_angle * ratio)

            outer_r = radius + 16
            inner_r = radius + (7 if index % 5 == 0 else 11)

            x1 = center_x + math.cos(angle) * inner_r
            y1 = center_y - math.sin(angle) * inner_r
            x2 = center_x + math.cos(angle) * outer_r
            y2 = center_y - math.sin(angle) * outer_r

            canvas.create_line(
                x1,
                y1,
                x2,
                y2,
                fill=(
                    self.MUTED
                    if index % 5 == 0
                    else self.BORDER_LIGHT
                ),
                width=2 if index % 5 == 0 else 1
            )

        canvas.create_text(
            center_x,
            center_y - 48,
            text="FUEL",
            fill=self.MUTED,
            font=("Arial", 9, "bold")
        )

        canvas.create_text(
            center_x,
            center_y - 7,
            text=f"{percent:.1f}",
            fill=self.TEXT,
            font=("Arial", 34, "bold")
        )

        canvas.create_text(
            center_x + 58,
            center_y + 3,
            text="%",
            fill=self.MUTED,
            font=("Arial", 12, "bold")
        )

        canvas.create_text(
            center_x,
            center_y + 35,
            text=f"약 {distance_after:,.0f} km",
            fill=gauge_color,
            font=("맑은 고딕", 12, "bold")
        )

        canvas.create_text(
            center_x,
            center_y + 59,
            text="주유 후 예상 주행 가능 거리",
            fill=self.MUTED,
            font=("맑은 고딕", 8)
        )

        before_angle = math.radians(
            start_angle + total_angle * before_percent / 100
        )

        marker_r = radius

        marker_x = center_x + math.cos(before_angle) * marker_r
        marker_y = center_y - math.sin(before_angle) * marker_r

        canvas.create_oval(
            marker_x - 4,
            marker_y - 4,
            marker_x + 4,
            marker_y + 4,
            fill=self.YELLOW,
            outline=self.CARD,
            width=2
        )

        canvas.create_text(
            center_x - radius + 5,
            center_y + radius + 27,
            text=f"BEFORE  {before_percent:.1f}%",
            anchor="w",
            fill=self.YELLOW,
            font=("Arial", 8, "bold")
        )

        canvas.create_text(
            center_x + radius - 5,
            center_y + radius + 27,
            text=f"AFTER  {after_percent:.1f}%",
            anchor="e",
            fill=gauge_color,
            font=("Arial", 8, "bold")
        )

    @staticmethod
    def parse_number(entry):
        value = entry.get().replace(",", "").strip()

        if not value:
            raise ValueError("모든 값을 입력하세요.")

        return float(value)

    @staticmethod
    def format_entry(entry):
        try:
            value = float(
                entry.get().replace(",", "").strip()
            )

            entry.delete(0, tk.END)

            if value.is_integer():
                entry.insert(0, f"{int(value):,}")
            else:
                entry.insert(0, f"{value:,.2f}")

        except ValueError:
            pass

    def calculate(self):
        try:
            current_distance = self.parse_number(
                self.distance_entry
            )
            fuel_price = self.parse_number(
                self.price_entry
            )
            payment = self.parse_number(
                self.payment_entry
            )

            if current_distance < 0:
                raise ValueError(
                    "현재 주행 가능 거리는 0 이상이어야 합니다."
                )

            if current_distance > FULL_RANGE_KM * 1.3:
                raise ValueError(
                    "현재 주행 가능 거리 값이 지나치게 큽니다."
                )

            if fuel_price <= 0:
                raise ValueError(
                    "휘발유 가격은 0보다 커야 합니다."
                )

            if payment < 0:
                raise ValueError(
                    "주유 금액은 0 이상이어야 합니다."
                )

            current_percent = (
                current_distance
                / FULL_RANGE_KM
                * 100
            )
            current_percent = max(
                0,
                min(current_percent, 100)
            )

            current_fuel_l = (
                TANK_CAPACITY_L
                * current_percent
                / 100
            )

            requested_fuel_l = payment / fuel_price

            available_space_l = max(
                0,
                TANK_CAPACITY_L - current_fuel_l
            )

            actual_added_l = min(
                requested_fuel_l,
                available_space_l
            )

            fuel_after_l = (
                current_fuel_l
                + actual_added_l
            )

            fuel_after_percent = (
                fuel_after_l
                / TANK_CAPACITY_L
                * 100
            )

            actual_payment = (
                actual_added_l
                * fuel_price
            )

            excess_payment = max(
                0,
                payment - actual_payment
            )

            reference_fuel_l = (
                TANK_CAPACITY_L
                * REFERENCE_FUEL_PERCENT
                / 100
            )

            estimated_efficiency = (
                REFERENCE_DISTANCE_KM
                / reference_fuel_l
            )

            added_distance = (
                actual_added_l
                * estimated_efficiency
            )

            distance_after = min(
                FULL_RANGE_KM,
                current_distance + added_distance
            )

            self.last_result = {
                "current_distance": current_distance,
                "current_percent": current_percent,
                "current_fuel_l": current_fuel_l,
                "requested_fuel_l": requested_fuel_l,
                "actual_added_l": actual_added_l,
                "fuel_after_l": fuel_after_l,
                "fuel_after_percent": fuel_after_percent,
                "actual_payment": actual_payment,
                "excess_payment": excess_payment,
                "estimated_efficiency": estimated_efficiency,
                "added_distance": added_distance,
                "distance_after": distance_after
            }

            self.update_status()
            self.animate_gauge()

            self.format_entry(self.distance_entry)
            self.format_entry(self.price_entry)
            self.format_entry(self.payment_entry)

        except ValueError as error:
            messagebox.showerror(
                "입력값 확인",
                str(error)
            )

    def update_status(self):
        result = self.last_result

        self.metric_labels["after_distance"][0].config(
            text=f"{result['distance_after']:,.0f}"
        )
        self.metric_labels["after_distance"][1].config(
            text=(
                f"기존보다 +{result['added_distance']:,.0f} km"
            )
        )

        self.metric_labels["after_percent"][0].config(
            text=f"{result['fuel_after_percent']:.1f}"
        )
        self.metric_labels["after_percent"][1].config(
            text=(
                f"주유 전 {result['current_percent']:.1f}%"
            )
        )

        self.metric_labels["added_fuel"][0].config(
            text=f"{result['actual_added_l']:.2f}"
        )
        self.metric_labels["added_fuel"][1].config(
            text=(
                f"실제 금액 {result['actual_payment']:,.0f}원"
            )
        )

        percent = result["fuel_after_percent"]

        if percent >= 99.5:
            self.status_chip.config(
                text="FULL",
                bg="#153E32",
                fg=self.GREEN
            )
            self.main_title.config(
                text="연료가 거의 가득 찼습니다"
            )

        elif percent <= 15:
            self.status_chip.config(
                text="LOW FUEL",
                bg="#42212A",
                fg=self.RED
            )
            self.main_title.config(
                text="연료 잔량이 부족합니다"
            )

        elif percent <= 30:
            self.status_chip.config(
                text="CAUTION",
                bg="#44361E",
                fg=self.YELLOW
            )
            self.main_title.config(
                text="주유 후 예상 연료 상태"
            )

        else:
            self.status_chip.config(
                text="NORMAL",
                bg="#153441",
                fg=self.CYAN
            )
            self.main_title.config(
                text="주유 후 예상 연료 상태"
            )

        if result["excess_payment"] > 1:
            self.metric_labels["added_fuel"][1].config(
                text=(
                    f"초과 예상 {result['excess_payment']:,.0f}원"
                ),
                fg=self.YELLOW
            )
        else:
            self.metric_labels["added_fuel"][1].config(
                fg=self.MUTED_DARK
            )

    def animate_gauge(self):
        if self.animation_id:
            self.root.after_cancel(self.animation_id)

        start = self.last_result["current_percent"]
        target = self.last_result["fuel_after_percent"]

        steps = 36
        current_step = 0

        def animate():
            nonlocal current_step

            current_step += 1
            progress = current_step / steps

            eased = 1 - pow(1 - progress, 3)

            percent = start + (target - start) * eased

            estimated_distance = (
                self.last_result["current_distance"]
                + self.last_result["added_distance"] * eased
            )

            self.draw_main_gauge(
                percent,
                estimated_distance,
                start
            )

            if current_step < steps:
                self.animation_id = self.root.after(
                    14,
                    animate
                )
            else:
                self.animation_id = None

        animate()

    def get_fuel_color(self, percent):
        if percent <= 15:
            return self.RED
        if percent <= 30:
            return self.YELLOW
        if percent >= 90:
            return self.CYAN
        return self.GREEN

    def reset_inputs(self):
        values = [
            (self.distance_entry, "234"),
            (self.price_entry, "1,650"),
            (self.payment_entry, "50,000")
        ]

        for entry, value in values:
            entry.delete(0, tk.END)
            entry.insert(0, value)

        self.calculate()


if __name__ == "__main__":
    root = tk.Tk()
    app = SportageFuelApp(root)
    root.mainloop()