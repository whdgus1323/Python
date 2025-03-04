{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "import tkinter as tk\n",
    "from tkinter import ttk\n",
    "from datetime import datetime, timedelta\n",
    "\n",
    "# 타이머 상태를 관리하기 위한 변수\n",
    "running = False\n",
    "start_time = None\n",
    "elapsed_seconds = 0\n",
    "\n",
    "# 타이머 업데이트 함수\n",
    "def update_timer():\n",
    "    if running:\n",
    "        global elapsed_seconds\n",
    "        now = datetime.now()\n",
    "        elapsed_seconds = int((now - start_time).total_seconds())\n",
    "        formatted_time = str(timedelta(seconds=elapsed_seconds))  # 시:분:초 포맷\n",
    "        timer_label.config(text=formatted_time)\n",
    "        root.after(1000, update_timer)\n",
    "\n",
    "# 타이머 시작\n",
    "def start_timer():\n",
    "    global running, start_time\n",
    "    if not running:\n",
    "        running = True\n",
    "        start_time = datetime.now()\n",
    "        update_timer()\n",
    "\n",
    "# 타이머 정지\n",
    "def stop_timer():\n",
    "    global running\n",
    "    if running:\n",
    "        running = False\n",
    "\n",
    "# 타이머 리셋\n",
    "def reset_timer():\n",
    "    global running, start_time, elapsed_seconds\n",
    "    running = False\n",
    "    start_time = None\n",
    "    elapsed_seconds = 0\n",
    "    timer_label.config(text=\"0:00:00\")\n",
    "    for i in tree.get_children():\n",
    "        tree.delete(i)\n",
    "\n",
    "# 시간 기록\n",
    "record_count = 1\n",
    "def record_time():\n",
    "    global record_count\n",
    "    if running:\n",
    "        formatted_time = str(timedelta(seconds=elapsed_seconds))\n",
    "        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')\n",
    "        tree.insert(\"\", \"end\", values=(record_count, current_time, formatted_time))\n",
    "        record_count += 1\n",
    "\n",
    "# GUI 생성\n",
    "root = tk.Tk()\n",
    "root.title(\"스톱워치\")\n",
    "root.geometry(\"400x350\")\n",
    "root.configure(bg=\"#ffffff\")\n",
    "\n",
    "# 창 투명도 설정\n",
    "root.attributes(\"-alpha\", 0.9)  # 90% 투명\n",
    "\n",
    "# 스타일 정의\n",
    "style = ttk.Style()\n",
    "style.configure(\"Treeview\", font=(\"Arial\", 10), rowheight=20)\n",
    "style.configure(\"Treeview.Heading\", font=(\"Arial\", 10, \"bold\"))\n",
    "style.configure(\"TButton\", font=(\"Arial\", 10), padding=6)\n",
    "\n",
    "# 타이머 표시 라벨\n",
    "timer_label = tk.Label(root, text=\"0:00:00\", font=(\"Arial\", 20, \"bold\"), bg=\"#ffffff\", fg=\"#333\")\n",
    "timer_label.pack(pady=5)\n",
    "\n",
    "# 버튼 프레임 (가로 배열)\n",
    "button_frame = tk.Frame(root, bg=\"#ffffff\")\n",
    "button_frame.pack(pady=5)\n",
    "\n",
    "# 버튼 생성 (가로 정렬)\n",
    "button_width = 8\n",
    "\n",
    "start_button = ttk.Button(button_frame, text=\"시작\", command=start_timer, width=button_width)\n",
    "start_button.grid(row=0, column=0, sticky=\"nsew\")\n",
    "\n",
    "stop_button = ttk.Button(button_frame, text=\"정지\", command=stop_timer, width=button_width)\n",
    "stop_button.grid(row=0, column=1, sticky=\"nsew\")\n",
    "\n",
    "reset_button = ttk.Button(button_frame, text=\"리셋\", command=reset_timer, width=button_width)\n",
    "reset_button.grid(row=0, column=2, sticky=\"nsew\")\n",
    "\n",
    "record_button = ttk.Button(button_frame, text=\"기록\", command=record_time, width=button_width)\n",
    "record_button.grid(row=0, column=3, sticky=\"nsew\")\n",
    "\n",
    "# 버튼과 테이블 크기 맞추기\n",
    "for col in range(4):\n",
    "    button_frame.columnconfigure(col, weight=1)\n",
    "\n",
    "# 테이블 (Treeview)\n",
    "columns = (\"Index\", \"Timestamp\", \"Elapsed Time\")\n",
    "tree = ttk.Treeview(root, columns=columns, show=\"headings\", height=10)\n",
    "tree.heading(\"Index\", text=\"번호\")\n",
    "tree.heading(\"Timestamp\", text=\"기록된 시간\")\n",
    "tree.heading(\"Elapsed Time\", text=\"경과 시간\")\n",
    "tree.column(\"Index\", width=50, anchor=\"center\")\n",
    "tree.column(\"Timestamp\", width=150, anchor=\"center\")\n",
    "tree.column(\"Elapsed Time\", width=100, anchor=\"center\")\n",
    "tree.pack(pady=5, padx=0)\n",
    "\n",
    "# 메인 루프 실행\n",
    "root.mainloop()\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "base",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
