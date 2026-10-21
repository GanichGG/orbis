"""Orbis — оконный инструмент: ролики слева, действия справа, прогресс и лог внизу. Сдержанная серая тема.

    pip install customtkinter
    python orbis_app.py            (или Orbis.exe, см. build_exe.bat)

Все действия запускают те же скрипты, что и из терминала (stick2r6.py, compare.py, tools/*.py, stickpose/train.py),
отдельным процессом: окно не зависает, «Стоп» прерывает. Собранный Orbis.exe — только окно,
скрипты он запускает установленным Python (тем, где стоят torch и opencv); лежать он должен в папке проекта.
"""
import glob
import json
import os
import queue
import re
import subprocess
import sys
import threading

HERE = os.path.dirname(os.path.abspath(sys.argv[0] if getattr(sys, "frozen", False) else __file__))
CFG = os.path.join(HERE, "orbis_app.json")
VIDEO_EXT = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")

# серая палитра без цветных акцентов: фон -> панели -> карточки/поля -> кнопки -> текст
C = {"bg": "#1b1b1c", "panel": "#232325", "card": "#2a2a2d", "field": "#1f1f21", "line": "#333336",
     "btn": "#313134", "btn_hover": "#3b3b3f", "main": "#e6e6e6", "main_hover": "#cfcfcf", "main_text": "#1b1b1c",
     "sel": "#3a3a3e", "text": "#e2e2e4", "dim": "#8a8a90", "faint": "#5f5f65", "bar": "#bdbdc2"}
R = 12          # скругление панелей
RB = 8          # скругление кнопок и полей


def font(size=13, weight="normal"):
    import customtkinter as ctk
    fam = "Segoe UI" if sys.platform.startswith("win") else "Inter"
    return ctk.CTkFont(family=fam, size=size, weight=weight)


class App:
    def __init__(self, root):
        import customtkinter as ctk
        self.ctk, self.root = ctk, root
        self.proc, self.q, self.then = None, queue.Queue(), None
        self.cfg = self.load_cfg()
        self.sel, self.rows, self._last_cr = set(), [], False
        root.title("Orbis")
        root.geometry(self.cfg.get("geometry", "1180x780"))
        root.minsize(980, 640)
        root.configure(fg_color=C["bg"])
        self.build()
        self.refresh_weights()
        self.refresh_videos()
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.after(80, self.pump)

    # ---------- элементы ----------
    def button(self, parent, text, cmd, main=False, width=0):
        ctk = self.ctk
        return ctk.CTkButton(parent, text=text, command=cmd, corner_radius=RB, height=34, width=width or 0,
                             fg_color=C["main"] if main else C["btn"], hover_color=C["main_hover"] if main else C["btn_hover"],
                             text_color=C["main_text"] if main else C["text"], font=font(13, "bold" if main else "normal"))

    def label(self, parent, text, size=13, dim=False, bold=False, **kw):
        return self.ctk.CTkLabel(parent, text=text, text_color=C["dim"] if dim else C["text"],
                                 font=font(size, "bold" if bold else "normal"), **kw)

    def entry(self, parent, var, width=90, hint=""):
        return self.ctk.CTkEntry(parent, textvariable=var, width=width, height=34, corner_radius=RB, fg_color=C["field"],
                                 border_color=C["line"], border_width=1, text_color=C["text"], font=font(13))

    def check(self, parent, text, var):
        return self.ctk.CTkCheckBox(parent, text=text, variable=var, corner_radius=5, border_width=2, checkbox_width=18,
                                    checkbox_height=18, fg_color=C["main"], hover_color=C["btn_hover"],
                                    border_color=C["faint"], checkmark_color=C["main_text"], text_color=C["text"],
                                    font=font(13))

    def card(self, parent, title, note=None):
        f = self.ctk.CTkFrame(parent, fg_color=C["card"], corner_radius=R)
        f.pack(fill="x", pady=(0, 12))
        inner = self.ctk.CTkFrame(f, fg_color="transparent")
        inner.pack(fill="x", padx=20, pady=16)
        self.label(inner, title, 15, bold=True).pack(anchor="w")
        if note:
            self.label(inner, note, 12, dim=True, wraplength=620, justify="left").pack(anchor="w", pady=(2, 12))
        return inner

    def hrow(self, parent, pady=(0, 10)):
        r = self.ctk.CTkFrame(parent, fg_color="transparent")
        r.pack(fill="x", pady=pady)
        return r

    def field(self, parent, title, var, width=90, hint=""):
        box = self.ctk.CTkFrame(parent, fg_color="transparent")
        box.pack(side="left", padx=(0, 14))
        self.label(box, title, 12, dim=True).pack(anchor="w")
        self.entry(box, var, width, hint).pack(anchor="w", pady=(2, 0))

    # ---------- раскладка ----------
    def build(self):
        ctk = self.ctk
        top = ctk.CTkFrame(self.root, fg_color="transparent")
        top.pack(fill="x", padx=20, pady=(16, 12))
        self.label(top, "Orbis", 20, bold=True).pack(side="left")
        self.label(top, "stickman → R6", 13, dim=True).pack(side="left", padx=(10, 0), pady=(4, 0))
        self.status = self.label(top, "", 12, dim=True)
        self.status.pack(side="right")
        self.weights = ctk.StringVar(value=self.cfg.get("weights", ""))
        self.wmenu = ctk.CTkOptionMenu(top, variable=self.weights, values=[""], width=200, height=32, corner_radius=RB,
                                       fg_color=C["btn"], button_color=C["btn"], button_hover_color=C["btn_hover"],
                                       dropdown_fg_color=C["card"], dropdown_hover_color=C["sel"],
                                       dropdown_text_color=C["text"], text_color=C["text"], font=font(13),
                                       dropdown_font=font(13))
        self.wmenu.pack(side="right", padx=(0, 16))
        self.label(top, "веса", 12, dim=True).pack(side="right", padx=(0, 8))

        body = ctk.CTkFrame(self.root, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=20)
        # слева — ролики
        left = ctk.CTkFrame(body, fg_color=C["panel"], corner_radius=R, width=300)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)
        head = ctk.CTkFrame(left, fg_color="transparent")
        head.pack(fill="x", padx=16, pady=(14, 6))
        self.label(head, "Ролики", 15, bold=True).pack(side="left")
        ctk.CTkButton(head, text="+", width=32, height=28, corner_radius=RB, fg_color=C["btn"],
                      hover_color=C["btn_hover"], text_color=C["text"], font=font(16), command=self.add_videos).pack(side="right")
        self.vbox = ctk.CTkScrollableFrame(left, fg_color="transparent", corner_radius=0,
                                           scrollbar_button_color=C["btn"], scrollbar_button_hover_color=C["btn_hover"])
        self.vbox.pack(fill="both", expand=True, padx=8)
        self.empty = self.label(self.vbox, "Нажмите «+», чтобы добавить ролики", 12, dim=True)
        foot = ctk.CTkFrame(left, fg_color="transparent")
        foot.pack(fill="x", padx=14, pady=(6, 6))
        self.button(foot, "Убрать", self.remove_videos).pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.button(foot, "Папка", self.open_folder).pack(side="left", fill="x", expand=True, padx=(4, 0))
        self.info = self.label(left, "", 12, dim=True, justify="left", anchor="w", wraplength=260)
        self.info.pack(fill="x", padx=18, pady=(4, 14))

        # справа — действия во вкладках
        right = ctk.CTkFrame(body, fg_color="transparent")
        right.pack(side="left", fill="both", expand=True, padx=(16, 0))
        tabs = ctk.CTkTabview(right, fg_color=C["panel"], corner_radius=R, segmented_button_fg_color=C["field"],
                              segmented_button_selected_color=C["btn_hover"],
                              segmented_button_selected_hover_color=C["btn_hover"],
                              segmented_button_unselected_color=C["field"],
                              segmented_button_unselected_hover_color=C["btn"], text_color=C["text"],
                              anchor="w")
        tabs.pack(fill="both", expand=True)
        tabs._segmented_button.configure(font=font(14), height=36, corner_radius=RB)
        for name, fill in (("Ролик", self.tab_video), ("Обучение", self.tab_train), ("Экзамен", self.tab_exam)):
            t = tabs.add(name)
            area = ctk.CTkScrollableFrame(t, fg_color="transparent", scrollbar_button_color=C["panel"],
                                          scrollbar_button_hover_color=C["btn"])
            area.pack(fill="both", expand=True, padx=4, pady=(4, 0))
            fill(area)

        # низ — прогресс и лог
        bottom = ctk.CTkFrame(self.root, fg_color=C["panel"], corner_radius=R)
        bottom.pack(fill="both", padx=20, pady=16)
        bar = ctk.CTkFrame(bottom, fg_color="transparent")
        bar.pack(fill="x", padx=16, pady=(12, 6))
        self.task = self.label(bar, "Готов", 13)
        self.task.pack(side="left")
        self.stop_btn = self.button(bar, "Стоп", self.stop, width=90)
        self.stop_btn.configure(state="disabled", text_color_disabled=C["faint"])
        self.stop_btn.pack(side="right")
        self.prog = ctk.CTkProgressBar(bar, width=300, height=6, corner_radius=3, fg_color=C["field"],
                                       progress_color=C["bar"])
        self.prog.set(0)
        self.prog.pack(side="right", padx=16)
        self.log = ctk.CTkTextbox(bottom, height=170, corner_radius=RB, fg_color=C["field"], text_color="#b4b4b8",
                                  border_width=0, font=ctk.CTkFont(family="Consolas" if sys.platform.startswith("win")
                                                                   else "DejaVu Sans Mono", size=12),
                                  scrollbar_button_color=C["btn"], wrap="word")
        self.log.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.log.configure(state="disabled")

    def tab_video(self, t):
        ctk = self.ctk
        s = self.card(t, "Разметка → анимация",
                      "Пипетка по персонажам, ответы на вопросы, экспорт .r6.json. Разметка сохраняется — можно продолжить.")
        r = self.hrow(s)
        self.start, self.end, self.ask = ctk.StringVar(), ctk.StringVar(), ctk.StringVar()
        self.field(r, "С кадра", self.start, 90)
        self.field(r, "По кадр (пусто — до конца)", self.end, 90)
        self.field(r, "Вопросов (пусто — авто)", self.ask, 90)
        self.fresh = ctk.BooleanVar(value=False)
        self.check(r, "Начать заново", self.fresh).pack(side="left", pady=(18, 0))
        r = self.hrow(s, (4, 0))
        self.button(r, "Разметить", self.do_label, main=True).pack(side="left")
        self.button(r, "Сравнить с R6", self.do_compare).pack(side="left", padx=8)
        self.button(r, "Открыть сравнение", self.open_compare).pack(side="left")

        s = self.card(t, "Blender", "Scripting → Open → blender_import_ikrig.py → Run Script → выбрать .r6.json ролика.")
        r = self.hrow(s, (0, 0))
        self.button(r, "Путь к импорту", self.copy_importer).pack(side="left")
        self.button(r, "Путь к .r6.json", self.copy_r6).pack(side="left", padx=8)

    def tab_train(self, t):
        ctk = self.ctk
        s = self.card(t, "Данные", "База движений людей — один раз (~15 мин). Самообучение — уверенные кадры выбранных роликов "
                                   "(или всех в списке) идут в обучение.")
        r = self.hrow(s, (0, 0))
        self.button(r, "Скачать базу движений",
                    lambda: self.run("Скачиваю базу движений", ["tools/fetch_mocap.py"])).pack(side="left")
        self.button(r, "Самообучение", self.do_selftrain, main=True).pack(side="left", padx=8)

        s = self.card(t, "Обучение сети", "Продолжает веса, выбранные сверху. Сохраняет каждые 200 шагов — если оборвётся, "
                                          "«Продолжить».")
        r = self.hrow(s)
        self.hours = ctk.StringVar(value=self.cfg.get("hours", "8"))
        self.sps = ctk.StringVar(value=self.cfg.get("sps", "0.08"))
        self.out = ctk.StringVar(value=self.cfg.get("out", "orbis2.pt"))
        self.field(r, "Часов", self.hours, 70)
        self.field(r, "с/шаг (из лога)", self.sps, 90)
        self.field(r, "Сохранить как", self.out, 170)
        r = self.hrow(s)
        self.use_real, self.use_pseudo = ctk.BooleanVar(value=True), ctk.BooleanVar(value=True)
        self.check(r, "Мои разметки", self.use_real).pack(side="left")
        self.check(r, "Самообучение", self.use_pseudo).pack(side="left", padx=(18, 0))
        r = self.hrow(s, (4, 0))
        self.button(r, "Обучить", self.do_train, main=True).pack(side="left")
        self.button(r, "Продолжить оборванное", self.do_resume).pack(side="left", padx=8)

    def tab_exam(self, t):
        ctk = self.ctk
        s = self.card(t, "Эталон", "Один раз на ролик: ~30 кадров, проверить и поправить каждую из 11 точек. "
                                   "В обучение не идёт.")
        r = self.hrow(s, (0, 0))
        self.button(r, "Разметить экзамен по ролику", self.do_exam_make).pack(side="left")
        s = self.card(t, "Сдать", "Средняя ошибка углов частей R6 — меньше лучше. Сравниваются веса сверху и перечисленные ниже.")
        r = self.hrow(s)
        self.cmp = ctk.StringVar(value=self.cfg.get("cmp", ""))
        self.field(r, "Сравнить ещё с (через запятую: orbis.pt, orbis2.pt)", self.cmp, 320)
        r = self.hrow(s, (4, 0))
        self.button(r, "Сдать экзамен", self.do_exam_run, main=True).pack(side="left")

    # ---------- список роликов ----------
    def refresh_videos(self):
        self.cfg["videos"] = [v for v in self.cfg.get("videos", []) if os.path.exists(v)]
        for w in self.rows:
            w.destroy()
        self.rows = []
        self.sel = {i for i in self.sel if i < len(self.cfg["videos"])}
        if self.cfg["videos"]:
            self.empty.pack_forget()
        else:
            self.empty.pack(pady=20)
        for i, v in enumerate(self.cfg["videos"]):
            row = self.ctk.CTkFrame(self.vbox, fg_color="transparent", corner_radius=RB, height=44)
            row.pack(fill="x", pady=2)
            name = self.label(row, os.path.basename(v), 13, anchor="w")
            name.pack(fill="x", padx=12, pady=(6, 0))
            sub = self.label(row, self.video_state(v), 11, dim=True, anchor="w")
            sub.pack(fill="x", padx=12, pady=(0, 6))
            for w in (row, name, sub):
                w.bind("<Button-1>", lambda e, i=i: self.click_row(i, e))
            self.rows.append(row)
        self.paint_rows()

    def video_state(self, v):
        base = os.path.splitext(v)[0]
        if os.path.exists(base + ".r6.json"):
            return "анимация готова"
        if os.path.exists(base + ".s2r.json"):
            return "размечается"
        return "новый"

    def click_row(self, i, e):
        if e.state & 0x4:            # Ctrl — добавить/убрать из выбора
            self.sel ^= {i}
        elif e.state & 0x1 and self.sel:   # Shift — диапазон
            a = min(self.sel)
            self.sel = set(range(min(a, i), max(a, i) + 1))
        else:
            self.sel = {i}
        self.paint_rows()
        self.show_info()

    def paint_rows(self):
        for i, row in enumerate(self.rows):
            row.configure(fg_color=C["sel"] if i in self.sel else "transparent")

    def selected(self, need=True):
        vids = [self.cfg["videos"][i] for i in sorted(self.sel)]
        if need and not vids:
            self.say("Выберите ролик слева.")
        return vids

    def show_info(self):
        v = self.selected(False)
        if len(v) != 1:
            self.info.configure(text=f"выбрано роликов: {len(v)}" if v else "")
            return
        base = os.path.splitext(v[0])[0]
        lines = [f"папка: {os.path.basename(os.path.dirname(v[0]))}"]
        if os.path.exists(base + ".s2r.json"):
            try:
                pj = json.load(open(base + ".s2r.json", encoding="utf-8"))
                lines.append(f"размечено кадров: {sum(len(L) for L in pj.get('labels', []))}, "
                             f"персонажей: {len(pj.get('colors', []))}")
            except Exception:
                pass
        if os.path.exists(base + "_compare.mp4"):
            lines.append("есть сравнение с R6")
        self.info.configure(text="\n".join(lines))

    # ---------- настройки ----------
    def load_cfg(self):
        try:
            return json.load(open(CFG, encoding="utf-8"))
        except Exception:
            return {}

    def save_cfg(self):
        self.cfg.update(weights=self.weights.get(), hours=self.hours.get(), out=self.out.get(), sps=self.sps.get(),
                        cmp=self.cmp.get(), geometry=self.root.geometry())
        try:
            json.dump(self.cfg, open(CFG, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        except OSError:
            pass

    def refresh_weights(self):
        files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(HERE, "weights", "*.pt")))
        self.wmenu.configure(values=files or [""])
        if self.weights.get() not in files and files:
            self.weights.set("orbis.pt" if "orbis.pt" in files else files[-1])

    def add_videos(self):
        from tkinter import filedialog
        files = filedialog.askopenfilenames(title="Ролики", filetypes=[("Видео", " ".join("*" + e for e in VIDEO_EXT))])
        for f in files:
            if f not in self.cfg.setdefault("videos", []):
                self.cfg["videos"].append(f)
        self.refresh_videos()
        self.save_cfg()

    def remove_videos(self):
        for v in self.selected():
            self.cfg["videos"].remove(v)
        self.sel = set()
        self.refresh_videos()
        self.show_info()
        self.save_cfg()

    def open_folder(self):
        v = self.selected(False)
        self.open_path(os.path.dirname(v[0]) if v else HERE)

    @staticmethod
    def open_path(p):
        if sys.platform.startswith("win"):
            os.startfile(p)
        else:
            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", p])

    def copy(self, text, what):
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.say(f"Скопировано: {what}")

    def copy_importer(self):
        self.copy(os.path.join(HERE, "blender_import_ikrig.py"), "путь к blender_import_ikrig.py")

    def copy_r6(self):
        v = self.selected()
        if v:
            self.copy(os.path.splitext(v[0])[0] + ".r6.json", "путь к .r6.json")

    def wpath(self, name=None):
        return os.path.join(HERE, "weights", name or self.weights.get())

    # ---------- действия ----------
    def do_label(self):
        v = self.selected()
        if not v:
            return
        args = ["stick2r6.py", v[0], "--weights", self.wpath()]
        for flag, var in (("--start", self.start), ("--end", self.end), ("--ask", self.ask)):
            if var.get().strip():
                args += [flag, var.get().strip()]
        if self.fresh.get():
            args.append("--fresh")
        self.run(f"Разметка: {os.path.basename(v[0])}", args)

    def do_compare(self):
        v = self.selected()
        if v:
            self.run(f"Сравнение: {os.path.basename(v[0])}", ["compare.py", v[0]], then=self.open_compare)

    def open_compare(self):
        v = self.selected()
        if v:
            p = os.path.splitext(v[0])[0] + "_compare.mp4"
            self.open_path(p) if os.path.exists(p) else self.say("Сравнения ещё нет — «Сравнить с R6».")

    def do_selftrain(self):
        vids = self.selected(False) or self.cfg.get("videos", [])
        if not vids:
            return self.say("Добавьте ролики.")
        self.run(f"Самообучение: роликов {len(vids)}", ["tools/selftrain.py", "--weights", self.wpath()] + vids)

    def train_args(self):
        args = []
        if self.use_real.get():
            args += ["--real", os.path.join(HERE, "weights", "real"), "--real_frac", "0.05"]
        if self.use_pseudo.get():
            args += ["--pseudo", os.path.join(HERE, "weights", "pseudo")]
        return args

    def do_train(self):
        try:
            steps = int(float(self.hours.get().replace(",", ".")) * 3600 / float(self.sps.get().replace(",", ".")))
        except ValueError:
            return self.say("Часы и с/шаг — числа.")
        self.save_cfg()
        self.run(f"Обучение → {self.out.get()}", ["stickpose/train.py", "--resume", self.wpath(), "--steps", str(steps),
                                                  "--out", self.wpath(self.out.get())] + self.train_args())

    def do_resume(self):
        out = self.wpath(self.out.get())
        if not os.path.exists(out):
            return self.say(f"Нет {self.out.get()} — нечего продолжать.")
        self.run(f"Продолжаю обучение → {self.out.get()}", ["stickpose/train.py", "--resume", out, "--out", out]
                 + self.train_args())

    def do_exam_make(self):
        v = self.selected()
        if v:
            self.run(f"Экзамен: разметка {os.path.basename(v[0])}",
                     ["tools/exam.py", "make", v[0], "--weights", self.wpath()])

    def do_exam_run(self):
        ws = [self.wpath()] + [self.wpath(w.strip()) for w in self.cmp.get().split(",") if w.strip()]
        self.save_cfg()
        self.run("Экзамен", ["tools/exam.py", "run"] + ws)

    # ---------- процессы ----------
    def run(self, title, args, then=None):
        if self.proc is not None:
            return self.say("Уже идёт задача — дождитесь или «Стоп».")
        script = os.path.join(HERE, args[0])
        py = sys.executable
        if getattr(sys, "frozen", False):  # .exe — это только окно; считает установленный Python
            import shutil
            py = shutil.which("python") or shutil.which("py") or "python"
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        self.clear_log()
        self.say(f"▸ {title}")
        self.task.configure(text=title)
        self.status.configure(text="")
        self.prog.set(0)
        self.stop_btn.configure(state="normal")
        self.proc = subprocess.Popen([py, "-u", script] + args[1:], cwd=os.path.dirname(script), stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, env=env,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.then = then
        threading.Thread(target=self.reader, args=(self.proc,), daemon=True).start()

    def reader(self, proc):
        buf = b""
        while True:
            ch = proc.stdout.read(1)
            if not ch:
                break
            if ch in (b"\n", b"\r"):
                if buf.strip():
                    self.q.put(("line", buf.decode("utf-8", "replace"), ch == b"\r"))
                buf = b""
            else:
                buf += ch
        if buf.strip():
            self.q.put(("line", buf.decode("utf-8", "replace"), False))
        self.q.put(("done", proc.wait(), None))

    def pump(self):
        try:
            while True:
                kind, a, b = self.q.get_nowait()
                if kind == "line":
                    self.progress(a)
                    self.say(a, replace_last=b)
                else:
                    ok = a == 0
                    self.task.configure(text="Готово" if ok else f"Остановлено (код {a})")
                    if ok:
                        self.prog.set(1)
                    self.stop_btn.configure(state="disabled")
                    self.proc = None
                    self.refresh_videos()
                    self.show_info()
                    self.refresh_weights()
                    if ok and self.then:
                        self.then()
        except queue.Empty:
            pass
        self.root.after(80, self.pump)

    def progress(self, line):
        m = re.search(r"(?:шаг|кадр)\s+(\d+)\s*/\s*(\d+)", line) or re.search(r"\[(?:вопрос|поза)\s+(\d+)\s*/\s*(\d+)\]", line)
        if m and int(m.group(2)) > 0:
            self.prog.set(int(m.group(1)) / int(m.group(2)))
        m = re.search(r"шаг\s+(\d+)/(\d+).*?([\d.]+)\s*с/шаг", line)
        if m:
            left = (int(m.group(2)) - int(m.group(1))) * float(m.group(3))
            self.status.configure(text=f"осталось ~{int(left // 3600)} ч {int(left % 3600 // 60)} мин")

    def stop(self):
        if self.proc is not None:
            self.proc.terminate()
            self.say("— остановлено —")

    def clear_log(self):
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def say(self, text, replace_last=False):
        self.log.configure(state="normal")
        if replace_last and self._last_cr:
            self.log.delete("end-2l", "end-1l")
        self.log.insert("end", text + "\n")
        self._last_cr = replace_last
        self.log.see("end")
        self.log.configure(state="disabled")

    def on_close(self):
        if self.proc is not None:
            self.proc.terminate()
        self.save_cfg()
        self.root.destroy()


def main():
    try:
        import customtkinter as ctk
    except ImportError:
        sys.exit("Нужен customtkinter:  pip install customtkinter")
    ctk.set_appearance_mode("dark")
    root = ctk.CTk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
