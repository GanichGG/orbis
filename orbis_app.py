"""Orbis — оконный инструмент: ролики слева, действия справа, прогресс и лог внизу. Серая тема без акцентов.

    python orbis_app.py            (или Orbis.exe, см. build_exe.bat)

Все действия запускают те же скрипты, что и из терминала (stick2r6.py, compare.py, tools/*.py, stickpose/train.py),
отдельным процессом: окно не зависает, «Стоп» прерывает. Собранный Orbis.exe (build_exe.bat) — только окно,
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

# серая палитра: от тёмного фона к светлому тексту, без цветных акцентов
C = {"bg": "#1d1d1d", "panel": "#252525", "field": "#1a1a1a", "line": "#323232", "btn": "#303030",
     "btn_hover": "#3a3a3a", "btn_press": "#444444", "main": "#3d3d3d", "main_hover": "#484848", "sel": "#3a3a3a",
     "text": "#d6d6d6", "dim": "#7d7d7d", "bar": "#8a8a8a"}
FONT = ("Segoe UI", 10)
HEAD = ("Segoe UI Semibold", 11)
MONO = ("Consolas", 9)


class App:
    def __init__(self, root):
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.ttk, self.root = tk, ttk, root
        self.proc, self.q = None, queue.Queue()
        self.cfg = self.load_cfg()
        root.title("Orbis")
        root.geometry(self.cfg.get("geometry", "1160x760"))
        root.minsize(900, 560)
        root.configure(bg=C["bg"])
        self.style()
        self.build()
        self.refresh_weights()
        self.refresh_videos()
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.after(80, self.pump)

    # ---------- оформление ----------
    def style(self):
        s = self.ttk.Style()
        s.theme_use("clam")
        s.configure(".", background=C["bg"], foreground=C["text"], fieldbackground=C["field"], bordercolor=C["line"],
                    lightcolor=C["line"], darkcolor=C["line"], troughcolor=C["field"], font=FONT, focuscolor=C["bg"])
        s.configure("TFrame", background=C["bg"])
        s.configure("Panel.TFrame", background=C["panel"])
        s.configure("TLabel", background=C["bg"], foreground=C["text"])
        s.configure("Panel.TLabel", background=C["panel"])
        s.configure("Dim.TLabel", background=C["panel"], foreground=C["dim"])
        s.configure("Head.TLabel", background=C["panel"], foreground=C["text"], font=HEAD)
        s.configure("Title.TLabel", background=C["bg"], foreground=C["text"], font=("Segoe UI Semibold", 13))
        s.configure("Bar.TLabel", background=C["bg"], foreground=C["dim"])
        for name, base, hover in (("TButton", C["btn"], C["btn_hover"]), ("Main.TButton", C["main"], C["main_hover"])):
            s.configure(name, background=base, foreground=C["text"], borderwidth=0, padding=(14, 7), relief="flat",
                        bordercolor=base, lightcolor=base, darkcolor=base, focusthickness=0)
            s.map(name, background=[("pressed", C["btn_press"]), ("active", hover), ("disabled", C["panel"])],
                  foreground=[("disabled", C["dim"])], bordercolor=[("active", hover)],
                  lightcolor=[("active", hover)], darkcolor=[("active", hover)])
        s.configure("TEntry", fieldbackground=C["field"], foreground=C["text"], insertcolor=C["text"], padding=5,
                    bordercolor=C["line"], lightcolor=C["field"], darkcolor=C["field"])
        s.map("TEntry", bordercolor=[("focus", "#5a5a5a")])
        s.configure("TSpinbox", fieldbackground=C["field"], foreground=C["text"], arrowcolor=C["dim"],
                    background=C["btn"], padding=3)
        s.configure("TCombobox", fieldbackground=C["field"], foreground=C["text"], arrowcolor=C["dim"],
                    background=C["btn"], padding=4, bordercolor=C["line"], lightcolor=C["field"], darkcolor=C["field"])
        s.map("TCombobox", fieldbackground=[("readonly", C["field"])], foreground=[("readonly", C["text"])],
              selectbackground=[("readonly", C["field"])], selectforeground=[("readonly", C["text"])])
        s.configure("TCheckbutton", background=C["panel"], foreground=C["text"], indicatorbackground=C["field"],
                    indicatorforeground=C["text"])
        s.map("TCheckbutton", background=[("active", C["panel"])], indicatorbackground=[("selected", C["sel"])])
        s.configure("Horizontal.TProgressbar", background=C["bar"], troughcolor=C["field"], bordercolor=C["line"],
                    lightcolor=C["bar"], darkcolor=C["bar"], thickness=6)
        s.configure("TNotebook", background=C["bg"], borderwidth=0, tabmargins=0, bordercolor=C["bg"],
                    lightcolor=C["panel"], darkcolor=C["panel"])
        s.configure("TNotebook.Tab", background=C["bg"], foreground=C["dim"], padding=(18, 8), borderwidth=0,
                    bordercolor=C["bg"], lightcolor=C["bg"], darkcolor=C["bg"])
        s.map("TNotebook.Tab", background=[("selected", C["panel"]), ("active", C["btn"])],
              foreground=[("selected", C["text"])], lightcolor=[("selected", C["panel"])],
              bordercolor=[("selected", C["panel"])])
        s.layout("TNotebook.Tab", [("Notebook.tab", {"sticky": "nswe", "children": [
            ("Notebook.padding", {"side": "top", "sticky": "nswe", "children": [
                ("Notebook.label", {"side": "top", "sticky": ""})]})]})])
        self.root.option_add("*TCombobox*Listbox.background", C["field"])
        self.root.option_add("*TCombobox*Listbox.foreground", C["text"])
        self.root.option_add("*TCombobox*Listbox.selectBackground", C["sel"])

    def build(self):
        tk, ttk = self.tk, self.ttk
        top = ttk.Frame(self.root, padding=(16, 14, 16, 10))
        top.pack(fill="x")
        ttk.Label(top, text="Orbis", style="Title.TLabel").pack(side="left", padx=(0, 24))
        ttk.Label(top, text="Веса", style="Bar.TLabel").pack(side="left")
        self.weights = tk.StringVar(value=self.cfg.get("weights", ""))
        self.wbox = ttk.Combobox(top, textvariable=self.weights, state="readonly", width=26)
        self.wbox.pack(side="left", padx=(8, 4))
        ttk.Button(top, text="Обновить", command=self.refresh_weights).pack(side="left")
        self.status = ttk.Label(top, text="", style="Bar.TLabel")
        self.status.pack(side="right")

        body = ttk.Frame(self.root, padding=(16, 0, 16, 0))
        body.pack(fill="both", expand=True)
        # слева — ролики
        left = ttk.Frame(body, style="Panel.TFrame", padding=14)
        left.pack(side="left", fill="y")
        ttk.Label(left, text="Ролики", style="Head.TLabel").pack(anchor="w")
        self.vlist = tk.Listbox(left, width=30, selectmode="extended", bg=C["field"], fg=C["text"],
                                selectbackground=C["sel"], selectforeground=C["text"], highlightthickness=0,
                                borderwidth=6, relief="flat", activestyle="none", font=FONT)
        self.vlist.pack(fill="both", expand=True, pady=8)
        self.vlist.bind("<<ListboxSelect>>", lambda e: self.show_info())
        row = ttk.Frame(left, style="Panel.TFrame")
        row.pack(fill="x")
        ttk.Button(row, text="Добавить", command=self.add_videos).pack(side="left")
        ttk.Button(row, text="Убрать", command=self.remove_videos).pack(side="left", padx=6)
        ttk.Button(row, text="Папка", command=self.open_folder).pack(side="right")
        self.info = ttk.Label(left, text="", style="Dim.TLabel", wraplength=250, justify="left")
        self.info.pack(anchor="w", pady=(8, 0))

        # справа — действия во вкладках
        nb = ttk.Notebook(body)
        nb.pack(side="left", fill="both", expand=True, padx=(14, 0))
        self.tab_video(nb)
        self.tab_train(nb)
        self.tab_exam(nb)

        # низ — прогресс и лог
        bottom = ttk.Frame(self.root, padding=(16, 12, 16, 16))
        bottom.pack(fill="both")
        bar = ttk.Frame(bottom)
        bar.pack(fill="x")
        self.task = ttk.Label(bar, text="Готов", style="Bar.TLabel")
        self.task.pack(side="left")
        self.stop_btn = ttk.Button(bar, text="Стоп", command=self.stop, state="disabled")
        self.stop_btn.pack(side="right")
        self.prog = ttk.Progressbar(bar, mode="determinate", length=280)
        self.prog.pack(side="right", padx=14)
        self.log = tk.Text(bottom, height=9, bg=C["field"], fg="#a8a8a8", insertbackground=C["text"],
                           highlightthickness=0, borderwidth=0, font=MONO, wrap="word", padx=12, pady=10,
                           spacing1=1, spacing3=1)
        self.log.pack(fill="both", expand=True, pady=(10, 0))
        self.log.configure(state="disabled")

    def section(self, parent, title, note=None):
        f = self.ttk.Frame(parent, style="Panel.TFrame", padding=(20, 16, 20, 6))
        f.pack(fill="x")
        self.ttk.Label(f, text=title, style="Head.TLabel").pack(anchor="w")
        if note:
            self.ttk.Label(f, text=note, style="Dim.TLabel", wraplength=600, justify="left").pack(anchor="w", pady=(3, 10))
        return f

    def field(self, parent, label, var, width=8, kind="entry", **kw):
        r = self.ttk.Frame(parent, style="Panel.TFrame")
        r.pack(side="left", padx=(0, 18))
        self.ttk.Label(r, text=label, style="Dim.TLabel").pack(anchor="w", pady=(0, 3))
        w = (self.ttk.Spinbox(r, textvariable=var, width=width, **kw) if kind == "spin" else
             self.ttk.Entry(r, textvariable=var, width=width))
        w.pack(anchor="w")
        return w

    def row(self, parent):
        r = self.ttk.Frame(parent, style="Panel.TFrame")
        r.pack(fill="x", pady=(6, 4))
        return r

    def tab_video(self, nb):
        tk, ttk = self.tk, self.ttk
        t = ttk.Frame(nb, style="Panel.TFrame")
        nb.add(t, text="Ролик")
        s = self.section(t, "Разметка → анимация",
                         "Пипетка по персонажам, ответы на вопросы, экспорт .r6.json. Разметка сохраняется — можно продолжить.")
        r = self.row(s)
        self.start, self.end, self.ask = tk.StringVar(value=""), tk.StringVar(value=""), tk.StringVar(value="")
        self.field(r, "С кадра", self.start, 7)
        self.field(r, "По кадр", self.end, 7)
        self.field(r, "Вопросов (пусто — авто)", self.ask, 7)
        self.fresh = tk.BooleanVar(value=False)
        ttk.Checkbutton(r, text="Начать заново", variable=self.fresh).pack(side="left", pady=(18, 0))
        r = self.row(s)
        ttk.Button(r, text="Разметить", command=self.do_label, style="Main.TButton").pack(side="left")
        ttk.Button(r, text="Сравнить с R6", command=self.do_compare).pack(side="left", padx=6)
        ttk.Button(r, text="Открыть сравнение", command=self.open_compare).pack(side="left")

        s = self.section(t, "Blender",
                         "Scripting → Open → blender_import_ikrig.py → Run Script → выбрать .r6.json ролика.")
        r = self.row(s)
        ttk.Button(r, text="Скопировать путь к импорту", command=self.copy_importer).pack(side="left")
        ttk.Button(r, text="Скопировать путь к .r6.json", command=self.copy_r6).pack(side="left", padx=6)

    def tab_train(self, nb):
        tk, ttk = self.tk, self.ttk
        t = ttk.Frame(nb, style="Panel.TFrame")
        nb.add(t, text="Обучение")
        s = self.section(t, "Данные",
                         "База движений людей (один раз, ~15 мин) и самообучение: уверенные кадры выбранных роликов "
                         "(или всех в списке) идут в обучение.")
        r = self.row(s)
        ttk.Button(r, text="Скачать базу движений", command=lambda: self.run("Скачиваю базу движений",
                   ["tools/fetch_mocap.py"])).pack(side="left")
        ttk.Button(r, text="Самообучение", command=self.do_selftrain, style="Main.TButton").pack(side="left", padx=6)

        s = self.section(t, "Обучение сети",
                         "Продолжает выбранные сверху веса. Каждые 200 шагов сохраняет — если оборвётся, «Продолжить».")
        r = self.row(s)
        self.hours, self.out = tk.StringVar(value=self.cfg.get("hours", "8")), tk.StringVar(value=self.cfg.get("out", "orbis2.pt"))
        self.sps = tk.StringVar(value=self.cfg.get("sps", "0.08"))
        self.field(r, "Часов", self.hours, 6)
        self.field(r, "с/шаг (из лога)", self.sps, 6)
        self.field(r, "Сохранить как", self.out, 18)
        self.use_real, self.use_pseudo = tk.BooleanVar(value=True), tk.BooleanVar(value=True)
        ttk.Checkbutton(r, text="Мои разметки", variable=self.use_real).pack(side="left", pady=(18, 0))
        ttk.Checkbutton(r, text="Самообучение", variable=self.use_pseudo).pack(side="left", padx=(12, 0), pady=(18, 0))
        r = self.row(s)
        ttk.Button(r, text="Обучить", command=self.do_train, style="Main.TButton").pack(side="left")
        ttk.Button(r, text="Продолжить оборванное", command=self.do_resume).pack(side="left", padx=6)

    def tab_exam(self, nb):
        ttk = self.ttk
        t = ttk.Frame(nb, style="Panel.TFrame")
        nb.add(t, text="Экзамен")
        s = self.section(t, "Эталон",
                         "Один раз на ролик: ~30 кадров, проверить и поправить каждую из 11 точек. В обучение не идёт.")
        r = self.row(s)
        ttk.Button(r, text="Разметить экзамен по ролику", command=self.do_exam_make).pack(side="left")
        s = self.section(t, "Сдать",
                         "Средняя ошибка углов частей R6 (меньше — лучше). Сравниваются выбранные сверху веса и все из списка ниже.")
        r = self.row(s)
        self.cmp = self.tk.StringVar(value=self.cfg.get("cmp", ""))
        self.field(r, "Сравнить ещё с (через запятую)", self.cmp, 40)
        r = self.row(s)
        ttk.Button(r, text="Сдать экзамен", command=self.do_exam_run, style="Main.TButton").pack(side="left")

    # ---------- данные ----------
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
        self.wbox["values"] = files
        if self.weights.get() not in files and files:
            self.weights.set("orbis.pt" if "orbis.pt" in files else files[-1])

    def refresh_videos(self):
        self.cfg["videos"] = [v for v in self.cfg.get("videos", []) if os.path.exists(v)]
        self.vlist.delete(0, "end")
        for v in self.cfg["videos"]:
            self.vlist.insert("end", os.path.basename(v))

    def selected(self, need=True):
        idx = self.vlist.curselection()
        vids = [self.cfg["videos"][i] for i in idx]
        if need and not vids:
            self.say("Выберите ролик слева.")
        return vids

    def show_info(self):
        v = self.selected(False)
        if len(v) != 1:
            self.info.configure(text=f"выбрано: {len(v)}" if v else "")
            return
        base = os.path.splitext(v[0])[0]
        marks = []
        if os.path.exists(base + ".s2r.json"):
            try:
                pj = json.load(open(base + ".s2r.json", encoding="utf-8"))
                marks.append(f"разметка: {sum(len(L) for L in pj.get('labels', []))} кадров, персонажей {len(pj.get('colors', []))}")
            except Exception:
                marks.append("разметка есть")
        marks += [t for ext, t in ((".r6.json", "анимация готова"), ("_compare.mp4", "сравнение есть"))
                  if os.path.exists(base + ext)]
        folder = os.path.basename(os.path.dirname(v[0])) or os.path.dirname(v[0])
        self.info.configure(text=f"папка: {folder}\n" + ("\n".join(marks) if marks else "ещё не размечен"))

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
        self.refresh_videos()
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
            steps = int(float(self.hours.get()) * 3600 / float(self.sps.get()))
        except ValueError:
            return self.say("Часы и с/шаг — числа.")
        out = self.wpath(self.out.get())
        self.save_cfg()
        self.run(f"Обучение → {self.out.get()} ({steps} шагов)",
                 ["stickpose/train.py", "--resume", self.wpath(), "--steps", str(steps), "--out", out] + self.train_args())

    def do_resume(self):
        out = self.wpath(self.out.get())
        if not os.path.exists(out):
            return self.say(f"Нет {self.out.get()} — нечего продолжать.")
        self.run(f"Продолжаю обучение → {self.out.get()}", ["stickpose/train.py", "--resume", out, "--out", out]
                 + self.train_args())

    def do_exam_make(self):
        v = self.selected()
        if v:
            self.run(f"Экзамен: разметка {os.path.basename(v[0])}", ["tools/exam.py", "make", v[0], "--weights", self.wpath()])

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
        cmd = [py, "-u", script] + args[1:]
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        self.clear_log()
        self.say(f"▸ {title}")
        self.task.configure(text=title)
        self.prog.configure(value=0)
        self.stop_btn.configure(state="normal")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.proc = subprocess.Popen(cmd, cwd=os.path.dirname(script), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     env=env, creationflags=flags)
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
                        self.prog.configure(value=100)
                    self.stop_btn.configure(state="disabled")
                    self.proc = None
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
            self.prog.configure(value=100 * int(m.group(1)) / int(m.group(2)))
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
        if replace_last and getattr(self, "_last_cr", False):
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
    import tkinter as tk
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
