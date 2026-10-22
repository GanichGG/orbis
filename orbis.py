"""Orbis — меню в терминале: ролики и все действия в одном месте.

    python orbis.py

Цифра — действие, Enter — назад. Список роликов и настройки хранятся в orbis.json рядом.
Действия запускают те же скрипты (stick2r6.py, compare.py, tools/*.py, stickpose/train.py), их вывод идёт сюда же;
Ctrl+C прерывает текущее действие и возвращает в меню.
"""
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = os.path.join(HERE, "orbis.json")
VIDEO_EXT = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")

if os.name == "nt":
    os.system("")  # включить ANSI-цвета в консоли Windows
DIM, BOLD, RST = "\033[90m", "\033[1m", "\033[0m"


def load():
    try:
        return json.load(open(CFG, encoding="utf-8"))
    except Exception:
        return {"videos": [], "weights": "orbis.pt", "hours": 8, "sps": 0.08}


def save(cfg):
    json.dump(cfg, open(CFG, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def ask(prompt, default=""):
    try:
        v = input(f"{prompt}{DIM}{f' [{default}]' if default != '' else ''}{RST}: ").strip()
    except EOFError:  # ввод закрыт — выходим
        sys.exit(0)
    return v or str(default)


def state(v):
    b = os.path.splitext(v)[0]
    return "анимация готова" if os.path.exists(b + ".r6.json") else "размечается" if os.path.exists(b + ".s2r.json") else "новый"


def run(args):
    print(f"{DIM}› {' '.join(os.path.basename(a) for a in args)}{RST}\n")
    try:
        subprocess.run([sys.executable, "-u", os.path.join(HERE, args[0])] + args[1:], cwd=os.path.dirname(os.path.join(HERE, args[0])))
    except KeyboardInterrupt:
        print(f"\n{DIM}— прервано —{RST}")
    input(f"\n{DIM}Enter — в меню{RST}")


def weights_list():
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(HERE, "weights", "*.pt")))


def pick_video(cfg, many=False):
    vids = cfg["videos"]
    if not vids:
        input(f"{DIM}Список пуст — сначала «добавить ролики». Enter{RST}")
        return []
    for i, v in enumerate(vids, 1):
        print(f"  {i:>2}  {os.path.basename(v)}  {DIM}{state(v)}{RST}")
    s = ask("Номер" + (" (несколько через пробел, пусто — все)" if many else ""), "" if many else 1)
    if many and not s:
        return list(vids)
    try:
        return [vids[int(x) - 1] for x in s.replace(",", " ").split()]
    except (ValueError, IndexError):
        return []


def header(cfg):
    os.system("cls" if os.name == "nt" else "clear")
    print(f"{BOLD}Orbis{RST}  {DIM}stickman → R6   веса: {cfg['weights']}   роликов: {len(cfg['videos'])}{RST}\n")


def menu(cfg):
    items = [
        ("Ролики", None),
        ("добавить ролики", "add"), ("убрать ролик", "remove"),
        ("Анимация", None),
        ("разметить ролик → .r6.json", "label"), ("сравнить с R6 (видео)", "compare"),
        ("оценить анимацию (0–10)", "anim"),
        ("Обучение", None),
        ("скачать базу движений (один раз)", "mocap"), ("самообучение по роликам", "self"),
        ("обучить на ночь", "train"), ("продолжить оборванное обучение", "resume"),
        ("Экзамен", None),
        ("разметить экзамен по ролику", "exam_make"), ("сдать экзамен", "exam_run"),
        ("", None), ("выбрать веса", "weights"),
    ]
    keys, n = {}, 0
    for title, key in items:
        if key is None:
            print(f"\n{DIM}{title}{RST}" if title else "")
            continue
        n += 1
        keys[str(n)] = key
        print(f"  {n:>2}  {title}")
    return keys


def main():
    cfg = load()
    while True:
        cfg["videos"] = [v for v in cfg.get("videos", []) if os.path.exists(v)]
        header(cfg)
        keys = menu(cfg)
        try:
            choice = ask("\nДействие (q — выход)").lower()
        except KeyboardInterrupt:
            break
        if choice in ("q", "й", "exit"):
            break
        k = keys.get(choice)
        if k is None:
            continue
        w = os.path.join(HERE, "weights", cfg["weights"])
        print()
        if k == "add":
            p = ask("Путь к ролику или папке (можно перетащить в окно)").strip('"')
            if os.path.isdir(p):
                new = sorted(os.path.join(p, f) for f in os.listdir(p) if f.lower().endswith(VIDEO_EXT))
            else:
                new = [p] if os.path.exists(p) else []
            cfg["videos"] += [v for v in new if v not in cfg["videos"]]
            print(f"{DIM}добавлено: {len(new)}{RST}")
        elif k == "remove":
            for v in pick_video(cfg):
                cfg["videos"].remove(v)
        elif k == "label":
            v = pick_video(cfg)
            if v:
                args = ["stick2r6.py", v[0], "--weights", w]
                s, e = ask("С кадра", 0), ask("По кадр (пусто — до конца)")
                args += ["--start", s] + (["--end", e] if e else [])
                if ask("Начать заново? (д/н)", "н").lower().startswith(("д", "y")):
                    args.append("--fresh")
                run(args)
        elif k == "anim":
            v = pick_video(cfg)
            if v:
                j = os.path.splitext(v[0])[0] + ".r6.json"
                run(["tools/anim_check.py", j]) if os.path.exists(j) else input(f"{DIM}Сначала «разметить ролик». Enter{RST}")
        elif k == "compare":
            v = pick_video(cfg)
            if v:
                run(["compare.py", v[0]])
                out = os.path.splitext(v[0])[0] + "_compare.mp4"
                if os.path.exists(out) and os.name == "nt":
                    os.startfile(out)
        elif k == "mocap":
            run(["tools/fetch_mocap.py"])
        elif k == "self":
            vids = pick_video(cfg, many=True)
            if vids:
                run(["tools/selftrain.py", "--weights", w] + vids)
        elif k in ("train", "resume"):
            out = ask("Сохранить как", cfg.get("out", "orbis2.pt"))
            cfg["out"] = out
            extra = ["--real", os.path.join(HERE, "weights", "real"), "--real_frac", "0.05",
                     "--pseudo", os.path.join(HERE, "weights", "pseudo")]
            if k == "train":
                h = float(ask("Часов", cfg.get("hours", 8)))
                sps = float(ask("Секунд на шаг (из прошлого лога)", cfg.get("sps", 0.08)))
                cfg["hours"], cfg["sps"] = h, sps
                save(cfg)
                run(["stickpose/train.py", "--resume", w, "--steps", str(int(h * 3600 / sps)),
                     "--out", os.path.join(HERE, "weights", out)] + extra)
            else:
                p = os.path.join(HERE, "weights", out)
                run(["stickpose/train.py", "--resume", p, "--out", p] + extra) if os.path.exists(p) else \
                    input(f"{DIM}Нет {out}. Enter{RST}")
        elif k == "exam_make":
            v = pick_video(cfg)
            if v:
                run(["tools/exam.py", "make", v[0], "--weights", w])
        elif k == "exam_run":
            others = ask("Сравнить ещё с (через пробел)", cfg.get("cmp", ""))
            cfg["cmp"] = others
            run(["tools/exam.py", "run", w] + [os.path.join(HERE, "weights", o) for o in others.split()])
        elif k == "weights":
            ws = weights_list()
            for i, n in enumerate(ws, 1):
                print(f"  {i:>2}  {n}")
            try:
                cfg["weights"] = ws[int(ask("Номер")) - 1]
            except (ValueError, IndexError):
                pass
        save(cfg)


if __name__ == "__main__":
    main()
