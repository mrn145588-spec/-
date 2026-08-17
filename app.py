import io
import json
import os
import re
import ssl
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, Wedge, Arc
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import requests
import streamlit as st

ssl._create_default_https_context = ssl._create_unverified_context
st.set_page_config(page_title="AI Builder PRO", page_icon="🏗️", layout="wide")

# ═══════════════ КАТАЛОГИ ═══════════════
OBJECT_TYPES = {
    "🏠 Частный дом (ИЖС)":         {"desc":"Жилой дом","snips":["СП 55.13330.2016"],"floors_max":3,"icon":"🏠"},
    "🏢 Многоквартирный дом":        {"desc":"МКД","snips":["СП 54.13330.2022"],"floors_max":25,"icon":"🏢"},
    "🛁 Баня":                       {"desc":"Баня","snips":["СанПиН 2.1.2.1188-03"],"floors_max":2,"icon":"🛁"},
    "🚗 Гараж":                      {"desc":"Гараж","snips":["СП 113.13330.2016"],"floors_max":5,"icon":"🚗"},
    "🏪 Офисное здание":             {"desc":"Офис","snips":["СП 118.13330.2022"],"floors_max":20,"icon":"🏪"},
    "🛒 Торговое здание":            {"desc":"Магазин","snips":["СП 118.13330.2022"],"floors_max":10,"icon":"🛒"},
    "🏭 Производственное здание":    {"desc":"Завод","snips":["СП 56.13330.2021"],"floors_max":6,"icon":"🏭"},
    "🏨 Гостиница / хостел":         {"desc":"Отель","snips":["СП 118.13330.2022"],"floors_max":15,"icon":"🏨"},
    "🏫 Образовательное учреждение": {"desc":"Школа","snips":["СП 251.1325800.2016"],"floors_max":4,"icon":"🏫"},
    "🏥 Медицинское учреждение":     {"desc":"Больница","snips":["СП 158.13330.2014"],"floors_max":9,"icon":"🏥"},
}

REGIONS = {
    "Москва и МО":      {"climate_zone":"II В","frost_depth":"1.4 м","snow_load":"III район, 1.5 кПа","wind_load":"I район, 0.23 кПа","seismicity":"5 баллов","thermal_resistance":"R0 3.13 м²·°C/Вт"},
    "Санкт-Петербург":  {"climate_zone":"II В","frost_depth":"1.4 м","snow_load":"III район, 1.5 кПа","wind_load":"II район","seismicity":"5 баллов","thermal_resistance":"R0 3.08 м²·°C/Вт"},
    "Екатеринбург":     {"climate_zone":"I В","frost_depth":"1.8-1.9 м","snow_load":"III район","wind_load":"I-II","seismicity":"5-6","thermal_resistance":"R0 3.5 м²·°C/Вт"},
    "Новосибирск":      {"climate_zone":"I В","frost_depth":"2.2 м","snow_load":"IV район","wind_load":"III район","seismicity":"6","thermal_resistance":"R0 3.7 м²·°C/Вт"},
    "Краснодар":        {"climate_zone":"III Б","frost_depth":"0.7 м","snow_load":"II район","wind_load":"IV район","seismicity":"6-8","thermal_resistance":"R0 2.1 м²·°C/Вт"},
    "Другой регион":    {"climate_zone":"уточняется","frost_depth":"уточняется","snow_load":"уточняется","wind_load":"уточняется","seismicity":"уточняется","thermal_resistance":"уточняется"},
}

SNIPS = [
    "СП 131.13330.2020 Строительная климатология",
    "СП 20.13330.2017 Нагрузки и воздействия",
    "СП 50.13330.2012 Тепловая защита",
    "СП 22.13330.2016 Основания",
    "СП 70.13330.2012 Несущие конструкции",
    "ГОСТ 21.501-2018 СПДС Архитектурные решения",
    "ГОСТ 21.101-2020 СПДС Основные требования",
    "ГОСТ 21.205-2016 Сантехника, условные знаки",
    "ГОСТ 21.614-88 Электроустановки, обозначения",
    "СП 256.1325800.2016 Электроустановки жилых зданий",
    "ПУЭ 7 изд. Правила устройства электроустановок",
    "СП 30.13330.2020 Водопровод и канализация",
    "СП 60.13330.2020 Отопление и вентиляция",
    "СП 1.13130.2020 Эвакуационные пути",
    "СП 484.1311500.2020 Пожарная сигнализация",
    "ФЗ-384 Технический регламент безопасности зданий",
]

# ── ГОСТ КОНСТАНТЫ (в метрах) ──────────────────────────────────────────────
WALL_OUTER = 0.51    # Наружная стена (кирпич 510 мм, СП 50/55)
WALL_INNER = 0.25    # Внутренняя несущая (250 мм)
WALL_PART  = 0.12    # Перегородка ГКЛ (120 мм)
DOOR_INT   = 0.90    # Внутренняя дверь (ГОСТ 6629)
DOOR_ENT   = 1.20    # Входная дверь (ГОСТ 24698)
WIN_STD    = 1.50    # Стандартное окно (ГОСТ 23166)
WIN_LARGE  = 2.10    # Большое окно гостиной
FLOOR_H    = 3.00    # Высота этажа (СП 55 — не менее 2.7 м)

# ── Толщины линий по ГОСТ 2.303-68 ─────────────────────────────────────────
LW_CONTOUR = 2.2     # Основная (стены в разрезе)
LW_THIN    = 0.5     # Тонкая (штриховка, размеры)
LW_DIM     = 0.7     # Размерная линия
LW_AXIS    = 0.7     # Осевая штрихпунктир
LW_HATCH   = 0.35    # Штриховка материала
LW_HIDDEN  = 0.5     # Невидимый контур (штриховая)

# ── Условные обозначения помещений (СП 55.13330 таблица Г.1) ───────────────
ROOM_TYPES = {
    "кухня":    {"color":"#FFF9C4","cat":"жилая","min_area":6.0},
    "гостиная": {"color":"#E3F2FD","cat":"жилая","min_area":12.0},
    "спальня":  {"color":"#FCE4EC","cat":"жилая","min_area":8.0},
    "санузел":  {"color":"#E0F2F1","cat":"мокрая","min_area":2.5},
    "ванная":   {"color":"#E0F2F1","cat":"мокрая","min_area":3.3},
    "туалет":   {"color":"#B2DFDB","cat":"мокрая","min_area":1.2},
    "прихожая": {"color":"#F3E5F5","cat":"вспомог","min_area":3.0},
    "тамбур":   {"color":"#EDE7F6","cat":"вспомог","min_area":1.2},
    "котельная":{"color":"#FBE9E7","cat":"техн","min_area":6.0},
    "гараж":    {"color":"#ECEFF1","cat":"техн","min_area":18.0},
    "терраса":  {"color":"#F1F8E9","cat":"летняя","min_area":6.0},
    "кабинет":  {"color":"#E8EAF6","cat":"жилая","min_area":8.0},
    "коридор":  {"color":"#EFEBE9","cat":"вспомог","min_area":2.0},
    "лестница": {"color":"#FFF8E1","cat":"верт","min_area":4.0},
    "гардероб": {"color":"#F9FBE7","cat":"вспомог","min_area":2.0},
    "default":  {"color":"#FAFAFA","cat":"вспомог","min_area":4.0},
}

def get_rtype(name):
    nl = (name or "").lower()
    for k, v in ROOM_TYPES.items():
        if k in nl:
            return v
    return ROOM_TYPES["default"]

def get_rc(name):    return get_rtype(name)["color"]
def is_wet(name):    return get_rtype(name)["cat"] == "мокрая"
def is_no_win(name):
    nl = (name or "").lower()
    return any(x in nl for x in ["санузел","туалет","гардероб","кладов","тамбур","лестниц","коридор"])

# ── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""<style>
.header{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
        padding:28px;border-radius:12px;color:white;margin-bottom:24px}
.box{background:#fff;padding:16px;border-radius:10px;margin:12px 0;border-left:5px solid #667eea}
.ok  {border-left-color:#2e7d32;background:#e8f5e9}
.warn{border-left-color:#e65100;background:#fff3e0}
.doc {border-left-color:#1565c0;background:#e3f2fd}
</style>""", unsafe_allow_html=True)

st.markdown("""<div class="header">
<h1>🏗️ AI Builder PRO</h1>
<p>Профессиональные чертежи по ГОСТ 21.501-2018, ГОСТ 21.101-2020, ПУЭ 7,
   СП 484.1311500, СП 60.13330 — со всеми допусками, осями и отметками</p>
<p>🇷🇺 YandexGPT · без VPN</p></div>""", unsafe_allow_html=True)

for _k, _v in [("messages",[]),("stage","select_type"),("project_data",""),
                ("parsed_project",None),("selected_region","Екатеринбург"),
                ("selected_object_type","🏠 Частный дом (ИЖС)")]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

def reset_project():
    st.session_state.messages        = []
    st.session_state.stage           = "select_type"
    st.session_state.project_data    = ""
    st.session_state.parsed_project  = None

with st.sidebar:
    st.header("Настройки")
    st.success("YandexGPT — без VPN")
    try:
        api_key   = st.secrets["YANDEX_API_KEY"]
        folder_id = st.secrets["YANDEX_FOLDER_ID"]
        st.success("Ключи настроены автоматически")
    except Exception:
        api_key   = st.text_input("YandexGPT API Key",type="password",
                                   placeholder="AQVN...",key="api_key_input").strip()
        folder_id = st.text_input("Folder ID",placeholder="b1g...",
                                   key="folder_id_input").strip()
    if st.button("Начать заново", key="btn_reset"):
        reset_project(); st.rerun()
    st.markdown("---")
    st.markdown("""**Нормативная база:**
- ГОСТ 21.501-2018 (архитектура)
- ГОСТ 21.101-2020 (штампы, оси)
- ГОСТ 21.614-88 (электрика)
- ГОСТ 21.205-2016 (сантехника)
- ГОСТ 2.303-68 (типы линий)
- ГОСТ 2.306-68 (штриховки)
- ГОСТ 2.307-2011 (размеры)
- СП 60.13330 (вентиляция)
- ПУЭ 7 (электроустановки)
- СП 484.1311500 (пожарка)
- СП 30.13330 (водопровод/к-ция)""")

if not api_key or not folder_id:
    st.warning("Введи API Key и Folder ID слева")
    st.stop()

# ═══════════════ УТИЛИТЫ ════════════════════════════════════════════════════
def looks_like_project(t):
    return bool(t) and "{" in t and "summary" in t and "layout" in t

def extract_json(text):
    c = (text or "").strip().replace("```json","").replace("```","")
    if "```" in c:
        c = max(c.split("```"), key=len)
    s, e = c.find("{"), c.rfind("}")
    if s == -1 or e <= s:
        raise ValueError("Нет JSON")
    return json.loads(c[s:e+1])

def safe_text(v, d="-"):
    if v is None: return d
    t = str(v)
    for o, n in {"₽":" руб.","–":"-","—":"-","«":'"',"»":'"',
                 "\u201c":'"',"\u201d":'"',"…":"...","•":"-",
                 "°":" °C","²":"2","³":"3","·":"*",
                 "\u00a0":" ","\u202f":" "}.items():
        t = t.replace(o, n)
    return t.strip() or d

def parse_area(s):
    n = re.findall(r"[\d]+(?:[.,]\d+)?", str(s or ""))
    return float(n[0].replace(",",".")) if n else 12.0

def fig_to_bytes(fig):
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    b.seek(0)
    return b.read()

def layout_rooms(rooms, hw=13.5, mh=10.0):
    """Раскладка помещений по сетке. Возвращает placed, HW, HH."""
    ta  = sum(parse_area(r.get("area")) for r in rooms)
    hh  = max(mh, ta / hw * 1.2)
    placed = []
    xc = yc = rh = 0.0
    for room in rooms:
        a   = parse_area(room.get("area"))
        rw  = max(2.8, min(a / hh * 3.0, hw * 0.5))
        rrh = max(2.4, a / rw)
        if xc + rw > hw + 0.01:
            xc = 0.0; yc += rh; rh = 0.0
        placed.append({"room":room,"x":xc,"y":yc,"w":rw,"h":rrh,
                        "cx":xc+rw/2,"cy":yc+rrh/2,"idx":len(placed)+1})
        xc += rw; rh = max(rh, rrh)
    return placed, hw, max(yc + rh, mh)

# ═══════════════ ГОСТ 2.301 — РАМКА И ШТАМП ════════════════════════════════
def draw_frame(ax, x0, x1, y0, y1):
    """Рамка листа по ГОСТ 2.301: поле подшивки 20 мм, остальные 5 мм."""
    # Внешняя граница листа
    ax.add_patch(patches.Rectangle(
        (x0, y0), x1-x0, y1-y0,
        fill=False, edgecolor="#000", lw=0.5, zorder=50))
    # Внутренняя рамка: слева +20 мм (≈0.6 у.е.), прочие +5 мм (≈0.15 у.е.)
    ax.add_patch(patches.Rectangle(
        (x0+0.6, y0+0.15), x1-x0-0.75, y1-y0-0.30,
        fill=False, edgecolor="#000", lw=1.8, zorder=50))

def draw_stamp(ax, xm, ym, proj, sheet_name, sn=1, st2=6):
    """
    ГОСТ 21.101-2020, форма 3 — основная надпись (штамп).
    xm, ym — правый нижний угол внутренней рамки.
    Размер штампа: 185×55 мм (у.е. 8.5×2.8).
    """
    SW, SH = 8.5, 2.8          # ширина/высота штампа в у.е.
    x0 = xm - 0.15 - SW        # лево штампа
    y0 = ym + 0.15              # низ штампа

    # Фон штампа
    ax.add_patch(patches.Rectangle(
        (x0, y0), SW, SH,
        facecolor="white", edgecolor="#000", lw=1.5, zorder=51))

    # ── Горизонтальные линии (высоты граф) ────────────────────────────────
    for h in [0.40, 0.80, 1.40, 2.00]:
        ax.plot([x0, x0+SW], [y0+h, y0+h], color="#000", lw=0.5, zorder=52)

    # ── Вертикальные линии (деление граф) ─────────────────────────────────
    # Позиции: ФИО/должность | подпись | дата | № изм | стадия/лист/листов
    for v in [1.50, 3.00, 5.50, 6.50, 7.20, 7.90]:
        ax.plot([x0+v, x0+v], [y0, y0+SH], color="#000", lw=0.5, zorder=52)

    # ── Вспомогательная функция текста ────────────────────────────────────
    def txt(x, y, tx, s=6, w="normal", a="center"):
        ax.text(x, y, tx, fontsize=s, ha=a, va="center",
                color="#000", fontweight=w, zorder=53)

    dt = datetime.now().strftime("%d.%m.%Y")
    ot = proj.get("object_type", "")

    # Строка 1 — наименование организации
    txt(x0+SW/2, y0+SH-0.22, "AI Builder PRO / ГОСТ 21.101-2020", 6)
    # Строка 2 — наименование объекта
    txt(x0+SW/2, y0+2.35, f"{proj.get('location','')}  ·  Объект: {ot}", 6)
    # Строка 3 — наименование документа (чертежа)
    txt(x0+SW/2, y0+1.65, sheet_name, 8, "bold")

    # Графы «Разработал / Проверил / Н.контр.»
    for row, label in zip([1.10, 0.60, 0.20], ["Разраб.", "Пров.", "Н.контр."]):
        txt(x0+0.75, y0+row, label, 5)
    txt(x0+2.25, y0+1.10, "AI Builder", 5)
    txt(x0+2.25, y0+0.60, "GPT-Arch",  5)
    txt(x0+4.25, y0+1.10, dt,  5)
    txt(x0+4.25, y0+0.60, dt,  5)

    # Стадия / Лист / Листов
    txt(x0+6.00, y0+1.10, "Стадия", 5)
    txt(x0+6.00, y0+0.65, "ЭП",     10, "bold")
    txt(x0+6.85, y0+0.80, "Лист",   5)
    txt(x0+6.85, y0+0.35, f"{sn}",  9,  "bold")
    txt(x0+7.55, y0+0.80, "Листов", 5)
    txt(x0+7.55, y0+0.35, f"{st2}", 9,  "bold")

    # Масштаб
    txt(x0+7.20, y0+1.70, "М 1:100", 7, "bold")

    # Номер чертежа / шифр
    proj_code = proj.get("code", "АИ-001")
    txt(x0+SW/2, y0+0.22, f"Шифр: {proj_code}", 5)

# ═══════════════ ГОСТ 2.303 — ОСИ И МАРКИ ══════════════════════════════════
def draw_axis(ax, x1, y1, x2, y2):
    """Штрихпунктирная ось по ГОСТ 2.303-68 тип «г» (длинный штрих — точка)."""
    ax.plot([x1, x2], [y1, y2],
            color="#000", lw=LW_AXIS,
            linestyle=(0, (12, 3, 2, 3)), zorder=8, alpha=0.80)

def draw_am(ax, x, y, lb):
    """Марка оси — кружок Ø8 мм с цифрой/буквой (ГОСТ 21.101-2020 п.5.6)."""
    ax.add_patch(plt.Circle((x, y), 0.30,
                             facecolor="white", edgecolor="#000", lw=1.0, zorder=15))
    ax.text(x, y, str(lb), ha="center", va="center",
            fontsize=9, fontweight="bold", color="#000", zorder=16)

# ═══════════════ ГОСТ 2.307-2011 — РАЗМЕРНЫЕ ЦЕПИ ══════════════════════════
def draw_dim(ax, pts, y, orient="h"):
    """
    Размерная цепь с засечками под 45° по ГОСТ 2.307-2011.
    pts  — список координат (начальные/конечные точки отрезков)
    y    — отступ от объекта (для горизонтали — Y-координата линии;
           для вертикали  — X-координата линии)
    orient — 'h' горизонтальный, 'v' вертикальный
    """
    TS = 0.09   # полудлина засечки
    EX = 0.12   # выносная линия (сверх засечки)

    if orient == "h":
        # Размерная линия
        ax.plot([pts[0], pts[-1]], [y, y], color="#000", lw=LW_DIM, zorder=8)
        # Выносные линии от точек объекта до размерной
        for p in pts:
            ax.plot([p, p], [y - EX, y + EX], color="#000", lw=LW_DIM*0.7, zorder=8,
                    linestyle=(0,(4,2)))
        # Засечки и подписи
        for i in range(len(pts) - 1):
            mid = (pts[i] + pts[i+1]) / 2
            val = abs(pts[i+1] - pts[i]) * 1000          # мм
            # Засечки (45°)
            for pt in [pts[i], pts[i+1]]:
                ax.plot([pt-TS, pt+TS], [y-TS, y+TS],
                        color="#000", lw=LW_DIM+0.3, zorder=9)
            # Подпись мм
            ax.text(mid, y + 0.13, f"{val:.0f}",
                    ha="center", va="bottom", fontsize=7.5, color="#000", zorder=10)
    else:  # vertical
        ax.plot([y, y], [pts[0], pts[-1]], color="#000", lw=LW_DIM, zorder=8)
        for p in pts:
            ax.plot([y - EX, y + EX], [p, p], color="#000", lw=LW_DIM*0.7, zorder=8,
                    linestyle=(0,(4,2)))
        for i in range(len(pts) - 1):
            mid = (pts[i] + pts[i+1]) / 2
            val = abs(pts[i+1] - pts[i]) * 1000
            for pt in [pts[i], pts[i+1]]:
                ax.plot([y-TS, y+TS], [pt-TS, pt+TS],
                        color="#000", lw=LW_DIM+0.3, zorder=9)
            ax.text(y + 0.15, mid, f"{val:.0f}",
                    ha="left", va="center", fontsize=7.5, color="#000",
                    rotation=90, zorder=10)

# ═══════════════ ГОСТ 21.501 — ОТМЕТКИ УРОВНЕЙ ══════════════════════════════
def draw_level(ax, x, y, v="±0.000"):
    """
    Отметка уровня — равносторонний треугольник + горизонтальная полка
    (ГОСТ 21.501-2018 п. 4.10).
    """
    d = 0.18   # полуразмер треугольника
    # Треугольник (вершиной вниз)
    tri = np.array([[x, y], [x - d, y + d*1.6], [x + d, y + d*1.6], [x, y]])
    ax.plot(tri[:,0], tri[:,1], color="#000", lw=0.9, zorder=15)
    ax.fill(tri[:,0], tri[:,1], color="#000", zorder=15, alpha=0.85)
    # Горизонтальная полка
    ax.plot([x - 0.45, x + 0.45], [y + d*1.6, y + d*1.6],
            color="#000", lw=0.9, zorder=15)
    # Подпись отметки
    ax.text(x, y + d*1.6 + 0.12, v,
            ha="center", va="bottom", fontsize=8,
            color="#000", fontweight="bold", zorder=16)

# ═══════════════ ГОСТ 2.306 — ШТРИХОВКИ МАТЕРИАЛОВ ══════════════════════════
def hatch_brick(ax, x, y, w, h):
    """Штриховка «кирпич» — диагональные линии 45° (ГОСТ 2.306-68 п.2)."""
    step = 0.18
    for off in np.arange(-h, w + h, step):
        x1 = max(x, x + off - h);  x2 = min(x + w, x + off)
        y1 = y + max(0, off - w);  y2 = y + min(h, off)
        if x2 > x1 and y2 > y1:
            ax.plot([x1, x2], [y1, y2],
                    color="#37474F", lw=LW_HATCH, alpha=0.70, zorder=6)

def hatch_conc(ax, x, y, w, h):
    """Штриховка «бетон» — точки + треугольники (ГОСТ 2.306-68 п.4)."""
    for xi in np.arange(x + 0.10, x + w, 0.22):
        for yi in np.arange(y + 0.10, y + h, 0.22):
            if (round(xi*10) + round(yi*10)) % 3 < 1:
                ax.plot(xi, yi, marker=".", markersize=1.6,
                        color="#455A64", zorder=6)

def hatch_insulation(ax, x, y, w, h):
    """Штриховка «утеплитель» — волнистые линии (ГОСТ 2.306-68 п.9)."""
    for yi in np.arange(y + 0.08, y + h, 0.14):
        xs = np.linspace(x, x + w, 60)
        ys = yi + 0.04 * np.sin((xs - x) / (w + 0.01) * 8 * np.pi)
        ax.plot(xs, ys, color="#80DEEA", lw=0.4, alpha=0.75, zorder=6)

def hatch_earth(ax, x, y, w, h):
    """Штриховка «грунт» — наклонные штрихи (ГОСТ 2.306-68 п.31)."""
    step = 0.25
    for xi in np.arange(x, x + w + step, step):
        ax.plot([xi, xi - 0.15], [y + h, y],
                color="#8D6E63", lw=0.5, alpha=0.60, zorder=5)

# ═══════════════ ГОСТ 21.501 — ДВЕРИ И ОКНА ════════════════════════════════
def draw_door(ax, x, y, w=DOOR_INT, direction="up_right"):
    """
    Условное обозначение двери — полотно + дуга (ГОСТ 21.501-2018 п.5.2).
    Проём в стене зачищается белой линией перед рисованием.
    """
    # Зачистка проёма
    ax.plot([x, x+w], [y, y], color="white", lw=8, zorder=6, solid_capstyle="butt")
    # Размерная засечка проёма
    for px in [x, x+w]:
        ax.plot([px, px], [y-0.08, y+0.08], color="#000", lw=1.2, zorder=11)

    if direction == "up_right":
        ax.plot([x, x+w*0.97], [y, y+w*0.12], color="#000", lw=1.6, zorder=10)
        th = np.linspace(0, np.pi/2 * 0.95, 50)
        ax.plot(x + w*np.cos(th), y + w*np.sin(th),
                color="#000", lw=0.7, linestyle=(0,(5,2)), zorder=10)
    elif direction == "up_left":
        ax.plot([x+w, x+w*0.03], [y, y+w*0.12], color="#000", lw=1.6, zorder=10)
        th = np.linspace(np.pi, np.pi/2*1.05, 50)
        ax.plot(x + w + w*np.cos(th), y + w*np.sin(th),
                color="#000", lw=0.7, linestyle=(0,(5,2)), zorder=10)
    elif direction == "down_right":
        ax.plot([x, x+w*0.97], [y, y-w*0.12], color="#000", lw=1.6, zorder=10)
        th = np.linspace(0, -np.pi/2*0.95, 50)
        ax.plot(x + w*np.cos(th), y + w*np.sin(th),
                color="#000", lw=0.7, linestyle=(0,(5,2)), zorder=10)

def draw_window(ax, x, y, d="h", w=WIN_STD):
    """
    Условное обозначение окна — тройная линия в проёме
    (ГОСТ 21.501-2018 п.5.3). Оконный блок по ГОСТ 23166.
    Аргументы: d='h' — горизонтальная стена, 'v' — вертикальная.
    """
    t = WALL_OUTER / 4        # шаг между линиями

    if d == "h":
        ax.plot([x, x+w], [y,   y  ], color="white", lw=7, zorder=6, solid_capstyle="butt")
        for dy, lw_ in [(-t, 0.9), (0, 0.5), (t, 0.9)]:
            ax.plot([x, x+w], [y+dy, y+dy], color="#000", lw=lw_, zorder=10)
        # Торцевые отбои
        for px in [x, x+w]:
            ax.plot([px, px], [y-t*1.6, y+t*1.6], color="#000", lw=0.7, zorder=10)
        # Подпись размера
        ax.text(x+w/2, y-t*2.2, f"{w*1000:.0f}", ha="center", fontsize=6.5,
                color="#01579B", zorder=11)
    else:
        ax.plot([x, x], [y, y+w], color="white", lw=7, zorder=6, solid_capstyle="butt")
        for dx, lw_ in [(-t, 0.9), (0, 0.5), (t, 0.9)]:
            ax.plot([x+dx, x+dx], [y, y+w], color="#000", lw=lw_, zorder=10)
        for py in [y, y+w]:
            ax.plot([x-t*1.6, x+t*1.6], [py, py], color="#000", lw=0.7, zorder=10)
        ax.text(x+t*2.4, y+w/2, f"{w*1000:.0f}", va="center", fontsize=6.5,
                color="#01579B", rotation=90, zorder=11)

def draw_stairs(ax, x, y, w=1.2, h=2.5, n=10):
    """
    Лестничный марш — ступени + ось + стрелка подъёма
    (ГОСТ 21.501-2018 п.5.4).
    """
    step_h = h / n
    for i in range(n):
        ax.plot([x, x+w], [y+i*step_h, y+i*step_h],
                color="#000", lw=0.8, zorder=8)
    # Ограждение (боковые линии)
    ax.plot([x,   x  ], [y, y+h], color="#000", lw=1.2, zorder=8)
    ax.plot([x+w, x+w], [y, y+h], color="#000", lw=1.2, zorder=8)
    # Осевая
    ax.plot([x+w/2, x+w/2], [y, y+h],
            color="#000", lw=0.5, linestyle=":", zorder=8)
    # Стрелка «вверх»
    ax.annotate("",
                xy=(x+w/2, y+h-0.15), xytext=(x+w/2, y+0.15),
                arrowprops=dict(arrowstyle="->", color="#000", lw=1.3),
                zorder=9)
    ax.text(x+w/2, y+h+0.18, "ВВЕРХ",
            ha="center", fontsize=6, fontweight="bold", color="#000", zorder=10)
    # Уклон ступени
    ax.text(x+w/2, y - 0.25,
            f"n={n}  h={h*1000/n:.0f} мм",
            ha="center", fontsize=5.5, color="#37474F", zorder=10)

# ═══════════════ ГОСТ 21.614-88 — УСЛОВНЫЕ ЗНАКИ ЭЛЕКТРИКИ ════════════════
def sym_socket(ax, x, y, count=1):
    """Штепсельная розетка (ГОСТ 21.614-88 поз.1)."""
    ax.add_patch(Wedge((x, y), 0.11, 180, 360,
                       facecolor="white", edgecolor="#000", lw=0.9, zorder=15))
    ax.plot([x-0.06, x+0.06], [y+0.01, y+0.01],
            color="#000", lw=0.6, zorder=15)
    if count >= 2:
        ax.text(x, y-0.05, str(count),
                ha="center", va="center", fontsize=5, color="#000", zorder=16)
    # Высотная отметка
    ax.text(x, y-0.22, "h=0.3", ha="center", fontsize=4.5,
            color="#37474F", zorder=16)

def sym_socket_pow(ax, x, y):
    """Силовая розетка с защитным заземлением УЗО (ПУЭ-7 гл.7.1)."""
    ax.add_patch(Wedge((x, y), 0.12, 180, 360,
                       facecolor="#FFF3E0", edgecolor="#E65100", lw=1.1, zorder=15))
    ax.plot([x-0.07, x+0.07], [y+0.01, y+0.01],
            color="#E65100", lw=0.7, zorder=15)
    # Штырёк PE
    ax.plot([x, x], [y+0.04, y+0.12],
            color="#E65100", lw=0.7, zorder=15)
    ax.text(x, y-0.22, "З/У h=1.0", ha="center", fontsize=4.5,
            color="#E65100", zorder=16)

def sym_switch(ax, x, y, count=1):
    """Выключатель одно/двухклавишный (ГОСТ 21.614-88 поз.10-11)."""
    ax.add_patch(plt.Circle((x, y), 0.09,
                             facecolor="white", edgecolor="#000", lw=0.9, zorder=15))
    ax.plot([x-0.14, x+0.14], [y-0.04, y+0.14],
            color="#000", lw=0.9, zorder=16)
    if count >= 2:
        ax.plot([x-0.14, x+0.14], [y-0.10, y+0.08],
                color="#000", lw=0.9, zorder=16)
    ax.text(x, y-0.22, "h=0.9", ha="center", fontsize=4.5,
            color="#37474F", zorder=16)

def sym_lamp(ax, x, y, kind="ceiling"):
    """Светильник потолочный — кружок с крестом (ГОСТ 21.614-88 поз.26)."""
    fc = "#FFF9C4" if kind == "wet" else "#FFF59D"
    ax.add_patch(plt.Circle((x, y), 0.15,
                             facecolor=fc, edgecolor="#000", lw=0.9, zorder=15))
    ax.plot([x-0.11, x+0.11], [y, y], color="#000", lw=0.7, zorder=16)
    ax.plot([x, x], [y-0.11, y+0.11], color="#000", lw=0.7, zorder=16)

def sym_lamp_led(ax, x, y):
    """LED-панель потолочная (ГОСТ 21.614-88 поз.30)."""
    ax.add_patch(patches.Rectangle(
        (x-0.22, y-0.09), 0.44, 0.18,
        facecolor="#FFF59D", edgecolor="#000", lw=0.9, zorder=15))
    ax.text(x, y, "LED", ha="center", va="center",
            fontsize=4, fontweight="bold", color="#000", zorder=16)

def sym_board(ax, x, y, label="ВРУ"):
    """Электрощит/щиток (ГОСТ 21.614-88 поз.54)."""
    ax.add_patch(patches.Rectangle(
        (x-0.22, y-0.22), 0.44, 0.44,
        facecolor="#455A64", edgecolor="#000", lw=1.2, zorder=15))
    ax.text(x, y, label, ha="center", va="center",
            fontsize=5.5, color="white", fontweight="bold", zorder=16)

# ═══════════════ ГОСТ 21.205-2016 — САНТЕХНИКА ══════════════════════════════
def sym_wc(ax, x, y):
    """Унитаз (ГОСТ 21.205-2016 поз.1)."""
    ax.add_patch(patches.Ellipse(
        (x, y+0.05), 0.30, 0.38,
        facecolor="white", edgecolor="#000", lw=0.9, zorder=15))
    ax.add_patch(patches.Rectangle(
        (x-0.14, y+0.21), 0.28, 0.11,
        facecolor="white", edgecolor="#000", lw=0.9, zorder=15))
    # Бачок
    ax.add_patch(patches.Rectangle(
        (x-0.10, y+0.31), 0.20, 0.06,
        facecolor="#ECEFF1", edgecolor="#000", lw=0.7, zorder=15))

def sym_sink(ax, x, y):
    """Умывальник/раковина (ГОСТ 21.205-2016 поз.5)."""
    ax.add_patch(patches.Rectangle(
        (x-0.21, y-0.14), 0.42, 0.28,
        facecolor="white", edgecolor="#000", lw=0.9, zorder=15))
    ax.add_patch(plt.Circle((x, y), 0.04,
                             facecolor="none", edgecolor="#000", lw=0.6, zorder=16))
    # Кран
    ax.plot([x, x], [y+0.08, y+0.15], color="#000", lw=1.3, zorder=16)
    ax.plot([x-0.06, x+0.06], [y+0.15, y+0.15], color="#000", lw=0.8, zorder=16)

def sym_bath(ax, x, y):
    """Ванна (ГОСТ 21.205-2016 поз.9)."""
    ax.add_patch(patches.FancyBboxPatch(
        (x-0.37, y-0.22), 0.74, 0.44,
        boxstyle="round,pad=0.03,rounding_size=0.09",
        facecolor="white", edgecolor="#000", lw=0.9, zorder=15))
    # Слив
    ax.add_patch(plt.Circle((x-0.26, y), 0.04,
                             facecolor="none", edgecolor="#000", lw=0.6, zorder=16))
    ax.plot([x-0.26, x-0.26], [y, y-0.22], color="#000", lw=0.5, zorder=16)

def sym_shower(ax, x, y):
    """Душевая кабина (ГОСТ 21.205-2016 поз.11)."""
    ax.add_patch(patches.Rectangle(
        (x-0.24, y-0.24), 0.48, 0.48,
        facecolor="#E1F5FE", edgecolor="#000", lw=0.9, zorder=15))
    # Штриховка воды — диагонали
    for d in np.arange(-0.20, 0.30, 0.12):
        ax.plot([x-0.22+d, x+0.15+d], [y-0.22, y+0.22],
                color="#0288D1", lw=0.4, alpha=0.6, zorder=16)

def sym_stove(ax, x, y):
    """Кухонная плита — 4 конфорки (ГОСТ 21.205-2016 поз.17)."""
    ax.add_patch(patches.Rectangle(
        (x-0.32, y-0.27), 0.64, 0.54,
        facecolor="#ECEFF1", edgecolor="#000", lw=0.9, zorder=15))
    for dx in [-0.16, 0.16]:
        for dy in [-0.11, 0.13]:
            ax.add_patch(plt.Circle((x+dx, y+dy), 0.08,
                                    facecolor="none", edgecolor="#000", lw=0.7, zorder=16))

# ═══════════════ СП 60.13330 — ВЕНТИЛЯЦИЯ ══════════════════════════════════
def sym_vent_sup(ax, x, y, size="Ø125"):
    """Приточная решётка (СП 60.13330.2020, ГОСТ 21.205 доп.)."""
    ax.add_patch(patches.Rectangle(
        (x-0.24, y-0.09), 0.48, 0.18,
        facecolor="#E3F2FD", edgecolor="#01579B", lw=1.1, zorder=15))
    # Решётка — горизонтальные штрихи
    for dy in [-0.04, 0, 0.04]:
        ax.plot([x-0.20, x+0.20], [y+dy, y+dy],
                color="#01579B", lw=0.5, zorder=16)
    # Стрелка приток ↓
    ax.annotate("", xy=(x, y-0.20), xytext=(x, y-0.10),
                arrowprops=dict(arrowstyle="->", color="#01579B", lw=1.1), zorder=16)
    ax.text(x, y+0.23, f"П {size}", ha="center", fontsize=5,
            color="#01579B", fontweight="bold", zorder=16)

def sym_vent_ext(ax, x, y, size="Ø100"):
    """Вытяжная решётка (СП 60.13330.2020)."""
    ax.add_patch(patches.Rectangle(
        (x-0.20, y-0.08), 0.40, 0.16,
        facecolor="#FBE9E7", edgecolor="#BF360C", lw=1.1, zorder=15))
    for dy in [-0.03, 0.03]:
        ax.plot([x-0.16, x+0.16], [y+dy, y+dy],
                color="#BF360C", lw=0.5, zorder=16)
    # Стрелка вытяжка ↑
    ax.annotate("", xy=(x, y+0.20), xytext=(x, y+0.09),
                arrowprops=dict(arrowstyle="->", color="#BF360C", lw=1.1), zorder=16)
    ax.text(x, y-0.22, f"В {size}", ha="center", fontsize=5,
            color="#BF360C", fontweight="bold", zorder=16)

# ═══════════════ СП 484.1311500 — ПОЖАРНАЯ СИГНАЛИЗАЦИЯ ════════════════════
def sym_fire_det(ax, x, y):
    """Дымовой пожарный извещатель ИП-212 (СП 484.1311500.2020 Приложение А)."""
    ax.add_patch(plt.Circle((x, y), 0.13,
                             facecolor="#FF5252", edgecolor="#000", lw=1.0, zorder=15))
    ax.text(x, y, "ДИ", ha="center", va="center",
            fontsize=4.5, color="white", fontweight="bold", zorder=16)

def sym_fire_heat(ax, x, y):
    """Тепловой извещатель ИП-101."""
    ax.add_patch(plt.Circle((x, y), 0.12,
                             facecolor="#FF8F00", edgecolor="#000", lw=1.0, zorder=15))
    ax.text(x, y, "ТИ", ha="center", va="center",
            fontsize=4.5, color="white", fontweight="bold", zorder=16)

def sym_fire_manual(ax, x, y):
    """Ручной пожарный извещатель ИПР (СП 484 п.6.6.2)."""
    ax.add_patch(patches.Rectangle(
        (x-0.13, y-0.13), 0.26, 0.26,
        facecolor="#B71C1C", edgecolor="#000", lw=1.0, zorder=15))
    ax.text(x, y, "ИПР", ha="center", va="center",
            fontsize=4, color="white", fontweight="bold", zorder=16)

def sym_fire_horn(ax, x, y):
    """Оповещатель световой+звуковой (СП 3.13130.2009)."""
    ax.add_patch(plt.Circle((x, y), 0.12,
                             facecolor="#FF6D00", edgecolor="#000", lw=1.0, zorder=15))
    ax.text(x, y, "ОП", ha="center", va="center",
            fontsize=4.5, color="white", fontweight="bold", zorder=16)

def sym_fire_ext(ax, x, y):
    """Огнетушитель ОП-5 (ГОСТ Р 51057)."""
    ax.add_patch(plt.Circle((x, y), 0.15,
                             facecolor="#D32F2F", edgecolor="#000", lw=1.2, zorder=15))
    ax.text(x, y, "ОП-5", ha="center", va="center",
            fontsize=4, color="white", fontweight="bold", zorder=16)

def sym_fire_hydrant(ax, x, y):
    """Пожарный кран ПК (СП 10.13130.2020)."""
    ax.add_patch(patches.Rectangle(
        (x-0.12, y-0.12), 0.24, 0.24,
        facecolor="#F44336", edgecolor="#000", lw=1.0, zorder=15))
    ax.text(x, y, "ПК", ha="center", va="center",
            fontsize=5, color="white", fontweight="bold", zorder=16)

def sym_exit_sign(ax, x, y):
    """Знак «ВЫХОД» (ГОСТ Р 12.4.026, СП 1.13130.2020)."""
    ax.add_patch(patches.Rectangle(
        (x-0.32, y-0.12), 0.64, 0.24,
        facecolor="#00C853", edgecolor="#1B5E20", lw=1.1, zorder=15))
    ax.text(x, y, "ВЫХОД", ha="center", va="center",
            fontsize=5.5, color="white", fontweight="bold", zorder=16)
# ═══════════════ ПЛАН ЭТАЖА (ГОСТ 21.501-2018) ══════════════════════════════
def make_floor_plan(fn, rooms, ot="", proj=None, sn=1, st2=6):
    """
    Архитектурный план этажа со всеми допусками, марками осей,
    размерными цепями, отметками уровня, штриховками материалов,
    условными знаками дверей и окон.
    Нормы: ГОСТ 21.501-2018, ГОСТ 21.101-2020, ГОСТ 2.303-68,
           ГОСТ 2.306-68, ГОСТ 2.307-2011, СП 55.13330.2016.
    """
    if not rooms:
        return None
    proj = proj or {}
    placed, HW, HH = layout_rooms(rooms)

    # ── Лист и рамка ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(20, 14))
    XL, XR, YB, YT = -5.5, HW + 10.5, -7.5, HH + 5.5
    ax.set_xlim(XL, XR); ax.set_ylim(YB, YT)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_facecolor("white"); fig.patch.set_facecolor("white")

    draw_frame(ax, XL, XR, YB, YT)
    draw_stamp(ax, XR, YB, proj, f"План {fn}", sn, st2)

    # ── Уникальные оси ───────────────────────────────────────────────────────
    raw_x = sorted({0.0} |
                   {p["x"] for p in placed} |
                   {p["x"]+p["w"] for p in placed} |
                   {HW})
    axu = [raw_x[0]]
    for xv in raw_x[1:]:
        if xv - axu[-1] > 0.4:
            axu.append(xv)

    raw_y = sorted({0.0} |
                   {p["y"] for p in placed} |
                   {p["y"]+p["h"] for p in placed} |
                   {HH})
    ayu = [raw_y[0]]
    for yv in raw_y[1:]:
        if yv - ayu[-1] > 0.4:
            ayu.append(yv)

    # Рисуем оси
    for xv in axu:
        draw_axis(ax, xv, YB+0.5, xv, YT-0.5)
    for yv in ayu:
        draw_axis(ax, XL+0.5, yv, XR-9.5, yv)

    # Марки цифровых осей (по X — снизу и сверху)
    for i, xv in enumerate(axu):
        draw_am(ax, xv, YB+0.3,  i+1)
        draw_am(ax, xv, HH+2.8,  i+1)

    # Марки буквенных осей (по Y — слева и справа)
    letters = "АБВГДЕЖИКЛМНПРСТ"
    for i, yv in enumerate(ayu):
        lb = letters[i] if i < len(letters) else str(i+1)
        draw_am(ax, -2.5, yv, lb)
        draw_am(ax, HW+2.8, yv, lb)

    # ── Наружные стены (кирпич 510 мм, СП 50.13330) ─────────────────────────
    ext_walls = [
        (-WALL_OUTER, -WALL_OUTER, HW + 2*WALL_OUTER, WALL_OUTER),   # низ
        (-WALL_OUTER,  HH,         HW + 2*WALL_OUTER, WALL_OUTER),   # верх
        (-WALL_OUTER,  0,          WALL_OUTER,         HH),           # лево
        ( HW,          0,          WALL_OUTER,         HH),           # право
    ]
    for xw, yw, ww, hw_ in ext_walls:
        ax.add_patch(patches.Rectangle(
            (xw, yw), ww, hw_,
            facecolor="#FFF8E1", edgecolor="#000", lw=LW_CONTOUR, zorder=5))
        hatch_brick(ax, xw, yw, ww, hw_)

    # Толщина стены — подпись
    ax.text(-WALL_OUTER/2, -0.3, f"{WALL_OUTER*1000:.0f}",
            ha="center", fontsize=6, color="#37474F", zorder=12)

    # ── Заливка и штриховка помещений ───────────────────────────────────────
    for pl in placed:
        c = get_rc(pl["room"].get("name",""))
        ax.add_patch(patches.Rectangle(
            (pl["x"], pl["y"]), pl["w"], pl["h"],
            facecolor=c, alpha=0.40, edgecolor="none", zorder=3))
        # Мокрая зона — точечная штриховка (СП 30.13330)
        if is_wet(pl["room"].get("name","")):
            for xi in np.arange(pl["x"]+0.25, pl["x"]+pl["w"], 0.32):
                for yi in np.arange(pl["y"]+0.25, pl["y"]+pl["h"], 0.32):
                    ax.plot(xi, yi, marker=".", markersize=2.0,
                            color="#0288D1", alpha=0.65, zorder=3)

    # ── Внутренние несущие стены ─────────────────────────────────────────────
    for pl in placed:
        if pl["x"] > 0.02:
            ax.add_patch(patches.Rectangle(
                (pl["x"]-WALL_INNER/2, pl["y"]), WALL_INNER, pl["h"],
                facecolor="#ECEFF1", edgecolor="#000",
                lw=LW_CONTOUR*0.85, zorder=5))
        if pl["y"] > 0.02:
            ax.add_patch(patches.Rectangle(
                (pl["x"], pl["y"]-WALL_INNER/2), pl["w"], WALL_INNER,
                facecolor="#ECEFF1", edgecolor="#000",
                lw=LW_CONTOUR*0.85, zorder=5))

    # ── Двери, окна ──────────────────────────────────────────────────────────
    for pl in placed:
        name = pl["room"].get("name","")
        # Внутренняя дверь по центру разделительной стены
        if pl["y"] > 0.02:
            dx = pl["x"] + pl["w"]/2 - DOOR_INT/2
            draw_door(ax, dx, pl["y"], DOOR_INT, "up_right")

        if is_no_win(name):
            continue

        # Окна на наружных стенах
        if pl["y"] + pl["h"] >= HH - 0.02:
            draw_window(ax, pl["x"]+pl["w"]/2 - WIN_STD/2, HH, "h", WIN_STD)
        elif pl["x"] + pl["w"] >= HW - 0.02:
            draw_window(ax, HW, pl["y"]+pl["h"]/2 - WIN_STD/2, "v", WIN_STD)
        elif pl["x"] <= 0.02:
            draw_window(ax, 0, pl["y"]+pl["h"]/2 - WIN_STD/2, "v", WIN_STD)

    # Входная дверь (по центру нижней стены, DOOR_ENT=1.2 м)
    ent_x = HW/2 - DOOR_ENT/2
    draw_door(ax, ent_x, 0, DOOR_ENT, "up_right")

    # ── Подписи помещений (номер в кружке, название, площадь) ───────────────
    for idx, pl in enumerate(placed, 1):
        a  = parse_area(pl["room"].get("area"))
        cx, cy = pl["cx"], pl["cy"]

        # Кружок с номером помещения (ГОСТ 21.501-2018 п.4.9)
        ax.add_patch(plt.Circle((cx, cy+0.65), 0.26,
                                facecolor="white", edgecolor="#000", lw=0.9, zorder=15))
        ax.text(cx, cy+0.65, f"{idx}",
                ha="center", va="center", fontsize=8,
                fontweight="bold", color="#000", zorder=16)
        # Название
        ax.text(cx, cy+0.10, pl["room"].get("name",""),
                ha="center", va="center", fontsize=8, color="#000", zorder=15)
        # Площадь (подчёркнутая) — ГОСТ 21.501-2018 п.4.8
        ax.text(cx, cy-0.38, f"{a:.1f}",
                ha="center", va="center", fontsize=10,
                fontweight="bold", color="#000", zorder=15)
        ax.plot([cx-0.55, cx+0.55], [cy-0.60, cy-0.60],
                color="#000", lw=0.8, zorder=15)
        # Минимальная площадь (допуск)
        min_a = get_rtype(pl["room"].get("name",""))["min_area"]
        ok_col = "#1B5E20" if a >= min_a else "#B71C1C"
        ax.text(cx, cy-0.85, f"min {min_a} м²",
                ha="center", fontsize=5, color=ok_col, zorder=15)

    # ── Отметка уровня пола ──────────────────────────────────────────────────
    draw_level(ax, HW/2, 0.35, "±0.000")

    # ── Размерная цепь (внешняя — оси; внутренняя — проёмы) ─────────────────
    # Горизонтальная
    draw_dim(ax, axu, -2.8, "h")
    # Горизонтальная сводная
    ax.plot([0, HW], [-3.8, -3.8], color="#000", lw=LW_DIM, zorder=8)
    for p in [0, HW]:
        ax.plot([p-0.07, p+0.07], [-3.86, -3.74],
                color="#000", lw=LW_DIM+0.3, zorder=9)
    ax.text(HW/2, -3.65, f"{HW*1000:.0f}",
            ha="center", va="bottom", fontsize=10,
            fontweight="bold", color="#000", zorder=10)

    # Вертикальная
    draw_dim(ax, ayu, HW+4.0, "v")
    # Вертикальная сводная
    ax.plot([HW+5.2, HW+5.2], [0, HH], color="#000", lw=LW_DIM, zorder=8)
    for p in [0, HH]:
        ax.plot([HW+5.14, HW+5.26], [p-0.07, p+0.07],
                color="#000", lw=LW_DIM+0.3, zorder=9)
    ax.text(HW+5.40, HH/2, f"{HH*1000:.0f}",
            ha="left", va="center", fontsize=10,
            fontweight="bold", color="#000", rotation=90, zorder=10)

    # ── Стрелка входа ────────────────────────────────────────────────────────
    ax.annotate("ВХОД", xy=(HW/2, -0.45), xytext=(HW/2, -1.85),
                fontsize=9, ha="center", color="#1B5E20", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#1B5E20", lw=2.0),
                zorder=15)

    # ── Компас (роза ветров) ─────────────────────────────────────────────────
    cn_x, cn_y = HW + 7.5, HH + 2.2
    ax.add_patch(plt.Circle((cn_x, cn_y), 0.60,
                             facecolor="white", edgecolor="#000", lw=1.0, zorder=15))
    ax.annotate("", xy=(cn_x, cn_y+0.45), xytext=(cn_x, cn_y-0.45),
                arrowprops=dict(arrowstyle="->", color="#000", lw=1.5),
                zorder=16)
    ax.text(cn_x, cn_y+0.82, "С", ha="center", fontsize=10,
            fontweight="bold", color="#000", zorder=16)
    for ang, lb in [(90,"В"),(270,"З"),(180,"Ю")]:
        rx = cn_x + 0.85*np.cos(np.radians(ang))
        ry = cn_y + 0.85*np.sin(np.radians(ang))
        ax.text(rx, ry, lb, ha="center", va="center", fontsize=8, color="#000", zorder=16)

    # ── Секущая А-А ──────────────────────────────────────────────────────────
    ax.plot([-1.2, HW+1.2], [HH/2, HH/2],
            color="#B71C1C", lw=1.8, linestyle="-.", zorder=12)
    for ex in [-1.8, HW+1.8]:
        ax.text(ex, HH/2, "А", fontsize=11, fontweight="bold", color="#B71C1C",
                ha="center", va="center",
                bbox=dict(facecolor="white", edgecolor="#B71C1C",
                          boxstyle="circle,pad=0.28"), zorder=13)

    # ── Экспликация (таблица помещений) ─────────────────────────────────────
    ex_x, ex_y = XR - 9.2, YT - 0.8
    ax.text(ex_x + 2.0, ex_y, "ЭКСПЛИКАЦИЯ ПОМЕЩЕНИЙ",
            ha="center", fontsize=8, fontweight="bold", color="#000", zorder=20)
    hdrs = ["№","Наименование","Площадь, м²","Категория","Примечание"]
    col_w = [0.4, 2.2, 1.1, 1.1, 1.5]
    for ci, (h, cw) in enumerate(zip(hdrs, col_w)):
        ex_xi = ex_x + sum(col_w[:ci])
        ax.add_patch(patches.Rectangle(
            (ex_xi, ex_y-0.45), cw, 0.40,
            facecolor="#B0BEC5", edgecolor="#000", lw=0.6, zorder=20))
        ax.text(ex_xi + cw/2, ex_y-0.25, h,
                ha="center", va="center", fontsize=5.5,
                fontweight="bold", color="#000", zorder=21)
    for ri, pl in enumerate(placed):
        ry = ex_y - 0.45 - ri*0.38
        row_c = get_rc(pl["room"].get("name",""))
        vals  = [str(pl["idx"]),
                 pl["room"].get("name",""),
                 f"{parse_area(pl['room'].get('area')):.1f}",
                 get_rtype(pl["room"].get("name",""))["cat"],
                 "мокрая зона" if is_wet(pl["room"].get("name","")) else ""]
        for ci, (v, cw) in enumerate(zip(vals, col_w)):
            ex_xi = ex_x + sum(col_w[:ci])
            fc = row_c if ci == 1 else "white"
            ax.add_patch(patches.Rectangle(
                (ex_xi, ry), cw, 0.35,
                facecolor=fc, edgecolor="#000", lw=0.4, alpha=0.6, zorder=20))
            ax.text(ex_xi + cw/2, ry+0.175, v,
                    ha="center", va="center", fontsize=5.5,
                    color="#000", zorder=21)

    plt.tight_layout()
    return fig

# ═══════════════ РАЗРЕЗ А-А (ГОСТ 21.501-2018, СП 55) ══════════════════════
def make_section(proj, sn=2, st2=6):
    """
    Вертикальный разрез здания. Показаны: фундамент, стены, перекрытия,
    крыша, окна, отметки уровней, размерные цепи высот.
    Нормы: ГОСТ 21.501-2018 п.5.7, ГОСТ 2.307-2011, СП 22.13330.
    """
    rooms = proj.get("rooms") or []
    if not rooms:
        return None
    placed, HW, _ = layout_rooms(rooms)

    fs = str(proj.get("floors","2"))
    fl = 1 if "1" in fs else (3 if "3" in fs else 2)
    ot = proj.get("object_type","")
    if "многоквартирный" in ot.lower():
        ns = re.findall(r'\d+', fs)
        fl = int(ns[0]) if ns else 5
    H = fl * FLOOR_H

    # Размеры фундамента
    FD = 1.50   # глубина фундамента (ниже промерзания)
    FW = 0.60   # ширина ленты фундамента
    CEIL_T = 0.25  # толщина ж/б перекрытия

    fig, ax = plt.subplots(figsize=(18, 12))
    XL, XR, YB, YT = -5.0, HW + 9.0, -3.5, H + 6.0
    ax.set_xlim(XL, XR); ax.set_ylim(YB, YT)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_facecolor("white"); fig.patch.set_facecolor("white")

    draw_frame(ax, XL, XR, YB, YT)
    draw_stamp(ax, XR, YB, proj, "Разрез А-А", sn, st2)

    # ── Грунт ───────────────────────────────────────────────────────────────
    ax.add_patch(patches.Rectangle(
        (-1.0, YB+0.3), HW+2.0, FD-0.1,
        facecolor="#D7CCC8", edgecolor="none", zorder=2, alpha=0.5))
    hatch_earth(ax, -1.0, YB+0.3, HW+2.0, FD-0.1)
    ax.plot([XL+0.5, XR-9.0], [-FD+FLOOR_H*0, -FD+FLOOR_H*0],
            color="#8D6E63", lw=1.5, linestyle="--", zorder=3)
    ax.text(XL+0.8, -FD+FLOOR_H*0+0.1, "Нат. грунт",
            fontsize=6, color="#8D6E63", zorder=4)

    # ── Фундамент (лента) — СП 22.13330 ────────────────────────────────────
    fd_rect = patches.Rectangle(
        (-FW, -FD), HW+2*FW, FD,
        facecolor="#90A4AE", edgecolor="#000", lw=1.5, zorder=3)
    ax.add_patch(fd_rect)
    hatch_conc(ax, -FW, -FD, HW+2*FW, FD)
    ax.text(HW/2, -FD/2, f"Ленточный монолитный фундамент\nB={FW*1000:.0f} мм, h={FD*1000:.0f} мм",
            ha="center", va="center", fontsize=7,
            color="#000", fontweight="bold", zorder=4)
    # Допуск на отметку подошвы
    draw_level(ax, -1.8, -FD, f"-{FD:.3f}")

    # ── Наружные стены (кирпич) ──────────────────────────────────────────────
    for wx in [0, HW-WALL_OUTER]:
        ax.add_patch(patches.Rectangle(
            (wx, 0), WALL_OUTER, H,
            facecolor="#FFF8E1", edgecolor="#000", lw=LW_CONTOUR, zorder=3))
        hatch_brick(ax, wx, 0, WALL_OUTER, H)
        # Толщина наружной стены
        ax.text(wx + WALL_OUTER/2, H/2,
                f"{WALL_OUTER*1000:.0f}\nмм",
                ha="center", va="center", fontsize=6,
                color="#37474F", rotation=90, zorder=4)

    # Утеплитель фасадный (минвата 150 мм)
    INS = 0.15
    for wx in [0, HW-WALL_OUTER]:
        ins_x = wx - INS if wx == 0 else wx + WALL_OUTER
        ax.add_patch(patches.Rectangle(
            (ins_x, 0), INS, H,
            facecolor="#B3E5FC", edgecolor="#000", lw=0.5, zorder=3, alpha=0.7))
        hatch_insulation(ax, ins_x, 0, INS, H)
        ax.text(ins_x + INS/2, H*0.25, "Утеп.\n150",
                ha="center", fontsize=4.5, color="#0277BD", rotation=90, zorder=4)

    # ── Перекрытия (ж/б) — СП 63.13330 ─────────────────────────────────────
    for f in range(fl + 1):
        z = f * FLOOR_H
        ax.add_patch(patches.Rectangle(
            (0, z), HW, CEIL_T,
            facecolor="#CFD8DC", edgecolor="#000", lw=1.0, zorder=3))
        hatch_conc(ax, 0, z, HW, CEIL_T)
        if f < fl:
            ax.text(HW/2, z + FLOOR_H/2,
                    f"Этаж {f+1}\nh = {FLOOR_H*1000:.0f} мм",
                    ha="center", va="center", fontsize=10,
                    color="#37474F", fontweight="bold", zorder=4)

    # ── Окна в разрезе ───────────────────────────────────────────────────────
    W_WIN_H = 1.50   # высота окна
    WIN_SILL = 0.90  # высота подоконника
    for f in range(fl):
        z = f * FLOOR_H
        for wx in np.linspace(WALL_OUTER + 1.2, HW - WALL_OUTER - 2.5,
                              max(2, int(HW / 4))):
            ax.add_patch(patches.Rectangle(
                (wx, z + WIN_SILL), 1.40, W_WIN_H,
                facecolor="#B3E5FC", edgecolor="#01579B", lw=1.0, zorder=4))
            # Рама — горизонтальный импост
            ax.plot([wx, wx+1.40],
                    [z+WIN_SILL+W_WIN_H/2, z+WIN_SILL+W_WIN_H/2],
                    color="#01579B", lw=0.6, zorder=5)
            # Подоконная доска
            ax.add_patch(patches.Rectangle(
                (wx-0.05, z+WIN_SILL-0.05), 1.50, 0.06,
                facecolor="#5D4037", edgecolor="#000", lw=0.5, zorder=4))

    # ── Крыша ───────────────────────────────────────────────────────────────
    if any(x in ot.lower() for x in ["частный дом","баня","гараж"]):
        rh, ov = 2.5, 0.6
        ridge_y = H + CEIL_T + rh
        tri = np.array([[-ov, H+CEIL_T], [HW+ov, H+CEIL_T],
                         [HW/2, ridge_y], [-ov, H+CEIL_T]])
        ax.add_patch(patches.Polygon(
            tri, facecolor="#B71C1C", edgecolor="#000", lw=1.5, zorder=4, alpha=0.93))
        ax.text(HW/2, H+CEIL_T+rh/2,
                "Металлочерепица\nСтропильная система\ni=28°",
                ha="center", va="center", fontsize=7,
                color="white", fontweight="bold", zorder=5)
        # Труба
        ax.add_patch(patches.Rectangle(
            (HW*0.7, H+CEIL_T), 0.40, rh*0.55,
            facecolor="#8D6E63", edgecolor="#000", lw=1.0, zorder=5))
        ax.text(HW*0.7+0.20, H+CEIL_T+rh*0.3, "Дымоход",
                ha="center", fontsize=5, color="white", rotation=90, zorder=6)
        # Конёк — отметка
        draw_level(ax, -1.8, ridge_y, f"+{ridge_y:.3f}")
    else:
        ax.add_patch(patches.Rectangle(
            (-0.30, H+CEIL_T), HW+0.60, 0.60,
            facecolor="#37474F", edgecolor="#000", lw=1.0, zorder=4))
        ax.text(HW/2, H+CEIL_T+0.30, "Плоская кровля  i=2%",
                ha="center", va="center", fontsize=7, color="white", zorder=5)
        draw_level(ax, -1.8, H+CEIL_T+0.60, f"+{H+CEIL_T+0.60:.3f}")

    # ── Отметки уровней ──────────────────────────────────────────────────────
    draw_level(ax, -1.8, 0,    "±0.000")
    for f in range(1, fl+1):
        draw_level(ax, -1.8, f*FLOOR_H, f"+{f*FLOOR_H:.3f}")
    draw_level(ax, -1.8, -FD,  f"-{FD:.3f}")

    # ── Размеры высот этажей ─────────────────────────────────────────────────
    DIM_X = HW + 1.8
    for f in range(fl):
        z = f * FLOOR_H
        ax.annotate("", xy=(DIM_X, z+FLOOR_H), xytext=(DIM_X, z),
                    arrowprops=dict(arrowstyle="<->", color="#000", lw=0.9))
        ax.text(DIM_X+0.25, z+FLOOR_H/2, f"{FLOOR_H*1000:.0f}",
                ha="left", va="center", fontsize=8, color="#000", rotation=90)
    # Общая высота здания
    ax.annotate("", xy=(DIM_X+1.5, H), xytext=(DIM_X+1.5, 0),
                arrowprops=dict(arrowstyle="<->", color="#B71C1C", lw=1.3))
    ax.text(DIM_X+1.85, H/2, f"H={H*1000:.0f}",
            ha="left", va="center", fontsize=10,
            color="#B71C1C", fontweight="bold", rotation=90)

    # Глубина фундамента
    ax.annotate("", xy=(DIM_X, -FD), xytext=(DIM_X, 0),
                arrowprops=dict(arrowstyle="<->", color="#8D6E63", lw=0.9))
    ax.text(DIM_X+0.25, -FD/2, f"{FD*1000:.0f}",
            ha="left", va="center", fontsize=8, color="#8D6E63", rotation=90)

    # ── Узловые выноски ──────────────────────────────────────────────────────
    # Узел 1 — угол фундамента (марка)
    ax.add_patch(plt.Circle((-FW/2, -FD/2), 0.20,
                             facecolor="white", edgecolor="#B71C1C", lw=1, zorder=15))
    ax.text(-FW/2, -FD/2, "1", ha="center", va="center",
            fontsize=7, color="#B71C1C", fontweight="bold", zorder=16)

    plt.tight_layout()
    return fig

# ═══════════════ ФАСАД (ГОСТ 21.501-2018, п.5.5) ════════════════════════════
def make_facade(proj, side="Фасад 1-1", sn=3, st2=6):
    """
    Архитектурный фасад с отметками уровней, межэтажными поясами,
    окнами с рамами, козырьком над входом, крышей.
    Нормы: ГОСТ 21.501-2018 п.5.5, ГОСТ 2.307, СП 55.13330.
    """
    rooms = proj.get("rooms") or []
    if not rooms:
        return None
    placed, HW, _ = layout_rooms(rooms)

    fs = str(proj.get("floors","2"))
    fl = 1 if "1" in fs else (3 if "3" in fs else 2)
    ot = proj.get("object_type","")
    if "многоквартирный" in ot.lower():
        ns = re.findall(r'\d+', fs)
        fl = int(ns[0]) if ns else 5
    H = fl * FLOOR_H

    fig, ax = plt.subplots(figsize=(18, 11))
    XL, XR, YB, YT = -3.5, HW + 6.0, -2.5, H + 6.5
    ax.set_xlim(XL, XR); ax.set_ylim(YB, YT)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_facecolor("white"); fig.patch.set_facecolor("white")

    draw_frame(ax, XL, XR, YB, YT)
    draw_stamp(ax, XR, YB, proj, side, sn, st2)

    # ── Отмостка / цоколь ────────────────────────────────────────────────────
    ax.add_patch(patches.Rectangle(
        (-0.50, -0.50), HW+1.00, 0.50,
        facecolor="#78909C", edgecolor="#000", lw=1.5, zorder=3))
    hatch_conc(ax, -0.50, -0.50, HW+1.00, 0.50)
    ax.text(HW/2, -0.25, "Цоколь / отмостка",
            ha="center", va="center", fontsize=6, color="white", zorder=4)

    # ── Фасадная плоскость ───────────────────────────────────────────────────
    ax.add_patch(patches.Rectangle(
        (0, 0), HW, H,
        facecolor="#FFFDE7", edgecolor="#000", lw=2.0, zorder=2))

    # ── Окна ─────────────────────────────────────────────────────────────────
    WIN_W, WIN_H_ = 1.40, 1.60
    WIN_SILL      = 0.90
    for f in range(fl):
        zb = f * FLOOR_H + WIN_SILL
        for wx in np.linspace(1.5, HW - 2.5, max(2, int(HW / 3.5))):
            # Оконный проём
            ax.add_patch(patches.Rectangle(
                (wx, zb), WIN_W, WIN_H_,
                facecolor="#B3E5FC", edgecolor="#01579B", lw=1.3, zorder=3))
            # Горизонтальный импост
            ax.plot([wx, wx+WIN_W], [zb+WIN_H_/2, zb+WIN_H_/2],
                    color="#01579B", lw=0.7, zorder=4)
            # Вертикальный импост
            ax.plot([wx+WIN_W/2, wx+WIN_W/2], [zb, zb+WIN_H_],
                    color="#01579B", lw=0.7, zorder=4)
            # Подоконник
            ax.add_patch(patches.Rectangle(
                (wx-0.06, zb-0.07), WIN_W+0.12, 0.09,
                facecolor="#5D4037", edgecolor="#000", lw=0.6, zorder=3))
            # Перемычка (над проёмом)
            ax.add_patch(patches.Rectangle(
                (wx-0.03, zb+WIN_H_), WIN_W+0.06, 0.12,
                facecolor="#90A4AE", edgecolor="#000", lw=0.5, zorder=3))
            # Размер окна
            ax.text(wx+WIN_W/2, zb-0.25,
                    f"{WIN_W*1000:.0f}×{WIN_H_*1000:.0f}",
                    ha="center", fontsize=5.5, color="#01579B", zorder=4)

    # ── Входная дверь с козырьком ────────────────────────────────────────────
    DX = HW/2 - 0.65
    ax.add_patch(patches.Rectangle(
        (DX, 0), 1.30, 2.30,
        facecolor="#4E342E", edgecolor="#000", lw=1.5, zorder=3))
    # Импост двери
    ax.plot([DX+0.65, DX+0.65], [0, 2.30],
            color="#3E2723", lw=0.7, zorder=4)
    # Козырёк
    ax.add_patch(patches.Polygon(
        [[DX-0.45, 2.30],[DX+1.75, 2.30],[DX+1.55, 2.72],[DX-0.25, 2.72]],
        facecolor="#607D8B", edgecolor="#000", lw=1.0, zorder=4))
    ax.text(DX+0.65, 2.51, "Козырёк",
            ha="center", fontsize=5.5, color="white", zorder=5)
    # Размер двери
    ax.text(DX+0.65, -0.25, f"{1300}×{2300}",
            ha="center", fontsize=5.5, color="#4E342E", zorder=4)
    # Ступени крыльца
    for si, sw_ in enumerate([1.60, 1.80, 2.00]):
        ax.add_patch(patches.Rectangle(
            (HW/2-sw_/2, -0.15*(si+1)), sw_, 0.15,
            facecolor="#9E9E9E", edgecolor="#000", lw=0.6, zorder=3))

    # ── Межэтажные пояса ─────────────────────────────────────────────────────
    for f in range(1, fl):
        ax.add_patch(patches.Rectangle(
            (-0.15, f*FLOOR_H-0.12), HW+0.30, 0.24,
            facecolor="#37474F", edgecolor="#000", lw=0.8, zorder=3))
        # Размер пояса
        ax.text(HW+0.25, f*FLOOR_H,
                f"Пояс {f}: {240} мм",
                va="center", fontsize=5.5, color="#37474F", zorder=4)

    # ── Крыша ───────────────────────────────────────────────────────────────
    if any(x in ot.lower() for x in ["частный дом","баня","гараж"]):
        rh, ov = 2.50, 0.65
        tri = np.array([[-ov, H], [HW+ov, H], [HW/2, H+rh], [-ov, H]])
        ax.add_patch(patches.Polygon(
            tri, facecolor="#B71C1C", edgecolor="#000",
            lw=1.5, zorder=4, alpha=0.94))
        # Водосток
        ax.add_patch(patches.Rectangle(
            (-ov-0.06, H-0.1), 0.08, -H+0.5,
            facecolor="#9E9E9E", edgecolor="#000", lw=0.6, zorder=3))
        ax.add_patch(patches.Rectangle(
            (HW+ov-0.02, H-0.1), 0.08, -H+0.5,
            facecolor="#9E9E9E", edgecolor="#000", lw=0.6, zorder=3))
        # Дымоход
        ax.add_patch(patches.Rectangle(
            (HW*0.70, H+rh*0.35), 0.40, rh*0.45,
            facecolor="#8D6E63", edgecolor="#000", lw=1.0, zorder=5))
        ax.text(HW*0.70+0.20, H+rh*0.60, "Дым.",
                ha="center", fontsize=5, color="white", rotation=90, zorder=6)
        draw_level(ax, -1.6, H+rh, f"+{H+rh:.3f}")
    else:
        ax.add_patch(patches.Rectangle(
            (-0.30, H), HW+0.60, 0.60,
            facecolor="#37474F", edgecolor="#000", lw=1.0, zorder=4))
        ax.text(HW/2, H+0.30, "Плоская кровля  i=2%",
                ha="center", va="center", fontsize=7, color="white", zorder=5)
        draw_level(ax, -1.6, H+0.60, f"+{H+0.60:.3f}")

    # ── Отметки уровней ──────────────────────────────────────────────────────
    draw_level(ax, -1.6, 0.0, "±0.000")
    draw_level(ax, -1.6, H,   f"+{H:.3f}")
    for f in range(1, fl):
        draw_level(ax, -1.6, f*FLOOR_H, f"+{f*FLOOR_H:.3f}")

    # ── Осевые линии наружных граней ─────────────────────────────────────────
    for xv, lb in [(0,"①"), (HW,"②")]:
        draw_axis(ax, xv, YB+0.3, xv, YT-0.3)
        draw_am(ax, xv, YB+0.15, lb.replace("①","1").replace("②","2"))
        draw_am(ax, xv, YT-0.15, lb.replace("①","1").replace("②","2"))

    # ── Общая ширина ─────────────────────────────────────────────────────────
    ax.plot([0, HW], [-2.0, -2.0], color="#000", lw=LW_DIM, zorder=8)
    for p in [0, HW]:
        ax.plot([p-0.07, p+0.07], [-2.06, -1.94],
                color="#000", lw=LW_DIM+0.3, zorder=9)
    ax.text(HW/2, -1.83, f"{HW*1000:.0f}",
            ha="center", va="bottom", fontsize=10,
            fontweight="bold", color="#000", zorder=10)

    ax.set_title(side, fontsize=13, fontweight="bold", color="#263238", pad=14)
    plt.tight_layout()
    return fig

# ═══════════════ ГЕНПЛАН (ГОСТ 21.508-93, М 1:500) ══════════════════════════
def make_genplan(proj, ot="", sn=4, st2=8):
    """
    Генеральный план участка: здание, парковка, септик, скважина,
    ограждение, деревья, красные линии, сети, розы ветров.
    Нормы: ГОСТ 21.508-93, СП 42.13330.2016, СП 30-102-99.
    """
    fig, ax = plt.subplots(figsize=(16, 15))
    ax.set_xlim(-5, 47); ax.set_ylim(-7, 57)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_facecolor("white"); fig.patch.set_facecolor("white")

    draw_frame(ax, -5, 47, -7, 57)
    draw_stamp(ax, 47, -7, proj, "Генеральный план  М 1:500", sn, st2)

    # ── Участок (красная линия) ──────────────────────────────────────────────
    ax.add_patch(patches.Rectangle(
        (0, 0), 40, 50,
        lw=2.0, edgecolor="#C62828", facecolor="#F1F8E9", linestyle="-"))
    ax.text(20, 52, "Красная линия застройки",
            ha="center", fontsize=7, color="#C62828", fontweight="bold")
    ax.plot([0, 40], [50, 50], color="#C62828", lw=2.0)
    ax.plot([0,  0], [ 0, 50], color="#C62828", lw=2.0)
    ax.plot([40,40], [ 0, 50], color="#C62828", lw=2.0)

    # ── Забор ────────────────────────────────────────────────────────────────
    for x in np.arange(0, 40, 2.5):
        ax.plot([x, x+1.5], [0, 0], color="#5D4037", lw=2.5)
        ax.plot([x, x+1.5], [50,50], color="#5D4037", lw=2.5)
    for y in np.arange(0, 50, 2.5):
        ax.plot([0, 0],   [y, y+1.5], color="#5D4037", lw=2.5)
        ax.plot([40, 40], [y, y+1.5], color="#5D4037", lw=2.5)
    # Ворота
    ax.add_patch(patches.Rectangle(
        (17, -0.3), 6, 0.6,
        facecolor="#795548", edgecolor="#000", lw=1, zorder=5))
    ax.text(20, -0.7, "Ворота", ha="center", fontsize=6, color="#5D4037")

    # ── Здание (штриховка крыши) ─────────────────────────────────────────────
    ax.add_patch(patches.Rectangle(
        (5, 5), 16, 13,
        lw=2.0, edgecolor="#000", facecolor="#E3F2FD", zorder=3))
    # Диагональная штриховка (условное обозначение здания на генплане)
    for d in np.arange(-13, 16+13, 1.5):
        x1 = max(5, 5+d-13); x2 = min(21, 5+d)
        y1 = 5+max(0, d-16); y2 = 5+min(13, d)
        if x2 > x1 and y2 > y1:
            ax.plot([x1,x2],[y1,y2], color="#90A4AE", lw=0.4, alpha=0.5, zorder=3)
    ax.text(13, 12.2, "ЗДАНИЕ", ha="center", fontsize=9,
            fontweight="bold", color="#000", zorder=4)
    ax.text(13, 10.5, f"{proj.get('area','')}",
            ha="center", fontsize=7, color="#37474F", zorder=4)
    # Отмостка
    ax.add_patch(patches.Rectangle(
        (4.2, 4.2), 17.6, 14.6,
        lw=1.0, edgecolor="#78909C", facecolor="none",
        linestyle="--", zorder=3))
    ax.text(13, 4.0, "Отмостка 800 мм",
            ha="center", fontsize=5.5, color="#78909C", zorder=4)

    # ── Отступы (противопожарные — СП 42.13330) ─────────────────────────────
    for x1,y1,x2,y2,tx,ty,lbl in [
        (0, 5, 5, 5, 2.5, 5.6, "5.0 м"),
        (5, 0, 5, 5, 5.8, 2.5, "5.0 м"),
        (21,5, 40,5, 30.5,5.6, "19.0 м"),
    ]:
        ax.annotate("", xy=(x2,y2), xytext=(x1,y1),
                    arrowprops=dict(arrowstyle="<->", color="#B71C1C", lw=0.9))
        ax.text(tx, ty, lbl, ha="center", fontsize=6.5,
                color="#B71C1C", fontweight="bold")

    # ── Парковка ─────────────────────────────────────────────────────────────
    ax.add_patch(patches.Rectangle(
        (23, 5), 7, 10,
        lw=1.5, edgecolor="#607D8B", facecolor="#ECEFF1", zorder=3))
    ax.text(26.5, 10.2, "Парковка\n3 м/места",
            ha="center", va="center", fontsize=7, color="#000")
    # Разметка машино-мест
    for pk in range(3):
        ax.plot([23, 30], [5+pk*3.3, 5+pk*3.3], color="#607D8B", lw=0.5)

    # ── Септик ───────────────────────────────────────────────────────────────
    ax.add_patch(plt.Circle((35, 8), 1.6,
                             facecolor="#FFCC80", edgecolor="#000", lw=1.5, zorder=3))
    ax.text(35, 8, "Септик\nЛОС-5", ha="center", va="center",
            fontsize=6, color="#000", fontweight="bold", zorder=4)
    # Труба к зданию
    ax.plot([21, 35], [6, 8], color="#8D6E63", lw=1.5,
            linestyle="-.", zorder=3)
    ax.text(28, 6.5, "Канализация Ø110",
            fontsize=5.5, color="#6D4C41", zorder=4)

    # ── Скважина ─────────────────────────────────────────────────────────────
    ax.add_patch(plt.Circle((5, 43), 1.3,
                             facecolor="#B3E5FC", edgecolor="#000", lw=1.5, zorder=3))
    ax.text(5, 43, "Скважина\n35 м", ha="center", va="center",
            fontsize=6, color="#000", fontweight="bold", zorder=4)
    # Ввод воды к зданию
    ax.plot([5, 5], [43-1.3, 18], color="#29B6F6", lw=1.5, zorder=3)
    ax.plot([5, 5], [18, 18],  color="#29B6F6", lw=1.5, zorder=3)
    ax.plot([5, 5], [18, 18],  color="#29B6F6", lw=1.5, zorder=3)
    ax.text(5.3, 30, "ВВод ХВС Ø32",
            fontsize=5.5, color="#0288D1", rotation=90, zorder=4)

    # ── Газ (если есть) ──────────────────────────────────────────────────────
    ax.plot([0, 5], [25, 25], color="#FF8F00", lw=2.0, linestyle="-.", zorder=3)
    ax.text(2.5, 25.5, "ГАЗ Ø32", fontsize=5.5, color="#FF8F00",
            ha="center", fontweight="bold", zorder=4)

    # ── Электровод ───────────────────────────────────────────────────────────
    ax.plot([0, 5], [22, 22], color="#FDD835", lw=1.5, linestyle="--", zorder=3)
    ax.text(2.5, 22.5, "СИП 4×16", fontsize=5.5, color="#F9A825",
            ha="center", fontweight="bold", zorder=4)

    # ── Деревья и кустарники ─────────────────────────────────────────────────
    for tx, ty, r_ in [(8,32,1.4),(16,37,1.2),(29,33,1.4),(33,37,1.0),(38,20,1.2)]:
        ax.add_patch(plt.Circle((tx,ty), r_,
                                facecolor="#A5D6A7", edgecolor="#388E3C",
                                lw=0.7, zorder=3, alpha=0.75))
    # Огород / газон
    ax.add_patch(patches.Rectangle(
        (5, 22), 30, 17,
        lw=0.5, edgecolor="#81C784", facecolor="#F9FBE7",
        linestyle=":", zorder=2))
    ax.text(20, 30, "Газон / огород", ha="center",
            fontsize=9, color="#558B2F", alpha=0.6, zorder=3)

    # ── Стороны света ────────────────────────────────────────────────────────
    for nx, ny, lb in [(20,54,"С"),(20,-3,"Ю"),(1,25,"З"),(39,25,"В")]:
        ax.add_patch(plt.Circle((nx,ny), 1.0,
                                facecolor="white", edgecolor="#000", lw=1.0, zorder=10))
        ax.text(nx, ny, lb, ha="center", va="center",
                fontsize=11, fontweight="bold", color="#000", zorder=11)

    # ── Легенда ──────────────────────────────────────────────────────────────
    leg_x, leg_y = 42, 50
    ax.text(leg_x, leg_y+0.5, "Условные\nобозначения",
            ha="center", fontsize=6.5, fontweight="bold", color="#000")
    items_g = [("Здание", "#E3F2FD"),
               ("Парковка", "#ECEFF1"),
               ("Газон", "#F9FBE7"),
               ("Деревья", "#A5D6A7")]
    for gi, (gl, gc) in enumerate(items_g):
        gy = leg_y - 1.0 - gi*0.9
        ax.add_patch(patches.Rectangle(
            (leg_x-1.8, gy-0.2), 1.0, 0.4,
            facecolor=gc, edgecolor="#000", lw=0.6))
        ax.text(leg_x-0.65, gy, gl, fontsize=5.5, va="center")

    # Масштабная линейка
    ax.plot([1, 11], [-4.5, -4.5], color="#000", lw=1.5)
    ax.plot([1,  1], [-4.3, -4.7], color="#000", lw=1.0)
    ax.plot([11,11], [-4.3, -4.7], color="#000", lw=1.0)
    ax.plot([6,  6], [-4.3, -4.7], color="#000", lw=0.6)
    ax.text(6, -5.2, "0          25         50 м",
            ha="center", fontsize=6, color="#000")

    plt.tight_layout()
    return fig

# ═══════════════ УЛУЧШЕННАЯ РАСКЛАДКА КОМНАТ ═══════════════
def layout_rooms_grid(rooms):
    """
    Раскладка комнат в виде реалистичного плана дома:
    прямоугольник, коридор по центру, комнаты по бокам.
    Возвращает: placed, HW, HH, corr_y, corr_h
    """
    if not rooms:
        return [], 12, 8, 0, 0

    CORR_W = 1.5
    all_rooms = list(rooms)
    n = len(all_rooms)
    cols = max(2, (n + 1) // 2)

    total_area = sum(parse_area(r.get("area")) for r in all_rooms)
    W = max(10, min(16, total_area / 6))
    H = max(7, total_area / W * 1.3)

    room_h = (H - CORR_W) / 2

    # Верхний ряд (сверху коридора)
    top_rooms = all_rooms[:cols]
    top_w = W / max(1, len(top_rooms))
    # Нижний ряд (снизу коридора)
    bot_rooms = all_rooms[cols:] if cols < n else []
    bot_w = W / max(1, len(bot_rooms)) if bot_rooms else W

    placed = []
    idx = 1

    for i, r in enumerate(top_rooms):
        a = parse_area(r.get("area"))
        rw = top_w
        ry = CORR_W + room_h
        placed.append({"room": r, "x": i * top_w, "y": ry,
                        "w": rw, "h": room_h,
                        "cx": i * top_w + rw / 2, "cy": ry + room_h / 2,
                        "idx": idx, "row": "top"})
        idx += 1

    for i, r in enumerate(bot_rooms):
        a = parse_area(r.get("area"))
        rw = bot_w
        ry = 0
        placed.append({"room": r, "x": i * bot_w, "y": ry,
                        "w": rw, "h": room_h,
                        "cx": i * bot_w + rw / 2, "cy": ry + room_h / 2,
                        "idx": idx, "row": "bot"})
        idx += 1

    # Коридор
    corr_name = "Коридор"
    corr_room = {"name": corr_name, "area": f"{W * CORR_W:.1f} м2", "floor": "1 этаж"}
    placed.append({"room": corr_room, "x": 0, "y": room_h,
                    "w": W, "h": CORR_W,
                    "cx": W / 2, "cy": room_h + CORR_W / 2,
                    "idx": idx, "row": "corr"})
    corr_y = room_h
    corr_h = CORR_W

    return placed, W, H, corr_y, corr_h


def draw_base_plan(ax, placed, HW, HH, corr_y, corr_h, show_wet_dots=True):
    """
    Рисует архитектурную основу плана:
    наружные стены, внутренние стены, заливку помещений,
    двери между коридором и комнатами, окна на наружных стенах,
    подписи помещений с номерами и площадями.
    """
    # ── Наружные стены (кирпич) ──
    for rect in [
        (-WALL_OUTER, -WALL_OUTER, HW + 2 * WALL_OUTER, WALL_OUTER),
        (-WALL_OUTER, HH, HW + 2 * WALL_OUTER, WALL_OUTER),
        (-WALL_OUTER, 0, WALL_OUTER, HH),
        (HW, 0, WALL_OUTER, HH),
    ]:
        x, y, w, h = rect
        ax.add_patch(patches.Rectangle((x, y), w, h,
                                       facecolor="#FFF8E1", edgecolor="#000",
                                       lw=LW_CONTOUR, zorder=5))
        hatch_brick(ax, x, y, w, h)

    # ── Заливка помещений + подписи ──
    for pl in placed:
        name = pl["room"].get("name", "")
        nl = name.lower()
        c = get_rc(name)
        ax.add_patch(patches.Rectangle(
            (pl["x"], pl["y"]), pl["w"], pl["h"],
            facecolor=c, alpha=0.38, edgecolor="#90A4AE",
            lw=0.7, zorder=3))

        # Мокрая зона — точки
        if show_wet_dots and is_wet(name):
            for xi in np.arange(pl["x"] + 0.25, pl["x"] + pl["w"], 0.32):
                for yi in np.arange(pl["y"] + 0.25, pl["y"] + pl["h"], 0.32):
                    ax.plot(xi, yi, marker=".", markersize=1.8,
                            color="#0288D1", alpha=0.6, zorder=3)

        # Номер в кружке
        cx, cy = pl["cx"], pl["cy"]
        ax.add_patch(plt.Circle((cx, cy + 0.45), 0.24,
                                facecolor="white", edgecolor="#000",
                                lw=0.8, zorder=15))
        ax.text(cx, cy + 0.45, f"{pl['idx']}",
                ha="center", va="center", fontsize=7.5,
                fontweight="bold", color="#000", zorder=16)
        # Название
        if nl != "коридор":
            ax.text(cx, cy, name,
                    ha="center", va="center", fontsize=7.5,
                    color="#000", fontweight="bold", zorder=15)
        # Площадь
        a = parse_area(pl["room"].get("area"))
        ax.text(cx, cy - 0.40, f"{a:.1f} м²",
                ha="center", va="center", fontsize=8.5,
                fontweight="bold", color="#000", zorder=15)
        ax.plot([cx - 0.50, cx + 0.50], [cy - 0.62, cy - 0.62],
                color="#000", lw=0.7, zorder=15)
        # Минимальная площадь (допуск)
        min_a = get_rtype(name)["min_area"]
        ok_col = "#1B5E20" if a >= min_a else "#B71C1C"
        ax.text(cx, cy - 0.85, f"min {min_a} м²",
                ha="center", fontsize=5, color=ok_col, zorder=15)

    # ── Внутренние стены ──
    for pl in placed:
        # Вертикальные стены (между комнатами в ряду)
        if pl["x"] > 0.02:
            ax.add_patch(patches.Rectangle(
                (pl["x"] - WALL_INNER / 2, pl["y"]),
                WALL_INNER, pl["h"],
                facecolor="#ECEFF1", edgecolor="#000",
                lw=LW_CONTOUR * 0.8, zorder=5))
        # Горизонтальные стены (между рядами и коридором)
        if pl["y"] > 0.02 and pl.get("row") != "corr":
            ax.add_patch(patches.Rectangle(
                (pl["x"], pl["y"] - WALL_PART / 2),
                pl["w"], WALL_PART,
                facecolor="#ECEFF1", edgecolor="#000",
                lw=LW_CONTOUR * 0.7, zorder=5))
        if pl["y"] + pl["h"] < HH - 0.02 and pl.get("row") != "corr":
            ax.add_patch(patches.Rectangle(
                (pl["x"], pl["y"] + pl["h"] - WALL_PART / 2),
                pl["w"], WALL_PART,
                facecolor="#ECEFF1", edgecolor="#000",
                lw=LW_CONTOUR * 0.7, zorder=5))

    # ── Двери между коридором и каждой комнатой ──
    for pl in placed:
        if pl.get("row") == "corr" or pl.get("row") == "top":
            continue
        if pl["room"].get("name", "").lower() == "коридор":
            continue
        # Нижний ряд — дверь в верхней стене комнаты (в коридор)
        door_x = pl["x"] + pl["w"] / 2 - DOOR_INT / 2
        draw_door(ax, door_x, pl["y"] + pl["h"], DOOR_INT, "up_right")

    for pl in placed:
        if pl.get("row") != "top":
            continue
        if pl["room"].get("name", "").lower() == "коридор":
            continue
        # Верхний ряд — дверь в нижней стене комнаты (в коридор)
        door_x = pl["x"] + pl["w"] / 2 - DOOR_INT / 2
        draw_door(ax, door_x, pl["y"], DOOR_INT, "down_right")

    # ── Входная дверь (центр нижней стены здания) ──
    ent_x = HW / 2 - DOOR_ENT / 2
    draw_door(ax, ent_x, 0, DOOR_ENT, "up_right")

    # ── Окна на наружных стенах ──
    for pl in placed:
        name = pl["room"].get("name", "")
        if is_no_win(name):
            continue
        # Верхний ряд — окно в верхней стене
        if pl.get("row") == "top":
            wx = pl["x"] + pl["w"] / 2 - WIN_STD / 2
            draw_window(ax, wx, HH, "h", WIN_STD)
        # Нижний ряд — окно в нижней стене
        elif pl.get("row") == "bot":
            wx = pl["x"] + pl["w"] / 2 - WIN_STD / 2
            draw_window(ax, wx, 0, "h", WIN_STD)
        # Крайние комнаты — окно в боковой стене
        if pl["x"] <= 0.02:
            wy = pl["y"] + pl["h"] / 2 - WIN_STD / 2
            draw_window(ax, 0, wy, "v", WIN_STD)
        if pl["x"] + pl["w"] >= HW - 0.02:
            wy = pl["y"] + pl["h"] / 2 - WIN_STD / 2
            draw_window(ax, HW, wy, "v", WIN_STD)

    # ── Стрелка входа ──
    ax.annotate("ВХОД", xy=(HW / 2, -0.5), xytext=(HW / 2, -1.8),
                fontsize=9, ha="center", color="#1B5E20", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#1B5E20", lw=2.0),
                zorder=15)


def draw_plan_axes_and_dims(ax, placed, HW, HH):
    """
    Рисует осевые линии с марками и размерные цепи.
    """
    raw_x = sorted({0.0} | {p["x"] for p in placed} |
                   {p["x"] + p["w"] for p in placed} | {HW})
    axu = [raw_x[0]]
    for xv in raw_x[1:]:
        if xv - axu[-1] > 0.4:
            axu.append(xv)

    raw_y = sorted({0.0} | {p["y"] for p in placed} |
                   {p["y"] + p["h"] for p in placed} | {HH})
    ayu = [raw_y[0]]
    for yv in raw_y[1:]:
        if yv - ayu[-1] > 0.4:
            ayu.append(yv)

    for xv in axu:
        draw_axis(ax, xv, -2.2, xv, HH + 2.8)
    for yv in ayu:
        draw_axis(ax, -2.2, yv, HW + 2.8, yv)

    letters = "АБВГДЕЖИКЛМН"
    for i, xv in enumerate(axu):
        draw_am(ax, xv, -2.5, i + 1)
        draw_am(ax, xv, HH + 3.0, i + 1)
    for i, yv in enumerate(ayu):
        lb = letters[i] if i < len(letters) else str(i + 1)
        draw_am(ax, -2.5, yv, lb)
        draw_am(ax, HW + 3.0, yv, lb)

    draw_dim(ax, axu, -3.5, "h")
    draw_dim(ax, ayu, HW + 4.2, "v")


def get_wet_rooms(placed):
    """Возвращает список мокрых комнат (без коридора)."""
    return [p for p in placed
            if p["room"].get("name", "").lower() != "коридор"
            and is_wet(p["room"].get("name", ""))]

# ═══════════════ УЛУЧШЕННАЯ РАСКЛАДКА КОМНАТ ═══════════════
def layout_rooms_grid(rooms):
    """
    Раскладка комнат в виде реалистичного плана дома:
    прямоугольник, коридор по центру, комнаты по бокам.
    Возвращает: placed, HW, HH, corr_y, corr_h
    """
    if not rooms:
        return [], 12, 8, 0, 0

    CORR_W = 1.5
    all_rooms = list(rooms)
    n = len(all_rooms)
    cols = max(2, (n + 1) // 2)

    total_area = sum(parse_area(r.get("area")) for r in all_rooms)
    W = max(10, min(16, total_area / 6))
    H = max(7, total_area / W * 1.3)

    room_h = (H - CORR_W) / 2

    # Верхний ряд (сверху коридора)
    top_rooms = all_rooms[:cols]
    top_w = W / max(1, len(top_rooms))
    # Нижний ряд (снизу коридора)
    bot_rooms = all_rooms[cols:] if cols < n else []
    bot_w = W / max(1, len(bot_rooms)) if bot_rooms else W

    placed = []
    idx = 1

    for i, r in enumerate(top_rooms):
        a = parse_area(r.get("area"))
        rw = top_w
        ry = CORR_W + room_h
        placed.append({"room": r, "x": i * top_w, "y": ry,
                        "w": rw, "h": room_h,
                        "cx": i * top_w + rw / 2, "cy": ry + room_h / 2,
                        "idx": idx, "row": "top"})
        idx += 1

    for i, r in enumerate(bot_rooms):
        a = parse_area(r.get("area"))
        rw = bot_w
        ry = 0
        placed.append({"room": r, "x": i * bot_w, "y": ry,
                        "w": rw, "h": room_h,
                        "cx": i * bot_w + rw / 2, "cy": ry + room_h / 2,
                        "idx": idx, "row": "bot"})
        idx += 1

    # Коридор
    corr_name = "Коридор"
    corr_room = {"name": corr_name, "area": f"{W * CORR_W:.1f} м2", "floor": "1 этаж"}
    placed.append({"room": corr_room, "x": 0, "y": room_h,
                    "w": W, "h": CORR_W,
                    "cx": W / 2, "cy": room_h + CORR_W / 2,
                    "idx": idx, "row": "corr"})
    corr_y = room_h
    corr_h = CORR_W

    return placed, W, H, corr_y, corr_h


def draw_base_plan(ax, placed, HW, HH, corr_y, corr_h, show_wet_dots=True):
    """
    Рисует архитектурную основу плана:
    наружные стены, внутренние стены, заливку помещений,
    двери между коридором и комнатами, окна на наружных стенах,
    подписи помещений с номерами и площадями.
    """
    # ── Наружные стены (кирпич) ──
    for rect in [
        (-WALL_OUTER, -WALL_OUTER, HW + 2 * WALL_OUTER, WALL_OUTER),
        (-WALL_OUTER, HH, HW + 2 * WALL_OUTER, WALL_OUTER),
        (-WALL_OUTER, 0, WALL_OUTER, HH),
        (HW, 0, WALL_OUTER, HH),
    ]:
        x, y, w, h = rect
        ax.add_patch(patches.Rectangle((x, y), w, h,
                                       facecolor="#FFF8E1", edgecolor="#000",
                                       lw=LW_CONTOUR, zorder=5))
        hatch_brick(ax, x, y, w, h)

    # ── Заливка помещений + подписи ──
    for pl in placed:
        name = pl["room"].get("name", "")
        nl = name.lower()
        c = get_rc(name)
        ax.add_patch(patches.Rectangle(
            (pl["x"], pl["y"]), pl["w"], pl["h"],
            facecolor=c, alpha=0.38, edgecolor="#90A4AE",
            lw=0.7, zorder=3))

        # Мокрая зона — точки
        if show_wet_dots and is_wet(name):
            for xi in np.arange(pl["x"] + 0.25, pl["x"] + pl["w"], 0.32):
                for yi in np.arange(pl["y"] + 0.25, pl["y"] + pl["h"], 0.32):
                    ax.plot(xi, yi, marker=".", markersize=1.8,
                            color="#0288D1", alpha=0.6, zorder=3)

        # Номер в кружке
        cx, cy = pl["cx"], pl["cy"]
        ax.add_patch(plt.Circle((cx, cy + 0.45), 0.24,
                                facecolor="white", edgecolor="#000",
                                lw=0.8, zorder=15))
        ax.text(cx, cy + 0.45, f"{pl['idx']}",
                ha="center", va="center", fontsize=7.5,
                fontweight="bold", color="#000", zorder=16)
        # Название
        if nl != "коридор":
            ax.text(cx, cy, name,
                    ha="center", va="center", fontsize=7.5,
                    color="#000", fontweight="bold", zorder=15)
        # Площадь
        a = parse_area(pl["room"].get("area"))
        ax.text(cx, cy - 0.40, f"{a:.1f} м²",
                ha="center", va="center", fontsize=8.5,
                fontweight="bold", color="#000", zorder=15)
        ax.plot([cx - 0.50, cx + 0.50], [cy - 0.62, cy - 0.62],
                color="#000", lw=0.7, zorder=15)
        # Минимальная площадь (допуск)
        min_a = get_rtype(name)["min_area"]
        ok_col = "#1B5E20" if a >= min_a else "#B71C1C"
        ax.text(cx, cy - 0.85, f"min {min_a} м²",
                ha="center", fontsize=5, color=ok_col, zorder=15)

    # ── Внутренние стены ──
    for pl in placed:
        # Вертикальные стены (между комнатами в ряду)
        if pl["x"] > 0.02:
            ax.add_patch(patches.Rectangle(
                (pl["x"] - WALL_INNER / 2, pl["y"]),
                WALL_INNER, pl["h"],
                facecolor="#ECEFF1", edgecolor="#000",
                lw=LW_CONTOUR * 0.8, zorder=5))
        # Горизонтальные стены (между рядами и коридором)
        if pl["y"] > 0.02 and pl.get("row") != "corr":
            ax.add_patch(patches.Rectangle(
                (pl["x"], pl["y"] - WALL_PART / 2),
                pl["w"], WALL_PART,
                facecolor="#ECEFF1", edgecolor="#000",
                lw=LW_CONTOUR * 0.7, zorder=5))
        if pl["y"] + pl["h"] < HH - 0.02 and pl.get("row") != "corr":
            ax.add_patch(patches.Rectangle(
                (pl["x"], pl["y"] + pl["h"] - WALL_PART / 2),
                pl["w"], WALL_PART,
                facecolor="#ECEFF1", edgecolor="#000",
                lw=LW_CONTOUR * 0.7, zorder=5))

    # ── Двери между коридором и каждой комнатой ──
    for pl in placed:
        if pl.get("row") == "corr" or pl.get("row") == "top":
            continue
        if pl["room"].get("name", "").lower() == "коридор":
            continue
        # Нижний ряд — дверь в верхней стене комнаты (в коридор)
        door_x = pl["x"] + pl["w"] / 2 - DOOR_INT / 2
        draw_door(ax, door_x, pl["y"] + pl["h"], DOOR_INT, "up_right")

    for pl in placed:
        if pl.get("row") != "top":
            continue
        if pl["room"].get("name", "").lower() == "коридор":
            continue
        # Верхний ряд — дверь в нижней стене комнаты (в коридор)
        door_x = pl["x"] + pl["w"] / 2 - DOOR_INT / 2
        draw_door(ax, door_x, pl["y"], DOOR_INT, "down_right")

    # ── Входная дверь (центр нижней стены здания) ──
    ent_x = HW / 2 - DOOR_ENT / 2
    draw_door(ax, ent_x, 0, DOOR_ENT, "up_right")

    # ── Окна на наружных стенах ──
    for pl in placed:
        name = pl["room"].get("name", "")
        if is_no_win(name):
            continue
        # Верхний ряд — окно в верхней стене
        if pl.get("row") == "top":
            wx = pl["x"] + pl["w"] / 2 - WIN_STD / 2
            draw_window(ax, wx, HH, "h", WIN_STD)
        # Нижний ряд — окно в нижней стене
        elif pl.get("row") == "bot":
            wx = pl["x"] + pl["w"] / 2 - WIN_STD / 2
            draw_window(ax, wx, 0, "h", WIN_STD)
        # Крайние комнаты — окно в боковой стене
        if pl["x"] <= 0.02:
            wy = pl["y"] + pl["h"] / 2 - WIN_STD / 2
            draw_window(ax, 0, wy, "v", WIN_STD)
        if pl["x"] + pl["w"] >= HW - 0.02:
            wy = pl["y"] + pl["h"] / 2 - WIN_STD / 2
            draw_window(ax, HW, wy, "v", WIN_STD)

    # ── Стрелка входа ──
    ax.annotate("ВХОД", xy=(HW / 2, -0.5), xytext=(HW / 2, -1.8),
                fontsize=9, ha="center", color="#1B5E20", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#1B5E20", lw=2.0),
                zorder=15)


def draw_plan_axes_and_dims(ax, placed, HW, HH):
    """
    Рисует осевые линии с марками и размерные цепи.
    """
    raw_x = sorted({0.0} | {p["x"] for p in placed} |
                   {p["x"] + p["w"] for p in placed} | {HW})
    axu = [raw_x[0]]
    for xv in raw_x[1:]:
        if xv - axu[-1] > 0.4:
            axu.append(xv)

    raw_y = sorted({0.0} | {p["y"] for p in placed} |
                   {p["y"] + p["h"] for p in placed} | {HH})
    ayu = [raw_y[0]]
    for yv in raw_y[1:]:
        if yv - ayu[-1] > 0.4:
            ayu.append(yv)

    for xv in axu:
        draw_axis(ax, xv, -2.2, xv, HH + 2.8)
    for yv in ayu:
        draw_axis(ax, -2.2, yv, HW + 2.8, yv)

    letters = "АБВГДЕЖИКЛМН"
    for i, xv in enumerate(axu):
        draw_am(ax, xv, -2.5, i + 1)
        draw_am(ax, xv, HH + 3.0, i + 1)
    for i, yv in enumerate(ayu):
        lb = letters[i] if i < len(letters) else str(i + 1)
        draw_am(ax, -2.5, yv, lb)
        draw_am(ax, HW + 3.0, yv, lb)

    draw_dim(ax, axu, -3.5, "h")
    draw_dim(ax, ayu, HW + 4.2, "v")


def get_wet_rooms(placed):
    """Возвращает список мокрых комнат (без коридора)."""
    return [p for p in placed
            if p["room"].get("name", "").lower() != "коридор"
            and is_wet(p["room"].get("name", ""))]

# ═══════════════ AI (YandexGPT) ═══════════════
def get_ai(inp, hist, rn, norms, ot):
    oi = OBJECT_TYPES.get(ot, OBJECT_TYPES["🏠 Частный дом (ИЖС)"])
    asnips = SNIPS + oi["snips"]
    sp = f"""Ты архитектор в России. Соблюдай СНиП,СП,ГОСТ,ПУЭ,ФЗ-384.
Тип:{ot} Регион:{rn} Нормативы:{",".join(asnips)}
Верни ТОЛЬКО JSON. Структура:
{{"object_type":"{ot}","summary":"Записка","location":"{rn}","budget":"Бюджет","area":"Площадь","floors":"Этажность","residents":"Пользователи",
"region_data":{{"climate_zone":"{norms['climate_zone']}","frost_depth":"{norms['frost_depth']}","snow_load":"{norms['snow_load']}","wind_load":"{norms['wind_load']}","seismicity":"{norms['seismicity']}","thermal_resistance":"{norms['thermal_resistance']}","applied_snips":{json.dumps(asnips,ensure_ascii=False)}}},
"pre_construction_docs":[{{"name":"ЕГРН","description":"МФЦ"}},{{"name":"ГПЗУ","description":"Госуслуги"}}],
"layout":{{"concept":"Концепция","floors_plan":[{{"floor":"1 этаж","idea":"Описание","rooms":[{{"name":"Прихожая","area":"6 м2","neighbors":"гостиная","windows":"нет","wet_zone":false,"description":"Вход"}}]}}],"zoning":"Зонирование","orientation":"Ориентация"}},
"rooms":[{{"name":"Прихожая","area":"6 м2","floor":"1 этаж","neighbors":"вход","windows":"нет","wet_zone":false,"electric":"розетка","water":"-","sewage":"-","description":"Вход"}}],
"features":["Терраса"],
"construction":{{"foundation":"Монолит","walls":"Газобетон","floors_type":"Ж/б","roof":"Металлочерепица","insulation":"Минвата 200мм"}},
"electricity":{{"source":"Ввод 15 кВт","allocated_power":"15 кВт","input_cable":"СИП 4×16","main_board":"ВРУ 3ф","grounding":"TN-C-S","groups":[{{"name":"Освещение","rooms":"все","protection":"АВ 10А"}}]}},
"water":{{"source":"Скважина Ø133","entry":"h=-1.8м ниже промерзания","pressure":"3 бар","filtration":"Ф1 Ф2","hot_water":"Бойлер 80л","risers":"ХВС Ø25 ГВС Ø25","points":[{{"room":"Кухня","fixtures":"мойка","pipe":"PPR Ø20"}}]}},
"sewage":{{"type":"Септик ЛОС","outdoor":"Выпуск Ø110","septic":"Топас-5","risers":"Ст. Ø110","indoor":"Ø50 i=0.03","points":[{{"room":"Кухня","fixture":"мойка","diameter":"D50","route":"стояк"}}]}},
"heating":{{"type":"Газовый котёл","distribution":"Тёплый пол + радиаторы","boiler_room":"Котельная 8м2"}},"ventilation":{{"type":"ПВУ с рекуп.","air_exchange":"3 об/ч","special":"Нет"}},"fire_safety":{{"fire_resistance_degree":"II","fire_danger_class":"С0","evacuation":"По плану","alarm":"ИП-212","extinguishers":"ОП-5"}},"accessibility":{{"ramp":"Нет","elevator":"Нет","parking_mgn":"Нет"}},"estimate":[{{"item":"Фундамент","cost":"500000 руб."}}],"total_cost":"5000000 руб."}}"""

    ym = []
    for m in hist:
        if m.get("role") in ("user", "assistant") and m.get("content"):
            ym.append({"role": m["role"], "text": m["content"]})
    ym.append({"role": "user", "text": inp})
    h = {"Authorization": f"Api-Key {api_key}", "Content-Type": "application/json"}
    for model in (
        f"gpt://{folder_id}/yandexgpt/latest",
        f"gpt://{folder_id}/yandexgpt-lite/latest",
    ):
        pl = {
            "modelUri": model,
            "completionOptions": {"stream": False, "temperature": 0.2, "maxTokens": "4000"},
            "messages": [{"role": "system", "text": sp}] + ym,
        }
        try:
            r = requests.post(
                "https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
                headers=h, json=pl, timeout=120,
            )
            if r.status_code == 200:
                return r.json()["result"]["alternatives"][0]["message"]["text"]
        except Exception:
            pass
    return None


# ═══════════════ PDF ═══════════════
def efont(ps):
    for p in ps:
        if p and os.path.exists(p) and os.path.getsize(p) > 20000:
            return p
    return None

def gfp():
    w = r"C:\Windows\Fonts"
    r_font = efont([os.path.join(w, "arial.ttf"), os.path.join(w, "calibri.ttf"), os.path.join(w, "tahoma.ttf")])
    b_font = efont([os.path.join(w, "arialbd.ttf"), os.path.join(w, "calibrib.ttf"), os.path.join(w, "tahomabd.ttf"), r_font])
    if not r_font:
        raise RuntimeError("Шрифт не найден")
    return r_font, b_font or r_font

def gen_pdf(proj):
    from reportlab.lib import colors as rlc
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    r_fnt, b_fnt = gfp()
    pdfmetrics.registerFont(TTFont("AF", r_fnt))
    pdfmetrics.registerFont(TTFont("AB", b_fnt))
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=16*mm, rightMargin=16*mm, topMargin=14*mm, bottomMargin=14*mm, title="Паспорт")
    ts = ParagraphStyle("ts", fontName="AB", fontSize=18, leading=22, alignment=1, spaceAfter=10)
    hs = ParagraphStyle("hs", fontName="AB", fontSize=13, leading=17, spaceBefore=8, spaceAfter=6)
    ps2 = ParagraphStyle("ps2", fontName="AF", fontSize=10, leading=14, spaceAfter=4)
    ss = ParagraphStyle("ss", fontName="AF", fontSize=8, leading=11, textColor=rlc.grey, alignment=1)
    def p(t, s=None): return Paragraph(safe_text(t).replace("\n", "<br/>"), s or ps2)
    def tbl(rows, widths):
        d = [[p(c) for c in row] for row in rows]
        t = Table(d, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),rlc.Color(0.9,0.9,0.96)),("GRID",(0,0),(-1,-1),0.4,rlc.grey),("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),4),("RIGHTPADDING",(0,0),(-1,-1),4)]))
        return t
    def fi(fig_obj, w=180*mm):
        b2 = io.BytesIO(); fig_obj.savefig(b2, format="png", dpi=110, bbox_inches="tight", facecolor="white"); b2.seek(0)
        return Image(b2, width=w, height=w*0.72)
    ly = proj.get("layout") or {}; rooms = proj.get("rooms") or []; ot = proj.get("object_type", "")
    st2 = 8 + len(ly.get("floors_plan") or [])
    story = [Spacer(1,20*mm), p("ТЕХНИЧЕСКИЙ ПАСПОРТ ПРОЕКТА",ts), p(ot,ts),
             p(f"{proj.get('area','')} | {proj.get('floors','')} | Регион: {proj.get('location','')}",ps2),
             p("ГОСТ 21.501-2018, СП, СНиП, ПУЭ 7",ss), PageBreak(),
             p("Раздел 1. Пояснительная записка",hs), p(proj.get("summary","-")),
             tbl([["Параметр","Значение"],["Тип",ot],["Регион",proj.get("location","-")],["Площадь",proj.get("area","-")],["Этажность",proj.get("floors","-")],["Бюджет",proj.get("budget","-")]], [70*mm,110*mm])]
    for fi2, fl in enumerate(ly.get("floors_plan") or []):
        story.append(PageBreak()); story.append(p(f"Раздел 2.{fi2+1}. {fl.get('floor',f'Этаж {fi2+1}')}",hs))
        fr = fl.get("rooms") or []
        merged = [{**r2,**(next((x for x in rooms if x.get("name","").lower()==r2.get("name","").lower()),{}))} for r2 in fr]
        fig_fp = make_floor_plan(fl.get("floor",f"Этаж {fi2+1}"),merged or fr,ot,proj,fi2+1,st2)
        if fig_fp: story.append(fi(fig_fp)); plt.close(fig_fp)
    for sec_i,(name,func) in enumerate([("Разрез А-А",lambda p3,sn3,st3:make_section(p3,sn3,st3)),("Фасад 1-1",lambda p3,sn3,st3:make_facade(p3,"Фасад 1-1",sn3,st3)),("Генплан",lambda p3,sn3,st3:make_genplan(p3,ot,sn3,st3)),("Вода/Канализация",lambda p3,sn3,st3:make_utility(p3,sn3,st3)),("Вентиляция",lambda p3,sn3,st3:make_vent(p3,sn3,st3)),("Электрика",lambda p3,sn3,st3:make_elec(p3,sn3,st3)),("Пожарная",lambda p3,sn3,st3:make_fire(p3,sn3,st3))],start=2):
        story.append(PageBreak()); story.append(p(f"Раздел {sec_i+1}. {name}",hs))
        fig_s = func(proj, sec_i+1, st2)
        if fig_s: story.append(fi(fig_s)); plt.close(fig_s)
    doc.build(story); return buf.getvalue()

def gen_html(proj):
    ot = proj.get("object_type","")
    rooms_html = "".join(f"<tr><td>{safe_text(r.get('name'))}</td><td>{safe_text(r.get('area'))}</td><td>{safe_text(r.get('floor'))}</td></tr>" for r in (proj.get("rooms") or []))
    html = f"""<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>Паспорт — {safe_text(ot)}</title>
<style>body{{font-family:Arial,sans-serif;max-width:960px;margin:0 auto;padding:20px}}h1{{color:#263238}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccc;padding:6px 10px;text-align:left}}th{{background:#e8eaf6}}</style></head>
<body><h1>Технический паспорт</h1><p><b>Тип:</b> {safe_text(ot)}</p><p><b>Площадь:</b> {safe_text(proj.get('area'))} | <b>Этажность:</b> {safe_text(proj.get('floors'))} | <b>Регион:</b> {safe_text(proj.get('location'))}</p>
<h2>Пояснительная записка</h2><p>{safe_text(proj.get('summary'))}</p>
<h2>Помещения</h2><table><tr><th>Помещение</th><th>Площадь</th><th>Этаж</th></tr>{rooms_html}</table>
<h3>Итого: {safe_text(proj.get('total_cost'))}</h3><p><small>ГОСТ 21.501-2018, СП, СНиП, ПУЭ 7</small></p></body></html>"""
    return html.encode("utf-8")


# ═══════════════ УЛУЧШЕННАЯ РАСКЛАДКА КОМНАТ ═══════════════
def layout_rooms_grid(rooms):
    if not rooms: return [], 12, 8, 0, 0
    CORR_W = 1.5
    all_rooms = list(rooms)
    n = len(all_rooms)
    cols = max(2, (n + 1) // 2)
    total_area = sum(parse_area(r.get("area")) for r in all_rooms)
    W = max(10, min(16, total_area / 6))
    H = max(7, total_area / W * 1.3)
    room_h = (H - CORR_W) / 2
    top_rooms = all_rooms[:cols]
    top_w = W / max(1, len(top_rooms))
    bot_rooms = all_rooms[cols:] if cols < n else []
    bot_w = W / max(1, len(bot_rooms)) if bot_rooms else W
    placed = []; idx = 1
    for i, r in enumerate(top_rooms):
        rw = top_w; ry = CORR_W + room_h
        placed.append({"room":r,"x":i*top_w,"y":ry,"w":rw,"h":room_h,"cx":i*top_w+rw/2,"cy":ry+room_h/2,"idx":idx,"row":"top"}); idx+=1
    for i, r in enumerate(bot_rooms):
        rw = bot_w; ry = 0
        placed.append({"room":r,"x":i*bot_w,"y":ry,"w":rw,"h":room_h,"cx":i*bot_w+rw/2,"cy":ry+room_h/2,"idx":idx,"row":"bot"}); idx+=1
    corr_room = {"name":"Коридор","area":f"{W*CORR_W:.1f} м2","floor":"1 этаж"}
    placed.append({"room":corr_room,"x":0,"y":room_h,"w":W,"h":CORR_W,"cx":W/2,"cy":room_h+CORR_W/2,"idx":idx,"row":"corr"})
    return placed, W, H, room_h, CORR_W


def draw_base_plan(ax, placed, HW, HH, corr_y, corr_h, show_wet_dots=True):
    for rect in [(-WALL_OUTER,-WALL_OUTER,HW+2*WALL_OUTER,WALL_OUTER),(-WALL_OUTER,HH,HW+2*WALL_OUTER,WALL_OUTER),(-WALL_OUTER,0,WALL_OUTER,HH),(HW,0,WALL_OUTER,HH)]:
        x,y,w,h = rect
        ax.add_patch(patches.Rectangle((x,y),w,h,facecolor="#FFF8E1",edgecolor="#000",lw=LW_CONTOUR,zorder=5))
        hatch_brick(ax,x,y,w,h)
    for pl in placed:
        name = pl["room"].get("name",""); nl = name.lower(); c = get_rc(name)
        ax.add_patch(patches.Rectangle((pl["x"],pl["y"]),pl["w"],pl["h"],facecolor=c,alpha=0.38,edgecolor="#90A4AE",lw=0.7,zorder=3))
        if show_wet_dots and is_wet(name):
            for xi in np.arange(pl["x"]+0.25,pl["x"]+pl["w"],0.32):
                for yi in np.arange(pl["y"]+0.25,pl["y"]+pl["h"],0.32):
                    ax.plot(xi,yi,marker=".",markersize=1.8,color="#0288D1",alpha=0.6,zorder=3)
        cx,cy = pl["cx"],pl["cy"]
        ax.add_patch(plt.Circle((cx,cy+0.45),0.24,facecolor="white",edgecolor="#000",lw=0.8,zorder=15))
        ax.text(cx,cy+0.45,f"{pl['idx']}",ha="center",va="center",fontsize=7.5,fontweight="bold",color="#000",zorder=16)
        if nl!="коридор": ax.text(cx,cy,name,ha="center",va="center",fontsize=7.5,color="#000",fontweight="bold",zorder=15)
        a = parse_area(pl["room"].get("area"))
        ax.text(cx,cy-0.40,f"{a:.1f} м²",ha="center",va="center",fontsize=8.5,fontweight="bold",color="#000",zorder=15)
        ax.plot([cx-0.50,cx+0.50],[cy-0.62,cy-0.62],color="#000",lw=0.7,zorder=15)
        min_a = get_rtype(name)["min_area"]
        ax.text(cx,cy-0.85,f"min {min_a} м²",ha="center",fontsize=5,color="#1B5E20" if a>=min_a else "#B71C1C",zorder=15)
    for pl in placed:
        if pl["x"]>0.02: ax.add_patch(patches.Rectangle((pl["x"]-WALL_INNER/2,pl["y"]),WALL_INNER,pl["h"],facecolor="#ECEFF1",edgecolor="#000",lw=LW_CONTOUR*0.8,zorder=5))
        if pl["y"]>0.02 and pl.get("row")!="corr": ax.add_patch(patches.Rectangle((pl["x"],pl["y"]-WALL_PART/2),pl["w"],WALL_PART,facecolor="#ECEFF1",edgecolor="#000",lw=LW_CONTOUR*0.7,zorder=5))
        if pl["y"]+pl["h"]<HH-0.02 and pl.get("row")!="corr": ax.add_patch(patches.Rectangle((pl["x"],pl["y"]+pl["h"]-WALL_PART/2),pl["w"],WALL_PART,facecolor="#ECEFF1",edgecolor="#000",lw=LW_CONTOUR*0.7,zorder=5))
    for pl in placed:
        if pl.get("row") in ("corr","top"): continue
        if pl["room"].get("name","").lower()=="коридор": continue
        door_x = pl["x"]+pl["w"]/2-DOOR_INT/2
        draw_door(ax,door_x,pl["y"]+pl["h"],DOOR_INT,"up_right")
    for pl in placed:
        if pl.get("row")!="top": continue
        if pl["room"].get("name","").lower()=="коридор": continue
        door_x = pl["x"]+pl["w"]/2-DOOR_INT/2
        draw_door(ax,door_x,pl["y"],DOOR_INT,"down_right")
    ent_x = HW/2-DOOR_ENT/2
    draw_door(ax,ent_x,0,DOOR_ENT,"up_right")
    for pl in placed:
        name = pl["room"].get("name","")
        if is_no_win(name): continue
        if pl.get("row")=="top":
            wx = pl["x"]+pl["w"]/2-WIN_STD/2
            draw_window(ax,wx,HH,"h",WIN_STD)
        elif pl.get("row")=="bot":
            wx = pl["x"]+pl["w"]/2-WIN_STD/2
            draw_window(ax,wx,0,"h",WIN_STD)
        if pl["x"]<=0.02:
            wy = pl["y"]+pl["h"]/2-WIN_STD/2
            draw_window(ax,0,wy,"v",WIN_STD)
        if pl["x"]+pl["w"]>=HW-0.02:
            wy = pl["y"]+pl["h"]/2-WIN_STD/2
            draw_window(ax,HW,wy,"v",WIN_STD)
    ax.annotate("ВХОД",xy=(HW/2,-0.5),xytext=(HW/2,-1.8),fontsize=9,ha="center",color="#1B5E20",fontweight="bold",arrowprops=dict(arrowstyle="->",color="#1B5E20",lw=2.0),zorder=15)


def draw_plan_axes_and_dims(ax, placed, HW, HH):
    raw_x = sorted({0.0}|{p["x"] for p in placed}|{p["x"]+p["w"] for p in placed}|{HW})
    axu = [raw_x[0]]
    for xv in raw_x[1:]:
        if xv-axu[-1]>0.4: axu.append(xv)
    raw_y = sorted({0.0}|{p["y"] for p in placed}|{p["y"]+p["h"] for p in placed}|{HH})
    ayu = [raw_y[0]]
    for yv in raw_y[1:]:
        if yv-ayu[-1]>0.4: ayu.append(yv)
    for xv in axu: draw_axis(ax,xv,-2.2,xv,HH+2.8)
    for yv in ayu: draw_axis(ax,-2.2,yv,HW+2.8,yv)
    letters = "АБВГДЕЖИКЛМН"
    for i,xv in enumerate(axu): draw_am(ax,xv,-2.5,i+1); draw_am(ax,xv,HH+3.0,i+1)
    for i,yv in enumerate(ayu): lb=letters[i] if i<len(letters) else str(i+1); draw_am(ax,-2.5,yv,lb); draw_am(ax,HW+3.0,yv,lb)
    draw_dim(ax,axu,-3.5,"h"); draw_dim(ax,ayu,HW+4.2,"v")


def get_wet_rooms(placed):
    return [p for p in placed if p["room"].get("name","").lower()!="коридор" and is_wet(p["room"].get("name",""))]


# ═══════════════ ЧЕРТЁЖ: КОММУНИКАЦИИ ═══════════════
def make_utility(proj, sn=5, st2=8):
    rooms = proj.get("rooms") or []
    if not rooms: return None
    placed,HW,HH,corr_y,corr_h = layout_rooms_grid(rooms)
    fig,ax = plt.subplots(figsize=(22,16))
    XL,XR,YB,YT = -5.5,HW+10,-7.0,HH+5.0
    ax.set_xlim(XL,XR); ax.set_ylim(YB,YT); ax.set_aspect("equal"); ax.axis("off")
    ax.set_facecolor("white"); fig.patch.set_facecolor("white")
    draw_frame(ax,XL,XR,YB,YT); draw_stamp(ax,XR,YB,proj,"План водоснабжения и канализации (СП 30.13330.2020)",sn,st2)
    draw_base_plan(ax,placed,HW,HH,corr_y,corr_h); draw_plan_axes_and_dims(ax,placed,HW,HH)
    wet = get_wet_rooms(placed)
    sx,sy = (wet[0]["cx"],wet[0]["cy"]) if wet else (HW-1.5,corr_y+corr_h/2)
    ax.add_patch(plt.Circle((sx-0.45,sy),0.22,facecolor="#29B6F6",edgecolor="#01579B",lw=1.5,zorder=10))
    ax.text(sx-0.45,sy,"ХВС",ha="center",va="center",fontsize=5,color="white",fontweight="bold",zorder=11)
    ax.text(sx-0.45,sy-0.42,"Ст1\nØ25",ha="center",fontsize=5.5,color="#01579B",fontweight="bold",zorder=11)
    ax.add_patch(plt.Circle((sx-0.05,sy),0.22,facecolor="#EF5350",edgecolor="#B71C1C",lw=1.5,zorder=10))
    ax.text(sx-0.05,sy,"ГВС",ha="center",va="center",fontsize=5,color="white",fontweight="bold",zorder=11)
    ax.add_patch(plt.Circle((sx+0.50,sy),0.26,facecolor="#8D6E63",edgecolor="#3E2723",lw=1.5,zorder=10))
    ax.text(sx+0.50,sy,"К",ha="center",va="center",fontsize=6,color="white",fontweight="bold",zorder=11)
    ax.text(sx+0.50,sy-0.48,"Ст3\nØ110",ha="center",fontsize=5.5,color="#3E2723",fontweight="bold",zorder=11)
    ax.add_patch(patches.Rectangle((sx+0.38,sy+0.38),0.25,0.18,facecolor="#BCAAA4",edgecolor="#3E2723",lw=0.8,zorder=10))
    ax.text(sx+0.50,sy+0.47,"Рев.",ha="center",fontsize=4.5,color="#3E2723",fontweight="bold",zorder=11)
    for pl in placed:
        nl=pl["room"].get("name","").lower(); cx,cy=pl["cx"],pl["cy"]
        if "санузел" in nl or "туалет" in nl: sym_wc(ax,cx-0.35,cy+0.20); sym_sink(ax,cx+0.35,cy+0.20)
        elif "ванная" in nl: sym_bath(ax,cx-0.40,cy+0.30); sym_sink(ax,cx+0.40,cy+0.30); sym_wc(ax,cx+0.40,cy-0.35)
        elif "кухня" in nl: sym_sink(ax,cx-0.30,cy+0.20); sym_stove(ax,cx+0.45,cy+0.20)
    for pl in placed:
        nl=pl["room"].get("name","").lower()
        if not(is_wet(pl["room"].get("name","")) or "кухня" in nl): continue
        cx,cy=pl["cx"],pl["cy"]
        ax.plot([sx-0.45,cx],[sy,cy],color="#29B6F6",lw=1.8,linestyle="-",zorder=6)
        mx,my=(sx-0.45+cx)/2,(sy+cy)/2
        ax.text(mx,my+0.15,"Ø20 PPR",fontsize=5.5,color="#01579B",ha="center",bbox=dict(facecolor="white",alpha=0.9,edgecolor="none",pad=1),zorder=7)
        if is_wet(pl["room"].get("name","")):
            ax.plot([sx-0.05,cx],[sy,cy],color="#EF5350",lw=1.8,linestyle="-",zorder=6)
            ax.text(mx+0.35,my-0.15,"Ø20 PPR",fontsize=5.5,color="#B71C1C",ha="center",bbox=dict(facecolor="white",alpha=0.9,edgecolor="none",pad=1),zorder=7)
        ax.plot([sx+0.50,cx],[sy,cy],color="#8D6E63",lw=2.0,linestyle="-.",zorder=6)
        ax.text(mx-0.35,my-0.30,"Ø50 i=0.03",fontsize=5,color="#3E2723",ha="center",bbox=dict(facecolor="white",alpha=0.9,edgecolor="none",pad=1),zorder=7)
    ax.annotate("Ввод ХВС\nØ32 h=-1.80 м",xy=(sx-0.45,0),xytext=(sx-0.45,-2.5),fontsize=6.5,ha="center",color="#0288D1",fontweight="bold",arrowprops=dict(arrowstyle="->",color="#0288D1",lw=1.8),zorder=12)
    ax.annotate("Выпуск К\nØ110 i=0.02",xy=(sx+0.50,0),xytext=(sx+0.50,-2.5),fontsize=6.5,ha="center",color="#795548",fontweight="bold",arrowprops=dict(arrowstyle="->",color="#795548",lw=1.8),zorder=12)
    draw_level(ax,sx-0.45,-1.80,"-1.800")
    lg_x,lg_y=HW+1.2,HH-0.3
    ax.add_patch(patches.Rectangle((lg_x-0.3,lg_y-5.0),8.0,5.6,facecolor="#FAFAFA",edgecolor="#000",lw=0.8,zorder=14))
    ax.text(lg_x+3.7,lg_y+0.4,"Условные обозначения (СП 30.13330.2020)",fontsize=8,fontweight="bold",color="#000",ha="center",zorder=15)
    for i,(lbl,clr,ls) in enumerate([("ХВС Ø20-32 PPR","#29B6F6","-"),("ГВС Ø20-32 PPR","#EF5350","-"),("Канализация Ø50-110 ПВХ","#8D6E63","-.")]):
        y=lg_y-0.4-i*0.50; ax.plot([lg_x,lg_x+0.8],[y,y],color=clr,lw=2.5,linestyle=ls,zorder=15); ax.text(lg_x+1.0,y,lbl,fontsize=7,va="center",zorder=15)
    notes=["Примечания:","1. Трубы ХВС/ГВС — PPR PN20 (ГОСТ Р 52134).","2. Трубы К — ПВХ Ø50/110 (ГОСТ 32414).","3. Уклон К ≥ i=0.02.","4. Ввод ХВС ниже промерзания.","5. Ревизии на каждом повороте."]
    ny=-5.5
    for note in notes: ax.text(0,ny,note,fontsize=6.5,color="#37474F",va="top",zorder=15,style="italic" if note.startswith("П") else "normal"); ny-=0.38
    plt.tight_layout(); return fig


# ═══════════════ ЧЕРТЁЖ: ЭЛЕКТРИКА ═══════════════
def make_elec(proj, sn=7, st2=8):
    rooms = proj.get("rooms") or []
    if not rooms: return None
    placed,HW,HH,corr_y,corr_h = layout_rooms_grid(rooms)
    fig,ax = plt.subplots(figsize=(22,16))
    XL,XR,YB,YT = -5.5,HW+10,-7.0,HH+5.0
    ax.set_xlim(XL,XR); ax.set_ylim(YB,YT); ax.set_aspect("equal"); ax.axis("off")
    ax.set_facecolor("white"); fig.patch.set_facecolor("white")
    draw_frame(ax,XL,XR,YB,YT); draw_stamp(ax,XR,YB,proj,"План электроснабжения (ПУЭ 7, ГОСТ 21.614-88)",sn,st2)
    draw_base_plan(ax,placed,HW,HH,corr_y,corr_h); draw_plan_axes_and_dims(ax,placed,HW,HH)
    bx,by=0.65,0.65; sym_board(ax,bx,by,"ВРУ")
    ax.text(bx,by-0.55,"h=1.8 м\n15 кВт\n3ф, 380В",ha="center",fontsize=5.5,color="#37474F",fontweight="bold",zorder=15)
    ax.annotate("СИП 4×16 мм²",xy=(bx,by+0.25),xytext=(bx-2.2,by+2.5),fontsize=6.5,color="#E65100",fontweight="bold",arrowprops=dict(arrowstyle="->",color="#E65100",lw=1.8),zorder=15)
    group_num=1
    for pl in placed:
        nl=pl["room"].get("name","").lower(); cx,cy=pl["cx"],pl["cy"]
        x0,y0,w,h=pl["x"],pl["y"],pl["w"],pl["h"]; wet=is_wet(pl["room"].get("name","")); is_corr=nl=="коридор"
        if not is_corr:
            if wet: sym_lamp(ax,cx,cy+0.3,"wet")
            elif "кухня" in nl: sym_lamp_led(ax,cx,cy+0.3)
            else: sym_lamp(ax,cx,cy+0.3)
        if not is_corr:
            sw_x=x0+w/2+DOOR_INT/2+0.20; sw_y=y0+h
            cnt=2 if("гостиная" in nl or "спальня" in nl) else 1
            sym_switch(ax,sw_x,sw_y-0.25,cnt)
        if not is_corr: ax.text(cx,cy-0.55,f"Гр.{group_num}",fontsize=5.5,ha="center",color="#263238",fontweight="bold",zorder=13); group_num+=1
        if "гостиная" in nl or "спальня" in nl or "кабинет" in nl:
            for i in range(min(4,max(2,int(w)))):
                sx2=x0+0.35+i*(w-0.70)/max(1,min(3,int(w)-1)); sym_socket(ax,sx2,y0+0.30,2)
        elif "кухня" in nl:
            for i in range(3): sx2=x0+0.35+i*(w-0.70)/3; sym_socket(ax,sx2,y0+0.30,2)
            sym_socket_pow(ax,x0+w-0.40,y0+h-0.40); ax.text(x0+w-0.40,y0+h-0.62,"Плита 4кВт\nh=1.0",fontsize=4.5,ha="center",color="#E65100",zorder=14)
        elif "прихожая" in nl or "коридор" in nl:
            if w>3: sym_socket(ax,x0+0.40,y0+0.30); sym_socket(ax,x0+w-0.40,y0+0.30)
            else: sym_socket(ax,cx,y0+0.30)
        elif wet:
            sym_socket_pow(ax,x0+0.38,y0+0.40); ax.text(x0+0.38,y0+0.15,"УЗО 30мА\nh=1.2",fontsize=4.5,ha="center",color="#E65100",fontweight="bold",zorder=14)
        if not is_corr: ax.plot([bx,cx],[by,cy],color="#FDD835",lw=0.7,linestyle="--",alpha=0.55,zorder=4)
    lg_x,lg_y=HW+1.2,HH-0.3
    ax.add_patch(patches.Rectangle((lg_x-0.3,lg_y-6.0),8.5,6.6,facecolor="#FAFAFA",edgecolor="#000",lw=0.8,zorder=14))
    ax.text(lg_x+3.95,lg_y+0.4,"Условные обозначения (ГОСТ 21.614-88)",fontsize=8,fontweight="bold",color="#000",ha="center",zorder=15)
    for i,(lb,k) in enumerate([("Розетка h=0.30 м","socket"),("Розетка силовая h=1.0 м","socket_pow"),("Выключатель 1кл. h=0.9 м","switch"),("Выключатель 2кл. h=0.9 м","switch2"),("Светильник потолочный","lamp"),("LED-панель","led"),("ВРУ h=1.80 м","board")]):
        y2=lg_y-0.5-i*0.72; sx2=lg_x+0.15
        if k=="socket": sym_socket(ax,sx2,y2)
        elif k=="socket_pow": sym_socket_pow(ax,sx2,y2)
        elif k=="switch": sym_switch(ax,sx2,y2,1)
        elif k=="switch2": sym_switch(ax,sx2,y2,2)
        elif k=="lamp": sym_lamp(ax,sx2,y2)
        elif k=="led": sym_lamp_led(ax,sx2,y2)
        elif k=="board": sym_board(ax,sx2,y2,"ЩЭ")
        ax.text(lg_x+0.65,y2,lb,fontsize=7,va="center",color="#000",zorder=15)
    notes_e=["Примечания:","1. Кабели — ВВГнг-LS, Cu.","2. Освещ. — 1.5мм², розетки — 2.5мм².","3. Мокрые зоны — УЗО 30мА.","4. Плита — 3×4мм², АВ 25А.","5. Заземление — TN-C-S."]
    ny=-5.5
    for note in notes_e: ax.text(0,ny,note,fontsize=6.5,color="#37474F",va="top",zorder=15); ny-=0.38
    plt.tight_layout(); return fig


# ═══════════════ ЧЕРТЁЖ: ВЕНТИЛЯЦИЯ ═══════════════
def make_vent(proj, sn=6, st2=8):
    rooms = proj.get("rooms") or []
    if not rooms: return None
    placed,HW,HH,corr_y,corr_h = layout_rooms_grid(rooms)
    fig,ax = plt.subplots(figsize=(22,16))
    XL,XR,YB,YT = -5.5,HW+10,-7.0,HH+5.0
    ax.set_xlim(XL,XR); ax.set_ylim(YB,YT); ax.set_aspect("equal"); ax.axis("off")
    ax.set_facecolor("white"); fig.patch.set_facecolor("white")
    draw_frame(ax,XL,XR,YB,YT); draw_stamp(ax,XR,YB,proj,"План вентиляции (СП 60.13330.2020)",sn,st2)
    draw_base_plan(ax,placed,HW,HH,corr_y,corr_h); draw_plan_axes_and_dims(ax,placed,HW,HH)
    for pl in placed:
        nl=pl["room"].get("name","").lower(); is_ex=any(x in nl for x in["санузел","ванная","туалет","кухня","котельная"])
        is_corr=nl=="коридор"; area=parse_area(pl["room"].get("area")); cx,cy=pl["cx"],pl["cy"]
        if is_ex:
            base=90 if "кухня" in nl else 50 if "ванная" in nl else 25
            sym_vent_ext(ax,cx,cy-0.30,"Ø100"); ax.text(cx,cy-0.68,f"В  L={base} м³/ч",ha="center",fontsize=6.5,color="#BF360C",fontweight="bold",zorder=4)
        elif not is_corr:
            l_h=max(30,int(area*3*2.7)); sym_vent_sup(ax,cx,cy-0.30,"Ø125"); ax.text(cx,cy-0.68,f"П  L={l_h} м³/ч",ha="center",fontsize=6.5,color="#01579B",fontweight="bold",zorder=4)
    pvu_x,pvu_y=HW/2,HH+2.2
    ax.add_patch(patches.FancyBboxPatch((pvu_x-1.8,pvu_y-0.55),3.6,1.1,boxstyle="round,pad=0.06",facecolor="#1565C0",edgecolor="#000",lw=1.8,zorder=8))
    ax.text(pvu_x,pvu_y+0.18,"ПВУ  L=500 м³/ч  N=0.5 кВт",ha="center",fontsize=7,color="white",fontweight="bold",zorder=9)
    ax.text(pvu_x,pvu_y-0.22,"рекуператор КПД≥70%  фильтр F7",ha="center",fontsize=6,color="#B3E5FC",zorder=9)
    ax.add_patch(patches.Rectangle((0.5,HH+0.55),HW-1.0,0.35,facecolor="#90CAF9",edgecolor="#1565C0",lw=1.5,zorder=5))
    ax.text(HW/2,HH+0.73,"Магистральный канал 200×100 мм",ha="center",fontsize=7,color="#0D47A1",fontweight="bold",zorder=6)
    lg_x,lg_y=HW+1.2,HH-0.3
    ax.add_patch(patches.Rectangle((lg_x-0.3,lg_y-3.8),8.0,4.4,facecolor="#FAFAFA",edgecolor="#000",lw=0.8,zorder=14))
    ax.text(lg_x+3.7,lg_y+0.4,"Условные обозначения (СП 60.13330.2020)",fontsize=8,fontweight="bold",color="#000",ha="center",zorder=15)
    sym_vent_sup(ax,lg_x+0.25,lg_y-0.50,"Ø125"); ax.text(lg_x+0.90,lg_y-0.50,"Приточная решётка П",fontsize=7,va="center",zorder=15)
    sym_vent_ext(ax,lg_x+0.25,lg_y-1.20,"Ø100"); ax.text(lg_x+0.90,lg_y-1.20,"Вытяжная решётка В",fontsize=7,va="center",zorder=15)
    notes_v=["Примечания:","1. Воздухообмен по СП 60.13330.2020.","2. Кратность ≥ 3 об/ч (жил.), 10 об/ч (кухня).","3. ПВУ с рекуператором КПД ≥ 70%.","4. Воздуховоды — оцинк. сталь δ=0.5мм."]
    ny=-5.5
    for note in notes_v: ax.text(0,ny,note,fontsize=6.5,color="#37474F",va="top",zorder=15); ny-=0.38
    plt.tight_layout(); return fig


# ═══════════════ ЧЕРТЁЖ: ПОЖАРНАЯ ═══════════════
def make_fire(proj, sn=8, st2=8):
    rooms = proj.get("rooms") or []
    if not rooms: return None
    placed,HW,HH,corr_y,corr_h = layout_rooms_grid(rooms)
    fig,ax = plt.subplots(figsize=(22,16))
    XL,XR,YB,YT = -5.5,HW+10,-7.0,HH+5.0
    ax.set_xlim(XL,XR); ax.set_ylim(YB,YT); ax.set_aspect("equal"); ax.axis("off")
    ax.set_facecolor("white"); fig.patch.set_facecolor("white")
    draw_frame(ax,XL,XR,YB,YT); draw_stamp(ax,XR,YB,proj,"План пожарной сигнализации (СП 484.1311500.2020)",sn,st2)
    draw_base_plan(ax,placed,HW,HH,corr_y,corr_h); draw_plan_axes_and_dims(ax,placed,HW,HH)
    for pl in placed:
        nl=pl["room"].get("name","").lower(); is_corr=nl=="коридор"
        is_path=any(x in nl for x in["прихожая","коридор","лестница","тамбур","холл"])
        x0,y0,w,h=pl["x"],pl["y"],pl["w"],pl["h"]; cx,cy=pl["cx"],pl["cy"]
        if is_corr: continue
        n_det=max(1,int(w/5.0)+1)
        for di in range(n_det):
            dx=x0+(di+0.5)*w/n_det; dy=y0+h-0.35; sym_fire_det(ax,dx,dy)
            if di>0: prev_dx=x0+(di-0.5)*w/n_det; ax.plot([prev_dx,dx],[dy,dy],color="#B71C1C",lw=0.8,zorder=6)
        sym_fire_horn(ax,x0+w-0.30,y0+h-0.30)
        if is_path: sym_fire_manual(ax,x0+0.30,cy); ax.text(x0+0.30,cy-0.28,"h=1.5 м",fontsize=4.5,ha="center",color="#B71C1C",zorder=13)
        a=parse_area(pl["room"].get("area"))
        if a>=25: sym_fire_ext(ax,x0+0.35,y0+0.35)
    ppk_x,ppk_y=HW+0.8,HH/2
    ax.add_patch(patches.FancyBboxPatch((ppk_x,ppk_y-0.65),1.6,2.0,boxstyle="round,pad=0.06",facecolor="#B71C1C",edgecolor="#000",lw=1.8,zorder=8))
    ax.text(ppk_x+0.8,ppk_y+0.72,"ППКП\nСигнал-20П",ha="center",fontsize=7,color="white",fontweight="bold",zorder=9)
    ax.text(ppk_x+0.8,ppk_y-0.38,"→ пульт 01\n(GSM)",ha="center",fontsize=5,color="#FFCDD2",zorder=9)
    for pl in placed:
        if pl["room"].get("name","").lower()=="коридор": continue
        ax.plot([ppk_x,pl["cx"]],[ppk_y,pl["y"]+pl["h"]-0.35],color="#B71C1C",lw=0.5,linestyle=":",alpha=0.50,zorder=4)
    ax.annotate("ЭВАКУАЦИЯ →",xy=(HW/2,0),xytext=(HW/2,-2.2),fontsize=10,ha="center",color="#1B5E20",fontweight="bold",arrowprops=dict(arrowstyle="->",color="#1B5E20",lw=3),zorder=10)
    sym_exit_sign(ax,HW/2,0.20)
    lg_x,lg_y=HW+2.8,HH-0.3
    ax.add_patch(patches.Rectangle((lg_x-0.3,lg_y-5.2),7.5,5.8,facecolor="#FAFAFA",edgecolor="#000",lw=0.8,zorder=14))
    ax.text(lg_x+3.4,lg_y+0.35,"Условные обозначения\n(СП 484.1311500.2020)",fontsize=8,fontweight="bold",color="#000",ha="center",zorder=15)
    for i,(lb,k) in enumerate([("ДИ — дымовой ИП-212","det"),("ИПР — ручной h=1.5 м","man"),("ОП — оповещатель","horn"),("ОП-5 — огнетушитель","ext"),("Знак ВЫХОД","exit")]):
        y2=lg_y-0.5-i*0.80
        if k=="det": sym_fire_det(ax,lg_x+0.15,y2)
        elif k=="man": sym_fire_manual(ax,lg_x+0.15,y2)
        elif k=="horn": sym_fire_horn(ax,lg_x+0.15,y2)
        elif k=="ext": sym_fire_ext(ax,lg_x+0.15,y2)
        elif k=="exit": sym_exit_sign(ax,lg_x+0.15,y2)
        ax.text(lg_x+0.60,y2,lb,fontsize=7,va="center",color="#000",zorder=15)
    notes_f=["Примечания:","1. Шаг ДИ ≤ 9 м.","2. Оповещение — 3-я зона.","3. Огнетушители — 1 шт/50м².","4. ППКП → пульт 01 (GSM).","5. Огнестойкость — II."]
    ny=-5.5
    for note in notes_f: ax.text(0,ny,note,fontsize=6.5,color="#37474F",va="top",zorder=15); ny-=0.38
    plt.tight_layout(); return fig


# ═══════════════ 3D ═══════════════
def make_3d(proj):
    fig = plt.figure(figsize=(14,10)); ax = fig.add_subplot(111, projection='3d')
    a=parse_area(proj.get("area","120")); fs=str(proj.get("floors","2"))
    fl=1 if "1" in fs else (3 if "3" in fs else 2); ot=proj.get("object_type","")
    if any(x in ot.lower() for x in["многоквартирный","офис","гостиница"]): ns=re.findall(r'\d+',fs); fl=int(ns[0]) if ns else 5
    W=14.0; L=max(9.0,a/W/fl); H=fl*3.2
    gx=np.linspace(-6,W+6,10); gy=np.linspace(-6,L+6,10); GX,GY=np.meshgrid(gx,gy)
    ax.plot_surface(GX,GY,np.zeros_like(GX),color='#81C784',alpha=0.45)
    ax.bar3d(-0.5,-0.5,-0.7,W+1,L+1,0.7,facecolor='#78909C',edgecolor='#37474F',alpha=0.95)
    ax.bar3d(0,0,0,W,L,H,facecolor='#F5F5F5',edgecolor='#455A64',alpha=0.92,linewidth=1.5)
    for f in range(1,fl): ax.bar3d(-0.1,-0.1,f*3.2-0.12,W+0.2,L+0.2,0.22,facecolor='#37474F',edgecolor='#000',alpha=0.9)
    if any(x in ot.lower() for x in["частный дом","баня","гараж"]):
        rh=2.8; ov=0.8; p_v=[[-ov,-ov,H],[W+ov,-ov,H],[W+ov,L+ov,H],[-ov,L+ov,H],[W/2,-ov,H+rh],[W/2,L+ov,H+rh]]
        rf=[[p_v[0],p_v[1],p_v[4]],[p_v[2],p_v[3],p_v[5]],[p_v[0],p_v[3],p_v[5],p_v[4]],[p_v[1],p_v[2],p_v[5],p_v[4]]]
        ax.add_collection3d(Poly3DCollection(rf,facecolor='#B71C1C',edgecolor='#212121',lw=1.2,alpha=0.95))
        ax.bar3d(W*0.68,L*0.5,H+0.5,0.45,0.45,1.2,facecolor='#8D6E63',edgecolor='#3E2723',alpha=1)
    else: ax.bar3d(-0.3,-0.3,H,W+0.6,L+0.6,0.55,facecolor='#37474F',edgecolor='#000',alpha=1)
    for f in range(fl):
        zb=f*3.2+0.9
        for wx in np.linspace(1.8,W-2.8,max(2,int(W/3))): ax.bar3d(wx,-0.08,zb,1.2,0.08,1.6,facecolor='#B3E5FC',edgecolor='#01579B',lw=1.2,alpha=0.9)
        for wy in np.linspace(1.8,L-2.8,max(2,int(L/3))): ax.bar3d(W,wy,zb,0.08,1.4,1.6,facecolor='#B3E5FC',edgecolor='#01579B',lw=1.2,alpha=0.9)
    dw=1.4; dx=W/2-dw/2
    ax.bar3d(dx,-0.12,0,dw,0.12,2.3,facecolor='#4E342E',edgecolor='#271C19',alpha=1)
    ax.bar3d(dx-0.4,-0.8,2.3,dw+0.8,0.8,0.15,facecolor='#B0BEC5',edgecolor='#37474F',alpha=1)
    for tx,ty in[(-3,-3),(-3,L/2),(-3,L+3),(W+3,-3),(W+3,L/2),(W+3,L+3)]:
        ax.bar3d(tx-0.2,ty-0.2,0,0.4,0.4,2.5,facecolor='#5D4037',edgecolor='#3E2723')
        for tzi,tri in zip(np.linspace(2.2,5.0,4),np.linspace(1.5,0.3,4)):
            u=np.linspace(0,2*np.pi,10); v=np.linspace(0,np.pi/2,5); UU,VV=np.meshgrid(u,v)
            ax.plot_surface(tx+tri*np.cos(UU)*np.sin(VV),ty+tri*np.sin(UU)*np.sin(VV),tzi+tri*np.cos(VV)*0.8,color='#388E3C',alpha=0.65)
    ax.set_xlabel('Ширина (м)',fontsize=9,color='#37474F'); ax.set_ylabel('Длина (м)',fontsize=9,color='#37474F'); ax.set_zlabel('Высота (м)',fontsize=9,color='#37474F')
    ax.set_xlim(-6,W+6); ax.set_ylim(-6,L+6); ax.set_zlim(-1,H+5)
    ax.set_title(f"3D: {safe_text(ot)}\nПлощадь: {safe_text(proj.get('area'))} | Этажей: {fl} | H={H:.1f}м",fontsize=12,fontweight='bold',pad=18,color="#263238")
    ax.view_init(elev=28,azim=135); plt.tight_layout(); return fig


# ═══════════════ ДИСПЕТЧЕР ═══════════════
def ask(ut,rn,norms,ot):
    st.session_state.messages.append({"role":"user","content":ut})
    with st.spinner(f"Архитектор проектирует {ot}..."):
        r=get_ai(ut,st.session_state.messages[:-1],rn,norms,ot)
    if not r: st.error("YandexGPT не ответил."); return
    st.session_state.messages.append({"role":"assistant","content":r})
    if looks_like_project(r):
        try: st.session_state.parsed_project=extract_json(r); st.session_state.project_data=r; st.session_state.stage="result"
        except: st.session_state.stage="questions"
    else: st.session_state.stage="questions"
    st.rerun()


# ═══════════════ ИНТЕРФЕЙС ═══════════════
if st.session_state.stage == "select_type":
    st.subheader("Шаг 1. Выбери тип объекта")
    cols = st.columns(3)
    for i,(on,oi) in enumerate(OBJECT_TYPES.items()):
        with cols[i%3]:
            lb = f"{oi['icon']} {on.split(' ',1)[1] if ' ' in on else on}"
            if st.button(lb,use_container_width=True,key=f"ob_{i}"):
                st.session_state.selected_object_type=on; st.session_state.stage="input"; st.rerun()

elif st.session_state.stage == "input":
    ot=st.session_state.selected_object_type; oi=OBJECT_TYPES.get(ot,OBJECT_TYPES["🏠 Частный дом (ИЖС)"])
    st.subheader(f"Шаг 2. Параметры: {ot}")
    rn=st.selectbox("Регион",list(REGIONS.keys()),index=2,key="rs"); st.session_state.selected_region=rn; norms=REGIONS[rn]
    c1,c2,c3=st.columns(3)
    with c1: area=st.text_input("Площадь",value="120 м2",key="ai"); fmax=oi["floors_max"]; fo=[f"{i} эт." for i in range(1,fmax+1)]; floors=st.selectbox("Этажность",fo,index=min(1,len(fo)-1),key="fi")
    with c2: walls=st.selectbox("Стены",["Газобетон","Кирпич","Монолит","Каркас"],key="wi"); res=st.text_input("Пользователи",value="4 жильца",key="ri")
    with c3: budget=st.text_input("Бюджет",value="10 млн руб.",key="bi"); extras=st.multiselect("Дополнительно",["Терраса","Гараж","Баня","Котельная"],default=["Терраса"],key="ei")
    with st.expander("📍 Климатические данные региона"):
        cc1,cc2,cc3=st.columns(3)
        cc1.info(f"🌡️ Климатический район: **{norms['climate_zone']}**"); cc2.info(f"❄️ Промерзание: **{norms['frost_depth']}**"); cc3.info(f"🌨️ Снег: **{norms['snow_load']}**")
        cc1.info(f"💨 Ветер: **{norms['wind_load']}**"); cc2.info(f"🏔️ Сейсмика: **{norms['seismicity']}**"); cc3.info(f"🔥 Термосопрот.: **{norms['thermal_resistance']}**")
    if st.button("Собрать проект",type="primary",use_container_width=True,key="bs"):
        pr=f"Тип:{ot}. Регион:{rn}. Площадь:{area}. Этажность:{floors}. Стены:{walls}. Пользователи:{res}. Бюджет:{budget}. Доп:{','.join(extras) if extras else 'нет'}."
        ask(pr,rn,norms,ot)

elif st.session_state.stage == "questions":
    ot=st.session_state.selected_object_type; st.subheader(f"Шаг 3. Уточни: {ot}")
    rn=st.selectbox("Регион",list(REGIONS.keys()),index=2,key="rq"); norms=REGIONS[rn]
    for m in st.session_state.messages:
        who="🧑 Ты" if m["role"]=="user" else "🤖 Архитектор"
        with st.chat_message(m["role"]): st.write(f"**{who}:** {m['content'][:600]}{'…' if len(m['content'])>600 else ''}")
    ans=st.text_area("Ответ",height=90,key="aq")
    if st.button("Продолжить",type="primary",use_container_width=True,key="bc"):
        if ans.strip(): ask(ans.strip(),rn,norms,ot)
        else: st.warning("Введи ответ.")

elif st.session_state.stage == "result":
    try:
        proj = st.session_state.parsed_project or extract_json(st.session_state.project_data)
        st.session_state.parsed_project = proj
        ot   = proj.get("object_type", st.session_state.selected_object_type)
        ly   = proj.get("layout") or {}
        ar   = proj.get("rooms")  or []

        st.subheader(f"✅ Готовый проект: {ot}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Площадь", proj.get("area","-"))
        c2.metric("Этажи", proj.get("floors","-"))
        c3.metric("Бюджет", proj.get("budget","-"))
        c4.metric("Регион", proj.get("location","-"))

        st.markdown(f"<div class='box'><b>Пояснительная записка</b><br><br>{safe_text(proj.get('summary', '-'))}</div>", unsafe_allow_html=True)

        cn = proj.get("construction") or {}
        if cn:
            st.markdown("### 🏗️ Конструктивные решения")
            cc1, cc2, cc3 = st.columns(3)
            cc1.markdown(f"<div class='box doc'>🏛️ Фундамент<br><b>{safe_text(cn.get('foundation'))}</b></div>", unsafe_allow_html=True)
            cc2.markdown(f"<div class='box doc'>🧱 Стены<br><b>{safe_text(cn.get('walls'))}</b></div>", unsafe_allow_html=True)
            cc3.markdown(f"<div class='box doc'>🏠 Кровля<br><b>{safe_text(cn.get('roof'))}</b></div>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("## 📐 Чертежи и визуализация")

        tabs = st.tabs([
            "🧊 3D","🏠 Планы этажей","✂️ Разрез А-А","🏛️ Фасад","🌍 Генплан",
            "🚿 Вода / Канализация","💨 Вентиляция","⚡ Электрика","🔥 Пожарная"
        ])
        t3d, t1, t2, t3, t4, t5, t6, t7, t8 = tabs
        fp  = ly.get("floors_plan") or []
        st2 = 8 + len(fp)

        with t3d:
            st.markdown("### 🧊 3D-визуализация")
            if 'make_3d' in globals():
                f3d = make_3d(proj)
                if f3d:
                    st.pyplot(f3d, use_container_width=True)
                    st.download_button("⬇️ Скачать 3D PNG", data=fig_to_bytes(f3d), file_name="3d_view.png", mime="image/png", key="d3d")
                    plt.close(f3d)
            else:
                st.info("ℹ️ Функция `make_3d` не загружена.")

        with t1:
            st.markdown("### 🏠 Поэтажные планы")
            if fp:
                for fi2, fl in enumerate(fp):
                    st.markdown(f"#### {fl.get('floor', f'Этаж {fi2+1}')}")
                    fr = fl.get("rooms") or []
                    merged = [{**r2, **(next((x for x in ar if x.get("name","").lower()==r2.get("name","").lower()), {}))} for r2 in fr]
                    fig_fp = make_floor_plan(fl.get("floor", f"Этаж {fi2+1}"), merged or fr, ot, proj, fi2+1, st2)
                    if fig_fp:
                        st.pyplot(fig_fp, use_container_width=True)
                        st.download_button(f"⬇️ План этажа {fi2+1}", data=fig_to_bytes(fig_fp), file_name=f"plan_floor_{fi2+1}.png", mime="image/png", key=f"df_{fi2}")
                        plt.close(fig_fp)
            else:
                st.info("Нет данных по этажам.")

        with t2:
            fs = make_section(proj, 2, st2)
            if fs:
                st.pyplot(fs, use_container_width=True)
                st.download_button("⬇️ Разрез", data=fig_to_bytes(fs), file_name="section_AA.png", mime="image/png", key="dsec")
                plt.close(fs)

        with t3:
            ff = make_facade(proj, "Фасад 1-1", 3, st2)
            if ff:
                st.pyplot(ff, use_container_width=True)
                st.download_button("⬇️ Фасад", data=fig_to_bytes(ff), file_name="facade.png", mime="image/png", key="dfac")
                plt.close(ff)

        with t4:
            fg = make_genplan(proj, ot, 4, st2)
            if fg:
                st.pyplot(fg, use_container_width=True)
                st.download_button("⬇️ Генплан", data=fig_to_bytes(fg), file_name="genplan.png", mime="image/png", key="dgp")
                plt.close(fg)

        with t5:
            fu = make_utility(proj, 5, st2) if 'make_utility' in globals() else None
            if fu:
                st.pyplot(fu, use_container_width=True)
                st.download_button("⬇️ План ВК", data=fig_to_bytes(fu), file_name="water_sewage.png", mime="image/png", key="dut")
                plt.close(fu)
            else:
                st.info("ℹ️ Функция `make_utility` не загружена.")

        with t6:
            fv = make_vent(proj, 6, st2)
            if fv:
                st.pyplot(fv, use_container_width=True)
                st.download_button("⬇️ Вентиляция", data=fig_to_bytes(fv), file_name="ventilation.png", mime="image/png", key="dven")
                plt.close(fv)

        with t7:
            fe = make_elec(proj, 7, st2)
            if fe:
                st.pyplot(fe, use_container_width=True)
                st.download_button("⬇️ Электрика", data=fig_to_bytes(fe), file_name="electrical.png", mime="image/png", key="delec")
                plt.close(fe)

        with t8:
            ff2 = make_fire(proj, 8, st2)
            if ff2:
                st.pyplot(ff2, use_container_width=True)
                st.download_button("⬇️ Пожарная", data=fig_to_bytes(ff2), file_name="fire_alarm.png", mime="image/png", key="dfire")
                plt.close(ff2)

        est = proj.get("estimate") or []
        if est:
            st.markdown("---")
            st.markdown("### 💰 Укрупнённая смета")
            st.table([["Позиция","Стоимость"]] + [[safe_text(e.get("item")), safe_text(e.get("cost"))] for e in est] + [["**ИТОГО**", f"**{safe_text(proj.get('total_cost','-'))}**"]])

        st.markdown("---")
        st.markdown("### 📦 Скачать")
        d1, d2, d3 = st.columns(3)
        with d1:
            try:
                pb = gen_pdf(proj)
                if pb:
                    st.download_button("📄 PDF", data=pb, file_name="project.pdf", mime="application/pdf", use_container_width=True, key="dp")
            except Exception as pdf_e:
                st.warning(f"PDF: {pdf_e}")
        with d2:
            st.download_button("🌐 HTML", data=gen_html(proj), file_name="project.html", mime="text/html", use_container_width=True, key="dh")
        with d3:
            st.download_button("📋 JSON", data=json.dumps(proj, ensure_ascii=False, indent=2).encode("utf-8"), file_name="project.json", mime="application/json", use_container_width=True, key="djson")

        st.markdown("---")
        if st.button("🔄 Новый проект", use_container_width=True, key="bn"):
            reset_project(); st.rerun()

    except Exception as e:
        import traceback
        st.error(f"Ошибка отображения проекта: {e}")
        st.code(traceback.format_exc(), language="python")
        st.info("Попробуйте уточнить запрос или начать заново.")
        if st.button("🔄 Начать заново", key="err_reset"):
            reset_project(); st.rerun()