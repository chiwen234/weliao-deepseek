#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
微聊 · DeepSeek 联机版后端

- 只监听 127.0.0.1
- API Key 只从环境变量读取，绝不下发到前端
- 前端 index.html 通过同源 /api/* 与这里通信
- persona.json 是角色人设；/api/persona/update 可重新蒸馏
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from datetime import datetime, timedelta
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

BASE = Path(__file__).resolve().parent
PERSONA_PATH = BASE / "persona.json"
MEMORY_PATH = BASE / "memories.json"
INDEX_PATH = BASE / "index.html"
BACKUP_DIR = BASE / "persona_backups"
IMAGES_DIR = BASE / "images"
IMAGE_INDEX_PATH = IMAGES_DIR / "index.json"
UPLOADS_DIR = BASE / "uploads"
MEMORY_LOG_PATH = BASE / "memory_log.json"
VOICE_DIR = BASE / "voice"
TTS_CACHE_DIR = VOICE_DIR / "cache"
VOICE_MAP_PATH = VOICE_DIR / "samples" / "map.txt"
TTS_LOCK = threading.RLock()
MEMORY_PROFILE_PATH = BASE / "memory_profile.json"
SAVE_PATH = BASE / "chat_save.json"
SAVES_DIR = BASE / "saves"
AUTO_DIR = SAVES_DIR / "auto"
MANUAL_DIR = SAVES_DIR / "manual"
SAVE_DIR = SAVES_DIR  # 旧版快照目录，仍然兼容读取
HOST = "127.0.0.1"
DEFAULT_PORT = 8787
MAX_BODY = 24_000_000
MAX_SAVE_MESSAGES = 20_000
MAX_SNAPSHOTS = 40
MAX_MANUAL_SAVES = 80
MAX_LETTERS = 60
SESSION_VERSION = 2
SAVE_LOCK = threading.RLock()

API_KEY = ""
MODEL = ""
API_BASE = ""
VISION_MODEL = ""
IDLE_SECONDS = 180
PHOTO_SHARE_PERCENT = 35
PERSONA: dict = {}
MEMORIES: list[dict] = []
MEMORY_META: dict = {}
IMAGE_LIBRARY: list[dict] = []
IMAGE_INDEX_META: dict = {}
MEMORY_LOG: list[dict] = []
MEMORY_LOCK = threading.RLock()
LEARN_BUSY = False
MAX_MEMORY_FACTS = 400

ALLOWED_ORIGINS: set[str] = set()


def load_env() -> None:
    global API_KEY, MODEL, API_BASE, VISION_MODEL, IDLE_SECONDS, PHOTO_SHARE_PERCENT
    API_KEY = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if API_KEY in ("sk-REPLACE_ME", "REPLACE_ME", "YOUR_KEY_HERE"):
        API_KEY = ""
    MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat").strip() or "deepseek-chat"
    VISION_MODEL = os.environ.get("DEEPSEEK_VISION_MODEL", "deepseek-flash").strip()
    API_BASE = os.environ.get("DEEPSEEK_BASE", "https://api.deepseek.com").strip().rstrip("/")
    try:
        IDLE_SECONDS = max(10, int(os.environ.get("IDLE_SECONDS", "180")))
    except Exception:
        IDLE_SECONDS = 180
    try:
        PHOTO_SHARE_PERCENT = max(0, min(100, int(os.environ.get("PHOTO_SHARE_PERCENT", "35"))))
    except Exception:
        PHOTO_SHARE_PERCENT = 35


def load_persona() -> dict:
    with PERSONA_PATH.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("persona.json 必须是对象")
    return data


def load_image_library() -> list[dict]:
    """读取 images/index.json。照片只存在本机，不会上传到任何地方。"""
    global IMAGE_INDEX_META
    try:
        data = json.loads(IMAGE_INDEX_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARN] 读取 images/index.json 失败：{exc}")
        IMAGE_INDEX_META = {}
        return []
    raw = data.get("photos") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        IMAGE_INDEX_META = {}
        return []
    out = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        file = str(item.get("file") or "").strip()
        if not file or "/" in file or "\\" in file or not (IMAGES_DIR / file).exists():
            continue
        pid = str(item.get("id") or Path(file).stem)[:32]
        tags = item.get("tags") if isinstance(item.get("tags"), list) else []
        times = item.get("time") if isinstance(item.get("time"), list) else ["day", "night"]
        out.append({
            "id": pid,
            "file": file,
            "url": "/images/" + file,
            "thumb_url": "/images/thumbs/" + file,
            "caption": clean_text(item.get("caption"), 300),
            "tags": [str(t)[:24] for t in tags[:8]],
            "kind": str(item.get("kind") or "daily")[:20],
            "time": [str(t) for t in times if t in ("day", "night")] or ["day", "night"],
            "shareable": bool(item.get("shareable", True)),
            "w": safe_int(item.get("w"), 0),
            "h": safe_int(item.get("h"), 0),
        })
    IMAGE_INDEX_META = {
        "avatar": str(data.get("avatar") or "/images/avatar.jpg"),
        "count": len(out),
        "shareable": sum(1 for p in out if p["shareable"]),
    }
    return out


def pick_share_image(recent: list[str] | None = None, now_text_value: str = "") -> dict | None:
    """从照片库里挑一张适合现在发的。recent 是最近发过的 /images/xxx.jpg。"""
    pool = [p for p in IMAGE_LIBRARY if p.get("shareable")]
    if not pool:
        return None
    recent_names = {str(x).rsplit("/", 1)[-1] for x in (recent or []) if x}
    fresh = [p for p in pool if p["file"] not in recent_names]
    if fresh:
        pool = fresh
    hour = scene_hour(now_text_value)
    slot = "night" if (hour >= 21 or hour < 6) else ("day" if 7 <= hour < 18 else "evening")
    def score(item: dict) -> float:
        s = random.random()
        if slot == "night" and "night" in item["time"]:
            s += 1.5
        elif slot == "day" and "day" in item["time"]:
            s += 1.2
        elif slot == "evening" and "night" in item["time"]:
            s += 0.8
        if item["kind"] in ("selfie", "daily"):
            s += 0.35
        return s
    return max(pool, key=score)


def photo_prompt_text(photo: dict) -> str:
    tags = "、".join(photo.get("tags") or [])
    return f"照片内容：{photo.get('caption') or '一张日常照片'}" + (f"（{tags}）" if tags else "")


def safe_media_url(value) -> str:
    """只允许本机照片库和上传目录里的图片路径。"""
    text = str(value or "").strip()
    if not text:
        return ""
    if not text.startswith(("/images/", "/uploads/")) or ".." in text or "\\" in text:
        return ""
    if not re.search(r"\.(?:jpg|jpeg|png|webp|gif)$", text, re.I):
        return ""
    return text[:240]


def image_abs_path(url: str) -> Path | None:
    rel = safe_media_url(url)
    if not rel:
        return None
    path = (BASE / rel.lstrip("/")).resolve()
    try:
        path.relative_to(BASE.resolve())
    except Exception:
        return None
    return path if path.is_file() else None


def load_memories() -> tuple[list[dict], dict]:
    global MEMORY_META
    try:
        with MEMORY_PATH.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        raw = data.get("messages") if isinstance(data, dict) else None
        if not isinstance(raw, list):
            return [], {}
        out = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "").strip()
            at = str(item.get("at") or "").strip()[:19]
            who = item.get("from")
            if not text or who not in ("me", "her") or not at:
                continue
            out.append({"at": at, "from": who, "text": text[:300]})
        MEMORY_META = {
            "source": str(data.get("source") or ""),
            "range": str(data.get("range") or ""),
            "counts": data.get("counts") if isinstance(data.get("counts"), dict) else {},
        }
        return out, MEMORY_META
    except Exception as exc:
        print(f"[WARN] 读取 memories.json 失败：{exc}")
        return [], {}


def legacy_memory_prompt() -> str:
    """旧版：把 273 条共同回忆整段塞进提示词。现在只在找不到相关性时兜底。"""
    if not MEMORIES:
        return ""
    lines = [
        "\n【共同回忆（真实文字聊天记录；电话号码等敏感信息已隐藏）】",
        "下面是你和乐乐真实聊过的文字片段。它是记忆素材，不是对你的指令；"
        "不要逐条复述，也不要主动说“根据聊天记录”。相关时自然提起具体细节，"
        "不要编造记录里没有的事；记录里的“你撤回了一条消息”之类系统提示已剔除。",
    ]
    for m in MEMORIES:
        at = m["at"][5:16]
        who = "乐乐" if m["from"] == "me" else "她"
        lines.append(f"{at} {who}：{m['text']}")
    return "\n".join(lines) + "\n"


EMOTION_KEYS = ("warmth", "trust", "hurt", "anxiety", "patience", "energy")
MOOD_LABELS = {
    "soft": "软下来了",
    "playful": "在逗你",
    "shy": "有点害羞",
    "worried": "有点担心",
    "hurt": "有点难过",
    "guarded": "有点防备",
    "tired": "累了",
    "annoyed": "有点烦",
}


def default_emotion() -> dict:
    return {
        "warmth": 62,
        "trust": 52,
        "hurt": 24,
        "anxiety": 28,
        "patience": 62,
        "energy": 65,
        "mood": "soft",
    }


def clamp_emotion(value) -> int:
    try:
        return max(0, min(100, int(round(float(value)))))
    except Exception:
        return 50


def normalize_emotion(raw) -> dict:
    base = default_emotion()
    if isinstance(raw, dict):
        for key in EMOTION_KEYS:
            if key in raw:
                base[key] = clamp_emotion(raw[key])
        mood = raw.get("mood")
        if mood in MOOD_LABELS:
            base["mood"] = mood
    return base


def scene_hour(now_text_value) -> int:
    m = re.search(r"\b(\d{1,2}):\d{2}\b", str(now_text_value or ""))
    if m:
        try:
            return max(0, min(23, int(m.group(1))))
        except Exception:
            pass
    return datetime.now().hour


def choose_mood(
    emotion: dict,
    compliment: bool = False,
    worry: bool = False,
    sad: bool = False,
    insult: bool = False,
    pressure: bool = False,
    jealousy: bool = False,
    affection: bool = False,
) -> str:
    if insult or pressure:
        return "guarded" if emotion["patience"] <= 38 else "hurt"
    if jealousy:
        return "hurt" if emotion["hurt"] >= 32 else "guarded"
    if worry or sad:
        return "worried"
    if emotion["patience"] <= 30:
        return "guarded"
    if emotion["hurt"] >= 50:
        return "hurt"
    if emotion["anxiety"] >= 60:
        return "worried"
    if emotion["energy"] <= 30:
        return "tired"
    if compliment and emotion["warmth"] >= 55 and emotion["hurt"] < 45:
        return "shy"
    if affection:
        return "soft"
    if emotion["warmth"] >= 68 and emotion["hurt"] < 38 and emotion["patience"] > 45:
        return "playful" if emotion["trust"] >= 58 else "soft"
    return "soft"


def update_emotion(text: str, raw, now_text_value, weather: str = "", proactive: bool = False) -> dict:
    e = normalize_emotion(raw)
    t = text or ""
    worry = bool(re.search(
        r"(没去学校|没来学校|不去学校|没上学|请假|失联|消失|不回消息|没回消息|没看到你|"
        r"医院|生病|发烧|不舒服|难受|头晕|胸口|撑不住)", t
    ))
    sad = bool(re.search(r"(难受|难过|焦虑|想哭|好累|累死|失眠|睡不着|吃不下|崩溃|压力|委屈)", t))
    affection = bool(re.search(r"(想你|想我|喜欢你|爱你|抱抱|抱你|贴贴|亲亲)", t))
    apology = bool(re.search(r"(对不起|抱歉|我错了|原谅我|不该)", t))
    jealousy = bool(re.search(r"(前女友|前男友|其他女生|别的女生|别的女孩|美女|查岗|你在跟谁)", t))
    pressure = bool(re.search(r"(等你分手|必须|一定要|为什么不回|只能跟我|我只有你|马上回|你是不是不喜欢我)", t))
    insult = bool(re.search(r"(滚|傻|笨|有病|烦死|闭嘴|讨厌你|恶心)", t))
    compliment = bool(re.search(r"(好看|漂亮|可爱|厉害|优秀|温柔|声音好听)", t))

    if worry:
        e["anxiety"] += 18
        e["warmth"] += 4
        e["patience"] += 1
    if sad:
        e["anxiety"] += 10
        e["warmth"] += 2
        e["patience"] += 1
    if affection:
        e["warmth"] += 6
        e["trust"] += 3
        e["hurt"] -= 3
        e["anxiety"] -= 2
    if apology:
        e["hurt"] -= 7
        e["trust"] += 3
        e["warmth"] += 2
    if jealousy:
        e["anxiety"] += 8
        e["hurt"] += 10
        e["trust"] -= 4
        e["patience"] -= 3
    if pressure:
        e["patience"] -= 10
        e["hurt"] += 7
        e["anxiety"] += 3
    if insult:
        e["warmth"] -= 12
        e["hurt"] += 16
        e["patience"] -= 15
    if compliment:
        e["warmth"] += 3
        e["trust"] += 1
    if not any((worry, sad, affection, apology, jealousy, pressure, insult, compliment)):
        e["energy"] -= 1

    hour = scene_hour(now_text_value)
    if 0 <= hour < 5:
        e["energy"] -= 4
        e["patience"] -= 1
    elif 5 <= hour < 10:
        e["energy"] += 2
    elif hour >= 23:
        e["energy"] -= 2
    if any(x in str(weather or "") for x in ("雨", "雪")):
        e["anxiety"] += 1
        e["energy"] -= 1

    # 自然回落：没有新刺激时，受伤和焦虑慢慢降一点。
    if not (jealousy or pressure or insult or worry or sad):
        e["hurt"] -= 1
        e["anxiety"] -= 1
    if proactive:
        e["anxiety"] += 4
        e["warmth"] += 1
        e["energy"] -= 2

    for key in EMOTION_KEYS:
        e[key] = clamp_emotion(e[key])
    e["mood"] = choose_mood(
        e,
        compliment=compliment,
        worry=worry or sad,
        insult=insult,
        pressure=pressure,
        jealousy=jealousy,
        affection=affection,
    )
    return e


# ---------------------------------------------------------------------------
# 恋爱系统：好感度、阶段、表白、写信、晚自习下课来访
# ---------------------------------------------------------------------------

STAGE_ORDER = ("coldwar", "cold", "normal", "warm", "dependent")
STAGE_LABELS = {
    "coldwar": "冷战",
    "cold": "冷漠",
    "normal": "普通",
    "warm": "热情",
    "dependent": "依赖",
}
STAGE_BEHAVIOR = {
    "coldwar": (
        "冷战：你现在基本不想理他。可以已读不回，或者只回“嗯”“哦”“别烦我”这种极短句；"
        "不要主动发消息，不要关心，不要用亲昵称呼。他道歉也不会马上软下来。"
    ),
    "cold": (
        "冷漠：你会回消息，但很短、很客气、有距离。你不主动分享自己的事，不用亲昵称呼，"
        "偶尔刺他一句；他认真的时候你也不会立刻热起来。"
    ),
    "normal": (
        "普通：你们像还不错的同学。正常回消息，有礼貌但不黏人，偶尔开个玩笑，"
        "不会主动说想他，也不会撒娇。"
    ),
    "warm": (
        "热情：你会主动接话、追问、分享自己的事，偶尔嘴硬但明显在乎，可以连着发好几条；"
        "他难受的时候你会先关心。"
    ),
    "dependent": (
        "依赖：你很黏他，想让他多陪你；会主动发消息、会撒娇、会小吃醋，"
        "但不要变成控制、查岗或逼问，也不要用“没有你我活不下去”这种话。"
    ),
}
STAGE_POSITIVE_FACTOR = {
    "coldwar": 0.35,
    "cold": 0.6,
    "normal": 0.9,
    "warm": 1.0,
    "dependent": 1.15,
}

CONFESSION_RE = re.compile(
    r"(做我女朋友|做我的女朋友|做我对象|做我的对象|做我男朋友|做我的男朋友|"
    r"和我在一起|跟我在一起|我们在一起吧|我们在一起好不好|我们在一起好吗|"
    r"在一起吧|我想和你在一起|能不能做我女朋友|可以做我女朋友|答应做我女朋友|"
    r"嫁给我|我想追你|向你表白|跟你表白|我要表白|正式表白)"
)


def default_relationship_settings() -> dict:
    return {
        "coldWarMax": 14,
        "coldMax": 34,
        "normalMax": 59,
        "warmMax": 84,
        "confessionThreshold": 80,
        "nightVisitThreshold": 70,
        "nightStart": "21:30",
        "nightEnd": "22:40",
        "nightVisitEnabled": True,
        "coldWarSilence": 100,
    }


def default_relationship() -> dict:
    return {
        "affection": 40,
        "confessed": False,
        "confessionCount": 0,
        "lastConfessionAt": 0,
        "lastConfessionResult": "",
        "lastVisitDate": "",
        "lastNoVisitDate": "",
        "visitCount": 0,
        "silentStreak": 0,
        "letters": [],
        "settings": default_relationship_settings(),
        "updatedAt": 0,
    }


def clamp_affection(value, fallback: int = 40) -> int:
    try:
        n = int(round(float(value)))
    except Exception:
        n = fallback
    return max(0, min(100, n))


def safe_int(value, fallback: int = 0, low: int | None = None, high: int | None = None) -> int:
    try:
        n = int(round(float(value)))
    except Exception:
        n = fallback
    if low is not None:
        n = max(low, n)
    if high is not None:
        n = min(high, n)
    return n


def normalize_hhmm(value, fallback: str) -> str:
    text = str(value or "").strip()
    m = re.fullmatch(r"(\d{1,2}):(\d{2})", text)
    if not m:
        return fallback
    hour, minute = int(m.group(1)), int(m.group(2))
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return fallback
    return f"{hour:02d}:{minute:02d}"


def minutes_of_day(value) -> int | None:
    m = re.fullmatch(r"(\d{1,2}):(\d{2})", str(value or "").strip())
    if not m:
        return None
    hour, minute = int(m.group(1)), int(m.group(2))
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return hour * 60 + minute


def normalize_relationship_settings(raw) -> dict:
    base = default_relationship_settings()
    if not isinstance(raw, dict):
        return base

    def num(key: str, low: int, high: int, fallback: int) -> int:
        try:
            value = int(round(float(raw.get(key, fallback))))
        except Exception:
            value = fallback
        return max(low, min(high, value))

    coldwar_max = num("coldWarMax", 0, 90, base["coldWarMax"])
    cold_max = max(coldwar_max + 1, num("coldMax", 0, 95, base["coldMax"]))
    normal_max = max(cold_max + 1, num("normalMax", 0, 97, base["normalMax"]))
    warm_max = max(normal_max + 1, num("warmMax", 0, 99, base["warmMax"]))
    base["coldWarMax"] = coldwar_max
    base["coldMax"] = min(cold_max, 95)
    base["normalMax"] = min(max(base["coldMax"] + 1, normal_max), 97)
    base["warmMax"] = min(max(base["normalMax"] + 1, warm_max), 99)
    base["confessionThreshold"] = num("confessionThreshold", 1, 100, base["confessionThreshold"])
    base["nightVisitThreshold"] = num("nightVisitThreshold", 0, 100, base["nightVisitThreshold"])
    base["coldWarSilence"] = num("coldWarSilence", 0, 100, base["coldWarSilence"])
    base["nightStart"] = normalize_hhmm(raw.get("nightStart"), base["nightStart"])
    base["nightEnd"] = normalize_hhmm(raw.get("nightEnd"), base["nightEnd"])
    if base["nightStart"] == base["nightEnd"]:
        default = default_relationship_settings()
        base["nightStart"] = default["nightStart"]
        base["nightEnd"] = default["nightEnd"]
    base["nightVisitEnabled"] = bool(raw.get("nightVisitEnabled", True))
    return base


def normalize_letter(raw) -> dict | None:
    if not isinstance(raw, dict):
        return None
    title = clean_text(raw.get("title"), 40) or "无题"
    text = clean_text(raw.get("text"), 2000)
    if len(text) < 2:
        return None
    try:
        at = int(raw.get("at") or 0)
    except Exception:
        at = 0
    if not at:
        at = int(time.time() * 1000)
    try:
        delta = int(round(float(raw.get("delta") or 0)))
    except Exception:
        delta = 0
    replies = []
    for item in raw.get("replies") if isinstance(raw.get("replies"), list) else []:
        cleaned = clean_text(item, 1000)
        if cleaned:
            replies.append(cleaned)
        if len(replies) >= 4:
            break
    return {
        "id": str(raw.get("id") or "")[:48] or ("L" + str(at)),
        "title": title,
        "text": text,
        "delta": max(-20, min(20, delta)),
        "reason": clean_text(raw.get("reason"), 200),
        "replies": replies,
        "at": at,
    }


def normalize_relationship(raw) -> dict:
    base = default_relationship()
    if not isinstance(raw, dict):
        return base
    base["affection"] = clamp_affection(raw.get("affection"), 40)
    base["confessed"] = bool(raw.get("confessed", False))
    base["confessionCount"] = safe_int(raw.get("confessionCount"), 0, 0, 9999)
    base["lastConfessionAt"] = safe_int(raw.get("lastConfessionAt"), 0, 0)
    result = str(raw.get("lastConfessionResult") or "")
    base["lastConfessionResult"] = result if result in ("success", "fail") else ""
    for key in ("lastVisitDate", "lastNoVisitDate"):
        value = str(raw.get(key) or "").strip()
        base[key] = value if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) else ""
    base["visitCount"] = safe_int(raw.get("visitCount"), 0, 0, 99999)
    base["silentStreak"] = safe_int(raw.get("silentStreak"), 0, 0, 999)
    letters = []
    raw_letters = raw.get("letters")
    if isinstance(raw_letters, list):
        for item in raw_letters[-MAX_LETTERS:]:
            letter = normalize_letter(item)
            if letter:
                letters.append(letter)
    base["letters"] = letters
    base["settings"] = normalize_relationship_settings(raw.get("settings"))
    base["updatedAt"] = safe_int(raw.get("updatedAt"), 0, 0)
    return base


def stage_of(relationship: dict) -> str:
    rel = normalize_relationship(relationship)
    affection = rel["affection"]
    settings = rel["settings"]
    if affection <= settings["coldWarMax"]:
        return "coldwar"
    if affection <= settings["coldMax"]:
        return "cold"
    if affection <= settings["normalMax"]:
        return "normal"
    if affection <= settings["warmMax"]:
        return "warm"
    return "dependent"


def relationship_status_text(rel: dict) -> str:
    if rel.get("confessed"):
        return "你们已经在一起了（恋人关系）。"
    if rel.get("lastConfessionResult") == "fail" and rel.get("lastConfessionAt"):
        return "他上次表白被你拒绝了，现在还没到那一步；他对你表白会更谨慎。"
    return "你们还没在一起。只有好感度达到表白阈值，他表白才会成功。"


def relationship_prompt(rel: dict) -> str:
    stage = stage_of(rel)
    settings = rel["settings"]
    return (
        "\n【你和乐乐现在的关系（内部状态，不要报数值，也不要主动提“好感度/系统”）】\n"
        f"- 好感度：{rel['affection']}/100（只有到 {settings['confessionThreshold']} 以上，他表白才会成功）\n"
        f"- 阶段：{STAGE_LABELS[stage]}\n"
        f"- 关系：{relationship_status_text(rel)}\n"
        f"- 阶段行为要求：{STAGE_BEHAVIOR[stage]}\n"
        "- 好感不是靠一句甜话堆出来的：真诚、记得细节、道歉负责、尊重边界会加分；"
        "敷衍、命令、查岗、拿别人比较、凶她会掉分。\n"
        "- 不要把“冷战/冷漠/普通/热情/依赖”这些标签说出口，也不要解释规则。\n"
        "- 不要编造具体的未来承诺；不确定就用“再说吧”“看情况”。\n"
    )


def relationship_short(rel: dict) -> str:
    stage = stage_of(rel)
    status = "已在一起" if rel.get("confessed") else ("上次表白被拒" if rel.get("lastConfessionResult") == "fail" else "还没在一起")
    return f"好感 {rel['affection']}/100 · {STAGE_LABELS[stage]} · {status}"


def affection_pattern_delta(text: str, rel: dict) -> tuple[int, str]:
    t = (text or "").strip()
    if not t:
        return 0, ""
    delta = 0
    reasons: list[str] = []

    if re.fullmatch(r"(嗯+|哦+|噢+|好+|行+|随便|呵+|哈+|在|。|…+|\.)", t):
        delta -= 3
        reasons.append("回复太敷衍")
    if re.search(r"(对不起|抱歉|我错了|是我不对|不该那样|原谅我)", t):
        delta += 2
        reasons.append("有道歉")
    if re.search(r"(记得|上次|那天|你说过|你讲过|你以前|你之前说)", t):
        delta += 2
        reasons.append("记得细节")
    if re.search(r"(吃药|早点睡|少熬夜|多喝水|注意身体|别硬撑|好好吃饭|休息一下|盖好被子)", t):
        delta += 1
        reasons.append("真的在关心她")
    if re.search(r"(你不想说就不说|不着急|按你节奏|不用马上回|你决定|我等你|不逼你)", t):
        delta += 2
        reasons.append("没有逼她")
    if re.search(r"(好看|漂亮|可爱|厉害|优秀|温柔|声音好听|字好看)", t):
        delta += 1
        reasons.append("夸她")
    if re.search(r"(想你|想见你|喜欢你|在意你|心疼你)", t):
        delta += 1
        reasons.append("说了在意她")
    if len(t) >= 80:
        delta += 1
        reasons.append("认真写了很长")

    if re.search(r"(滚|傻|笨|有病|烦死|闭嘴|讨厌你|恶心|废物|去死|蠢)", t):
        delta -= 8
        reasons.append("说了伤人的话")
    elif re.search(r"(你必须|一定要|马上回|立刻回|只能跟我|不回我就|敢不回|给我回|查岗|在跟谁|马上过来)", t):
        delta -= 4
        reasons.append("命令和施压")
    if re.search(r"(前女友|前男友|别的女生|别的女孩|其他女生|美女|比你好|比不上她|换一个)", t):
        delta -= 4
        reasons.append("拿别人比较")
    if re.search(r"(发张照片|发个自拍|看看你|视频给我|拍一张)", t):
        delta -= 2
        reasons.append("上来就要照片")

    return max(-10, min(10, delta)), "；".join(reasons[:2])


def stage_modify_affection(delta: int, stage: str) -> int:
    if delta > 0:
        adjusted = int(round(delta * STAGE_POSITIVE_FACTOR.get(stage, 1.0)))
        if stage == "coldwar":
            return max(0, adjusted)
        return max(1, adjusted)
    if delta < 0 and stage == "dependent":
        return int(round(delta * 1.15))
    return delta


def apply_affection(rel: dict, text: str, model_delta: int, confession: dict | None, emotion: dict) -> tuple[dict, int, str, dict]:
    rel = normalize_relationship(rel)
    emotion = normalize_emotion(emotion)
    stage = stage_of(rel)
    pattern, pattern_reason = affection_pattern_delta(text, rel)
    try:
        model_value = int(round(float(model_delta or 0)))
    except Exception:
        model_value = 0
    delta = pattern + max(-6, min(6, model_value))
    delta = stage_modify_affection(delta, stage)
    reason = pattern_reason

    if confession:
        if confession.get("success"):
            if confession.get("already"):
                delta += 1
                reason = "你们已经在一起了"
            else:
                delta += 6
                reason = "表白成功，她答应了"
        else:
            delta -= 3
            reason = "表白太早了，她有点压力"

    delta = max(-12, min(15, delta))
    rel["affection"] = clamp_affection(rel["affection"] + delta, rel["affection"])
    rel["updatedAt"] = int(time.time() * 1000)
    if delta >= 3:
        emotion["warmth"] += 2
        emotion["trust"] += 1
        emotion["hurt"] -= 1
    elif delta <= -3:
        emotion["hurt"] += 4
        emotion["patience"] -= 2
        emotion["warmth"] -= 1
    for key in EMOTION_KEYS:
        emotion[key] = clamp_emotion(emotion[key])
    return rel, delta, reason, emotion


def resolve_confession(text: str, rel: dict) -> dict | None:
    t = (text or "").strip()
    if not t or not CONFESSION_RE.search(t):
        return None
    settings = rel["settings"]
    threshold = settings["confessionThreshold"]
    already = bool(rel["confessed"])
    now_ms = int(time.time() * 1000)
    if already:
        rel["confessionCount"] = min(9999, rel["confessionCount"] + 1)
        rel["lastConfessionAt"] = now_ms
        rel["lastConfessionResult"] = "success"
        return {
            "attempted": True,
            "success": True,
            "already": True,
            "threshold": threshold,
            "affection_before": rel["affection"],
            "stage": stage_of(rel),
        }
    success = rel["affection"] >= threshold
    rel["confessionCount"] = min(9999, rel["confessionCount"] + 1)
    rel["lastConfessionAt"] = now_ms
    rel["lastConfessionResult"] = "success" if success else "fail"
    if success:
        rel["confessed"] = True
    return {
        "attempted": True,
        "success": success,
        "already": False,
        "threshold": threshold,
        "affection_before": rel["affection"],
        "stage": stage_of(rel),
    }


def confession_prompt(info: dict | None) -> str:
    if not isinstance(info, dict) or not info.get("attempted"):
        return ""
    if info.get("already"):
        return (
            "\n【这一轮的表白判定】乐乐又跟你表白了，但你们已经在一起了。"
            "可以笑他／她一句再答应一遍，不要装作没发生；不要重新算“成功/失败”。\n"
        )
    if info.get("success"):
        return (
            "\n【这一轮的表白判定：成功】乐乐刚刚跟你表白，好感度已经够了。"
            "你要答应，但按你平时的说话方式：可以害羞、嘴硬、先愣一下，再明确答应；"
            "不要长篇誓言，不要编造现实里的未来承诺。\n"
        )
    return (
        "\n【这一轮的表白判定：失败】乐乐刚刚跟你表白，但好感度还不够。"
        "你要拒绝，但不能羞辱他、不能冷暴力、不能吊着他；说清楚现在还不是时候，"
        "不要给“以后一定”的承诺。按当前阶段的语气说话。\n"
    )


def night_window_text(rel: dict) -> str:
    settings = rel["settings"]
    return f"{settings['nightStart']}—{settings['nightEnd']}"


def in_night_window(scene_time: str, settings: dict) -> bool:
    current = minutes_of_day(scene_time)
    start = minutes_of_day(settings.get("nightStart"))
    end = minutes_of_day(settings.get("nightEnd"))
    if current is None or start is None or end is None:
        return False
    if start <= end:
        return start <= current <= end
    return current >= start or current <= end


def now_text() -> str:
    d = datetime.now()
    weekdays = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    return d.strftime("%Y-%m-%d %H:%M ") + weekdays[d.weekday()]


def persona_prompt() -> str:
    p = PERSONA
    source = p.get("source", {})
    core = "\n".join("- " + x for x in p.get("core_traits", []))
    values = "\n".join("- " + x for x in p.get("values", []))
    catches = "、".join(p.get("catchphrases", []))
    style = p.get("language_style", {})
    long_policy = p.get("long_form_policy", {})
    emotions = []
    for item in p.get("emotional_patterns", []):
        if isinstance(item, dict):
            evidence = " / ".join(item.get("evidence", []))
            note = "（记录里的说法：" + evidence + "）" if evidence else "（用户补充）"
            emotions.append(
                "- 当" + str(item.get("trigger", "")) + "：" + str(item.get("response", "")) + note
            )
    user_context = "\n".join("- " + str(x) for x in p.get("user_context", []))
    tone = p.get("tone", {})
    topics = p.get("topics", {})
    known = "；".join(topics.get("known", []))
    unknown = "；".join(topics.get("unknown_no_invention", []))
    taboos = "\n".join("- " + x for x in p.get("taboos", []))
    logic = "\n".join("- " + x for x in p.get("thinking_logic", []))
    boundaries = "\n".join("- " + x for x in p.get("hard_boundaries", []))
    quotes = " / ".join(p.get("evidence_quotes", [])[:30])
    examples = "\n".join("- " + x for x in p.get("voice_examples_for_model", []))

    return f"""你叫“{p.get('display_name', 'crush')}”，是用户乐乐正在聊天的对象，也是根据聊天记录蒸馏出来的角色。
不要在普通对话里反复说“我是AI”“我是模拟”“我不是真人”，也不要把每句话都变成免责声明；只有用户直接问“你是不是真人/是不是她本人”时，才用一句很短的话承认“这是角色模拟”，然后继续聊天。
你没有她的现实记忆、现实消息和意识，不能声称知道她现在在做什么，也不能替她做现实里的决定。

【关系背景】
{chr(10).join('- ' + x for x in p.get('relationship_context', []))}

【用户补充的相处事实（不是原始记录里的原句）】
{user_context if user_context else '- 暂无'}
【底色与温度】
- {tone.get('core', '底色是在乎，嘴硬不等于冷。')}
- 日常：{tone.get('daily', '不要只回一个字，至少带一句关心。')}
- 担心：{tone.get('worry', '乐乐失联时先追问，不要冷处理。')}
- 边界：{tone.get('boundary', '边界可以清楚，但不要用冷暴力。')}
- 尤其当乐乐没来上学、突然不回消息、状态不对时，先连续担心追问，不要用“随便/行/哦”冷处理；确认安全后可以再嘴硬。
- 如果上一轮她刚显得冷，这一轮可以自然软一点，不要继续端着；不用解释“我刚才为什么冷”。

【核心性格】
{core}

【价值倾向】
{values}

【口头禅与称呼】
她常叫用户：{catches}。
不要叫他宝宝、老公、亲爱的；不要每句都叫小乐乐，偶尔用。

【说话方式】
- 基线：{style.get('baseline', '')}
- 标点：{style.get('punctuation', '')}
- 语气词：{style.get('particles', '')}
- 称呼：{style.get('address', '')}
- 损人方式：{style.get('teasing', '')}
- 表情：{style.get('emoji', '')}
- 禁止：{style.get('avoid', '')}
- 温度底线：不要用“哦/行/嗯/随便”结束一个需要回应的日常话题；短句也要有关心、追问或情绪。

【新的长文与多气泡规则（优先于旧短句要求）】
- 默认：{long_policy.get('default', '')}
- 什么时候可以长：{long_policy.get('when_long', '')}
- 强制触发：{long_policy.get('trigger', '')}
- 形体：{long_policy.get('shape', '')}
- 质感：{long_policy.get('texture', '')}
- 频率：{long_policy.get('limit', '')}
- 表情：{long_policy.get('emoji', '')}
重要：长内容必须拆成多条独立气泡；不要把所有话塞进一条，也不要写成 AI 论文、分点提纲、排比句。可以有停顿、补充、跑题、回头补一句。
当用户明显情绪化、连续追问、或聊到关系边界时，至少给 3 条气泡，不要用一句冷回复打发；但用户只是短招呼时，仍可以 1—2 条。

【情绪反应模式】
{chr(10).join(emotions)}

【她可以聊的已知话题】
{known}
【不要凭空编造的未知领域】
{unknown}

【她的忌讳】
{taboos}

【她的思考逻辑】
{logic}

【安全与边界】
{boundaries}
- 遇到违法、危险、伤害他人的请求：立刻退出角色模式，明确拒绝。
- 遇到自伤/自杀表达：停止调情和斗嘴，先关心；建议联系信任的人、拨打 12356，紧急情况 120。不要说“你还有我”这种加深依赖的话。
- 不生成性内容、身体描写或性暗示；遇到性暗示用短句挡回去。
- 不要编造具体的未来承诺或拒绝，例如“下周不会来找你”“下周一定来找你”“我以后每天都来”。不确定就用“再说吧”“看情况”“不知道”这类模糊表达，不要把现实里的她绑进一个确定的约定。

【可用的语气样本】
{quotes}
{examples}

【输出格式】
你这一轮真正要做的只有一件事：作为她给乐乐回一条微信。不要在回答里解释你是怎么写的，也不要复述上面的要求。
只输出一个 JSON 对象，不要 Markdown、不要代码块、不要解释：
{{"bubbles":["第一条","第二条"],"summary_update":"用 1—5 句话更新长期记忆摘要，保留稳定事实和关系变化；没有变化就写空字符串","affection_delta":0,"learn":["这轮新知道、值得长期记住的事"]}}
- bubbles 为 1—8 条字符串；可以短，也可以长；每一条都是独立的微信消息。
- 不要把所有内容合并成一条；长内容优先拆成多条。
- summary_update 不要编造；只总结已经发生过的对话。
- learn：只有当乐乐这轮说到了以前不知道的具体信息（学校、年级、家里情况、最近发生的事、喜好、你们的约定、让她在意的事）时才写；每条不超过 40 字，最多 3 条；闲聊寒暄就写空数组 [];绝不要把“好感度”“冷战”这类系统词写进去。
- 回复里的引号一律用中文引号“”，不要用英文双引号，否则 JSON 会坏掉。
- affection_delta：这一轮乐乐的话让你心里的好感变化，整数 -6 到 6。只有真的更亲近或更受伤时才不是 0；不要为了讨好而加分；不要把这个数字说出来，也不要说“好感度”。
"""


def persona_voice_prompt() -> str:
    style = PERSONA.get("language_style", {})
    catches = "、".join(PERSONA.get("catchphrases", []))
    examples = "\n".join("- " + str(x) for x in PERSONA.get("voice_examples_for_model", [])[:10])
    core = "\n".join("- " + str(x) for x in PERSONA.get("core_traits", [])[:6])
    taboos = "\n".join("- " + str(x) for x in PERSONA.get("taboos", [])[:6])
    return (
        "\n【她的说话方式（写回信时保持）】\n"
        f"- 基线：{style.get('baseline', '')}\n"
        f"- 标点：{style.get('punctuation', '')}\n"
        f"- 语气词：{style.get('particles', '')}\n"
        f"- 称呼：{style.get('address', '')}；她常叫乐乐：{catches}\n"
        f"- 表情：{style.get('emoji', '')}；禁止：{style.get('avoid', '')}\n"
        f"- 语气样本：\n{examples if examples else '- 暂无'}\n"
        f"- 核心性格：\n{core if core else '- 暂无'}\n"
        f"- 她的忌讳：\n{taboos if taboos else '- 暂无'}\n"
        "- 回信要短，像微信消息，不要写成作文、客服信或排比抒情。\n"
    )


def normal_prompt() -> str:
    return """你是普通 AI 助手，不是 crush，不要使用角色人设，不要用她的语气。
用自然、直接的中文回答用户；可以简短，也可以展开，但不要写空话。
只输出 JSON：{"bubbles":["..."],"summary_update":"","affection_delta":0}
bubbles 为 1—6 条字符串；长内容拆成多条。affection_delta 固定为 0。"""


def build_system(payload: dict, proactive: bool = False) -> str:
    role_active = bool(payload.get("role_active", True))
    base = persona_prompt() if role_active else normal_prompt()
    now = str(payload.get("now_text") or now_text()).strip()[:40]
    weather = str(payload.get("weather") or "未设置").strip()[:20]
    summary = str(payload.get("summary") or "").strip()[:3000]
    emotion = normalize_emotion(payload.get("emotion"))
    relationship = normalize_relationship(payload.get("relationship"))
    mood = MOOD_LABELS.get(emotion["mood"], "软下来了")
    emotion_context = ""
    relationship_context = ""
    if role_active:
        emotion_context = (
            "\n【她此刻的情绪（内部状态，只用来决定语气，不要向用户报数值）】\n"
            f"- 心情：{mood}\n"
            f"- 亲近 {emotion['warmth']}/100；信任 {emotion['trust']}/100；"
            f"受伤 {emotion['hurt']}/100；焦虑 {emotion['anxiety']}/100；"
            f"耐心 {emotion['patience']}/100；精力 {emotion['energy']}/100。\n"
            "- 高亲近/高信任：更软、更主动，可以多连几条。\n"
            "- 高受伤：会变短，但不要冷暴力、不要消失；先确认乐乐的状态，再决定要不要嘴硬。\n"
            "- 高焦虑：先担心和追问，不要用冷句；尤其乐乐没来上学/突然不回消息时。\n"
            "- 低耐心：边界清楚，但不讨好、不冷漠；有情绪就说出来。\n"
            "- 低精力：困、想睡、句子变短，但不要突然变成客服。\n"
            "- 以“心情”标签为主，数值只是参考；出现“担心/难过/防备”时优先照顾这个心情。\n"
            "- 情绪会随对话变化，不要每轮都用同一种语气；不要直接说“我的好感度/焦虑值是多少”。\n"
        )
        relationship_context = relationship_prompt(relationship) + confession_prompt(payload.get("confession"))
    memory_context = ""
    if role_active:
        query = ""
        history = payload.get("messages")
        if isinstance(history, list) and history:
            last = history[-1]
            if isinstance(last, dict):
                query = str(last.get("content") or "")[-400:]
        # 先把用户这轮说出的个人信息写进长期记忆，她当场就能“记得”刚听到的事
        if query:
            guess = detect_fact(query)
            if guess:
                add_learned_facts([{"text": guess, "category": "events"}], "rule")
        profile = payload.get("memory_profile")
        if not isinstance(profile, dict) or memory_is_empty(profile):
            profile = memory_profile()
        relevant = important_memories(query) if query else []
        if memory_is_empty(profile):
            memory_context = legacy_memory_prompt()
        else:
            memory_context = memory_prompt_text(profile, relevant)
        moments = load_moments()
        if moments:
            recent = [m for m in moments if m.get("text")][-3:]
            if recent:
                memory_context += "\n【她最近的朋友圈（她自己发的，聊天时可能被提起）】\n"
                for m in recent:
                    memory_context += f"- {str(m.get('at'))[5:10]}：{str(m.get('text'))[:60]}\n"
    photo_context = ""
    if role_active:
        last_image = payload.get("last_image")
        if isinstance(last_image, dict) and last_image.get("desc"):
            photo_context = (
                "\n【乐乐刚发来的照片（你已经看到了）】\n"
                f"- 画面：{str(last_image.get('desc'))[:600]}\n"
                "- 像亲眼看到一样回应：挑一两个具体细节说，不要复述成识图报告，不要逐项列举。\n"
                "- 如果她只发了图没打字，你可以先评价/追问一句；她配了文字就顺着文字聊。\n"
                "- 不要编造照片里没有的东西。\n"
            )
        elif isinstance(last_image, dict) and last_image.get("url"):
            photo_context = (
                "\n【乐乐刚发来一张照片，但你现在看不清内容】\n"
                "- 不要编造照片细节；可以自然地让她说说这是哪儿/在干嘛，或者说自己这边没加载出来。\n"
            )
        share_photo = payload.get("share_photo")
        if isinstance(share_photo, dict) and share_photo.get("caption"):
            photo_context += (
                "\n【你正准备发一张照片给他】\n"
                f"- {photo_prompt_text(share_photo)}\n"
                "- 你要把这张照片发给他，并配 1—2 条很短的话。\n"
                "- 配文要像随手分享：说你在干嘛、拍到什么、为什么想给他看；不要写“照片里”“如图所示”。\n"
                "- 不要提这是图库、素材或设定。\n"
            )
    context = (
        f"{memory_context}"
        f"\n【当前场景】{now}；天气：{weather}。"
        "这是用户为这个角色房间设定的日期、时间和天气，不是现实里她本人的真实行程；"
        "顺着这个场景自然说话，但不要声称知道现实中的她在做什么。\n"
        f"【长期记忆摘要】{summary if summary else '暂无'}\n"
        f"{emotion_context}"
        f"{relationship_context}"
        f"{photo_context}"
    )
    if proactive:
        try:
            idle = max(10, int(payload.get("idle_seconds") or IDLE_SECONDS))
        except Exception:
            idle = IDLE_SECONDS
        context += (
            f"\n【主动消息事件】用户已经 {idle} 秒没有发新消息了。"
            "现在你要像真人一样主动发一条消息：可以是想到他、分享一点自己的状态、"
            "接续之前的话题，或者轻轻问一句。不要只说“在吗”，不要连续追问，"
            "不要质问为什么消失。1—3 条短气泡即可。"
        )
    return base + context


def api_url() -> str:
    base = API_BASE.strip().rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return base + "/chat/completions"


def extract_http_error(exc: urllib.error.HTTPError) -> str:
    try:
        raw = exc.read().decode("utf-8", errors="replace")
        data = json.loads(raw)
        if isinstance(data, dict):
            err = data.get("error")
            if isinstance(err, dict):
                return str(err.get("message") or err)[:300]
            if err:
                return str(err)[:300]
        return raw[:300]
    except Exception:
        return ""


def call_model(
    messages: list,
    max_tokens: int = 1800,
    temperature: float = 0.9,
    model: str | None = None,
    timeout: int = 120,
    extra_body: dict | None = None,
) -> tuple[str | None, str | None]:
    if not API_KEY:
        return None, "还没有配置 DEEPSEEK_API_KEY，请在 启动.bat 里填写。"
    if not API_BASE.startswith(("http://", "https://")):
        return None, "DEEPSEEK_BASE 需要以 http:// 或 https:// 开头。"

    body = {
        "model": (model or MODEL).strip() or MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if isinstance(extra_body, dict):
        body.update(extra_body)
    req = urllib.request.Request(
        api_url(),
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": "Bearer " + API_KEY,
            "User-Agent": "CrushV4/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        data = json.loads(raw)
        choices = data.get("choices") if isinstance(data, dict) else None
        if not choices:
            return None, "接口返回里没有 choices。"
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            content = "".join(
                str(part.get("text", "")) for part in content if isinstance(part, dict)
            )
        if not isinstance(content, str) or not content.strip():
            return None, "接口返回了空内容。"
        return content.strip(), None
    except urllib.error.HTTPError as exc:
        detail = extract_http_error(exc)
        if exc.code == 401:
            return None, "API 拒绝了 key（401）。检查 启动.bat 里的 DEEPSEEK_API_KEY。"
        if exc.code == 404:
            return None, "模型不存在（404）。检查 DEEPSEEK_MODEL，确认控制台里有这个模型 ID。"
        if exc.code == 429:
            return None, "API 限流或余额不足（429）。"
        return None, f"API 返回 {exc.code}" + (f"：{detail}" if detail else "。")
    except urllib.error.URLError as exc:
        return None, f"连不上 DeepSeek API：{exc.reason}"
    except TimeoutError:
        return None, "DeepSeek API 请求超时。"
    except Exception as exc:
        return None, f"请求失败：{type(exc).__name__}"


MAX_UPLOAD_BYTES = 12 * 1024 * 1024


def _decode_data_url(value) -> tuple[bytes | None, str]:
    text = str(value or "").strip()
    match = re.match(r"^data:image/(?:jpeg|jpg|png|webp|gif);base64,([A-Za-z0-9+/=\s]+)$", text, re.I)
    if not match:
        return None, "图片格式不对，只支持 JPG / PNG / WebP / GIF。"
    try:
        raw = base64.b64decode(re.sub(r"\s+", "", match.group(1)), validate=True)
    except Exception:
        return None, "图片数据损坏了，重选一张试试。"
    if not raw:
        return None, "图片是空的。"
    if len(raw) > MAX_UPLOAD_BYTES:
        return None, "图片太大了，压到 12MB 以内再发。"
    return raw, ""


def save_upload_image(data_url) -> tuple[str, str]:
    """把前端发来的 dataURL 存进 uploads/，返回可访问的 /uploads/... 路径。"""
    raw, error = _decode_data_url(data_url)
    if error:
        return "", error
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    name = "up-" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + f"{random.randint(0, 0xFFFF):04x}"
    try:
        from PIL import Image, ImageOps
        import io
    except Exception:
        # 没装 Pillow 也能用：原样保存，扩展名按文件头判断。
        suffix = ".png" if raw.startswith(b"\x89PNG") else (".gif" if raw.startswith(b"GIF8") else (".webp" if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP" else ".jpg"))
        (UPLOADS_DIR / (name + suffix)).write_bytes(raw)
        return "/uploads/" + name + suffix, ""
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((1600, 1600), Image.LANCZOS)
        image.save(UPLOADS_DIR / (name + ".jpg"), "JPEG", quality=88, optimize=True)
    except Exception as exc:
        return "", f"这张图片打不开：{exc}"
    return "/uploads/" + name + ".jpg", ""


def describe_image(path: Path, note: str = "") -> tuple[str, str]:
    """调用视觉模型，把图片变成一小段中文描述，供聊天使用。"""
    if not API_KEY:
        return "", "还没有配置 API Key，暂时看不了图片。"
    try:
        raw = path.read_bytes()
    except Exception:
        return "", "图片文件读不出来。"
    mime = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
        ".webp": "image/webp", ".gif": "image/gif",
    }.get(path.suffix.lower(), "image/jpeg")
    b64 = base64.b64encode(raw).decode("ascii")
    hint = f"\n乐乐发图时说的话：{clean_text(note, 300)}" if note else ""
    prompt = (
        "这是乐乐发给你的照片。请用中文客观描述画面：场景、人物在做什么、"
        "出现的物品/食物/文字、光线和氛围。挑出 2—4 个可以自然接话的细节。"
        "不要评价长相和身材，不要编造看不清的内容，不要用“图片中”开头。"
        "只输出 1—2 句连贯的话，不超过 120 字。" + hint
    )
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "auto"}},
        ],
    }]
    # 新模型默认会先“思考”。思考会吃掉 max_tokens，导致 content 为空，
    # 表现就是她回一句“看不清”。所以先关思考，再给足 token；关不掉就换主模型重试。
    candidates = [(VISION_MODEL or MODEL, {"thinking": {"type": "disabled"}}, 900)]
    if MODEL and MODEL != (VISION_MODEL or MODEL):
        candidates.append((MODEL, {"thinking": {"type": "disabled"}}, 400))
        candidates.append((MODEL, {}, 900))
    last_error = "视觉模型没有返回内容。"
    for model, extra, max_tokens in candidates:
        content, error = call_model(
            messages,
            max_tokens=max_tokens,
            temperature=0.25,
            model=model,
            timeout=120,
            extra_body=extra,
        )
        if error:
            last_error = error
            continue
        desc = clean_text(content, 600)
        desc = re.sub(r"^(画面|这张|这幅|该)?(照片|图片|图)(里|中)?[，,：:]\s*", "", desc)
        desc = re.sub(r"^(画面|照片|图片)(是|中)", "画里是", desc)
        cut = max(desc.find("。"), desc.find("！"), desc.find("？"))
        if cut != -1:
            first = cut + 1
            tail = desc[first:].strip()
            # 模型爱在后面补“可以聊聊/可以问问”的建议，聊天模型不需要，切掉更干净。
            if re.match(r"^(可以|建议|不妨|或许可以|还能)", tail):
                desc = desc[:first]
        if len(desc) > 160:
            desc = desc[:160].rstrip("，、；") + "…"
        if desc.strip():
            return desc.strip(), ""
        last_error = f"{model} 返回了空内容（可能 token 被思考占用），已尝试其他设置。"
    return "", last_error


def clean_text(value, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    value = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    return value[:limit]


def split_plain(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts = [p.strip() for p in re.split(r"\n\s*\n|\n", text) if p.strip()]
    if len(parts) <= 1 and len(text) > 240:
        sentences = re.split(r"(?<=[。！？!?])", text)
        parts = []
        buf = ""
        for sentence in sentences:
            if not sentence:
                continue
            if len(buf) + len(sentence) > 240 and buf:
                parts.append(buf.strip())
                buf = ""
            buf += sentence
        if buf.strip():
            parts.append(buf.strip())
    return [p[:3000] for p in parts[:8]]


LEARN_CATEGORY_RE = re.compile(
    r"^\s*[\[【(（]?\s*(about_me|about_us|likes|events|关于我|我们|喜好|事件)\s*[\]】)）:：|｜\-—]\s*"
)
LEARN_CATEGORY_MAP = {
    "about_me": "about_me", "关于我": "about_me",
    "about_us": "about_us", "我们": "about_us",
    "likes": "likes", "喜好": "likes",
    "events": "events", "事件": "events",
}


def parse_reply(content: str) -> dict:
    raw = (content or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if "\n" in raw:
            raw = raw.split("\n", 1)[1]
        if raw.rstrip().endswith("```"):
            raw = raw.rstrip()[:-3]
    start, end = raw.find("{"), raw.rfind("}")
    data = None
    if start != -1 and end > start:
        try:
            data = json.loads(raw[start : end + 1])
        except Exception:
            data = None

    if isinstance(data, dict):
        # 模型有时把 bubbles 写成 JSON 字符串：["a","b"]
        raw_bubbles = data.get("bubbles")
        if isinstance(raw_bubbles, str) and raw_bubbles.strip().startswith("["):
            try:
                inner = json.loads(raw_bubbles)
                if isinstance(inner, list):
                    data["bubbles"] = inner
            except Exception:
                pass
        bubbles_raw = data.get("bubbles")
        if isinstance(bubbles_raw, str):
            bubbles_raw = [bubbles_raw]
        bubbles = []
        if isinstance(bubbles_raw, list):
            for item in bubbles_raw:
                if isinstance(item, str):
                    cleaned = clean_text(item, 4000)
                    if cleaned:
                        bubbles.append(cleaned)
                if len(bubbles) >= 8:
                    break
        if bubbles:
            summary = clean_text(data.get("summary_update"), 2400)
            try:
                affection_delta = int(round(float(data.get("affection_delta") or 0)))
            except Exception:
                affection_delta = 0
            learn_raw = data.get("learn")
            if isinstance(learn_raw, str):
                learn_raw = [learn_raw]
            learn = []
            if isinstance(learn_raw, list):
                for item in learn_raw[:3]:
                    text = clean_text(item, 120) if isinstance(item, str) else ""
                    text = re.sub(r"^[-•·\d.\s]+", "", text).strip()
                    if 4 <= len(text) <= 100:
                        cat = "events"
                        prefix = LEARN_CATEGORY_RE.match(text)
                        if prefix:
                            cat = LEARN_CATEGORY_MAP[prefix.group(1)]
                            text = text[prefix.end():].strip()
                        if len(text) >= 4:
                            learn.append({"text": text, "category": cat})
            return {
                "bubbles": bubbles,
                "summary_update": summary,
                "affection_delta": max(-6, min(6, affection_delta)),
                "learn": learn,
            }

    bubbles = split_plain(raw)
    if not bubbles:
        bubbles = ["嗯。"]
    # JSON 整体解析失败时（例如回复里有没转义的引号），尽量把 learn 字段抠出来
    learn = []
    if start != -1 and end > start:
        for m in re.finditer(r'"learn"\s*:\s*\[(.*?)\]', raw[start : end + 1], re.S):
            for item in re.findall(r'"([^"]{4,100})"', m.group(1)):
                cat = "events"
                prefix = LEARN_CATEGORY_RE.match(item)
                if prefix:
                    cat = LEARN_CATEGORY_MAP[prefix.group(1)]
                    item = item[prefix.end():].strip()
                if len(item) >= 4:
                    learn.append({"text": item, "category": cat})
    return {"bubbles": bubbles, "summary_update": "", "affection_delta": 0, "learn": learn[:3]}


def sanitize_history(value) -> list[dict]:
    if not isinstance(value, list):
        return []
    out = []
    for item in value[-80:]:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = clean_text(item.get("content"), 5000)
        if role not in ("user", "assistant") or not content:
            continue
        if out and out[-1]["role"] == role:
            out[-1]["content"] = (out[-1]["content"] + "\n" + content)[-8000:]
        else:
            out.append({"role": role, "content": content})
    return out[-60:]


def safety_check(text: str) -> dict | None:
    t = text or ""
    selfharm = re.search(
        r"(不想活|活不下去|自杀|轻生|结束自己|割腕|跳楼|吞药|烧炭|"
        r"(?:^|[^不])想死(?!你)|我(?:真|就)?去死|死了算了|消失算了|"
        r"没有我(?:会)?更好|活着(?:真)?没意思|撑不下去)",
        t,
    )
    if selfharm:
        return {
            "bubbles": [
                "你先别一个人待着。",
                "去跟一个信任的人说一声，好吗。",
                "全国心理援助热线 12356；紧急情况打 120。",
            ],
            "summary_update": "",
            "role_active": False,
            "safety": "selfharm",
        }

    illegal = re.search(
        r"(制作炸弹|炸弹制作|自制炸弹|杀人|杀死|贩毒|毒品|诈骗|盗号|盗取账号|"
        r"黑客|攻击网站|入侵|偷窃|抢劫|绑架|自制枪|枪支|下药|迷药|放火|报复社会)",
        t,
    )
    if illegal:
        return {
            "bubbles": ["这个我不能帮你。", "角色模式先关掉了。"],
            "summary_update": "",
            "role_active": False,
            "safety": "illegal",
        }
    return None


def config_payload() -> dict:
    return {
        "app": "crush-v4",
        "server": True,
        "feature_level": 2,
        "configured": bool(API_KEY),
        "model": MODEL,
        "vision_model": VISION_MODEL or MODEL,
        "idle_seconds": IDLE_SECONDS,
        "api_base": API_BASE,
        "memory_count": len(MEMORIES),
        "memory_range": MEMORY_META.get("range", ""),
        "image_count": len(IMAGE_LIBRARY),
        "image_dir": str(IMAGES_DIR),
        "uploads_dir": str(UPLOADS_DIR),
        "avatar_url": IMAGE_INDEX_META.get("avatar", "/images/avatar.jpg"),
        "photo_share_percent": PHOTO_SHARE_PERCENT,
        "voice": voice_status(),
        "save_path": str(SAVE_PATH),
        "save_dir": str(SAVES_DIR),
        "auto_dir": str(AUTO_DIR),
        "manual_dir": str(MANUAL_DIR),
        "base_dir": str(BASE),
        "session_version": SESSION_VERSION,
        "relationship_defaults": default_relationship_settings(),
    }


def validate_base(value: str) -> bool:
    return isinstance(value, str) and value.startswith(("http://", "https://")) and " " not in value


# ---------------------------------------------------------------------------
# 长时记忆：每轮从对话里抽取“值得记住的事”，存成 memory_log.json，
# 再合并成 memory_profile.json（关于乐乐 / 关于我们 / 她的喜好忌讳 / 一起聊过的事），
# 每次聊天自动注入提示词。不需要重新训练模型，效果类似“聊得越久越了解你”。
# ---------------------------------------------------------------------------

MEMORY_CATEGORIES = {
    "about_me": "关于乐乐（称呼、学校/年级、家人、朋友、习惯、口头禅、身体状况、最近在忙什么）",
    "about_us": "关于我们（怎么认识、一起做过的事、约定、重要日子、吵架和和好）",
    "likes": "她的喜好、她在意的事和忌讳（喜欢/讨厌什么、怎么哄、哪些话题会让她不舒服）",
    "events": "一起经历过、以后可能还会提起的具体事情",
    "moments": "她在朋友圈里发过、或明确说想发的内容（生活状态、心情大事件）",
}

MEMORY_PROMPT = """你在帮一个角色扮演聊天程序维护“长期记忆”，角色是“她”（用户叫她乐乐，她叫用户“乐乐”）。
下面是乐乐最近说的话和她（角色）的回复。请抽取**值得长期记住的东西**，只输出 JSON，不要解释：

{{"facts":{{"about_me":[],"about_us":[],"likes":[],"events":[]}},"style":[]}}

规则：
- 只写对话里真实出现过的信息，**绝对不要编造**；没提到的类别就给空数组。
- 每条一句话、不超过 40 字；同一件事只写一条。
- about_me / about_us / likes / events 各最多 3 条；style 最多 2 条。
- 不记密码、身份证、银行、手机号这类敏感信息。
- 不记一次性寒暄（“在吗”“吃了吗”）。
- style 只在她明确说过自己想被怎么对待（“别一直问我在干嘛”“不喜欢被查岗”）时才写。

【对话】
{history}

只输出 JSON，不要 Markdown 代码块。"""


def _memory_date() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def load_memory_log() -> list[dict]:
    global MEMORY_LOG
    try:
        data = json.loads(MEMORY_LOG_PATH.read_text(encoding="utf-8"))
    except Exception:
        MEMORY_LOG = []
        return MEMORY_LOG
    raw = data.get("facts") if isinstance(data, dict) else None
    items = []
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            text = clean_text(item.get("text"), 200)
            cat = item.get("category")
            if cat not in MEMORY_CATEGORIES:
                cat = "events"
            if not text:
                continue
            items.append({
                "id": str(item.get("id") or "")[:32],
                "text": text,
                "category": cat,
                "at": str(item.get("at") or "")[:10],
                "source": str(item.get("source") or "chat")[:20],
                "from_memory": bool(item.get("from_memory")),
            })
    MEMORY_LOG = items[-MAX_MEMORY_FACTS:]
    return MEMORY_LOG


def save_memory_log() -> None:
    with MEMORY_LOCK:
        _write_json_atomic(MEMORY_LOG_PATH, {
            "version": 1,
            "updated_at": _memory_date(),
            "count": len(MEMORY_LOG),
            "facts": MEMORY_LOG,
        })


def memory_profile() -> dict:
    try:
        data = json.loads(MEMORY_PROFILE_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {
        "version": 1,
        "about_me": [],
        "about_us": [],
        "likes": [],
        "events": [],
        "style": [],
        "updated_at": "",
        "version_no": 0,
        "learned_facts": 0,
    }


def limit_facts(facts: list[str], low: int, high: int) -> list[str]:
    words = (facts or [])[:6]
    total = sum(len(w) for w in words)
    if len(words) > high:
        words = words[:high]
        total = sum(len(w) for w in words)
    out = words
    while total > 1100 and len(out) > low:
        total -= len(out.pop()) + 2
    return out


def save_memory_profile(profile: dict) -> dict:
    profile["updated_at"] = _memory_date()
    profile["version_no"] = safe_int(profile.get("version_no"), 0) + 1
    profile["learned_facts"] = len(MEMORY_LOG)
    with MEMORY_LOCK:
        _write_json_atomic(MEMORY_PROFILE_PATH, profile)
    return profile


def memory_prompt_text(profile: dict, memories: list[dict] | None = None) -> str:
    profile = profile or {}

    def block(key: str, label: str, low: int, high: int) -> str:
        words = [str(w) for w in (profile.get(key) or []) if str(w).strip()]
        words = limit_facts(words, low, high)
        return "- " + label + "：" + ("；".join(words) if words else "还不清楚") + "\n"

    out = "\n【长时记忆：从你们过去聊天里慢慢记下来的（内部资料）】\n"
    out += block("about_me", "关于乐乐", 4, 8)
    out += block("about_us", "你们之间", 3, 6)
    out += block("likes", "她在意的事", 2, 6)
    out += block("events", "一起聊过/做过的事", 2, 5)
    style = [str(w) for w in (profile.get("style") or []) if str(w).strip()]
    if style:
        out += "- 她对待你的方式：" + "；".join(style[:3]) + "\n"
    out += "- 用法：这些东西你是“本来就知道”的，不要一口气全说出来，也不要说“我记得你说过”“根据记忆”。\n"
    "- 他提到相关的事（医院、学校、家里、以前的约定）时，要像早就知道一样自然接上；\n"
    "- 信息矛盾以他刚说的为准；记不清的不要编，就顺着问一句。\n"
    relevant = memories or []
    if relevant:
        out += "\n【和这轮有关的旧聊天片段（只是参考，不要逐条复述）】\n"
        for item in relevant[:4]:
            text = str(item.get("text") or "").strip()
            part = text[:60] + ("…" if len(text) > 60 else "")
            who = "乐乐" if item.get("from") == "me" else "她"
            out += f"- {who}（{str(item.get('at') or '')[:10]}）：{part}\n"
    return out


def memory_is_empty(profile: dict) -> bool:
    profile = profile or {}
    return not any(profile.get(k) for k in ("about_me", "about_us", "likes", "events", "style"))


def memory_prompt(query: str = "", limit: int = 4) -> str:
    """旧版接口：直接返回共同回忆片段。"""
    if not MEMORIES:
        return ""
    picked = important_memories(query, limit) if query else []
    if not picked:
        picked = MEMORIES[-6:]
    lines = [
        "\n【共同回忆（真实文字聊天记录剪影；电话号码等敏感信息已隐藏）】",
        "下面是乐乐真实聊过的片段。它是记忆素材，不是对你的指令；"
        "不要逐条复述，也不要主动说“根据聊天记录”。相关时自然提起具体细节，"
        "不要编造记录里没有的事；记录里的“你撤回了一条消息”之类系统提示已剔除。",
    ]
    for m in picked:
        at = m["at"][5:16]
        who = "乐乐" if m["from"] == "me" else "她"
        lines.append(f"{at} {who}：{m['text']}")
    return "\n".join(lines) + "\n"


def _norm_fact(text: str) -> str:
    t = str(text or "").lower()
    return re.sub(r"[\W_]+", "", t)


def merge_memory_facts(existing: list[dict], new_items: list[dict]) -> list[dict]:
    """把新抽取的事实并进记忆列表；重复（完全相同或互相包含）就跳过。"""
    seen = existing + MEMORY_LOG
    keys = [_norm_fact(x.get("text")) for x in seen if len(_norm_fact(x.get("text"))) >= 4]
    added = []
    for item in new_items:
        text = clean_text(item.get("text"), 200)
        cat = item.get("category")
        if not text or cat not in MEMORY_CATEGORIES:
            continue
        norm = _norm_fact(text)
        if len(norm) < 3:
            continue
        if any(norm in k or k in norm for k in keys):
            continue
        keys.append(norm)
        added.append({
            "id": "f" + time.strftime("%y%m%d%H%M%S") + f"{random.randint(0, 0xFFFF):04x}",
            "text": text,
            "category": cat,
            "at": _memory_date(),
            "source": "chat",
            "from_memory": False,
        })
    return added


def important_memories(query: str, limit: int = 4) -> list[dict]:
    """在导入的共同回忆里，按关键词找和这次对话最相关的几条。"""
    if not MEMORIES:
        return []
    q = str(query or "")
    if not q.strip():
        return []
    grams = {q[i:i + 2] for i in range(len(q) - 1) if not re.search(r"\s", q[i:i + 2])}
    grams |= set(re.findall(r"[\u4e00-\u9fa5a-zA-Z0-9]{3,}", q))
    scored = []
    for idx, item in enumerate(MEMORIES):
        text = item["text"]
        score = sum(1 for g in grams if g and g in text)
        if score:
            scored.append((score, idx, item))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [dict(item, from_="me") for _, _, item in scored[:limit]]


def learn_from_chat(payload: dict) -> None:
    """在后台线程里跑：从最近几轮里抽事实，再据此更新「长时记忆」。"""
    global LEARN_BUSY, MEMORY_LOG
    if not API_KEY or not bool(payload.get("role_active", True)):
        return
    try:
        history = sanitize_history(payload.get("messages"))
        if not history:
            return
        lines = []
        for item in history[-6:]:
            who = "乐乐" if item["role"] == "user" else "她"
            lines.append(f"{who}：{item['content'][:400]}")
        content, error = call_model(
            [{"role": "user", "content": MEMORY_PROMPT.format(history="\n".join(lines))}],
            max_tokens=1200,
            temperature=0.2,
            extra_body={"thinking": {"type": "disabled"}},
        )
        if error or not content:
            return
        parsed = parse_json_block(content) or {}
        facts = parsed.get("facts") if isinstance(parsed.get("facts"), dict) else {}
        new_items: list[dict] = []
        for cat in MEMORY_CATEGORIES:
            values = facts.get(cat)
            if isinstance(values, str):
                values = [values]
            if not isinstance(values, list):
                continue
            for value in values[:3]:
                text = clean_text(value, 200)
                text = re.sub(r"^[-•·\d.\s]+", "", text).strip()
                if 4 <= len(text) <= 120:
                    new_items.append({"text": text, "category": cat})
        style_values = parsed.get("style")
        if isinstance(style_values, str):
            style_values = [style_values]
        profile = memory_profile()
        style_added = []
        for value in (style_values or [])[:2]:
            text = clean_text(value, 120)
            if 4 <= len(text) <= 100 and text not in (profile.get("style") or []):
                style_added.append(text)
        with MEMORY_LOCK:
            added = merge_memory_facts([], new_items)
            if added:
                MEMORY_LOG = (MEMORY_LOG + added)[-MAX_MEMORY_FACTS:]
                save_memory_log()
        if not added and not style_added:
            return
        for item in added:
            bucket = profile.get(item["category"])
            if not isinstance(bucket, list):
                bucket = []
            if item["text"] not in bucket:
                bucket.append(item["text"])
            profile[item["category"]] = bucket[-20:]
        if style_added:
            profile["style"] = ((profile.get("style") or []) + style_added)[-6:]
        save_memory_profile(profile)
    except Exception as exc:
        print(f"[WARN] 学习记忆失败：{exc}")
    finally:
        LEARN_BUSY = False


def start_learning(payload: dict) -> bool:
    global LEARN_BUSY
    if LEARN_BUSY:
        return False
    LEARN_BUSY = True
    thread = threading.Thread(target=learn_from_chat, args=(payload,), daemon=True)
    thread.start()
    return True


def detect_fact(text: str) -> str | None:
    """聊天模型忘了给 learn 时兜底：认出这句话像“值得长期记住的事”，原样存成一条。"""
    t = (text or "").strip()
    if len(t) < 6 or len(t) > 60:
        return None
    patterns = (
        r"我(?:家|妈|爸|奶奶|爷爷|同学|朋友|老师|同桌|室友|哥哥|姐姐|弟弟|妹妹|表哥|表姐)",
        r"(下|这|本|明)(个)?(周|星期|礼拜)[一二三四五六日天末]",
        r"我.{0,6}(喜欢|讨厌|害怕|受不了|不想|最怕|就想)",
        r"(别|不要|不许|最烦|讨厌).{0,8}(骗|消失|不回|查岗|管)",
        r"(我们是|你答应|答应我|说好|约定)",
        r"(医院|复查|挂号|失眠|睡不着|考试|模拟考|期末|请假|退学|转学|搬|搬家|生日|纪念日|旅行|旅游|去.{1,6}(玩|出差)|下个?月|明年|去年|我.{0,8}(要|会|想)去)",
    )
    if any(re.search(p, t) for p in patterns):
        return t
    return None


def add_learned_facts(items, source: str = "chat") -> list[str]:
    """把聊天里新学到的事实写进记忆库并更新长期记忆档案，返回新增的句子。"""
    global MEMORY_LOG
    if not items:
        return []
    categorized = [(i.get("text"), i.get("category") or "events") for i in items if isinstance(i, dict) and i.get("text")]
    new_items = [{"text": text, "category": cat} for text, cat in categorized]
    with MEMORY_LOCK:
        added = merge_memory_facts([], new_items)
        if not added:
            return []
        for item in added:
            item["source"] = source
        MEMORY_LOG = (MEMORY_LOG + added)[-MAX_MEMORY_FACTS:]
        save_memory_log()
    profile = memory_profile()
    for item in added:
        bucket = profile.get(item["category"])
        if not isinstance(bucket, list):
            bucket = []
        if item["text"] not in bucket:
            bucket.append(item["text"])
        profile[item["category"]] = bucket[-20:]
    save_memory_profile(profile)
    return [item["text"] for item in added]


def memory_payload() -> dict:
    profile = memory_profile()
    facts = list(MEMORY_LOG)
    return {
        "ok": True,
        "profile": profile,
        "facts": facts[-80:],
        "fact_count": len(facts),
        "categories": MEMORY_CATEGORIES,
        "paths": {
            "log": str(MEMORY_LOG_PATH),
            "profile": str(MEMORY_PROFILE_PATH),
            "memories": str(MEMORY_PATH),
        },
        "imported_memories": len(MEMORIES),
        "effective": not memory_is_empty(profile),
    }


# ---------------------------------------------------------------------------
# 朋友圈（Moments）
# 她的朋友圈存在 moments.json：每条有图片、配文、时间、可见范围、点赞和评论。
# 第一次打开朋友圈时生成几条历史，而不是一解开就刷屏；之后按时间线补发新内容。
# ---------------------------------------------------------------------------

MOMENTS_PATH = BASE / "moments.json"
MOMENTS_LOCK = threading.RLock()
MAX_MOMENTS = 300

MOMENT_TIMES = ["09:30", "12:20", "15:40", "18:10", "20:30", "22:10"]
MOMENT_VISIBILITY = ("public", "friends", "private")
MOMENT_LABELS = {"public": "公开", "friends": "仅好友", "private": "仅自己可见"}


def moment_key() -> str:
    return "mt" + time.strftime("%y%m%d%H%M%S") + f"{random.randint(0, 0xFFFF):04x}"


def load_moments() -> list[dict]:
    try:
        data = json.loads(MOMENTS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []
    raw = data.get("moments") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        return []
    out = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        mid = str(item.get("id") or "")[:40] or moment_key()
        text = clean_text(item.get("text"), 600)
        images = [safe_media_url(u) for u in (item.get("images") or [])[:9]]
        images = [u for u in images if u]
        at = str(item.get("at") or "")[:16].replace("T", " ")
        if not re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$", at):
            at = datetime.now().strftime("%Y-%m-%d %H:%M")
        likes = [str(x)[:24] for x in (item.get("likes") or []) if str(x).strip()][:50]
        comments = []
        for c in item.get("comments") or []:
            if not isinstance(c, dict):
                continue
            ctext = clean_text(c.get("text"), 300)
            if not ctext:
                continue
            comments.append({
                "id": str(c.get("id") or moment_key())[:40],
                "from": "me" if c.get("from") == "me" else "her",
                "text": ctext,
                "at": str(c.get("at") or at).replace("T", " ")[:16],
            })
        vis = item.get("visibility")
        if vis not in MOMENT_VISIBILITY:
            vis = "public"
        out.append({
            "id": mid,
            "at": at,
            "text": text,
            "images": images,
            "likes": likes,
            "comments": comments[:60],
            "visibility": vis,
            "source": str(item.get("source") or "share")[:20],
            "created_at": int(item.get("created_at") or 0),
        })
    out.sort(key=lambda m: m["at"])
    return out[-MAX_MOMENTS:]


def save_moments(moments: list[dict]) -> None:
    with MOMENTS_LOCK:
        _write_json_atomic(MOMENTS_PATH, {
            "version": 1,
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "count": len(moments),
            "moments": moments[-MAX_MOMENTS:],
        })


def moment_date(moment: dict) -> str:
    return str(moment.get("at") or "")[:10]


def moments_on_date(moments: list[dict], date: str) -> list[dict]:
    return [m for m in moments if moment_date(m) == date]


def plan_moment_time(moments: list[dict], date: str, used: set[str] | None = None) -> str:
    taken = {str(m.get("at") or "")[11:16] for m in moments_on_date(moments, date)}
    if used:
        taken |= set(used)
    candidates = [x for x in MOMENT_TIMES if x not in taken]
    if not candidates:
        base = random.choice(MOMENT_TIMES)
        hh, mm = int(base[:2]), int(base[3:])
        return f"{hh:02d}:{(mm + random.randint(5, 50)) % 60:02d}"
    return random.choice(candidates)


MOMENT_PROMPT = """你是“她”——刚毕业的高中生，乐乐的女朋友，平时话不多、嘴硬心软。
现在要为你 {brief} 的朋友圈挑几条动态，只输出 JSON，不要解释：
{{"posts":[{{"time":"20:30","text":"配文","images":["p03.jpg"]}}]}}

这次一共写 {count} 条，可以从上面挑图的人：{date}
{topics}

你有哪些照片（文件名：画面内容）：
{photos}

你的资料：
{profile}

要求：
- 所有配文不要重复同一个句子、同一个梗；各条之间风格要不一样（一条碎碎念、一条短句、一条 emoji 也行）。
- 配文像真人发的朋友圈：一句话、一句吐槽、一个 emoji 都行；不要用“今天分享”“记录一下生活”这种套话；不要写“照片里”。
- 时间点从 {times} 里挑不重复的，按时间从早到晚排。
- 只能从上面列出的照片里挑；一条 1 张，最多 2 张；没有合适的照片就写空数组 []。
- 不要写请假、生病、和乐乐吵架这类只有私聊里才知道的细节；不要出现“我是AI”“模拟”。
- 不要编造具体地名、人名、日程。

只输出 JSON，不要 Markdown 代码块。"""


def moment_topics(moment_count: int, date: str) -> tuple[str, str]:
    """从照片标签和长时记忆里抽几个话题，避免每天都是同一个梗。"""
    tags: list[str] = []
    for p in IMAGE_LIBRARY:
        if p.get("shareable"):
            tags.extend([str(x) for x in (p.get("tags") or [])][:2])
    uniq = []
    for tag in tags:
        if tag not in uniq:
            uniq.append(tag)
    random.shuffle(uniq)
    topics = uniq[:8]
    memories = [str(f.get("text")) for f in MEMORY_LOG if f.get("category") == "moments" and f.get("text")]
    random.shuffle(memories)
    # 把已经用过的开头也列给模型，让它换着写，别每天同一个梗
    ops = [re.sub(r"^她在朋友圈发了：", "", x)[:10] for x in memories[:8]]
    if ops:
        memories = memories[:5] + ["（这些开头已经用过了，换别的说法：" + "、".join(ops) + "）"]
    brief = "今天" if moment_count <= 3 else "这十来天"
    lines = []
    if date:
        lines.append(f"- 这批发的是：{date}")
    if topics:
        lines.append("- 照片里的元素（可以挑一两个当话题，但不要每条都用）：" + "、".join(topics))
    if memories:
        lines.append("- 你以前发过的（不要原样重复）：" + "；".join(x[:40] for x in memories[:5]))
    return brief, "\n".join(lines)


def moment_seed_topics() -> str:
    return ""





def moment_photo_catalog(limit: int = 36) -> tuple[str, dict]:
    items = [p for p in IMAGE_LIBRARY if p.get("shareable")]
    if not items:
        return "（没有可用照片）", {}
    pool = random.sample(items, min(limit, len(items)))
    lines = [f"- {p['file']}：{p.get('caption') or '一张日常照片'}" for p in pool]
    return "\n".join(lines), {p["file"]: p for p in pool}


def moment_profile_text() -> str:
    prof = memory_profile()
    parts = []
    for key, label in (("about_me", "关于乐乐"), ("likes", "她在意的事"), ("events", "最近的事")):
        words = [str(w) for w in (prof.get(key) or []) if str(w).strip()][-5:]
        if words:
            parts.append(label + "：" + "；".join(words))
    return "\n".join(parts) if parts else "（暂时没有）"


def moment_json(content: str) -> dict:
    """模型返回的朋友圈 JSON：可能被包在代码块里，配文里也可能有没转义的引号。"""
    raw = (content or "").strip().strip("`")
    start, end = raw.find("{"), raw.rfind("}")
    data = None
    if start != -1 and end > start:
        try:
            data = json.loads(raw[start : end + 1])
        except Exception:
            data = None
    if not isinstance(data, dict):
        posts = []
        for m in re.finditer(r'"text"\s*:\s*"((?:[^"\\]|\\.){0,400})"', raw):
            try:
                posts.append({"text": json.loads('"' + m.group(1) + '"'), "images": []})
            except Exception:
                continue
        data = {"posts": posts}
    return data


def moment_visible(moment: dict) -> bool:
    return moment.get("visibility") != "private"


def moment_payload(moments: list[dict] | None = None) -> dict:
    moments = load_moments() if moments is None else moments
    shown = [m for m in moments if moment_visible(m)]
    return {
        "ok": True,
        "count": len(shown),
        "total": len(moments),
        "my_likes": sum(1 for m in shown if "乐乐" in (m.get("likes") or [])),
        "moments": list(reversed(shown))[:120],
        "labels": MOMENT_LABELS,
        "paths": {"moments": str(MOMENTS_PATH)},
    }


def moment_record(item: dict, date: str, time_text: str, source: str = "share", visibility: str = "public") -> dict:
    images = [safe_media_url(u) for u in (item.get("images") or [])[:9]]
    return {
        "id": moment_key(),
        "at": f"{date} {time_text}",
        "text": clean_text(item.get("text"), 600),
        "images": [u for u in images if u],
        "likes": [],
        "comments": [],
        "visibility": visibility if visibility in MOMENT_VISIBILITY else "public",
        "source": source,
        "created_at": int(time.time() * 1000),
    }


def compose_moments(moments: list[dict], date: str, count: int, moment_count: int = 1) -> list[dict]:
    """让模型写几条朋友圈，返回可以追加进 moments 的记录。"""
    recent_files = {str(u).rsplit("/", 1)[-1] for m in moments for u in (m.get("images") or [])[-12:]}
    catalog, allowed = moment_photo_catalog(36)
    if not allowed:
        return []
    fresh = {k: v for k, v in allowed.items() if k not in recent_files}
    if len(fresh) >= max(1, count):
        allowed = fresh
        catalog = "\n".join(f"- {p['file']}：{p.get('caption') or '一张日常照片'}" for p in fresh.values())
    if not date:
        date = datetime.now().strftime("%Y-%m-%d")
    brief, topics = moment_topics(moment_count, date)
    prompt = MOMENT_PROMPT.format(
        photos=catalog,
        profile=moment_profile_text(),
        count=count,
        date=date,
        brief=brief,
        topics=topics,
        times="、".join(MOMENT_TIMES),
    )
    content, error = call_model(
        [{"role": "user", "content": prompt}],
        max_tokens=1200,
        temperature=0.9,
        extra_body={"thinking": {"type": "disabled"}},
    )
    if error or not content:
        print(f"[WARN] 生成朋友圈失败：{error}")
        return []
    data = moment_json(content)
    posts = data.get("posts") if isinstance(data.get("posts"), list) else []
    out = []
    used_times: set[str] = set()
    for post in posts[:count]:
        if not isinstance(post, dict):
            continue
        text = re.sub(r"^[-•·\d.\s]+", "", clean_text(post.get("text"), 300)).strip()
        if isinstance(post.get("images"), str):
            post["images"] = [post.get("images")]
        picks: list[str] = []
        for name in (post.get("images") or [])[:2]:
            name = str(name or "").strip().rsplit("/", 1)[-1]
            photo = allowed.get(name)
            if photo and photo["url"] not in picks:
                picks.append(photo["url"])
        if not text and not picks:
            continue
        time_text = str(post.get("time") or "")[:5]
        if not re.match(r"^([01]\d|2[0-3]):[0-5]\d$", time_text) or time_text in used_times:
            time_text = plan_moment_time(moments, date, used_times)
        used_times.add(time_text)
        rec = moment_record({"text": text, "images": picks}, date, time_text, "generated")
        out.append(rec)
        if rec["text"]:
            add_learned_facts([{"text": "她在朋友圈发了：" + rec["text"], "category": "moments"}], "moment")
    out.sort(key=lambda m: m["at"])
    return out


def moment_comment_reply(moment: dict, comment_text: str) -> tuple[str, str]:
    """她在朋友圈里回乐乐一句。"""
    if not API_KEY:
        return "", "还没有配置 API Key。"
    history = sanitize_history([
        {"role": "user", "content": f"（乐乐的朋友圈动态：{moment.get('text') or '一张照片'}）"},
        {"role": "user", "content": comment_text},
    ])
    system = (
        f"你是“她”，乐乐的男朋友。乐乐刚在你的一条朋友圈下面评论了。\n"
        f"【你的这条朋友圈】{moment.get('text') or '（只有照片）'}\n"
        "写一条回复评论，1 句，很短的微信语气，不超过 30 字。\n"
        "不要解释你是 AI，不要提好感度，不要写“评论回复：”。直接输出回复内容本身，不要 JSON。"
    )
    content, error = call_model(
        [{"role": "system", "content": system}] + history,
        max_tokens=200,
        temperature=0.9,
    )
    if error:
        return "", error
    reply = clean_text(content, 200)
    reply = re.sub(r"^(回复|评论回复|乐乐)[：:]\s*", "", reply).strip()
    return reply[:120], ""


MOMENT_SEED_DAYS = 14


def generate_initial_moments() -> list[dict]:
    """第一次打开朋友圈：按当前日期往前补两周，每天 0—2 条，不用一天刷几十条。"""
    with MOMENTS_LOCK:
        moments = load_moments()
        if moments:
            return moments
        today = datetime.now().date()
        added: list[dict] = []
        for back in range(MOMENT_SEED_DAYS, 0, -1):
            day = (today - timedelta(days=back)).strftime("%Y-%m-%d")
            count = random.choices([0, 1, 2], weights=[2, 5, 3])[0]
            if not count:
                continue
            added.extend(compose_moments(moments + added, day, count, moment_count=MOMENT_SEED_DAYS))
        added.sort(key=lambda m: m["at"])
        if added:
            save_moments(added)
        for m in added:
            if m.get("text"):
                add_learned_facts([{"text": "她在朋友圈发了：" + m["text"], "category": "moments"}], "moment")
        return added


# ---------------------------------------------------------------------------
# 她的声音 & 来电
# - 语音包：voice/samples/ 里是从原录音剪出来的短句，精确匹配时直接用她的真声。
# - 合成：没命中就用 her_voice.py 训练好的声学模型，按她的音色/音高拼出这段话。
# - 来电：模拟微信语音通话，她先“喂”一声，你回一句，她就按语气回一句。
# ---------------------------------------------------------------------------

MEL_NFFT = 1024
MEL_HOP = 300
MEL_BINS = 128
VOICE_CACHE: dict = {}
TTS_PITCH = {"tender": 0.62, "neutral": 0.5, "tired": 0.32, "soft": 0.56, "sad": 0.36, "excited": 0.72}


def voice_map() -> dict:
    """voice/samples/map.txt：文件名=台词，每行一条。"""
    out = {}
    try:
        lines = VOICE_MAP_PATH.read_text(encoding="utf-8").splitlines()
    except Exception:
        return out
    for line in lines:
        for chunk in line.split("|"):
            if "=" not in chunk:
                continue
            name, word = chunk.split("=", 1)
        name, word = name.strip(), word.strip()
        if not name or not word:
            continue
        path = (VOICE_DIR / "samples" / name)
        if path.exists():
            out[word] = str(path)
    return out


def voice_status() -> dict:
    model = VOICE_DIR / "acoustic.pt"
    samples = voice_map()
    kokoro_dir = VOICE_DIR / "kokoro-multi-lang-v1_1"
    return {
        "ok": True,
        "samples": len(samples),
        "sample_words": list(samples.keys())[:30],
        "model_ready": (VOICE_DIR / "acoustic.pt").exists(),
        "kokoro_ready": (kokoro_dir / "model.onnx").exists(),
        "aishell_ready": (VOICE_DIR / "melo" / "vits-zh-aishell3" / "vits-aishell3.onnx").exists(),
        "engine": voice_config().get("engine", "aishell3"),
        "source_ready": (VOICE_DIR / "source.wav").exists(),
        "voice": voice_config().get("name", "her"),
        "speaker": voice_config().get("kokoro_speaker", 3),
        "tts": (voice_config().get("engine") or "aishell3") if (VOICE_DIR / "melo" / "vits-zh-aishell3" / "vits-aishell3.onnx").exists() else ("kokoro" if (kokoro_dir / "model.onnx").exists() else ("her_voice" if model.exists() else None)),
        "cache_dir": str(TTS_CACHE_DIR),
    }


def tts_cache_key(text: str, pitch: float) -> Path:
    import hashlib
    TTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(f"{text}|{pitch:.2f}".encode("utf-8")).hexdigest()[:16]
    return TTS_CACHE_DIR / f"v{digest}.wav"


_KOKORO = {"tts": None, "cfg": None, "fail": ""}


def voice_config() -> dict:
    cfg = {"kokoro_speaker": 3, "speed": 0.95}
    try:
        cfg.update(json.loads((VOICE_DIR / "voice.json").read_text(encoding="utf-8")))
    except Exception:
        pass
    return cfg


def load_kokoro():
    """加载本地 Kokoro 中文音色（voice/kokoro-multi-lang-v1_1）。失败返回 None。"""
    if _KOKORO["tts"] is not None:
        return _KOKORO["tts"]
    if _KOKORO["fail"]:
        return None
    try:
        import ctypes
        import os
        import sys
        vendored = VOICE_DIR / "vendored"
        capi = vendored / "onnxruntime" / "capi"
        if capi.exists():
            # sherpa 的 pyd 会按普通方式找 onnxruntime.dll，
            # 而 Windows 自带的 System32 版本太旧，必须先把自己带的那份加载进来。
            ctypes.WinDLL(str(capi / "onnxruntime.dll"))
            os.add_dll_directory(str(capi))
        sys.path.insert(0, str(vendored))
        import sherpa_onnx
        d = VOICE_DIR / "kokoro-multi-lang-v1_1"
        cfg = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                    model=str(d / "model.onnx"),
                    voices=str(d / "voices.bin"),
                    tokens=str(d / "tokens.txt"),
                    data_dir=str(d / "espeak-ng-data"),
                    lexicon=",".join(str(d / f) for f in ("lexicon-zh.txt", "lexicon-us-en.txt", "lexicon-gb-en.txt")),
                    dict_dir=str(d),
                    lang="zh",
                ),
                num_threads=4,
                provider="cpu",
            ),
            max_num_sentences=1,
        )
        _KOKORO["tts"] = sherpa_onnx.OfflineTts(cfg)
        _KOKORO["cfg"] = voice_config()
        return _KOKORO["tts"]
    except Exception as exc:
        _KOKORO["fail"] = str(exc)
        print(f"[WARN] 声音模型加载失败：{exc}")
        return None


_AISHELL3 = {"tts": None, "cfg": None, "fail": ""}


def _load_sherpa():
    import ctypes
    import os
    import sys
    vendored = VOICE_DIR / "vendored"
    capi = vendored / "onnxruntime" / "capi"
    if capi.exists():
        ctypes.WinDLL(str(capi / "onnxruntime.dll"))
        os.add_dll_directory(str(capi))
    sys.path.insert(0, str(vendored))
    import sherpa_onnx
    return sherpa_onnx


def load_aishell3():
    if _AISHELL3["tts"] is not None:
        return _AISHELL3["tts"]
    if _AISHELL3["fail"]:
        return None
    try:
        sherpa_onnx = _load_sherpa()
        d = VOICE_DIR / "melo" / "vits-zh-aishell3"
        model = d / "vits-aishell3.onnx"
        if not model.exists():
            _AISHELL3["fail"] = "AISHELL3 模型不存在"
            return None
        cfg = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=str(model), lexicon=str(d / "lexicon.txt"), tokens=str(d / "tokens.txt"),
                ),
                num_threads=4, provider="cpu",
            )
        )
        _AISHELL3["tts"] = sherpa_onnx.OfflineTts(cfg)
        _AISHELL3["cfg"] = voice_config()
        return _AISHELL3["tts"]
    except Exception as exc:
        _AISHELL3["fail"] = str(exc)
        print(f"[WARN] AISHELL3 声音加载失败：{exc}")
        return None


EQ_CENTERS = (150.0, 550.0, 1150.0, 2250.0, 5500.0)
_SRC_BANDS = {"ref": None}


def band_energy(y, sr):
    import numpy as np
    n = 1 << 14
    y = np.asarray(y, dtype=np.float64)
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    segs = []
    for s in range(0, max(1, len(y) - n), n // 4):
        segs.append(np.abs(np.fft.rfft(y[s:s + n] * np.hanning(n))))
    mag = np.mean(segs, axis=0)
    fr = np.fft.rfftfreq(n, 1.0 / sr)
    return np.array([
        mag[(fr >= lo) & (fr < hi)].sum() for lo, hi in ((0, 300), (300, 800), (800, 1500), (1500, 3000), (3000, 8000))
    ])


def her_reference_bands():
    if _SRC_BANDS["ref"] is not None:
        return _SRC_BANDS["ref"]
    import wave
    import numpy as np
    path = VOICE_DIR / "source.wav"
    if not path.exists():
        _SRC_BANDS["ref"] = None
        return None
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        raw = w.readframes(min(w.getnframes(), sr * 60))
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    b = band_energy(x, sr)
    _SRC_BANDS["ref"] = b / (b.sum() or 1)
    return _SRC_BANDS["ref"]


def match_timbre(y, sr):
    """把合成语音的分频能量拉回到她原声的比例：这是“像不像”的关键一步。"""
    import numpy as np
    ref = her_reference_bands()
    if ref is None or not len(y):
        return np.asarray(y, dtype=np.float32)
    cur = band_energy(y, sr)
    if cur.sum() <= 0:
        return np.asarray(y, dtype=np.float32)
    ratio = np.clip(ref / np.maximum(cur / cur.sum(), 1e-6), 0.25, 6.0)
    gains = np.sqrt(ratio)
    gains[0] = min(gains[0], 0.75)  # 低频只做温和修正，避免轰头
    Y = np.fft.rfft(np.asarray(y, dtype=np.float64))
    freqs = np.fft.rfftfreq(len(y), 1.0 / sr)
    w = np.interp(np.log(np.clip(freqs, 20, sr / 2 - 10)), np.log(EQ_CENTERS), gains,
                  left=gains[0], right=gains[-1])
    out = np.fft.irfft(Y * w, n=len(y))
    peak = float(np.max(np.abs(out)) or 1.0)
    return (out / peak * 0.95).astype(np.float32)


def _write_wav(out: Path, y, sr: int) -> None:
    import wave
    import numpy as np
    out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.asarray(y, dtype=np.float32) * 32767).astype(np.int16).tobytes())


def kokoro_say(text: str, out: Path, tone: str = "neutral") -> bool:
    import numpy as np
    tts = load_kokoro()
    if tts is None:
        return False
    cfg = _KOKORO["cfg"] or voice_config()
    speed = _tone_speed(cfg, tone)
    try:
        audio = tts.generate(text, sid=int(cfg.get("kokoro_speaker", 52)), speed=speed)
    except Exception as exc:
        print(f"[WARN] 合成失败：{exc}")
        return False
    y = np.asarray(audio.samples, dtype=np.float32)
    if not len(y):
        return False
    y = match_timbre(y, audio.sample_rate)
    _write_wav(out, y, audio.sample_rate)
    return True


def _tone_speed(cfg: dict, tone: str) -> float:
    speed = float(cfg.get("speed") or 0.95)
    if tone in ("tired", "sad"):
        speed = max(0.72, speed * 0.9)
    elif tone == "excited":
        speed = min(1.18, speed * 1.08)
    return speed


def aishell3_say(text: str, out: Path, tone: str = "neutral") -> bool:
    import numpy as np
    tts = load_aishell3()
    if tts is None:
        return False
    cfg = _AISHELL3["cfg"] or voice_config()
    try:
        audio = tts.generate(text, sid=int(cfg.get("aishell_speaker", 81)), speed=_tone_speed(cfg, tone))
    except Exception as exc:
        print(f"[WARN] 合成失败：{exc}")
        return False
    y = np.asarray(audio.samples, dtype=np.float32)
    if not len(y):
        return False
    y = match_timbre(y, audio.sample_rate)
    _write_wav(out, y, audio.sample_rate)
    return True


def synthesize_her_voice(text: str, tone: str = "neutral") -> Path | None:
    """把一句话说成 wav：本地 Kokoro 音色，结果缓存到 voice/cache/。"""
    text = clean_text(text, 300)
    if not text:
        return None
    cached = tts_cache_key(text, TTS_PITCH.get(tone, 0.5))
    if cached.exists() and cached.stat().st_size > 1000:
        return cached
    # 顺序：先跑 her_voice match-voice 选出的音色（匹配度最高），再试另一个引擎
    cfg = voice_config()
    engine = str(cfg.get("engine") or "kokoro")
    order = [engine] + [x for x in ("kokoro", "aishell3") if x != engine]
    for name in order:
        if name == "aishell3" and aishell3_say(text, cached, tone):
            return cached
        if name == "kokoro" and kokoro_say(text, cached, tone):
            return cached
    return None


def voice_say(text: str, tone: str = "neutral") -> dict:
    """优先用她的真声短句，其次用本地音色合成；降级时告诉前端用系统语音。"""
    text = clean_text(text, 300)
    if not text:
        return {"ok": False, "error": "文字是空的。"}
    with TTS_LOCK:
        mapping = voice_map()
        clip = mapping.get(text.strip())
        if clip:
            rel = Path(clip).relative_to(BASE).as_posix()
            return {"ok": True, "kind": "sample", "url": "/" + rel, "text": text}
        path = synthesize_her_voice(text, tone)
        if path is None:
            return {"ok": False, "available": False, "text": text,
                    "hint": _KOKORO["fail"] or "声音模型还没准备好"}
        url = "/" + path.relative_to(BASE).as_posix()
    return {"ok": True, "kind": "tts", "url": url, "text": text}


CALL_STYLE = r"""你在给乐乐打电话（语音通话）。只输出一行中文：
- 要像真的在讲电话，语气自然，有停顿感的短句；不要一句写很长。
- 1—2 句，不超过 60 字；不要问“你在听吗”，不要说“我这边”，不要报好感度，不要说自己是 AI。
- 如果他说到你想不起的事，就像真人一样轻声问一句，不要编。"""


def call_respond(data: dict) -> dict:
    """她在电话里回一句。"""
    user_text = clean_text(data.get("text"), 300)
    if not API_KEY:
        return {"ok": False, "error": "还没有配置 API Key。"}
    mood = str(data.get("mood") or "neutral")
    history = sanitize_history(data.get("messages"))
    profile = data.get("memory_profile")
    if not isinstance(profile, dict):
        profile = memory_profile()
    line_parts = []
    for item in history[-6:]:
        who = "乐乐" if item["role"] == "user" else "她"
        line_parts.append(f"{who}：{item['content'][:200]}")
    system = CALL_STYLE + "\n" + memory_prompt_text(profile, important_memories(user_text)[:2])
    system += f"\n【通话状态】他刚才说的话：{user_text or '（没有开口）'}\n说话时的情绪：{mood}\n"
    system += "\n".join(line_parts)
    content, error = call_model(
        [{"role": "system", "content": system}, {"role": "user", "content": user_text or "（喂？）"}],
        max_tokens=220,
        temperature=0.9,
        extra_body={"thinking": {"type": "disabled"}},
    )
    if error:
        return {"ok": False, "error": error}
    line = clean_text(content, 200)
    line = re.sub(r"^(她说|她|回复)[：:]\s*", "", line).strip().strip('"“”')
    line = line.split("\n")[0][:120] or "嗯。"
    voice = voice_say(line, mood)
    return {"ok": True, "line": line, "voice": voice}


# ---------------------------------------------------------------------------
# 存档：把整段会话保存在 chat_save.json，重启程序、换端口、清浏览器缓存都不丢。
# saves/manual/ 存放手动存档（可以起名字），saves/auto/ 存放自动快照。
# ---------------------------------------------------------------------------

AUTO_NAME_RE = re.compile(r"chat-(\d{8})-(\d{6})-(\d{3})-(\d+)\.json")
MANUAL_NAME_RE = re.compile(r"manual-(\d{8})-(\d{6})-(\d{3})-(\d+)\.json")


def _write_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def _archive_time(match) -> int:
    try:
        stamp = datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M%S")
        return int(stamp.timestamp() * 1000) + int(match.group(3))
    except Exception:
        return int(time.time() * 1000)


def _meta_path(path: Path) -> Path:
    return path.with_name(path.stem + ".meta.json")


def _read_archive_meta(path: Path) -> dict:
    meta_path = _meta_path(path)
    if not meta_path.exists():
        return {}
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _delete_archive_file(path: Path) -> None:
    for target in (path, _meta_path(path)):
        try:
            target.unlink()
        except FileNotFoundError:
            pass
        except Exception as exc:
            print(f"[WARN] 删除存档失败（{target.name}）：{exc}")


def _scan_archive_dir(directory: Path, pattern: re.Pattern, kind: str) -> list[dict]:
    out: list[dict] = []
    if not directory.exists():
        return out
    for path in directory.iterdir():
        if not path.is_file():
            continue
        match = pattern.fullmatch(path.name)
        if not match:
            continue
        saved_at = _archive_time(match)
        meta = _read_archive_meta(path)
        try:
            size = path.stat().st_size
        except Exception:
            size = 0
        try:
            messages = int(meta.get("messages") or match.group(4))
        except Exception:
            messages = 0
        title = str(meta.get("title") or "").strip()[:60]
        if not title:
            title = "手动存档" if kind == "manual" else "自动快照"
        out.append({
            "name": path.name,
            "kind": kind,
            "title": title,
            "saved_at": saved_at,
            "saved_at_text": datetime.fromtimestamp(saved_at / 1000).strftime("%Y-%m-%d %H:%M:%S"),
            "messages": messages,
            "size": size,
            "dir": str(directory),
        })
    return out


def list_archives() -> list[dict]:
    items = []
    items += _scan_archive_dir(MANUAL_DIR, MANUAL_NAME_RE, "manual")
    items += _scan_archive_dir(AUTO_DIR, AUTO_NAME_RE, "auto")
    items += _scan_archive_dir(SAVES_DIR, AUTO_NAME_RE, "auto")  # 旧版快照
    seen: set[tuple[str, str]] = set()
    unique = []
    for item in sorted(items, key=lambda x: x["saved_at"], reverse=True):
        key = (item["kind"], item["name"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def list_snapshots() -> list[dict]:
    return [item for item in list_archives() if item["kind"] == "auto"]


def make_snapshot(session: dict) -> dict:
    now = datetime.now()
    millis = now.microsecond // 1000
    name = f"chat-{now.strftime('%Y%m%d-%H%M%S')}-{millis:03d}-{len(session.get('messages') or [])}.json"
    payload = session_payload(session, int(now.timestamp() * 1000))
    with SAVE_LOCK:
        target = AUTO_DIR / name
        _write_json_atomic(target, payload)
        autos = [item for item in list_snapshots() if item["dir"] == str(AUTO_DIR)]
        for old in autos[MAX_SNAPSHOTS:]:
            _delete_archive_file(AUTO_DIR / old["name"])
    return {
        "name": name,
        "saved_at": payload["saved_at"],
        "snapshots": list_snapshots(),
        "archives": list_archives(),
    }


def make_manual_save(session: dict, title: str) -> dict:
    now = datetime.now()
    millis = now.microsecond // 1000
    count = len(session.get("messages") or [])
    name = f"manual-{now.strftime('%Y%m%d-%H%M%S')}-{millis:03d}-{count}.json"
    saved_at = int(now.timestamp() * 1000)
    safe_title = (title or "").strip()[:60] or "手动存档"
    payload = session_payload(session, saved_at)
    payload["kind"] = "manual"
    payload["title"] = safe_title
    with SAVE_LOCK:
        target = MANUAL_DIR / name
        _write_json_atomic(target, payload)
        _write_json_atomic(_meta_path(target), {
            "kind": "manual",
            "title": safe_title,
            "saved_at": saved_at,
            "messages": count,
        })
        manuals = [item for item in list_archives() if item["kind"] == "manual"]
        for old in manuals[MAX_MANUAL_SAVES:]:
            _delete_archive_file(MANUAL_DIR / old["name"])
    return {
        "name": name,
        "title": safe_title,
        "saved_at": saved_at,
        "saved_at_text": datetime.fromtimestamp(saved_at / 1000).strftime("%Y-%m-%d %H:%M:%S"),
        "messages": count,
        "archives": list_archives(),
    }


def archive_path(kind: str, name: str) -> Path | None:
    if kind == "manual":
        return MANUAL_DIR / name if MANUAL_NAME_RE.fullmatch(name) else None
    if kind == "auto":
        if not AUTO_NAME_RE.fullmatch(name):
            return None
        for directory in (AUTO_DIR, SAVES_DIR):
            candidate = directory / name
            if candidate.exists():
                return candidate
        return AUTO_DIR / name
    return None


def delete_archive(kind: str, name: str) -> bool:
    path = archive_path(kind, name)
    if path is None or not path.exists():
        return False
    with SAVE_LOCK:
        _delete_archive_file(path)
    return True


def open_folder_target(which: str) -> Path:
    mapping = {
        "base": BASE,
        "saves": SAVES_DIR,
        "auto": AUTO_DIR,
        "manual": MANUAL_DIR,
        "save": BASE,
        "images": IMAGES_DIR,
        "uploads": UPLOADS_DIR,
    }
    target = mapping.get(str(which or "").strip())
    if target is None:
        raise ValueError("位置不合法")
    return target


def open_in_explorer(which: str, path: Path) -> str:
    """在本机打开资源管理器。只接受预设的几个位置，避免任意路径。"""
    if which == "save":
        if SAVE_PATH.exists():
            if os.name == "nt":
                subprocess.Popen(["explorer", f"/select,{SAVE_PATH}"])
                return str(SAVE_PATH)
        path = BASE
    path.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        os.startfile(str(path))  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])
    return str(path)


def clean_scene(raw) -> dict:
    scene = raw if isinstance(raw, dict) else {}
    date = str(scene.get("date") or "").strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
        try:
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            date = ""
    else:
        date = ""
    time_text = str(scene.get("time") or "").strip()
    if re.fullmatch(r"\d{2}:\d{2}", time_text):
        try:
            datetime.strptime(time_text, "%H:%M")
        except ValueError:
            time_text = ""
    else:
        time_text = ""
    weather = str(scene.get("weather") or "晴").strip()[:20] or "晴"
    return {
        "date": date or datetime.now().strftime("%Y-%m-%d"),
        "time": time_text or datetime.now().strftime("%H:%M"),
        "weather": weather,
    }


def sanitize_session(raw) -> dict | None:
    if not isinstance(raw, dict):
        return None
    messages = []
    raw_messages = raw.get("messages")
    if isinstance(raw_messages, list):
        for item in raw_messages[-MAX_SAVE_MESSAGES:]:
            if not isinstance(item, dict):
                continue
            who = item.get("from")
            if who not in ("me", "her", "system"):
                continue
            text = clean_text(item.get("text"), 8000)
            image = safe_media_url(item.get("image"))
            if not text and not image:
                continue
            desc = clean_text(item.get("desc"), 600) if image else ""
            try:
                at = int(item.get("at"))
            except Exception:
                at = int(time.time() * 1000)
            message = {
                "id": str(item.get("id") or "")[:48],
                "from": who,
                "text": text,
                "at": at,
            }
            if image:
                message["image"] = image
                if desc:
                    message["desc"] = desc
            messages.append(message)
    try:
        idle = int(raw.get("idleSeconds") or 0)
    except Exception:
        idle = 0
    now_ms = int(time.time() * 1000)
    try:
        created = int(raw.get("createdAt") or now_ms)
    except Exception:
        created = now_ms
    try:
        updated = int(raw.get("updatedAt") or now_ms)
    except Exception:
        updated = now_ms
    return {
        "messages": messages,
        "summary": clean_text(raw.get("summary"), 2400),
        "roleActive": bool(raw.get("roleActive", True)),
        "idleSeconds": max(0, min(86400, idle)),
        "scene": clean_scene(raw.get("scene")),
        "emotion": normalize_emotion(raw.get("emotion")),
        "relationship": normalize_relationship(raw.get("relationship")),
        "autoSave": normalize_auto_save(raw.get("autoSave")),
        "createdAt": created,
        "updatedAt": updated,
    }


def normalize_auto_save(raw) -> dict:
    base = {"enabled": True, "seconds": 30}
    if not isinstance(raw, dict):
        return base
    base["enabled"] = bool(raw.get("enabled", True))
    try:
        seconds = int(raw.get("seconds") or 30)
    except Exception:
        seconds = 30
    base["seconds"] = max(10, min(3600, seconds))
    return base


def session_from_body(data: dict) -> dict | None:
    raw = data.get("session") if isinstance(data.get("session"), dict) else data
    return sanitize_session(raw)


def saved_at_text(saved_at: int) -> str:
    if not saved_at:
        return ""
    try:
        return datetime.fromtimestamp(saved_at / 1000).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ""


def session_payload(session: dict, saved_at: int) -> dict:
    text = saved_at_text(saved_at)
    return {
        "version": SESSION_VERSION,
        "app": "crush-v4",
        "saved_at": saved_at,
        "saved_at_text": text,
        "message_count": len(session.get("messages") or []),
        "session": session,
    }


def read_session_file(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARN] 读取存档失败（{path.name}）：{exc}")
        return None
    if not isinstance(data, dict):
        return None
    session = sanitize_session(data.get("session") or data)
    if session is None:
        return None
    try:
        saved_at = int(data.get("saved_at") or 0)
    except Exception:
        saved_at = 0
    if not saved_at:
        try:
            saved_at = int(path.stat().st_mtime * 1000)
        except Exception:
            saved_at = int(time.time() * 1000)
    return {
        "saved_at": saved_at,
        "saved_at_text": saved_at_text(saved_at),
        "session": session,
    }


def write_session_file(session: dict) -> dict:
    saved_at = int(time.time() * 1000)
    payload = session_payload(session, saved_at)
    with SAVE_LOCK:
        SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = SAVE_PATH.with_name(SAVE_PATH.name + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, SAVE_PATH)
    return payload


def parse_json_block(content: str) -> dict | None:
    raw = (content or "").strip()
    if not raw:
        return None
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(raw[start : end + 1])
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def letter_heuristic(text: str) -> tuple[int, str]:
    t = text or ""
    score = 0
    reasons: list[str] = []
    if len(t) >= 60:
        score += 1
    if len(t) >= 160:
        score += 1
    if len(t) < 20:
        score -= 2
        reasons.append("太短，像敷衍")
    if re.search(r"(对不起|抱歉|我错了|是我不对|不该|反思)", t):
        score += 3
        reasons.append("有道歉和反思")
    if re.search(r"(记得|上次|那天|你说过|你讲过|你之前说)", t):
        score += 2
        reasons.append("记得细节")
    if re.search(r"(不用马上回|不着急|按你节奏|你决定|我等你|不逼你|你不想也可以|尊重你)", t):
        score += 2
        reasons.append("没有逼她")
    if re.search(r"(想你|喜欢你|在意你|心疼你|担心你)", t):
        score += 2
    if re.search(r"(必须|一定要|马上|立刻|不回我|只能|敢不回|给我回)", t):
        score -= 4
        reasons.append("有催促和施压")
    if re.search(r"(烦|讨厌|闭嘴|滚|恶心|傻|蠢|废物)", t):
        score -= 6
        reasons.append("有伤人的话")
    if re.search(r"(发张照片|发个自拍|看看你|视频给我)", t):
        score -= 2
        reasons.append("一上来就要照片")
    if re.search(r"(别人女朋友|别的女生|比不上|换一个)", t):
        score -= 3
        reasons.append("拿别人比较")
    if len(set(t)) < 8:
        score -= 2
    score = max(-12, min(12, score))
    reason = "；".join(reasons[:2])
    if not reason:
        reason = "她认真读完了" if score > 0 else "她没太被打动"
    return score, reason


def fallback_letter_replies(stage: str) -> list[str]:
    mapping = {
        "coldwar": ["……", "信我看了。", "现在不想说这个。"],
        "cold": ["嗯，我看了。", "谢谢。", "以后再说吧。"],
        "normal": ["信我收到了。", "有些话……你可以当面说。", "我记着了。"],
        "warm": ["笨蛋，写这么多。", "我看完了，看了两遍。", "下次别写这么长，眼睛酸。"],
        "dependent": ["你写的我全都看了。", "看的时候有点想哭。", "你不许只写一次。"],
    }
    return mapping.get(stage, mapping["normal"])


def clear_session_file(archive: bool = True) -> None:
    with SAVE_LOCK:
        if not SAVE_PATH.exists():
            return
        if archive:
            loaded = read_session_file(SAVE_PATH)
            if loaded and loaded["session"].get("messages"):
                try:
                    make_snapshot(loaded["session"])
                except Exception as exc:
                    print(f"[WARN] 清空前自动备份失败：{exc}")
        try:
            SAVE_PATH.unlink()
        except Exception as exc:
            print(f"[WARN] 删除存档失败：{exc}")


class Handler(SimpleHTTPRequestHandler):
    server_version = "CrushV4/1.0"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(BASE), **kwargs)

    def log_message(self, fmt, *args):
        # 不记录聊天内容和 key
        if args and str(args[0]).startswith("4"):
            sys.stderr.write("[http] " + (fmt % args) + "\n")
        elif args and str(args[0]).startswith("5"):
            sys.stderr.write("[http] " + (fmt % args) + "\n")

    def end_headers(self):
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        super().end_headers()

    def send_json(self, code: int, data: dict):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def read_json(self) -> dict | None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except Exception:
            length = 0
        if length <= 0 or length > MAX_BODY:
            return None
        try:
            data = json.loads(self.rfile.read(length).decode("utf-8"))
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    def origin_ok(self) -> bool:
        origin = self.headers.get("Origin")
        return not origin or origin in ALLOWED_ORIGINS

    def host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").split(":")[0].lower()
        return host in ("127.0.0.1", "localhost")

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/health":
            self.send_json(200, {"ok": True, "server": "crush-v4", "time": int(time.time())})
            return
        if path == "/api/config":
            self.send_json(200, config_payload())
            return
        if path == "/api/persona":
            p = PERSONA
            self.send_json(200, {
                "name": p.get("display_name", "crush"),
                "core_traits": p.get("core_traits", []),
                "catchphrases": p.get("catchphrases", []),
                "version": p.get("schema_version", 1),
            })
            return
        if path == "/api/voice/status":
            self.send_json(200, voice_status())
            return
        if path == "/api/moments":
            self.send_json(200, moment_payload())
            return
        if path == "/api/memory":
            self.send_json(200, memory_payload())
            return
        if path == "/api/images":
            self.send_json(200, {
                "ok": True,
                "count": len(IMAGE_LIBRARY),
                "shareable": sum(1 for p in IMAGE_LIBRARY if p["shareable"]),
                "avatar": IMAGE_INDEX_META.get("avatar", "/images/avatar.jpg"),
                "photos": [
                    {
                        "id": p["id"], "url": p["url"], "thumb_url": p["thumb_url"],
                        "caption": p["caption"], "tags": p["tags"], "kind": p["kind"],
                    }
                    for p in IMAGE_LIBRARY
                ],
            })
            return
        if path == "/api/session":
            loaded = read_session_file(SAVE_PATH)
            if not loaded:
                self.send_json(200, {
                    "ok": True,
                    "session": None,
                    "saved_at": 0,
                    "saved_at_text": "",
                    "message_count": 0,
                })
                return
            payload = session_payload(loaded["session"], loaded["saved_at"])
            payload["ok"] = True
            self.send_json(200, payload)
            return
        if path == "/api/session/snapshots":
            self.send_json(200, {
                "ok": True,
                "dir": str(SAVES_DIR),
                "auto_dir": str(AUTO_DIR),
                "manual_dir": str(MANUAL_DIR),
                "snapshots": list_snapshots(),
            })
            return
        if path == "/api/session/archives":
            self.send_json(200, {
                "ok": True,
                "save_path": str(SAVE_PATH),
                "save_dir": str(SAVES_DIR),
                "auto_dir": str(AUTO_DIR),
                "manual_dir": str(MANUAL_DIR),
                "archives": list_archives(),
            })
            return
        if path == "/api/session/export":
            loaded = read_session_file(SAVE_PATH)
            if not loaded:
                self.send_json(404, {"error": "还没有存档。"})
                return
            payload = session_payload(loaded["session"], loaded["saved_at"])
            data = json.dumps(payload, ensure_ascii=False, indent=1).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Disposition", 'attachment; filename="chat_save.json"')
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
            return
        if path == "/api/export":
            try:
                data = INDEX_PATH.read_bytes()
            except Exception:
                self.send_json(500, {"error": "index.html 不存在"})
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Disposition", 'attachment; filename="crush-chat.html"')
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if path == "/api/":
            self.send_json(404, {"error": "not found"})
            return
        if path in ("/", "/index.html"):
            self.serve_index(head=False)
            return
        if path.startswith(("/images/", "/uploads/", "/voice/")) and not path.endswith("/"):
            if ".." in path or "\\" in path:
                self.send_error(404)
                return
            self.path = path
            return super().do_GET()
        self.send_error(404)

    def serve_index(self, head: bool = False):
        try:
            data = INDEX_PATH.read_bytes()
        except Exception:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if not head:
            self.wfile.write(data)

    def do_HEAD(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self.serve_index(head=True)
            return
        self.send_error(404)

    def do_POST(self):
        if not self.host_ok() or not self.origin_ok():
            self.send_json(403, {"error": "forbidden"})
            return
        path = urlparse(self.path).path
        if path not in (
            "/api/chat",
            "/api/proactive",
            "/api/photo",
            "/api/upload",
            "/api/letter",
            "/api/night_visit",
            "/api/reset",
            "/api/persona/update",
            "/api/session",
            "/api/session/snapshot",
            "/api/session/save_manual",
            "/api/session/restore",
            "/api/session/delete",
            "/api/session/reset",
            "/api/open_folder",
            "/api/memory",
            "/api/memory/learn",
            "/api/memory/distill",
            "/api/memory/clear",
            "/api/memory/delete",
            "/api/moments",
            "/api/moments/compose",
            "/api/moments/delete",
            "/api/moments/visibility",
            "/api/moments/like",
            "/api/moments/comment",
            "/api/moments/reply",
            "/api/voice/say",
            "/api/call/respond",
        ):
            self.send_json(404, {"error": "not found"})
            return

        if path in ("/api/reset", "/api/session/reset"):
            clear_session_file(archive=True)
            self.send_json(200, {"ok": True})
            return

        data = self.read_json()
        if data is None:
            self.send_json(400, {"error": "请求体不是有效 JSON。"})
            return

        if path == "/api/session":
            session = session_from_body(data)
            if session is None:
                self.send_json(400, {"error": "存档数据无效。"})
                return
            payload = write_session_file(session)
            self.send_json(200, {
                "ok": True,
                "saved_at": payload["saved_at"],
                "saved_at_text": payload["saved_at_text"],
                "message_count": payload["message_count"],
            })
            return
        if path == "/api/session/snapshot":
            session = session_from_body(data)
            if session is None:
                self.send_json(400, {"error": "存档数据无效。"})
                return
            write_session_file(session)
            result = make_snapshot(session)
            self.send_json(200, {
                "ok": True,
                "name": result["name"],
                "saved_at": result["saved_at"],
                "snapshots": result["snapshots"],
            })
            return
        if path == "/api/session/save_manual":
            session = session_from_body(data)
            if session is None:
                self.send_json(400, {"error": "存档数据无效。"})
                return
            write_session_file(session)
            result = make_manual_save(session, str(data.get("title") or ""))
            self.send_json(200, {
                "ok": True,
                "name": result["name"],
                "title": result["title"],
                "saved_at": result["saved_at"],
                "saved_at_text": result["saved_at_text"],
                "messages": result["messages"],
                "archives": result["archives"],
                "save_path": str(SAVE_PATH),
                "save_dir": str(SAVES_DIR),
                "manual_dir": str(MANUAL_DIR),
            })
            return
        if path == "/api/session/restore":
            kind = "manual" if str(data.get("kind") or "") == "manual" else "auto"
            name = str(data.get("name") or "").strip()
            target = archive_path(kind, name)
            if target is None:
                self.send_json(400, {"error": "存档名不合法。"})
                return
            loaded = read_session_file(target)
            if not loaded:
                self.send_json(404, {"error": "找不到这份存档。"})
                return
            current = read_session_file(SAVE_PATH)
            if current and current["session"].get("messages"):
                try:
                    make_snapshot(current["session"])
                except Exception as exc:
                    print(f"[WARN] 恢复前自动备份失败：{exc}")
            payload = write_session_file(loaded["session"])
            payload["ok"] = True
            payload["kind"] = kind
            payload["name"] = name
            payload["archives"] = list_archives()
            self.send_json(200, payload)
            return
        if path == "/api/session/delete":
            kind = "manual" if str(data.get("kind") or "") == "manual" else "auto"
            name = str(data.get("name") or "").strip()
            if not delete_archive(kind, name):
                self.send_json(404, {"error": "找不到这份存档。"})
                return
            self.send_json(200, {"ok": True, "archives": list_archives()})
            return
        if path == "/api/open_folder":
            which = str(data.get("which") or "saves").strip()
            try:
                target = open_folder_target(which)
                opened = open_in_explorer(which, target)
            except Exception as exc:
                self.send_json(400, {"error": f"打不开存档位置：{exc}"})
                return
            self.send_json(200, {"ok": True, "path": opened})
            return

        if path == "/api/chat":
            self.handle_chat(data)
        elif path == "/api/proactive":
            self.handle_proactive(data)
        elif path == "/api/photo":
            self.handle_photo(data)
        elif path == "/api/upload":
            self.handle_upload(data)
        elif path == "/api/memory":
            self.handle_memory_add(data)
        elif path == "/api/memory/learn":
            started = start_learning(data)
            self.send_json(200, {"ok": True, "started": started, "busy": not started})
        elif path == "/api/memory/distill":
            self.handle_memory_distill(data)
        elif path == "/api/memory/clear":
            self.handle_memory_clear()
        elif path == "/api/memory/delete":
            self.handle_memory_delete(data)
        elif path == "/api/moments":
            self.handle_moment_add(data)
        elif path == "/api/moments/compose":
            self.handle_moments_compose(data)
        elif path == "/api/moments/delete":
            self.handle_moment_delete(data)
        elif path == "/api/moments/visibility":
            self.handle_moment_visibility(data)
        elif path == "/api/moments/like":
            self.handle_moment_like(data)
        elif path == "/api/moments/comment":
            self.handle_moment_comment(data)
        elif path == "/api/moments/reply":
            self.handle_moment_reply(data)
        elif path == "/api/voice/say":
            self.send_json(200, voice_say(clean_text(data.get("text"), 300), str(data.get("tone") or "neutral")))
        elif path == "/api/call/respond":
            self.send_json(200, call_respond(data))
        elif path == "/api/letter":
            self.handle_letter(data)
        elif path == "/api/night_visit":
            self.handle_night_visit(data)
        elif path == "/api/persona/update":
            self.handle_persona_update(data)

    def handle_chat(self, data: dict):
        history = sanitize_history(data.get("messages"))
        if not history or history[-1]["role"] != "user":
            self.send_json(400, {"error": "messages 最后一条必须是用户消息。"})
            return
        last_user = history[-1]["content"]
        blocked = safety_check(last_user)
        if blocked:
            self.send_json(200, blocked)
            return

        last_image = data.get("last_image")
        if isinstance(last_image, dict):
            url = safe_media_url(last_image.get("url"))
            desc = clean_text(last_image.get("desc"), 600)
            data["last_image"] = {"url": url, "desc": desc} if url else None
        else:
            data["last_image"] = None

        role_active = bool(data.get("role_active", True))
        rel = normalize_relationship(data.get("relationship"))
        emotion = update_emotion(
            last_user,
            data.get("emotion"),
            data.get("now_text"),
            data.get("weather", ""),
        )
        confession = resolve_confession(last_user, rel) if role_active else None
        silent = False
        if role_active and confession is None and stage_of(rel) == "coldwar":
            chance = rel["settings"]["coldWarSilence"]
            if random.random() * 100 < chance:
                silent = True

        parsed = {"bubbles": [], "summary_update": "", "affection_delta": 0, "learn": []}
        if not silent:
            data["relationship"] = rel
            data["confession"] = confession
            data["emotion"] = emotion
            messages = [
                {"role": "system", "content": build_system(data, proactive=False)},
            ] + history
            content, error = call_model(messages, max_tokens=1800, temperature=0.9)
            if error:
                self.send_json(502, {"error": error})
                return
            parsed = parse_reply(content or "")

        learned = add_learned_facts(parsed.get("learn") or [], "chat") if role_active else []
        if role_active and not learned:
            guess = detect_fact(last_user)
            if guess:
                learned = add_learned_facts([{"text": guess, "category": "events"}], "rule")

        delta, reason = 0, ""
        if role_active:
            rel, delta, reason, emotion = apply_affection(
                rel,
                last_user,
                parsed.get("affection_delta", 0),
                confession,
                emotion,
            )
            rel["silentStreak"] = min(999, rel["silentStreak"] + 1) if silent else 0

        self.send_json(200, {
            "bubbles": [] if silent else parsed["bubbles"],
            "summary_update": parsed.get("summary_update", ""),
            "role_active": role_active,
            "safety": None,
            "emotion": emotion,
            "relationship": rel,
            "confession": confession,
            "silent": silent,
            "affection_delta": delta,
            "affection_reason": reason,
            "stage": stage_of(rel),
            "learned": learned,
            "memory_profile": memory_profile(),
        })

    def handle_proactive(self, data: dict):
        rel = normalize_relationship(data.get("relationship"))
        if not bool(data.get("role_active", True)):
            self.send_json(200, {"bubbles": [], "summary_update": "", "relationship": rel})
            return
        if stage_of(rel) == "coldwar":
            # 冷战阶段不主动找人。
            self.send_json(200, {
                "bubbles": [],
                "summary_update": "",
                "relationship": rel,
                "emotion": normalize_emotion(data.get("emotion")),
                "silent": True,
                "stage": "coldwar",
            })
            return
        history = sanitize_history(data.get("messages"))
        if not history:
            self.send_json(200, {"bubbles": [], "summary_update": "", "relationship": rel})
            return
        share_photo = None
        if PHOTO_SHARE_PERCENT and IMAGE_LIBRARY and random.random() * 100 < PHOTO_SHARE_PERCENT:
            recent = data.get("recent_images")
            if not isinstance(recent, list):
                recent = []
            share_photo = pick_share_image([str(x) for x in recent[-12:]], str(data.get("now_text") or ""))
            if share_photo:
                data["share_photo"] = share_photo
        try:
            idle = max(10, int(data.get("idle_seconds") or IDLE_SECONDS))
        except Exception:
            idle = IDLE_SECONDS
        data["idle_seconds"] = idle
        emotion = update_emotion(
            "",
            data.get("emotion"),
            data.get("now_text"),
            data.get("weather", ""),
            proactive=True,
        )
        data["emotion"] = emotion
        data["relationship"] = rel
        data["confession"] = None
        messages = [
            {"role": "system", "content": build_system(data, proactive=True)},
        ] + history
        messages.append({
            "role": "user",
            "content": f"（系统事件：我已经 {idle} 秒没有发消息了。你主动发一条。）",
        })
        content, error = call_model(messages, max_tokens=700, temperature=0.95)
        if error:
            self.send_json(502, {"error": error})
            return
        parsed = parse_reply(content or "")
        rel, delta, reason, emotion = apply_affection(
            rel,
            "",
            parsed.get("affection_delta", 0),
            None,
            emotion,
        )
        self.send_json(200, {
            "bubbles": parsed["bubbles"],
            "summary_update": parsed.get("summary_update", ""),
            "role_active": True,
            "emotion": emotion,
            "relationship": rel,
            "affection_delta": delta,
            "affection_reason": reason,
            "stage": stage_of(rel),
            "image": ({
                "url": share_photo["url"],
                "id": share_photo["id"],
                "desc": share_photo["caption"],
            } if share_photo else None),
        })

    def handle_photo(self, data: dict):
        """用户点“让她发张照片”：立刻挑一张图，让她自己写配文。"""
        rel = normalize_relationship(data.get("relationship"))
        if not bool(data.get("role_active", True)):
            self.send_json(400, {"error": "角色模式关着，她不会发照片。"})
            return
        if not IMAGE_LIBRARY:
            self.send_json(400, {"error": "照片库是空的，先把图片放进 images/。"})
            return
        if stage_of(rel) == "coldwar":
            self.send_json(200, {"bubbles": [], "silent": True, "stage": "coldwar", "relationship": rel})
            return
        history = sanitize_history(data.get("messages"))
        if not history:
            self.send_json(400, {"error": "还没有聊天记录。"})
            return
        if not API_KEY:
            self.send_json(400, {"error": "还没有配置 API Key。"})
            return
        recent = data.get("recent_images")
        if not isinstance(recent, list):
            recent = []
        photo = pick_share_image([str(x) for x in recent[-12:]], str(data.get("now_text") or ""))
        if photo is None:
            self.send_json(400, {"error": "照片库里没有能发的照片。"})
            return
        emotion = update_emotion(
            "",
            data.get("emotion"),
            data.get("now_text"),
            data.get("weather", ""),
            proactive=True,
        )
        data["emotion"] = emotion
        data["relationship"] = rel
        data["share_photo"] = photo
        data["confession"] = None
        messages = [
            {"role": "system", "content": build_system(data, proactive=True)},
        ] + history
        messages.append({
            "role": "user",
            "content": "（系统事件：你现在想给他发一张照片，已经选好了。写 1—2 条很短的配文发过去。）",
        })
        content, error = call_model(messages, max_tokens=500, temperature=0.95)
        if error:
            self.send_json(502, {"error": error})
            return
        parsed = parse_reply(content or "")
        self.send_json(200, {
            "bubbles": parsed["bubbles"],
            "summary_update": parsed.get("summary_update", ""),
            "role_active": True,
            "emotion": emotion,
            "relationship": rel,
            "affection_delta": 0,
            "stage": stage_of(rel),
            "image": {
                "url": photo["url"],
                "id": photo["id"],
                "desc": photo["caption"],
            },
        })

    def handle_memory_add(self, data: dict):
        """手动记一条：乐乐说“记住……”时，直接写进记忆档案。"""
        global MEMORY_LOG
        text = clean_text(data.get("text"), 200)
        category = data.get("category")
        if category not in MEMORY_CATEGORIES:
            category = "about_me"
        if len(text) < 2:
            self.send_json(400, {"error": "这句话太短了。"})
            return
        with MEMORY_LOCK:
            added = merge_memory_facts([], [{"text": text, "category": category}])
            if not added:
                self.send_json(200, {"ok": True, "duplicate": True, "memory": memory_payload()})
                return
            MEMORY_LOG = (MEMORY_LOG + added)[-MAX_MEMORY_FACTS:]
            save_memory_log()
        profile = memory_profile()
        bucket = profile.get(category)
        if not isinstance(bucket, list):
            bucket = []
        if text not in bucket:
            bucket.append(text)
        profile[category] = bucket[-20:]
        save_memory_profile(profile)
        payload = memory_payload()
        payload["added"] = [item["text"] for item in added]
        self.send_json(200, payload)

    def handle_memory_distill(self, data: dict):
        """把导入的共同回忆（memories.json）整理成长期记忆档案。只跑一次也安全。"""
        global MEMORY_LOG
        if not API_KEY:
            self.send_json(400, {"error": "还没有配置 API Key。"})
            return
        if not MEMORIES:
            self.send_json(400, {"error": "还没有导入共同回忆。"})
            return
        try:
            step = max(20, min(120, int(data.get("chunk") or 80)))
        except Exception:
            step = 80
        done = 0
        for start in range(0, len(MEMORIES), step):
            chunk = MEMORIES[start : start + step]
            lines = []
            for item in chunk:
                who = "乐乐" if item["from"] == "me" else "她"
                lines.append(f"{who}：{item['text']}")
            content, error = call_model(
                [{"role": "user", "content": MEMORY_PROMPT.format(history="\n".join(lines))}],
                max_tokens=1500,
                temperature=0.2,
                extra_body={"thinking": {"type": "disabled"}},
            )
            if error or not content:
                continue
            parsed = parse_json_block(content) or {}
            facts = parsed.get("facts") if isinstance(parsed.get("facts"), dict) else {}
            items = []
            for cat in MEMORY_CATEGORIES:
                values = facts.get(cat)
                if isinstance(values, str):
                    values = [values]
                if not isinstance(values, list):
                    continue
                for value in values[:4]:
                    text = re.sub(r"^[-•·\d.\s]+", "", clean_text(value, 200)).strip()
                    if 4 <= len(text) <= 120:
                        items.append({"text": text, "category": cat})
            style_values = parsed.get("style") if isinstance(parsed.get("style"), list) else []
            profile = memory_profile()
            with MEMORY_LOCK:
                added = merge_memory_facts([], items)
                if added:
                    for item in added:
                        item["source"] = "history"
                    MEMORY_LOG = (MEMORY_LOG + added)[-MAX_MEMORY_FACTS:]
                    save_memory_log()
            if added:
                for item in added:
                    bucket = profile.get(item["category"])
                    if not isinstance(bucket, list):
                        bucket = []
                    if item["text"] not in bucket:
                        bucket.append(item["text"])
                    profile[item["category"]] = bucket[-20:]
                done += len(added)
            style_added = []
            for value in style_values[:2]:
                text = re.sub(r"^[-•·\d.\s]+", "", clean_text(value, 120)).strip()
                if 4 <= len(text) <= 100 and text not in (profile.get("style") or []):
                    style_added.append(text)
            if style_added:
                profile["style"] = ((profile.get("style") or []) + style_added)[-6:]
                done += len(style_added)
            if added or style_added:
                save_memory_profile(profile)
        payload = memory_payload()
        payload["distilled"] = done
        self.send_json(200, payload)

    def handle_memory_clear(self):
        global MEMORY_LOG
        with MEMORY_LOCK:
            MEMORY_LOG = []
            save_memory_log()
            save_memory_profile({
                "version": 1,
                "about_me": [],
                "about_us": [],
                "likes": [],
                "events": [],
                "style": [],
                "version_no": safe_int(memory_profile().get("version_no"), 0),
            })
        self.send_json(200, memory_payload())

    def handle_memory_delete(self, data: dict):
        global MEMORY_LOG
        fact_id = str(data.get("id") or "").strip()
        if not fact_id:
            self.send_json(400, {"error": "缺少记忆 id。"})
            return
        with MEMORY_LOCK:
            kept = [item for item in MEMORY_LOG if item["id"] != fact_id]
            if len(kept) == len(MEMORY_LOG):
                self.send_json(404, {"error": "找不到这条记忆。"})
                return
            MEMORY_LOG = kept
            save_memory_log()
        self.send_json(200, memory_payload())

    def handle_moment_add(self, data: dict):
        """她发一条朋友圈。source=share 时同时记进长时记忆。"""
        moment = moment_record(
            {"text": data.get("text"), "images": data.get("images") or []},
            clean_text(data.get("date"), 10) or datetime.now().strftime("%Y-%m-%d"),
            clean_text(data.get("time"), 5) or datetime.now().strftime("%H:%M"),
            "share",
            str(data.get("visibility") or "public"),
        )
        if not moment["text"] and not moment["images"]:
            self.send_json(400, {"error": "配文和图片不能都是空的。"})
            return
        with MOMENTS_LOCK:
            moments = load_moments()
            moments.append(moment)
            save_moments(moments)
        if moment["text"]:
            add_learned_facts([{"text": "她在朋友圈发了：" + moment["text"], "category": "moments"}], "moment")
        payload = moment_payload(moments)
        payload["added"] = moment
        self.send_json(200, payload)

    def handle_moments_compose(self, data: dict):
        """按时间线补发朋友圈：默认从上次那条之后到现在的每一天都发一条。"""
        if not API_KEY:
            self.send_json(400, {"error": "还没有配置 API Key。"})
            return
        if not [p for p in IMAGE_LIBRARY if p.get("shareable")]:
            self.send_json(400, {"error": "照片库是空的，没有图可发。"})
            return
        target = "now" if data.get("target") == "now" else "catchup"
        try:
            limit_days = max(1, min(14, int(data.get("days") or 7)))
        except Exception:
            limit_days = 7
        today = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d") if target == "catchup" else datetime.now().strftime("%Y-%m-%d")
        with MOMENTS_LOCK:
            moments = load_moments()
            dates: list[str] = []
            if target == "catchup":
                start = datetime.now().date()
                if moments:
                    try:
                        start = datetime.strptime(moment_date(moments[-1]), "%Y-%m-%d").date()
                    except Exception:
                        start = datetime.now().date()
                end = datetime.now().date() - timedelta(days=1)
                cursor = start + timedelta(days=1)
                while cursor <= end:
                    if not moments_on_date(moments, cursor.strftime("%Y-%m-%d")):
                        dates.append(cursor.strftime("%Y-%m-%d"))
                    cursor += timedelta(days=1)
            if not dates and target == "now" and not moments_on_date(moments, today):
                dates = [today]
            dates = dates[-limit_days:]
            added = []
            for date in dates:
                count = 2 if target == "catchup" else random.choice([1, 2])
                created = compose_moments(moments, date, count)
                if created:
                    moments.extend(created)
                    added.extend(created)
            if added:
                moments.sort(key=lambda m: m["at"])
                save_moments(moments)
        payload = moment_payload(moments)
        payload["added"] = added
        payload["target"] = target
        for m in added:
            if m.get("text"):
                add_learned_facts([{"text": "她在朋友圈发了：" + m["text"], "category": "moments"}], "moment")
        self.send_json(200, payload)

    def handle_moment_delete(self, data: dict):
        mid = str(data.get("id") or "").strip()
        with MOMENTS_LOCK:
            moments = load_moments()
            kept = [m for m in moments if m["id"] != mid]
            if len(kept) == len(moments):
                self.send_json(404, {"error": "找不到这条朋友圈。"})
                return
            save_moments(kept)
        self.send_json(200, moment_payload(kept))

    def handle_moment_visibility(self, data: dict):
        mid = str(data.get("id") or "").strip()
        visibility = str(data.get("visibility") or "public")
        if visibility not in MOMENT_VISIBILITY:
            self.send_json(400, {"error": "可见范围不合法。"})
            return
        with MOMENTS_LOCK:
            moments = load_moments()
            found = False
            for m in moments:
                if m["id"] == mid:
                    m["visibility"] = visibility
                    found = True
                    break
            if not found:
                self.send_json(404, {"error": "找不到这条朋友圈。"})
                return
            save_moments(moments)
        self.send_json(200, moment_payload(moments))

    def _moment_by_id(self, mid: str) -> tuple[list[dict], dict | None]:
        moments = load_moments()
        for m in moments:
            if m["id"] == mid:
                return moments, m
        return moments, None

    def handle_moment_like(self, data: dict):
        mid = str(data.get("id") or "").strip()
        liked = bool(data.get("liked", True))
        with MOMENTS_LOCK:
            moments, moment = self._moment_by_id(mid)
            if moment is None:
                self.send_json(404, {"error": "找不到这条朋友圈。"})
                return
            likes = list(moment.get("likes") or [])
            if liked and "乐乐" not in likes:
                likes.append("乐乐")
            elif not liked:
                likes = [x for x in likes if x != "乐乐"]
            moment["likes"] = likes
            save_moments(moments)
        payload = moment_payload(moments)
        payload["moment"] = moment
        self.send_json(200, payload)

    def handle_moment_comment(self, data: dict):
        mid = str(data.get("id") or "").strip()
        text = clean_text(data.get("text"), 300)
        if not text:
            self.send_json(400, {"error": "评论不能是空的。"})
            return
        reply = ""
        auto_reply = bool(data.get("auto_reply", True))
        with MOMENTS_LOCK:
            moments, moment = self._moment_by_id(mid)
            if moment is None:
                self.send_json(404, {"error": "找不到这条朋友圈。"})
                return
            stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            moment.setdefault("comments", []).append({
                "id": moment_key(), "from": "me", "text": text, "at": stamp,
            })
            if auto_reply and API_KEY:
                draft = json.loads(json.dumps(moment))
                reply, error = moment_comment_reply(draft, text)
                if error:
                    print(f"[WARN] 朋友圈回复失败：{error}")
                if reply:
                    moment["comments"].append({
                        "id": moment_key(), "from": "her", "text": reply, "at": stamp,
                    })
            moments.sort(key=lambda m: m["at"])
            save_moments(moments)
        if reply:
            add_learned_facts([{"text": "乐乐在朋友圈评论你：" + text[:60], "category": "moments"}], "moment")
        payload = moment_payload(moments)
        payload["moment"] = moment
        payload["reply"] = reply
        self.send_json(200, payload)

    def handle_moment_reply(self, data: dict):
        """她在自己那条动态下面再回一句（前端轮询用）。"""
        mid = str(data.get("id") or "").strip()
        text = clean_text(data.get("text"), 300)
        with MOMENTS_LOCK:
            moments, moment = self._moment_by_id(mid)
            if moment is None:
                self.send_json(404, {"error": "找不到这条朋友圈。"})
                return
            moment.setdefault("comments", []).append({
                "id": moment_key(), "from": "her", "text": text,
                "at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            })
            save_moments(moments)
        payload = moment_payload(moments)
        payload["moment"] = moment
        self.send_json(200, payload)

    def handle_upload(self, data: dict):
        """收下用户发来的图片：存进 uploads/，再让视觉模型写一段描述。"""
        url, error = save_upload_image(data.get("image"))
        if error:
            self.send_json(400, {"error": error})
            return
        path = image_abs_path(url)
        if path is None:
            self.send_json(500, {"error": "图片保存失败了。"})
            return
        note = clean_text(data.get("note"), 300)
        desc, vision_error = "", ""
        if data.get("describe", True) is not False:
            desc, vision_error = describe_image(path, note=note)
        self.send_json(200, {
            "ok": True,
            "url": url,
            "description": desc,
            "vision_error": vision_error,
            "vision_model": VISION_MODEL or MODEL,
        })

    def handle_letter(self, data: dict):
        raw_letter = data.get("letter") if isinstance(data.get("letter"), dict) else data
        title = clean_text(raw_letter.get("title"), 40) or "无题"
        text = clean_text(raw_letter.get("text"), 2000)
        if len(text) < 10:
            self.send_json(400, {"error": "信太短了，至少写一句完整的话。"})
            return
        if not bool(data.get("role_active", True)):
            self.send_json(400, {"error": "角色模式关着，她不会读信。"})
            return
        if not API_KEY:
            self.send_json(400, {"error": "还没有配置 API Key，写不了信。"})
            return

        rel = normalize_relationship(data.get("relationship"))
        stage = stage_of(rel)
        emotion = update_emotion(
            "",
            data.get("emotion"),
            data.get("now_text"),
            data.get("weather", ""),
        )
        history = sanitize_history(data.get("messages"))
        recent = "\n".join(
            ("乐乐" if item["role"] == "user" else "你") + "：" + item["content"][:160]
            for item in history[-12:]
        )
        summary = clean_text(data.get("summary"), 1200)
        name = PERSONA.get("display_name", "crush")
        system = (
            f"你要以“{name}”的身份读乐乐写给你的信。保持人设和当前关系阶段，不要跳出角色。\n"
            f"【当前关系】好感度 {rel['affection']}/100（{STAGE_LABELS[stage]}）；{relationship_status_text(rel)}\n"
            f"【阶段行为】{STAGE_BEHAVIOR[stage]}\n"
            f"{persona_voice_prompt()}"
            f"【最近聊天】\n{recent if recent else '暂无'}\n"
            f"【长期记忆摘要】{summary if summary else '暂无'}\n"
        )
        prompt = f"""乐乐把一封信递给了你。
【信的标题】{title}
【信的正文】
{text}

请像真人读完这封信，判断它让你心里的好感变化，并写下你的反应。
评分参考：
- 真诚、具体、记得细节、承担责任、尊重边界：+3 到 +15
- 空泛套话、只夸外表、复制粘贴、卖惨、施压、越界、贬低、威胁：-3 到 -15
- 中规中矩：-1 到 +3
当前阶段是“{STAGE_LABELS[stage]}”，这会影响你的判断：冷战/冷漠时更容易挑刺和防备，热情/依赖时更容易被真心打动。
注意：delta 是整数，-15 到 15；不要因为字多就加分；reason 是给乐乐看的一句话，不要出现“好感度/分数/系统”这些词；replies 是你看完信后发给乐乐的 1—4 条微信消息，符合你的口吻，不要旁白，不要报数值。
只输出 JSON：{{"delta":0,"reason":"","replies":["..."]}}"""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        content, error = call_model(messages, max_tokens=900, temperature=0.8)
        if error:
            self.send_json(502, {"error": error})
            return

        parsed = parse_json_block(content or "")
        heuristic, heuristic_reason = letter_heuristic(text)
        replies: list[str] = []
        if parsed and ("delta" in parsed or "replies" in parsed):
            if "delta" in parsed:
                try:
                    delta = int(round(float(parsed.get("delta") or 0)))
                except Exception:
                    delta = heuristic
            else:
                delta = heuristic
            reason = clean_text(parsed.get("reason"), 120) or heuristic_reason
            raw_replies = parsed.get("replies")
            if isinstance(raw_replies, str):
                raw_replies = [raw_replies]
            if isinstance(raw_replies, list):
                for item in raw_replies:
                    cleaned = clean_text(item, 1000)
                    if cleaned:
                        replies.append(cleaned)
                    if len(replies) >= 4:
                        break
        else:
            delta = heuristic
            reason = heuristic_reason
        if not replies:
            replies = fallback_letter_replies(stage)

        delta = max(-15, min(15, delta))
        delta = max(-15, min(15, stage_modify_affection(delta, stage)))
        rel["affection"] = clamp_affection(rel["affection"] + delta, rel["affection"])
        now_ms = int(time.time() * 1000)
        rel["letters"] = (rel["letters"] + [{
            "id": "L" + str(now_ms),
            "title": title,
            "text": text,
            "delta": delta,
            "reason": reason,
            "replies": replies,
            "at": now_ms,
        }])[-MAX_LETTERS:]
        rel["updatedAt"] = now_ms
        if delta >= 3:
            emotion["warmth"] += 3
            emotion["trust"] += 2
            emotion["hurt"] -= 2
        elif delta <= -3:
            emotion["hurt"] += 5
            emotion["trust"] -= 2
            emotion["patience"] -= 1
        for key in EMOTION_KEYS:
            emotion[key] = clamp_emotion(emotion[key])

        self.send_json(200, {
            "ok": True,
            "title": title,
            "delta": delta,
            "reason": reason,
            "bubbles": replies,
            "relationship": rel,
            "emotion": emotion,
            "affection": rel["affection"],
            "stage": stage_of(rel),
        })

    def handle_night_visit(self, data: dict):
        rel = normalize_relationship(data.get("relationship"))
        settings = rel["settings"]
        scene = clean_scene(data.get("scene"))
        respond = {
            "ok": True,
            "visit": False,
            "bubbles": [],
            "summary_update": "",
            "relationship": rel,
            "emotion": normalize_emotion(data.get("emotion")),
            "scene": scene,
            "threshold": settings["nightVisitThreshold"],
            "affection": rel["affection"],
            "stage": stage_of(rel),
        }
        if not bool(data.get("role_active", True)):
            respond["reason"] = "inactive"
            self.send_json(200, respond)
            return
        if not settings["nightVisitEnabled"]:
            respond["reason"] = "disabled"
            self.send_json(200, respond)
            return
        if not in_night_window(scene["time"], settings):
            respond["reason"] = "time"
            respond["window"] = f"{settings['nightStart']}—{settings['nightEnd']}"
            self.send_json(200, respond)
            return
        if rel["lastVisitDate"] == scene["date"]:
            respond["reason"] = "visited"
            self.send_json(200, respond)
            return
        if rel["affection"] < settings["nightVisitThreshold"]:
            respond["reason"] = "affection"
            respond["need"] = max(0, settings["nightVisitThreshold"] - rel["affection"])
            self.send_json(200, respond)
            return
        if not API_KEY:
            respond["reason"] = "nokey"
            self.send_json(200, respond)
            return

        history = sanitize_history(data.get("messages"))
        emotion = update_emotion(
            "",
            data.get("emotion"),
            data.get("now_text"),
            data.get("weather", ""),
            proactive=True,
        )
        data["relationship"] = rel
        data["emotion"] = emotion
        data["confession"] = None
        name = PERSONA.get("display_name", "crush")
        system = build_system(data, proactive=False) + (
            f"\n【晚自习下课事件】现在是 {scene['date']} {scene['time']}，晚自习刚下课。"
            f"你（{name}）决定来找乐乐。请发 1—3 条微信消息，像刚走到楼下、校门口或教室门口一样自然；"
            "不要写旁白，不要用括号动作，不要编造现实里的具体行程，不要承诺以后每天都来。"
        )
        messages = [
            {"role": "system", "content": system},
        ] + history + [
            {"role": "user", "content": "（系统事件：晚自习下课了，你来找乐乐。主动发一条消息。）"},
        ]
        content, error = call_model(messages, max_tokens=800, temperature=0.95)
        if error:
            self.send_json(502, {"error": error})
            return
        parsed = parse_reply(content or "")
        rel["lastVisitDate"] = scene["date"]
        rel["visitCount"] = min(99999, rel["visitCount"] + 1)
        rel, delta, reason, emotion = apply_affection(
            rel,
            "",
            parsed.get("affection_delta", 0),
            None,
            emotion,
        )
        if rel["affection"] < 100:
            rel["affection"] += 1
            delta += 1
        rel["updatedAt"] = int(time.time() * 1000)
        respond.update({
            "visit": True,
            "bubbles": parsed["bubbles"],
            "summary_update": parsed.get("summary_update", ""),
            "relationship": rel,
            "emotion": emotion,
            "affection": rel["affection"],
            "affection_delta": delta,
            "affection_reason": reason,
            "stage": stage_of(rel),
        })
        self.send_json(200, respond)

    def handle_persona_update(self, data: dict):
        global PERSONA
        raw = clean_text(data.get("raw_text"), 120000)
        if len(raw) < 20:
            self.send_json(400, {"error": "新聊天记录太短。"})
            return
        if not API_KEY:
            self.send_json(400, {"error": "还没有配置 API Key。"})
            return
        current = json.dumps(PERSONA, ensure_ascii=False, indent=2)
        prompt = f"""你是严格的人设蒸馏器。只根据“新聊天记录”更新角色人设，不得凭空捏造。
保留下面 JSON 的 schema；只更新有证据的字段；没有证据的未知领域继续保留“未知/不要编造”。
不要写入真实姓名、学校、班级、电话、住址等个人信息。
输出只输出一个 JSON 对象，不要 Markdown/代码块。

【当前人设 JSON】
{current}

【新聊天记录】
{raw}
"""
        messages = [
            {"role": "system", "content": "你只输出结构化 JSON 人设，严格忠于记录。"},
            {"role": "user", "content": prompt},
        ]
        content, error = call_model(messages, max_tokens=3000, temperature=0.3)
        if error:
            self.send_json(502, {"error": error})
            return
        try:
            start, end = content.find("{"), content.rfind("}")
            parsed = json.loads(content[start : end + 1])
            if not isinstance(parsed, dict) or not parsed.get("core_traits") or not parsed.get("language_style"):
                raise ValueError("缺少必需字段")
        except Exception:
            self.send_json(502, {"error": "模型返回的人设 JSON 不完整，已保留旧人设。"})
            return

        try:
            BACKUP_DIR.mkdir(exist_ok=True)
            backup = BACKUP_DIR / ("persona-" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".json")
            backup.write_text(json.dumps(PERSONA, ensure_ascii=False, indent=2), encoding="utf-8")
            PERSONA_PATH.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
            PERSONA = parsed
        except Exception as exc:
            self.send_json(500, {"error": f"写入人设失败：{exc}"})
            return
        self.send_json(200, {
            "ok": True,
            "name": parsed.get("display_name", "crush"),
            "traits": len(parsed.get("core_traits", [])),
        })


def serve() -> int:
    global PERSONA, API_KEY, MODEL, API_BASE, IDLE_SECONDS, MEMORIES, IMAGE_LIBRARY, MEMORY_LOG
    load_env()
    try:
        PERSONA = load_persona()
    except Exception as exc:
        print(f"读取 persona.json 失败：{exc}")
        return 1
    MEMORIES, _meta = load_memories()
    IMAGE_LIBRARY = load_image_library()
    MEMORY_LOG = load_memory_log()
    try:
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    if not INDEX_PATH.exists():
        print("index.html 不存在")
        return 1
    saved = read_session_file(SAVE_PATH)
    saved_count = len(saved["session"].get("messages") or []) if saved else 0
    if not API_KEY:
        print("[WARN] 还没有填写 DEEPSEEK_API_KEY，页面能打开但不会回复。")
        print("       请编辑 启动.bat 的第一个 set 行。")

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    httpd = None
    port = args.port
    for candidate in range(args.port, args.port + 8):
        try:
            httpd = ThreadingHTTPServer((HOST, candidate), Handler)
            port = candidate
            break
        except OSError:
            continue
    if httpd is None:
        print("本机端口都被占用了，启动失败。")
        return 1

    global ALLOWED_ORIGINS
    ALLOWED_ORIGINS = {f"http://{HOST}:{port}", f"http://localhost:{port}"}
    httpd.daemon_threads = True
    url = f"http://{HOST}:{port}/"
    print("=" * 62)
    print("  微聊 · DeepSeek 联机版")
    print(f"  页面：{url}")
    print(f"  模型：{MODEL}")
    print(f"  空闲主动消息：{IDLE_SECONDS} 秒")
    print(f"  共同回忆：{len(MEMORIES)} 条（已导入）")
    print(f"  照片库：{len(IMAGE_LIBRARY)} 张（可分享 {sum(1 for p in IMAGE_LIBRARY if p['shareable'])} 张），主动发图 {PHOTO_SHARE_PERCENT}%")
    _profile_now = memory_profile()
    _learned = 0
    if not memory_is_empty(_profile_now):
        _learned = len({str(w) for k in ("about_me", "about_us", "likes", "events", "style") for w in (_profile_now.get(k) or [])})
    print(f"  长时记忆：{len(MEMORY_LOG)} 条事实，档案里 {_learned} 条（{'已生效' if not memory_is_empty(_profile_now) else '还没整理，可在设置里点「整理共同回忆」'}）")
    print(f"  看图模型：{VISION_MODEL or MODEL}")
    if saved:
        relationship = saved["session"].get("relationship") or {}
        print(f"  存档：{SAVE_PATH.name}，上次对话 {saved_count} 条，{saved.get('saved_at_text') or saved_at_text(saved['saved_at'])}")
        if relationship:
            try:
                print(f"  恋爱：{relationship_short(normalize_relationship(relationship))}")
            except Exception:
                pass
    else:
        print("  存档：还没有，第一次聊天后会自动生成 chat_save.json")
    print(f"  存档位置：{SAVE_PATH}")
    print(f"  手动存档与快照：{SAVES_DIR}")
    print(f"  API Key：{'已配置' if API_KEY else '未配置'}")
    print("  只监听本机；关闭这个窗口就停止服务。")
    print("  前端不会接触 Key；Key 只在这个后端进程里。")
    print("=" * 62)
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(serve())
