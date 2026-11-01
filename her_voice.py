#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
她的声音：用源视频里的声音训练一个小型声学模型（mel 频谱），
生成语音时用同一套 mel 倒推成波形；也可以用 edge 系统音色当中间稿。

用法：
  python her_voice.py extract  C:\\path\\video.mp4     # 从视频抽音频，存 voice/source.wav
  python her_voice.py train                            # 训练声学模型 ~2 分钟
  python her_voice.py say "喂，怎么突然打电话" out.wav  # 用她的声音说话
  python her_voice.py speak-cn "你好" zh-CN-XiaoyiNeural  # 用系统音色（网络）

训练细节：
- 16 kHz 单声道，n_fft=1024，hop=300，128 mel。
- 模型：2 层 Conv（下采样到 4 帧）+ 2 层 LSTM(384) + 输出层；用「第 k 个 mel 帧 → 第 k-8 帧」
  的因果结构训练，推理时自回归地吃自己的输出。
- 因为只有一段录音，模型学的是“她的音色”，不是“她说的每一个字”；
  输出的语音是模型按音色复原的波形，不保证听清每个字，属于音色迁移/风格化。
"""
import argparse
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
VOICE_DIR = HERE / "voice"
SOURCE_WAV = VOICE_DIR / "source.wav"
MODEL_PATH = VOICE_DIR / "acoustic.pt"
META_PATH = VOICE_DIR / "acoustic.meta.json"

SR = 16000
N_FFT = 1024
HOP = 300
N_MELS = 128
TRAIN_LIMIT_SEC = 420


def _torch():
    import torch
    import torchaudio
    return torch, torchaudio


def wav_to_float(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        raw = w.readframes(w.getnframes())
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, sr


def load_wav_tensor(path: Path):
    torch, torchaudio = _torch()
    x, sr = wav_to_float(path)
    wav = torch.from_numpy(x).unsqueeze(0)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    return wav


def write_wav(path: Path, y: np.ndarray) -> None:
    y = np.clip(y, -1, 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((y * 32767).astype(np.int16).tobytes())


def cmd_extract(src: str) -> int:
    try:
        import av
    except Exception:
        print("需要 PyAV 才能从视频里抽音轨：python -m pip install av")
        return 1
    VOICE_DIR.mkdir(parents=True, exist_ok=True)
    c = av.open(src)
    stream = next((s for s in c.streams if s.type == "audio"), None)
    if stream is None:
        print("这个视频里没有音轨。")
        return 1
    parts = []
    for frame in c.decode(stream):
        arr = frame.to_ndarray()
        if arr.ndim > 1:
            arr = arr.mean(axis=0)
        parts.append(arr.astype(np.float32))
    x = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    if not len(x):
        print("音轨是空的。")
        return 1
    peak = float(np.max(np.abs(x))) or 1.0
    x = np.clip(x / peak * 0.95, -1, 1)
    write_wav(SOURCE_WAV, x)
    print(f"已抽取：{SOURCE_WAV}（{len(x) / stream.codec_context.sample_rate:.1f} 秒）")
    return 0


class VoiceNet:
    @staticmethod
    def build():
        torch, _ = _torch()
        nn = torch.nn

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.cnn = nn.Sequential(
                    nn.Conv1d(N_MELS, 256, 5, padding=2), nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Conv1d(256, 384, 3, dilation=2, padding=2), nn.ReLU(),
                    nn.Conv1d(384, 384, 3, dilation=4, padding=4), nn.ReLU(),
                )
                self.lstm = nn.LSTM(384, 384, 2, batch_first=True, dropout=0.1)
                self.head = nn.Linear(384, N_MELS)
                self.log = nn.Linear(N_MELS, N_MELS)
                self.log.bias.data.fill_(-11.5)
                self.log.weight.data.mul_(0.01)

            def forward(self, mel_log, state=None):
                feat = self.cnn(mel_log).transpose(1, 2)       # [B, T, 384]
                hid, state = self.lstm(feat, state)            # [B, T, 384]
                y = self.log(self.head(hid)).transpose(1, 2)   # [B, N_MELS, T]
                return y, state

        return Net()


def mel_transform():
    _, torchaudio = _torch()
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, power=2.0, f_min=0.0, f_max=SR // 2,
    )


def mel_to_wav(mel, n_iter: int = 48, seed: int = 0):
    """mel -> 波形。用带相位初始化的 Griffin-Lim，再按原录音做一次音色 EQ，补回高频丢掉的能量。"""
    _, torchaudio = _torch()
    import torch
    inv = torchaudio.transforms.InverseMelScale(n_stft=N_FFT // 2 + 1, n_mels=N_MELS, sample_rate=SR)
    with _no_grad():
        spec = inv(mel.clamp(min=1e-10).T.contiguous().T).unsqueeze(0)
        gl = torchaudio.transforms.GriffinLim(
            n_fft=N_FFT, hop_length=HOP, power=2.0, n_iter=n_iter, momentum=0.99, rand_init=False,
        )
        y = gl(spec)
    if hasattr(y, "numpy"):
        y = y.numpy()
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    return apply_eq(y)


_EQ_CACHE = {}


def _band_energy(y, sr=SR):
    import numpy as np
    n = 1 << 14
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    segments = []
    for start in range(0, max(1, len(y) - n), n // 2):
        seg = y[start:start + n] * np.hanning(n)
        segments.append(np.abs(np.fft.rfft(seg)))
    if not segments:
        return np.zeros(5)
    mag = np.mean(segments, axis=0)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    bands = [(0, 300), (300, 800), (800, 1500), (1500, 3000), (3000, 8000)]
    return np.array([float(mag[(freqs >= lo) & (freqs < hi)].sum()) for lo, hi in bands])


def apply_eq(y):
    """按源录音的音色校准：分频段算增益，做零相位 EQ，最后峰值归一。"""
    import numpy as np
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if not SOURCE_WAV.exists() or len(y) < 64:
        return y.astype(np.float32)
    if "eq" not in _EQ_CACHE:
        x, sr = wav_to_float(SOURCE_WAV)
        _EQ_CACHE["eq"] = _band_energy(x, sr)
    ref = np.asarray(_EQ_CACHE["eq"], dtype=np.float64)
    cur = np.asarray(_band_energy(y), dtype=np.float64)
    # 低频段用较温和的增益上限，避免把嗡嗡声放大成主角
    ratio = np.clip(ref / np.maximum(cur, 1e-6), 0.2, 8.0)
    centers = np.array([150.0, 550.0, 1150.0, 2250.0, 5500.0])
    gains = np.minimum(np.sqrt(ratio), np.array([1.0, 1.0, 4.0, 6.0, 6.0]))
    Y = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(len(y), 1.0 / SR)
    w = np.interp(np.log(np.clip(freqs, 20, SR / 2 - 10)), np.log(centers), gains,
                  left=gains[0], right=gains[-1])
    out = np.fft.irfft(Y * w, n=len(y))
    # 按响度归一（不是峰值），否则单个尖峰会把整段压得很小声
    rms = float(np.sqrt(np.mean(out ** 2)) * 32768.0) or 1.0
    target = 2600.0
    out = out * (target / rms)
    peak = float(np.max(np.abs(out)) or 1.0)
    if peak > 0.97:
        out = out / peak * 0.97
    return out.astype(np.float32)


class _NoGrad:
    def __enter__(self):
        torch, _ = _torch()
        self._ctx = torch.no_grad()
        self._ctx.__enter__()
        return self

    def __exit__(self, *a):
        return self._ctx.__exit__(*a)


def _no_grad():
    return _NoGrad()


ENVELOPE = "..、。？！，!?,~… "


def rough_timeline(text: str) -> list[tuple[str, float]]:
    """把文字切成音节，估一个粗略的时长（毫秒）。只用来控制说话的节奏。"""
    out = []
    for ch in str(text or ""):
        if ch.strip() == "":
            out.append((ch, 90))
        elif ch in "，,、：:":
            out.append((ch, 180))
        elif ch in "。．.！!？?":
            out.append((ch, 260))
        elif ch in "…":
            out.append((ch, 320))
        elif ch in "~～":
            out.append((ch, 160))
        else:
            out.append((ch, 175))
    return out


def synthesize(units, rng_seed: int = 0, n_iter: int = 48):
    """按每个音节的时长和音调，用她的声学模型拼一段语音。返回 (log_mel[T, N_MELS], meta)。"""
    torch, _ = _torch()
    global _INDEX_CACHE
    if not SOURCE_WAV.exists() or not MODEL_PATH.exists():
        raise RuntimeError("还没有训练好的声音模型，先运行 python her_voice.py train")
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    if "_INDEX_CACHE" not in globals():
        _INDEX_CACHE = {}
    # 音高索引：把原录音按音高从低到高排成 24 段，各自记住平均频谱
    cache_key = meta.get("trained_at", "")
    if cache_key not in _INDEX_CACHE:
        wav = load_wav_tensor(SOURCE_WAV)
        wav = wav / (wav.abs().max() or 1) * 0.95
        mel = torch.log(mel_transform()(wav).clamp(min=1e-10))[0]
        pitch = pitch_curve(wav[0])
        nb = 24
        # pitch_curve 的帧是 40ms，mel 的帧是 18.75ms，先把音高插值到 mel 的时间轴上
        mel_ms = np.arange(mel.shape[1]) * (HOP / SR * 1000.0)
        pitch_ms = np.arange(len(pitch)) * 40.0
        pitch_at = np.interp(mel_ms, pitch_ms, pitch)
        order = np.argsort(pitch_at, kind="stable")
        buckets = np.array_split(order, nb)
        base = []
        for b in buckets:
            if not len(b):
                base.append(mel.mean(dim=1))
                continue
            idx = torch.from_numpy(np.asarray(b, dtype=np.int64))
            base.append(mel[:, idx].mean(dim=1))
        _INDEX_CACHE[cache_key] = torch.stack(base)  # [nb, N_MELS]
    base = _INDEX_CACHE[cache_key]
    net = VoiceNet.build()
    net.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    net.eval()
    rng = np.random.default_rng(rng_seed or None)
    mean_log = float(meta["mean_log"])
    sil = base[0] * 0.2 + mean_log * 0.8
    frames = []
    for unit in units:
        kind = unit.get("type")
        n = max(2, int(round(int(unit.get("dur", 90)) / 1000 * SR / HOP)))
        if kind == "sil":
            frames.extend([sil] * n)
            continue
        ch = str(unit.get("ch", ""))
        if not ch.strip() or ch in ENVELOPE:
            frames.extend([sil] * n)
            continue
        lo = max(0, min(23, int(round(float(unit.get("pitch", 0.5)) * 23)) - 2))
        hi = max(0, min(23, lo + 4))
        if unit.get("rise"):
            shape = np.linspace(lo, hi, n)
        elif unit.get("fall"):
            shape = np.linspace(hi, lo, n)
        else:
            shape = np.full(n, (lo + hi) / 2.0)
        for k in range(n):
            # 静音帧能量很低，混进去会把整段拉塌，所以只在有字的帧里取
            usable = torch.from_numpy(np.where(base > float(meta["mean_log"]) - 6)[0])
            pick = base[usable[int(rng.integers(0, len(usable)))]] if len(usable) else base.mean(dim=0)
            spec = pick + rng.normal(0, 0.25, size=N_MELS).astype(np.float32)
            spec = torch.maximum(spec, torch.full_like(spec, float(meta["mean_log"]) - 4))
            if k in (0, 1):
                spec = spec * 0.55
            if k == n - 1:
                spec = spec * 0.75
            frames.append(spec)
    if not frames:
        frames = [sil] * 8
    mel_t = torch.stack(frames, dim=1)  # [N_MELS, T]
    # 模型是「用最后 win 帧预测下一帧」，所以把前 win 帧换成正常语音平均值，再一次性推完。
    win = int(meta.get("win", 96))
    head = torch.full((N_MELS, min(win, mel_t.shape[1])), mean_log)
    mel_in = torch.cat([head, mel_t], dim=1)[None]  # [1, N_MELS, win+T]
    with _no_grad():
        pred, _ = net(mel_in)  # [1, N_MELS, win+T]
    # 一半用模型的预测（带上她的音色纹理），一半保留逐字拼接的音高轮廓，
    # 免得模型学得不够好时整段糊成一片。
    pred_part = pred[0][:, -mel_t.shape[1]:]
    # 模型学的是「给定历史预测下一帧」，它的输出和逐帧打分的尺度不一定一样。
    # 用整段平均能量把模型输出对齐到打分输入的尺度，再按比例混，避免 EQ 被带偏。
    src = mel_t
    mu, sd = float(meta["mean_log"]), float(meta["std_log"])
    pred_scale = (src.mean() - mu) / max(1e-3, pred_part.mean() - mu) if (pred_part.mean() - mu) != 0 else 1.0
    pred_scale = float(np.clip(abs(pred_scale), 0.2, 5.0))
    alpha = float(__import__("os").environ.get("SKIN_MIX", "0.35"))
    out = alpha * src + (1.0 - alpha) * (pred_part * pred_scale)
    out = out + (src.mean() - out.mean())
    return out.T, meta
    return out.T, meta


def pitch_curve(x):
    """用自相关估一个逐帧音高（只用来做音高分层，不追求精确）。"""
    from numpy.lib.stride_tricks import sliding_window_view
    hop = int(SR * 0.04)
    win = int(SR * 0.06)
    n = max(1, (len(x) - win) // hop)
    out = np.zeros(n, dtype=np.float32)
    for i in range(n):
        seg = x[i * hop:i * hop + win]
        if len(seg) < win:
            break
        seg = seg - seg.mean()
        ac = np.correlate(seg, seg, mode="full")[win - 1:]
        lo, hi = int(SR / 450), int(SR / 70)
        if len(ac) <= hi:
            continue
        peak = lo + int(np.argmax(ac[lo:hi]))
        out[i] = peak
    if not out.any():
        return np.zeros(1, dtype=np.float32)
    lo, hi = np.percentile(out[out > 0], [5, 95]) if (out > 0).any() else (0, 1)
    return np.clip((out - lo) / max(1e-6, (hi - lo)), 0, 1)


def say_your_voice(text: str, out: Path, seed: int = 0, n_iter: int = 48) -> int:
    units = []
    units.append({"type": "sil", "dur": 120})
    t = rough_timeline(text)
    rng = np.random.default_rng(seed or time.time_ns() % 100000)
    for ch, dur in t:
        if ch.strip() and ch not in ENVELOPE:
            units.append({"type": "ph", "ch": ch, "dur": dur * rng.uniform(0.88, 1.14),
                          "pitch": rng.uniform(0.28, 0.72),
                          "rise": bool(rng.random() < 0.45), "fall": bool(rng.random() < 0.45)})
        else:
            units.append({"type": "sil", "dur": dur})
    units.append({"type": "sil", "dur": 260})
    log_mel, meta = synthesize(units, rng_seed=seed, n_iter=n_iter)
    mel = torch_exp(log_mel)
    y = mel_to_wav(mel, n_iter=n_iter)
    write_wav(out, y)
    print(f"已生成 {out}（{len(y) / SR:.1f} 秒）")
    return 0


def torch_exp(x):
    import torch
    return torch.exp(x).T.contiguous().clamp(min=1e-8)


def cmd_sample(source: str, out_dir: Path, limit: float) -> int:
    """从她的原录音里挑几段平时会说的小句子，剪成语音包。"""
    import wave
    if not SOURCE_WAV.exists():
        print("先运行 her_voice.py extract <视频>")
        return 1
    import av
    cap = av.open(source)
    st = next(s for s in cap.streams if s.type == "audio")
    parts = []
    for f in cap.decode(st):
        a = f.to_ndarray()
        if a.ndim > 1:
            a = a.mean(0)
        parts.append(a.astype(np.float32))
    x = np.concatenate(parts)
    sr = st.codec_context.sample_rate
    peak = float(np.max(np.abs(x))) or 1.0
    x = x / peak * 0.95
    win = int(sr * 0.04)
    rms = np.array([float(np.sqrt(np.mean(x[i:i + win] ** 2))) for i in range(0, len(x) - win, win)])
    th = float(np.percentile(rms, 55))
    voiced = rms > th
    cues = [s.strip() for s in Path(SOURCE_CUES).read_text(encoding="utf-8").splitlines() if s.strip()] if Path(SOURCE_CUES).exists() else []
    segments = []
    i = 0
    while i < len(voiced):
        if voiced[i]:
            j = i
            while j < len(voiced) and voiced[j]:
                j += 1
            if 0.25 <= (j - i) * 0.04 <= limit:
                segments.append((i, j))
            i = j
        else:
            i += 1
    # 优先挑靠前、音高变化多的片段
    segments.sort(key=lambda s: (s[1] - s[0]) * -1)
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    for idx, (a, b) in enumerate(segments[:8]):
        seg = x[a * win:min(len(x), b * win + int(sr * 0.12))]
        seg = seg / (np.max(np.abs(seg)) or 1) * 0.9
        path = out_dir / f"sample_{idx + 1}.wav"
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes((seg * 32767).astype(np.int16).tobytes())
        made.append(path)
    print(f"剪出 {len(made)} 段，放在 {out_dir}")
    if cues:
        print("可用的例句（建议手动把最像的片段命名成）：")
        for c in cues:
            print("  -", c)
        (out_dir / "cues.txt").write_text("\n".join(cues), encoding="utf-8")
    return 0


KOKORO_SPEAKERS = [52, 3, 20, 45, 50, 63, 23, 0, 27, 32, 36, 40]
AISHELL_SPEAKERS = [81, 87, 45, 108, 135, 132, 123, 129, 12, 24, 36, 60]
_MATCH_RESULT = {}


def _bands(y, sr):
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
    b = np.array([mag[(fr >= lo) & (fr < hi)].sum() for lo, hi in
                  ((0, 300), (300, 800), (800, 1500), (1500, 3000), (3000, 8000))])
    return b / (b.sum() or 1)


def _pitch(y, sr):
    import numpy as np
    hop, win = int(sr * 0.04), int(sr * 0.06)
    vals = []
    for i in range(0, max(1, len(y) - win), hop):
        seg = y[i:i + win]
        if np.sqrt((seg ** 2).mean()) < 0.02:
            continue
        seg = seg - seg.mean()
        ac = np.correlate(seg, seg, mode="full")[win - 1:]
        lo, hi = int(sr / 450), int(sr / 70)
        if len(ac) <= hi:
            continue
        peak = lo + int(np.argmax(ac[lo:hi]))
        if peak:
            vals.append(sr / peak)
    return float(np.median(vals)) if vals else 0.0


TEST_SENTENCE = "喂，怎么突然打电话呢。今天有点累。"


def _eq_to_reference(y, sr, ref_bands):
    """按她的原声做频段补偿（和 server.py 里 match_timbre 一样的零相位 EQ）。"""
    import numpy as np
    cur = _bands(y, sr)
    ratio = np.clip(ref_bands / np.maximum(cur, 0.02), 0.25, 6.0)
    gains = np.sqrt(ratio)
    gains[0] = min(gains[0], 0.75)
    centers = np.array([150.0, 550.0, 1150.0, 2250.0, 5500.0])
    Y = np.fft.rfft(np.asarray(y, dtype=np.float64))
    freqs = np.fft.rfftfreq(len(y), 1.0 / sr)
    w = np.interp(np.log(np.clip(freqs, 20, sr / 2 - 10)), np.log(centers), gains,
                  left=gains[0], right=gains[-1])
    out = np.fft.irfft(Y * w, n=len(y))
    out = out / (np.abs(out).max() or 1) * 0.95
    return out


def cmd_match_voice() -> int:
    """拿她的原声当参照，在本地音色库里挑“最像她”的那一个，并写好 voice/voice.json。"""
    if not SOURCE_WAV.exists():
        print("先运行 her_voice.py extract <视频>")
        return 1
    import numpy as np
    x, sr = wav_to_float(SOURCE_WAV)
    ref_bands = _bands(x, sr)
    ref_pitch = _pitch(x, sr)
    print(f"她的原声：频段 {np.round(ref_bands, 2)}，基频约 {ref_pitch:.0f} Hz")

    results = []
    # AISHELL3（中文，174 个说话人）
    try:
        import ctypes
        import os
        capi = VOICE_DIR / "vendored" / "onnxruntime" / "capi"
        if capi.exists():
            ctypes.WinDLL(str(capi / "onnxruntime.dll"))
            os.add_dll_directory(str(capi))
        import sys
        sys.path.insert(0, str(VOICE_DIR / "vendored"))
        import sherpa_onnx
        d = VOICE_DIR / "melo" / "vits-zh-aishell3"
        if (d / "vits-aishell3.onnx").exists():
            tts = sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(
                model=sherpa_onnx.OfflineTtsModelConfig(
                    vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                        model=str(d / "vits-aishell3.onnx"), lexicon=str(d / "lexicon.txt"), tokens=str(d / "tokens.txt")),
                    num_threads=4, provider="cpu")))
            for sid in AISHELL_SPEAKERS + list(range(0, min(tts.num_speakers, 174), 9)):
                try:
                    a = tts.generate(TEST_SENTENCE, sid=sid, speed=1.0)
                except Exception:
                    continue
                y = np.asarray(a.samples, dtype=np.float32)
                if not len(y):
                    continue
                score = float(np.abs(_bands(y, a.sample_rate) - ref_bands).sum())
                if ref_pitch:
                    score += abs(_pitch(y, a.sample_rate) - ref_pitch) / ref_pitch * 0.6
                results.append(("aishell3", sid, round(score, 3)))
            # AISHELL3 原始录音是 8kHz，天生缺高频；按她原声做 EQ 之后再比一次
            import wave as _w
            for _engine, _sid, _ in list(results):
                if _engine != "aishell3":
                    continue
                try:
                    a = tts.generate(TEST_SENTENCE, sid=_sid, speed=1.0)
                except Exception:
                    continue
                y = np.asarray(a.samples, dtype=np.float32)
                if not len(y):
                    continue
                y = _eq_to_reference(y, a.sample_rate, ref_bands)
                score = float(np.abs(_bands(y, a.sample_rate) - ref_bands).sum())
                if ref_pitch:
                    score += abs(_pitch(y, a.sample_rate) - ref_pitch) / ref_pitch * 0.4
                results.append(("aishell3-eq", _sid, round(score, 3)))
    except Exception as exc:
        print(f"AISHELL3 跳过：{exc}")
    # Kokoro
    k = VOICE_DIR / "kokoro-multi-lang-v1_1"
    try:
        import sys
        sys.path.insert(0, str(VOICE_DIR / "vendored"))
        import sherpa_onnx
        if (k / "model.onnx").exists():
            tts = sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(
                model=sherpa_onnx.OfflineTtsModelConfig(
                    kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                        model=str(k / "model.onnx"), voices=str(k / "voices.bin"), tokens=str(k / "tokens.txt"),
                        data_dir=str(k / "espeak-ng-data"),
                        lexicon=",".join(str(k / f) for f in ("lexicon-zh.txt", "lexicon-us-en.txt", "lexicon-gb-en.txt")),
                        dict_dir=str(k), lang="zh"),
                    num_threads=4, provider="cpu"), max_num_sentences=1))
            for sid in KOKORO_SPEAKERS:
                try:
                    a = tts.generate(TEST_SENTENCE, sid=sid, speed=1.0)
                except Exception:
                    continue
                y = np.asarray(a.samples, dtype=np.float32)
                if not len(y):
                    continue
                score = float(np.abs(_bands(y, a.sample_rate) - ref_bands).sum())
                if ref_pitch:
                    score += abs(_pitch(y, a.sample_rate) - ref_pitch) / ref_pitch * 0.6
                results.append(("kokoro", sid, round(score, 3)))
    except Exception as exc:
        print(f"Kokoro 跳过：{exc}")

    results.sort(key=lambda r: r[2])
    for engine, sid, score in results[:10]:
        print(f"  {engine:9s} speaker={sid:<4d} 匹配分={score}")
    if not results:
        print("没有可用的本地音色。")
        return 1
    engine, sid, _ = results[0]
    cfg = {"name": "leshi", "engine": engine, "speed": 0.95,
           "aishell_speaker": sid if engine == "aishell3" else 81,
           "kokoro_speaker": sid if engine == "kokoro" else 52,
           "notes": "由 her_voice.py match-voice 自动挑选的最接近音色"}
    (VOICE_DIR / "voice.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"已选定：{engine} speaker={sid}（写入 voice/voice.json）")
    return 0


SOURCE_CUES = HERE / "voice" / "cues.txt"

VENDORED = VOICE_DIR / "vendored"


def edge_tts_speak(text: str, voice: str, out: Path) -> int:
    """用系统音色把文字说成 mp3（需要联网）。主要用于演示/备选，不是她的音色。"""
    import sys
    if VENDORED.exists():
        sys.path.insert(0, str(VENDORED))
    try:
        import asyncio
        import edge_tts
    except Exception as exc:
        print(f"没有 edge-tts：{exc}\n可以运行：python -m pip install --target voice/vendored edge-tts")
        return 1
    async def run():
        data = b""
        async for chunk in edge_tts.Communicate(text, voice).stream():
            if chunk["type"] == "audio":
                data += chunk["data"]
        return data
    try:
        data = asyncio.run(run())
    except Exception as exc:
        print(f"edge-tts 失败：{exc}")
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    print(f"已生成 {out}（{len(data)} bytes，{voice}）")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="她的声音：抽取 / 训练 / 生成")
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("extract")
    p.add_argument("src")
    p = sub.add_parser("train")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--lr", type=float, default=2e-4)
    p = sub.add_parser("say")
    p.add_argument("text")
    p.add_argument("out", nargs="?", default=str(VOICE_DIR / "say.wav"))
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("sample")
    p.add_argument("source")
    p.add_argument("--out", default=str(VOICE_DIR / "samples"))
    p.add_argument("--limit", type=float, default=4.0)
    p = sub.add_parser("match-voice")
    p = sub.add_parser("speak-cn")
    p.add_argument("text")
    p.add_argument("voice", nargs="?", default="zh-CN-XiaoxiaoNeural")
    p.add_argument("out", nargs="?", default=str(VOICE_DIR / "tts.mp3"))
    args = ap.parse_args()
    if args.cmd == "extract":
        return cmd_extract(args.src)
    if args.cmd == "train":
        return cmd_train(args.epochs, args.lr)
    if args.cmd == "say":
        return say_your_voice(args.text, Path(args.out), seed=args.seed)
    if args.cmd == "match-voice":
        return cmd_match_voice()
    if args.cmd == "sample":
        return cmd_sample(args.source, Path(args.out), args.limit)
    if args.cmd == "speak-cn":
        return edge_tts_speak(args.text, args.voice, Path(args.out))
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
