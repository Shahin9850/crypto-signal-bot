"""
ربات تحلیل و سیگنال‌دهی ارز دیجیتال (نسخه پیشرفته + پیگیری نتیجه سیگنال‌ها)
- سیگنال فیوچرز: تحلیل چندتایم‌فریمی (روند ۴ ساعته → تایید ۱ ساعته → ورود ۱۵ دقیقه)
- سیگنال اسپات: مناسب خرید/نگهداری هفتگی، بر پایه تایم‌فریم روزانه
- پیگیری خودکار: بعد از هر سیگنال فیوچرز، ربات چک می‌کند TP خورده یا SL
- دستورات تلگرام: "وضعیت" (گزارش امروز) و "وضعیت ماهانه" (گزارش ۳۰ روز اخیر)
منبع قیمت: Binance Public Data Mirror (بدون نیاز به API Key)
"""

import os
import json
from datetime import datetime, timedelta, timezone

import requests
import numpy as np
import pandas as pd

# ---------------------------------------------------------------
# تنظیمات کلی
# ---------------------------------------------------------------
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]
RISK_REWARD = 2.0

FUTURES_SCORE_THRESHOLD = 65
SPOT_SCORE_THRESHOLD = 65
SPOT_CHECK_EVERY_HOURS = 6

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

BINANCE_KLINES_URL = "https://data-api.binance.vision/api/v3/klines"
FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=1"

LOG_PATH = "data/signals_log.json"
OFFSET_PATH = "data/telegram_offset.json"
WEIGHTS_PATH = "data/factor_weights.json"

DEFAULT_FACTOR_WEIGHTS = {
    "trend_4h": 30,
    "trend_1h": 20,
    "rsi_1h": 15,
    "macd_1h": 15,
    "entry_15m": 15,
    "sr_proximity": 5,
    "sentiment": 5,
}


# ---------------------------------------------------------------
# ذخیره‌سازی ساده (فایل JSON)
# ---------------------------------------------------------------
def load_json(path, default):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default
    return default


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_factor_weights() -> dict:
    weights = load_json(WEIGHTS_PATH, dict(DEFAULT_FACTOR_WEIGHTS))
    for key, val in DEFAULT_FACTOR_WEIGHTS.items():
        weights.setdefault(key, val)
    return weights


def update_factor_weights(log: list, weights: dict, min_samples: int = 8,
                           step: float = 2.0, lo: float = 5.0, hi: float = 40.0) -> dict:
    """
    یادگیری واقعی و ساده (آماری، نه هوش مصنوعی عمیق):
    برای هر فاکتور تحلیلی (روند، RSI، MACD و...) نرخ موفقیت سیگنال‌هایی که آن فاکتور
    در جهت درست فعال بوده را با میانگین کلی مقایسه می‌کند. اگر فاکتوری مدام همراه با
    برد بوده، وزنش بالا می‌رود؛ اگر همراه با باخت بوده، وزنش کم می‌شود.
    حداقل نمونه لازم است تا از تصمیم‌گیری زودهنگام روی داده کم جلوگیری شود.
    """
    closed = [r for r in log if r.get("status") in ("win", "loss") and "factors" in r]
    if len(closed) < min_samples:
        return weights

    overall_win_rate = sum(1 for r in closed if r["status"] == "win") / len(closed)

    for factor in list(weights.keys()):
        occurrences = [r for r in closed if r.get("factors", {}).get(factor)]
        if len(occurrences) < min_samples:
            continue
        factor_win_rate = sum(1 for r in occurrences if r["status"] == "win") / len(occurrences)
        diff = factor_win_rate - overall_win_rate
        if diff > 0.10:
            weights[factor] = min(weights[factor] + step, hi)
        elif diff < -0.10:
            weights[factor] = max(weights[factor] - step, lo)

    return weights


# ---------------------------------------------------------------
# دریافت داده قیمتی
# ---------------------------------------------------------------
def fetch_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    resp = requests.get(BINANCE_KLINES_URL, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_base", "taker_quote", "ignore"
    ])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df


# ---------------------------------------------------------------
# اندیکاتورهای تکنیکال
# ---------------------------------------------------------------
def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series):
    ema12 = ema(series, 12)
    ema26 = ema(series, 26)
    macd_line = ema12 - ema26
    signal_line = ema(macd_line, 9)
    return macd_line, signal_line


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    df["ema20"] = ema(close, 20)
    df["ema50"] = ema(close, 50)
    df["rsi"] = rsi(close)
    macd_line, signal_line = macd(close)
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["atr"] = atr(df)
    return df


def find_support_resistance(df: pd.DataFrame, lookback: int = 30):
    recent = df.tail(lookback)
    return recent["low"].min(), recent["high"].max()


def get_fear_greed_index():
    try:
        resp = requests.get(FEAR_GREED_URL, timeout=10)
        resp.raise_for_status()
        data = resp.json()["data"][0]
        return int(data["value"]), data["value_classification"]
    except Exception:
        return None, None


# ---------------------------------------------------------------
# تحلیل فیوچرز: چندتایم‌فریمی
# ---------------------------------------------------------------
def analyze_futures(symbol: str, fng_value, score_threshold: float = FUTURES_SCORE_THRESHOLD,
                     weights: dict = None):
    if weights is None:
        weights = DEFAULT_FACTOR_WEIGHTS

    df4h = add_indicators(fetch_klines(symbol, "4h", 100))
    df1h = add_indicators(fetch_klines(symbol, "1h", 200))
    df15m = add_indicators(fetch_klines(symbol, "15m", 200))

    last4h, last1h, last15m = df4h.iloc[-1], df1h.iloc[-1], df15m.iloc[-1]
    price = last15m["close"]
    support, resistance = find_support_resistance(df1h, 30)

    bull, bear = 0.0, 0.0
    reasons = []
    bull_factors, bear_factors = {}, {}

    if last4h["ema20"] > last4h["ema50"]:
        bull += weights["trend_4h"]
        bull_factors["trend_4h"] = 1
        reasons.append("روند اصلی (۴ساعته) صعودی")
    elif last4h["ema20"] < last4h["ema50"]:
        bear += weights["trend_4h"]
        bear_factors["trend_4h"] = 1
        reasons.append("روند اصلی (۴ساعته) نزولی")

    if last1h["ema20"] > last1h["ema50"]:
        bull += weights["trend_1h"]
        bull_factors["trend_1h"] = 1
        reasons.append("روند میان‌مدت (۱ساعته) صعودی")
    elif last1h["ema20"] < last1h["ema50"]:
        bear += weights["trend_1h"]
        bear_factors["trend_1h"] = 1
        reasons.append("روند میان‌مدت (۱ساعته) نزولی")

    if last1h["rsi"] < 35:
        bull += weights["rsi_1h"]
        bull_factors["rsi_1h"] = 1
        reasons.append(f"RSI یک‌ساعته در ناحیه اشباع فروش ({last1h['rsi']:.1f})")
    elif last1h["rsi"] > 65:
        bear += weights["rsi_1h"]
        bear_factors["rsi_1h"] = 1
        reasons.append(f"RSI یک‌ساعته در ناحیه اشباع خرید ({last1h['rsi']:.1f})")

    if last1h["macd"] > last1h["macd_signal"]:
        bull += weights["macd_1h"]
        bull_factors["macd_1h"] = 1
        reasons.append("MACD یک‌ساعته مثبت")
    elif last1h["macd"] < last1h["macd_signal"]:
        bear += weights["macd_1h"]
        bear_factors["macd_1h"] = 1
        reasons.append("MACD یک‌ساعته منفی")

    if last15m["close"] > last15m["ema20"] and last15m["rsi"] > df15m.iloc[-2]["rsi"]:
        bull += weights["entry_15m"]
        bull_factors["entry_15m"] = 1
        reasons.append("قیمت بالای EMA20 در ۱۵ دقیقه و مومنتوم رو به رشد")
    elif last15m["close"] < last15m["ema20"] and last15m["rsi"] < df15m.iloc[-2]["rsi"]:
        bear += weights["entry_15m"]
        bear_factors["entry_15m"] = 1
        reasons.append("قیمت زیر EMA20 در ۱۵ دقیقه و مومنتوم رو به کاهش")

    if (price - support) / price < 0.01:
        bull += weights["sr_proximity"]
        bull_factors["sr_proximity"] = 1
        reasons.append("قیمت نزدیک حمایت کلیدی")
    elif (resistance - price) / price < 0.01:
        bear += weights["sr_proximity"]
        bear_factors["sr_proximity"] = 1
        reasons.append("قیمت نزدیک مقاومت کلیدی")

    if fng_value is not None:
        if fng_value <= 20:
            bull += weights["sentiment"]
            bull_factors["sentiment"] = 1
            reasons.append(f"ترس شدید در بازار ({fng_value})")
        elif fng_value >= 80:
            bear += weights["sentiment"]
            bear_factors["sentiment"] = 1
            reasons.append(f"طمع شدید در بازار ({fng_value})")

    confidence = max(bull, bear)
    direction = "BUY" if bull > bear else "SELL" if bear > bull else "NEUTRAL"

    if confidence < score_threshold:
        return None

    atr15 = last15m["atr"]
    if direction == "BUY":
        sl = price - 1.2 * atr15
        tp = price + 1.2 * atr15 * RISK_REWARD
    else:
        sl = price + 1.2 * atr15
        tp = price - 1.2 * atr15 * RISK_REWARD

    winning_factors = bull_factors if direction == "BUY" else bear_factors

    return {
        "symbol": symbol,
        "price": price,
        "direction": direction,
        "confidence": round(confidence),
        "reasons": reasons,
        "sl": sl,
        "tp": tp,
        "support": support,
        "resistance": resistance,
        "factors": winning_factors,
    }


# ---------------------------------------------------------------
# تحلیل اسپات
# ---------------------------------------------------------------
def analyze_spot(symbol: str, fng_value):
    df1d = add_indicators(fetch_klines(symbol, "1d", 120))
    last = df1d.iloc[-1]
    price = last["close"]
    support, resistance = find_support_resistance(df1d, 30)

    score = 0.0
    reasons = []

    if price > last["ema50"] and last["ema20"] > last["ema50"]:
        score += 35
        reasons.append("روند بلندمدت (روزانه) صعودی و قیمت بالای EMA50")
    elif price > last["ema50"]:
        score += 20
        reasons.append("قیمت بالای EMA50 روزانه (روند نسبتاً مثبت)")

    if last["rsi"] < 40:
        score += 20
        reasons.append(f"RSI روزانه در محدوده مناسب خرید ({last['rsi']:.1f})")
    elif last["rsi"] < 55:
        score += 10
        reasons.append(f"RSI روزانه خنثی ({last['rsi']:.1f})")

    dist_support = (price - support) / price
    if dist_support < 0.05:
        score += 20
        reasons.append("قیمت نزدیک حمایت ۳۰ روزه")
    elif dist_support < 0.10:
        score += 10
        reasons.append("قیمت در فاصله معقول از حمایت ۳۰ روزه")

    if last["macd"] > last["macd_signal"]:
        score += 15
        reasons.append("MACD روزانه مثبت")

    if fng_value is not None:
        if fng_value <= 35:
            score += 10
            reasons.append(f"ترس در بازار ({fng_value}) → فرصت خرید بلندمدت")
        elif fng_value <= 55:
            score += 5

    if score < SPOT_SCORE_THRESHOLD:
        return None

    return {
        "symbol": symbol,
        "price": price,
        "score": round(score),
        "reasons": reasons,
        "target": resistance,
        "invalidation": support,
    }


# ---------------------------------------------------------------
# پیگیری نتیجه سیگنال‌های قبلی (رسیدن به TP یا خوردن SL)
# ---------------------------------------------------------------
def check_open_signals(log: list) -> list:
    now_iso = datetime.now(timezone.utc).isoformat()
    for record in log:
        if record.get("status") != "open":
            continue
        try:
            entry_dt = datetime.fromisoformat(record["timestamp"])
            entry_ms = int(entry_dt.timestamp() * 1000)
            df = fetch_klines(record["symbol"], "15m", 500)
            df = df[df["open_time"] >= entry_ms]
            for _, candle in df.iterrows():
                if record["direction"] == "BUY":
                    if candle["low"] <= record["sl"]:
                        record["status"] = "loss"
                        record["closed_time"] = now_iso
                        break
                    if candle["high"] >= record["tp"]:
                        record["status"] = "win"
                        record["closed_time"] = now_iso
                        break
                else:  # SELL
                    if candle["high"] >= record["sl"]:
                        record["status"] = "loss"
                        record["closed_time"] = now_iso
                        break
                    if candle["low"] <= record["tp"]:
                        record["status"] = "win"
                        record["closed_time"] = now_iso
                        break
        except Exception as e:
            print(f"خطا در بررسی وضعیت سیگنال {record.get('symbol')}: {e}")
    return log


def build_status_summary(log: list, period: str, current_threshold: float = None) -> str:
    now = datetime.now(timezone.utc)
    if period == "day":
        cutoff = now.replace(hour=0, minute=0, second=0, microsecond=0)
        label = "امروز"
        relevant = [r for r in log if datetime.fromisoformat(r["timestamp"]) >= cutoff]
    elif period == "month":
        cutoff = now - timedelta(days=30)
        label = "۳۰ روز اخیر (ماهانه)"
        relevant = [r for r in log if datetime.fromisoformat(r["timestamp"]) >= cutoff]
    else:  # "last20"
        label = "۲۰ سیگنال اخیر"
        relevant = sorted(log, key=lambda r: r["timestamp"])[-20:]

    wins = sum(1 for r in relevant if r["status"] == "win")
    losses = sum(1 for r in relevant if r["status"] == "loss")
    still_open = sum(1 for r in relevant if r["status"] == "open")
    closed = wins + losses
    winrate = (wins / closed * 100) if closed else 0

    text = (
        f"📊 <b>وضعیت سیگنال‌ها | {label}</b>\n\n"
        f"✅ سود گرفته (TP): {wins}\n"
        f"🛑 استاپ خورده (SL): {losses}\n"
        f"⏳ هنوز باز (مشخص نشده): {still_open}\n"
        f"📈 تعداد کل سیگنال: {len(relevant)}\n"
        f"🎯 نرخ موفقیت (از بسته‌شده‌ها): {winrate:.1f}%"
    )
    if current_threshold is not None:
        text += f"\n\n🛡️ آستانه فعلی پذیرش سیگنال: {current_threshold:.0f} / 100 (خودکار بر اساس عملکرد اخیر تنظیم می‌شود)"
    return text


def compute_adaptive_threshold(log: list, base: float = 65, min_threshold: float = 60,
                                max_threshold: float = 85, lookback: int = 20) -> float:
    """
    مکانیزم خودتنظیم (نه هوش مصنوعی، بلکه یک قانون ساده و شفاف):
    نرخ موفقیت N سیگنال بسته‌شده اخیر را حساب می‌کند و بر اساس آن آستانه پذیرش سیگنال را
    بالا/پایین می‌برد. وقتی عملکرد ضعیف بوده، ربات سخت‌گیرتر می‌شود (کیفیت را فدای کمیت نمی‌کند).
    """
    closed = [r for r in log if r.get("status") in ("win", "loss")]
    closed = sorted(closed, key=lambda r: r["timestamp"])[-lookback:]
    if len(closed) < 5:
        return base  # هنوز داده کافی برای قضاوت نیست

    wins = sum(1 for r in closed if r["status"] == "win")
    win_rate = wins / len(closed)

    if win_rate < 0.40:
        return min(base + 15, max_threshold)
    if win_rate < 0.50:
        return min(base + 8, max_threshold)
    if win_rate > 0.65:
        return max(base - 5, min_threshold)
    return base


# ---------------------------------------------------------------
# مدیریت دستورات تلگرام (وضعیت / وضعیت ماهانه)
# ---------------------------------------------------------------
def get_telegram_updates(offset: int):
    if not TELEGRAM_TOKEN:
        return []
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
    params = {"offset": offset, "timeout": 0}
    resp = requests.get(url, params=params, timeout=15)
    if resp.status_code != 200:
        return []
    return resp.json().get("result", [])


def process_commands(log: list, current_threshold: float = None):
    offset_data = load_json(OFFSET_PATH, {"offset": 0})
    updates = get_telegram_updates(offset_data["offset"])

    for update in updates:
        offset_data["offset"] = update["update_id"] + 1
        message = update.get("message", {})
        text = (message.get("text") or "").strip()
        if not text:
            continue
        if "ماهانه" in text:
            send_telegram_message(build_status_summary(log, "month", current_threshold))
        elif "وضعیت" in text:
            send_telegram_message(build_status_summary(log, "day", current_threshold))

    save_json(OFFSET_PATH, offset_data)


# ---------------------------------------------------------------
# ارسال پیام به تلگرام
# ---------------------------------------------------------------
def send_telegram_message(text: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("توکن یا چت آیدی تلگرام تنظیم نشده است.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
    resp = requests.post(url, data=payload, timeout=15)
    if resp.status_code != 200:
        print("خطا در ارسال به تلگرام:", resp.text)


def format_futures_message(r, fng_value, fng_label) -> str:
    direction_fa = "خرید (BUY) 🟢" if r["direction"] == "BUY" else "فروش (SELL) 🔴"
    lines = [
        f"<b>⚡️ سیگنال فیوچرز | {r['symbol']}</b>",
        f"امتیاز اطمینان: <b>{r['confidence']} / 100</b>",
        f"جهت: <b>{direction_fa}</b>",
        f"قیمت ورود: {r['price']:.4f}",
        f"🎯 حد سود: {r['tp']:.4f}",
        f"🛑 حد ضرر: {r['sl']:.4f}",
        f"📈 مقاومت نزدیک: {r['resistance']:.4f}",
        f"📉 حمایت نزدیک: {r['support']:.4f}",
        "",
        "<i>دلایل تحلیل (تایم‌فریم ۴ساعته → ۱ساعته → ۱۵دقیقه):</i>",
    ]
    lines += [f"• {x}" for x in r["reasons"]]
    if fng_value is not None:
        lines.append(f"\n😨/🤑 شاخص ترس و طمع: {fng_value} ({fng_label})")
    lines.append(
        "\n📌 این سیگنال ثبت شد و ربات خودش نتیجه‌اش رو پیگیری می‌کنه. "
        "برای دیدن گزارش، بنویس «وضعیت» یا «وضعیت ماهانه»."
    )
    lines.append("\n⚠️ صرفاً تحلیل خودکار است، نه توصیه مالی قطعی. مدیریت ریسک را رعایت کنید.")
    return "\n".join(lines)


def format_spot_message(r, fng_value, fng_label) -> str:
    lines = [
        f"<b>💰 فرصت خرید اسپات (دید هفتگی) | {r['symbol']}</b>",
        f"امتیاز جذابیت: <b>{r['score']} / 100</b>",
        f"قیمت فعلی: {r['price']:.4f}",
        f"🎯 هدف قیمتی (مقاومت): {r['target']:.4f}",
        f"⚠️ سطح ابطال تحلیل: {r['invalidation']:.4f}",
        "",
        "<i>دلایل تحلیل (تایم‌فریم روزانه):</i>",
    ]
    lines += [f"• {x}" for x in r["reasons"]]
    if fng_value is not None:
        lines.append(f"\n😨/🤑 شاخص ترس و طمع: {fng_value} ({fng_label})")
    lines.append("\n⚠️ صرفاً تحلیل خودکار است، نه توصیه مالی قطعی. مدیریت ریسک را رعایت کنید.")
    return "\n".join(lines)


def main():
    log = load_json(LOG_PATH, [])
    weights = load_factor_weights()

    # ۱. اول وضعیت سیگنال‌های قبلی رو چک کن (TP خورده یا SL)
    log = check_open_signals(log)

    # ۲. یادگیری: وزن هر فاکتور رو بر اساس نتیجه واقعی سیگنال‌های قبلی تنظیم کن
    weights = update_factor_weights(log, weights)
    save_json(WEIGHTS_PATH, weights)

    # ۳. بر اساس عملکرد اخیر، آستانه پذیرش سیگنال رو هم خودکار تنظیم کن
    adaptive_threshold = compute_adaptive_threshold(log, base=FUTURES_SCORE_THRESHOLD)
    if adaptive_threshold != FUTURES_SCORE_THRESHOLD:
        print(f"آستانه به‌صورت خودکار به {adaptive_threshold:.0f} تغییر کرد (بر اساس عملکرد اخیر).")

    # ۴. اگه کاربر دستور "وضعیت" یا "وضعیت ماهانه" فرستاده بود، جواب بده
    process_commands(log, current_threshold=adaptive_threshold)

    fng_value, fng_label = get_fear_greed_index()
    current_hour = datetime.now(timezone.utc).hour
    run_spot_check = (current_hour % SPOT_CHECK_EVERY_HOURS == 0)

    any_signal = False
    count_before = len(log)

    for symbol in SYMBOLS:
        try:
            futures_result = analyze_futures(symbol, fng_value, score_threshold=adaptive_threshold, weights=weights)
            if futures_result:
                send_telegram_message(format_futures_message(futures_result, fng_value, fng_label))
                any_signal = True
                print(f"سیگنال فیوچرز {symbol} ارسال شد (امتیاز {futures_result['confidence']}).")

                log.append({
                    "symbol": futures_result["symbol"],
                    "direction": futures_result["direction"],
                    "entry": futures_result["price"],
                    "sl": futures_result["sl"],
                    "tp": futures_result["tp"],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "status": "open",
                    "factors": futures_result["factors"],
                })
            else:
                print(f"{symbol}: امتیاز فیوچرز زیر آستانه بود، سیگنالی ارسال نشد.")
        except Exception as e:
            print(f"خطا در تحلیل فیوچرز {symbol}: {e}")

        if run_spot_check:
            try:
                spot_result = analyze_spot(symbol, fng_value)
                if spot_result:
                    send_telegram_message(format_spot_message(spot_result, fng_value, fng_label))
                    any_signal = True
                    print(f"سیگنال اسپات {symbol} ارسال شد (امتیاز {spot_result['score']}).")
                else:
                    print(f"{symbol}: امتیاز اسپات زیر آستانه بود، سیگنالی ارسال نشد.")
            except Exception as e:
                print(f"خطا در تحلیل اسپات {symbol}: {e}")

    if not any_signal:
        print("در این اجرا هیچ ارزی شرایط لازم برای سیگنال با کیفیت را نداشت.")

    # اگه از یک مضرب ۲۰ سیگنال رد شدیم، خودکار یه گزارش کلی بفرست
    count_after = len(log)
    if count_after // 20 > count_before // 20:
        send_telegram_message(build_status_summary(log, "last20", adaptive_threshold))
        print("گزارش خودکار بعد از ۲۰ سیگنال ارسال شد.")

    # ۳. لاگ به‌روزشده رو ذخیره کن (workflow این فایل رو کامیت می‌کند)
    save_json(LOG_PATH, log)


if __name__ == "__main__":
    main()
