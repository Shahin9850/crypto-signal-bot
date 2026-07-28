"""
ربات تحلیل و سیگنال‌دهی ارز دیجیتال (نسخه پیشرفته)
- سیگنال فیوچرز: تحلیل چندتایم‌فریمی (روند ۴ ساعته → تایید ۱ ساعته → ورود ۱۵ دقیقه‌ای)
- سیگنال اسپات: مناسب خرید/نگهداری هفتگی، بر پایه تایم‌فریم روزانه
- امتیازدهی از ۱۰۰؛ زیر آستانه مشخص، سیگنالی ارسال نمی‌شود
منبع قیمت: Binance Public Data Mirror (بدون نیاز به API Key)
ارسال: تلگرام
"""

import os
from datetime import datetime, timezone

import requests
import numpy as np
import pandas as pd

# ---------------------------------------------------------------
# تنظیمات کلی
# ---------------------------------------------------------------
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]
RISK_REWARD = 2.0

FUTURES_SCORE_THRESHOLD = 65   # زیر این امتیاز، سیگنال فیوچرز ارسال نمی‌شود
SPOT_SCORE_THRESHOLD = 65      # زیر این امتیاز، سیگنال اسپات ارسال نمی‌شود
SPOT_CHECK_EVERY_HOURS = 6     # تحلیل اسپات هر چند ساعت یک‌بار انجام شود (دید هفتگی، نیازی به هر ۱۵ دقیقه نیست)

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

BINANCE_KLINES_URL = "https://data-api.binance.vision/api/v3/klines"
FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=1"


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
# تحلیل فیوچرز: چندتایم‌فریمی (۴ ساعته -> ۱ ساعته -> ۱۵ دقیقه)
# ---------------------------------------------------------------
def analyze_futures(symbol: str, fng_value):
    df4h = add_indicators(fetch_klines(symbol, "4h", 100))
    df1h = add_indicators(fetch_klines(symbol, "1h", 200))
    df15m = add_indicators(fetch_klines(symbol, "15m", 200))

    last4h, last1h, last15m = df4h.iloc[-1], df1h.iloc[-1], df15m.iloc[-1]
    price = last15m["close"]
    support, resistance = find_support_resistance(df1h, 30)

    bull, bear = 0.0, 0.0
    reasons = []

    # ۱. روند اصلی روی تایم‌فریم ۴ ساعته (وزن ۳۰)
    if last4h["ema20"] > last4h["ema50"]:
        bull += 30
        reasons.append("روند اصلی (۴ساعته) صعودی")
    elif last4h["ema20"] < last4h["ema50"]:
        bear += 30
        reasons.append("روند اصلی (۴ساعته) نزولی")

    # ۲. تایید روند روی ۱ ساعته (وزن ۲۰)
    if last1h["ema20"] > last1h["ema50"]:
        bull += 20
        reasons.append("روند میان‌مدت (۱ساعته) صعودی")
    elif last1h["ema20"] < last1h["ema50"]:
        bear += 20
        reasons.append("روند میان‌مدت (۱ساعته) نزولی")

    # ۳. RSI روی ۱ ساعته (وزن ۱۵)
    if last1h["rsi"] < 35:
        bull += 15
        reasons.append(f"RSI یک‌ساعته در ناحیه اشباع فروش ({last1h['rsi']:.1f})")
    elif last1h["rsi"] > 65:
        bear += 15
        reasons.append(f"RSI یک‌ساعته در ناحیه اشباع خرید ({last1h['rsi']:.1f})")

    # ۴. MACD روی ۱ ساعته (وزن ۱۵)
    if last1h["macd"] > last1h["macd_signal"]:
        bull += 15
        reasons.append("MACD یک‌ساعته مثبت")
    elif last1h["macd"] < last1h["macd_signal"]:
        bear += 15
        reasons.append("MACD یک‌ساعته منفی")

    # ۵. نقطه ورود دقیق روی ۱۵ دقیقه (وزن ۱۵)
    if last15m["close"] > last15m["ema20"] and last15m["rsi"] > df15m.iloc[-2]["rsi"]:
        bull += 15
        reasons.append("قیمت بالای EMA20 در ۱۵ دقیقه و مومنتوم رو به رشد")
    elif last15m["close"] < last15m["ema20"] and last15m["rsi"] < df15m.iloc[-2]["rsi"]:
        bear += 15
        reasons.append("قیمت زیر EMA20 در ۱۵ دقیقه و مومنتوم رو به کاهش")

    # ۶. نزدیکی به حمایت/مقاومت کلیدی روی ۱ ساعته (وزن ۵)
    if (price - support) / price < 0.01:
        bull += 5
        reasons.append("قیمت نزدیک حمایت کلیدی")
    elif (resistance - price) / price < 0.01:
        bear += 5
        reasons.append("قیمت نزدیک مقاومت کلیدی")

    if fng_value is not None:
        if fng_value <= 20:
            bull += 5
            reasons.append(f"ترس شدید در بازار ({fng_value})")
        elif fng_value >= 80:
            bear += 5
            reasons.append(f"طمع شدید در بازار ({fng_value})")

    confidence = max(bull, bear)
    direction = "BUY 🟢" if bull > bear else "SELL 🔴" if bear > bull else "NEUTRAL"

    if confidence < FUTURES_SCORE_THRESHOLD:
        return None

    atr15 = last15m["atr"]
    if direction.startswith("BUY"):
        sl = price - 1.2 * atr15
        tp = price + 1.2 * atr15 * RISK_REWARD
    else:
        sl = price + 1.2 * atr15
        tp = price - 1.2 * atr15 * RISK_REWARD

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
    }


# ---------------------------------------------------------------
# تحلیل اسپات: مناسب خرید/نگهداری با دید هفتگی (تایم‌فریم روزانه)
# ---------------------------------------------------------------
def analyze_spot(symbol: str, fng_value):
    df1d = add_indicators(fetch_klines(symbol, "1d", 120))
    last = df1d.iloc[-1]
    price = last["close"]
    support, resistance = find_support_resistance(df1d, 30)

    score = 0.0
    reasons = []

    # ۱. روند بلندمدت (وزن ۳۵)
    if price > last["ema50"] and last["ema20"] > last["ema50"]:
        score += 35
        reasons.append("روند بلندمدت (روزانه) صعودی و قیمت بالای EMA50")
    elif price > last["ema50"]:
        score += 20
        reasons.append("قیمت بالای EMA50 روزانه (روند نسبتاً مثبت)")

    # ۲. RSI روزانه (وزن ۲۰) - منطقه‌ی مناسب برای خرید، نه اشباع خرید
    if last["rsi"] < 40:
        score += 20
        reasons.append(f"RSI روزانه در محدوده مناسب خرید ({last['rsi']:.1f})")
    elif last["rsi"] < 55:
        score += 10
        reasons.append(f"RSI روزانه خنثی ({last['rsi']:.1f})")

    # ۳. فاصله تا حمایت ۳۰ روزه (وزن ۲۰)
    dist_support = (price - support) / price
    if dist_support < 0.05:
        score += 20
        reasons.append("قیمت نزدیک حمایت ۳۰ روزه")
    elif dist_support < 0.10:
        score += 10
        reasons.append("قیمت در فاصله معقول از حمایت ۳۰ روزه")

    # ۴. MACD روزانه (وزن ۱۵)
    if last["macd"] > last["macd_signal"]:
        score += 15
        reasons.append("MACD روزانه مثبت")

    # ۵. سنتیمنت بازار (وزن ۱۰)
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
    lines = [
        f"<b>⚡️ سیگنال فیوچرز | {r['symbol']}</b>",
        f"امتیاز اطمینان: <b>{r['confidence']} / 100</b>",
        f"جهت: <b>{r['direction']}</b>",
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
    lines.append("\n⚠️ صرفاً تحلیل خودکار است، نه توصیه مالی قطعی. مدیریت ریسک را رعایت کنید.")
    return "\n".join(lines)


def format_spot_message(r, fng_value, fng_label) -> str:
    lines = [
        f"<b>💰 فرصت خرید اسپات (دید هفتگی) | {r['symbol']}</b>",
        f"امتیاز جذابیت: <b>{r['score']} / 100</b>",
        f"قیمت فعلی: {r['price']:.4f}",
        f"🎯 هدف قیمتی (مقاومت): {r['target']:.4f}",
        f"⚠️ سطح ابطال تحلیل (زیر این یعنی تحلیل نقض شده): {r['invalidation']:.4f}",
        "",
        "<i>دلایل تحلیل (تایم‌فریم روزانه):</i>",
    ]
    lines += [f"• {x}" for x in r["reasons"]]
    if fng_value is not None:
        lines.append(f"\n😨/🤑 شاخص ترس و طمع: {fng_value} ({fng_label})")
    lines.append("\n⚠️ صرفاً تحلیل خودکار است، نه توصیه مالی قطعی. مدیریت ریسک را رعایت کنید.")
    return "\n".join(lines)


def main():
    fng_value, fng_label = get_fear_greed_index()
    current_hour = datetime.now(timezone.utc).hour
    run_spot_check = (current_hour % SPOT_CHECK_EVERY_HOURS == 0)

    any_signal = False

    for symbol in SYMBOLS:
        try:
            futures_result = analyze_futures(symbol, fng_value)
            if futures_result:
                send_telegram_message(format_futures_message(futures_result, fng_value, fng_label))
                any_signal = True
                print(f"سیگنال فیوچرز {symbol} ارسال شد (امتیاز {futures_result['confidence']}).")
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


if __name__ == "__main__":
    main()
