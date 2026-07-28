"""
ربات تحلیل و سیگنال‌دهی ارز دیجیتال
منبع قیمت: Binance Public API (بدون نیاز به API Key)
ارسال سیگنال: تلگرام
اجرا: از طریق GitHub Actions به‌صورت زمان‌بندی‌شده (هر ۱۵ دقیقه)
"""

import os
import requests
import numpy as np
import pandas as pd

# ---------------------------------------------------------------
# تنظیمات
# ---------------------------------------------------------------
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]
INTERVAL = "1h"          # تایم‌فریم کندل‌ها
LIMIT = 200              # تعداد کندل برای تحلیل
RISK_REWARD = 2.0        # نسبت حد سود به حد ضرر

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

BINANCE_KLINES_URL = "https://data-api.binance.vision/api/v3/klines"
FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=1"


# ---------------------------------------------------------------
# دریافت داده قیمتی
# ---------------------------------------------------------------
def fetch_klines(symbol: str) -> pd.DataFrame:
    params = {"symbol": symbol, "interval": INTERVAL, "limit": LIMIT}
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
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger_bands(series: pd.Series, period: int = 20, std_mult: float = 2.0):
    mid = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    return upper, mid, lower


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def find_support_resistance(df: pd.DataFrame, lookback: int = 30):
    recent = df.tail(lookback)
    resistance = recent["high"].max()
    support = recent["low"].min()
    return support, resistance


# ---------------------------------------------------------------
# سنتیمنت بازار (جایگزین ساده‌ی فاندامنتال)
# ---------------------------------------------------------------
def get_fear_greed_index():
    try:
        resp = requests.get(FEAR_GREED_URL, timeout=10)
        resp.raise_for_status()
        data = resp.json()["data"][0]
        return int(data["value"]), data["value_classification"]
    except Exception:
        return None, None


# ---------------------------------------------------------------
# منطق تولید سیگنال
# ---------------------------------------------------------------
def analyze_symbol(symbol: str, fng_value):
    df = fetch_klines(symbol)
    close = df["close"]

    df["ema20"] = ema(close, 20)
    df["ema50"] = ema(close, 50)
    df["rsi"] = rsi(close)
    macd_line, signal_line, hist = macd(close)
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    upper, mid, lower = bollinger_bands(close)
    df["bb_upper"] = upper
    df["bb_lower"] = lower
    df["atr"] = atr(df)

    last = df.iloc[-1]
    price = last["close"]
    support, resistance = find_support_resistance(df)

    score = 0
    reasons = []

    # روند (EMA)
    if last["ema20"] > last["ema50"]:
        score += 1
        reasons.append("روند کوتاه‌مدت صعودی (EMA20 بالای EMA50)")
    else:
        score -= 1
        reasons.append("روند کوتاه‌مدت نزولی (EMA20 زیر EMA50)")

    # RSI
    if last["rsi"] < 30:
        score += 1
        reasons.append(f"RSI در ناحیه اشباع فروش ({last['rsi']:.1f})")
    elif last["rsi"] > 70:
        score -= 1
        reasons.append(f"RSI در ناحیه اشباع خرید ({last['rsi']:.1f})")
    else:
        reasons.append(f"RSI خنثی ({last['rsi']:.1f})")

    # MACD
    if last["macd"] > last["macd_signal"]:
        score += 1
        reasons.append("MACD مثبت (تقاطع صعودی)")
    else:
        score -= 1
        reasons.append("MACD منفی (تقاطع نزولی)")

    # پرایس اکشن نسبت به باند بولینگر
    if price <= last["bb_lower"]:
        score += 1
        reasons.append("قیمت نزدیک باند پایین بولینگر (احتمال بازگشت)")
    elif price >= last["bb_upper"]:
        score -= 1
        reasons.append("قیمت نزدیک باند بالای بولینگر (احتمال اصلاح)")

    # فاصله تا حمایت/مقاومت
    dist_to_support = (price - support) / price
    dist_to_resistance = (resistance - price) / price
    if dist_to_support < 0.01:
        score += 1
        reasons.append("قیمت روی سطح حمایت کلیدی")
    if dist_to_resistance < 0.01:
        score -= 1
        reasons.append("قیمت روی سطح مقاومت کلیدی")

    # سنتیمنت کلی بازار
    if fng_value is not None:
        if fng_value <= 25:
            score += 1
            reasons.append(f"ترس شدید در بازار (شاخص {fng_value}) → فرصت احتمالی خرید")
        elif fng_value >= 75:
            score -= 1
            reasons.append(f"طمع شدید در بازار (شاخص {fng_value}) → احتیاط در خرید")

    # تعیین سیگنال نهایی
    if score >= 2:
        direction = "BUY 🟢"
        sl = price - 1.5 * last["atr"]
        tp = price + 1.5 * last["atr"] * RISK_REWARD
    elif score <= -2:
        direction = "SELL 🔴"
        sl = price + 1.5 * last["atr"]
        tp = price - 1.5 * last["atr"] * RISK_REWARD
    else:
        direction = "NEUTRAL ⚪ (منتظر بمانید)"
        sl = None
        tp = None

    return {
        "symbol": symbol,
        "price": price,
        "direction": direction,
        "score": score,
        "reasons": reasons,
        "support": support,
        "resistance": resistance,
        "sl": sl,
        "tp": tp,
    }


# ---------------------------------------------------------------
# ارسال پیام به تلگرام
# ---------------------------------------------------------------
def send_telegram_message(text: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("توکن یا چت آیدی تلگرام تنظیم نشده است.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
    }
    resp = requests.post(url, data=payload, timeout=15)
    if resp.status_code != 200:
        print("خطا در ارسال به تلگرام:", resp.text)


def format_message(result, fng_value, fng_label) -> str:
    lines = []
    lines.append(f"<b>📊 {result['symbol']}</b>")
    lines.append(f"قیمت فعلی: <b>{result['price']:.4f}</b>")
    lines.append(f"سیگنال: <b>{result['direction']}</b> (امتیاز {result['score']})")
    if result["sl"] and result["tp"]:
        lines.append(f"🎯 حد سود (TP): {result['tp']:.4f}")
        lines.append(f"🛑 حد ضرر (SL): {result['sl']:.4f}")
    lines.append(f"📈 مقاومت نزدیک: {result['resistance']:.4f}")
    lines.append(f"📉 حمایت نزدیک: {result['support']:.4f}")
    lines.append("")
    lines.append("<i>دلایل تحلیل:</i>")
    for r in result["reasons"]:
        lines.append(f"• {r}")
    if fng_value is not None:
        lines.append("")
        lines.append(f"😨/🤑 شاخص ترس و طمع بازار: {fng_value} ({fng_label})")
    lines.append("")
    lines.append("⚠️ این سیگنال صرفاً تحلیل خودکار است، نه توصیه مالی قطعی. مدیریت ریسک را رعایت کنید.")
    return "\n".join(lines)


def main():
    fng_value, fng_label = get_fear_greed_index()
    for symbol in SYMBOLS:
        try:
            result = analyze_symbol(symbol, fng_value)
            message = format_message(result, fng_value, fng_label)
            send_telegram_message(message)
            print(f"سیگنال {symbol} ارسال شد.")
        except Exception as e:
            print(f"خطا در تحلیل {symbol}: {e}")


if __name__ == "__main__":
    main()
