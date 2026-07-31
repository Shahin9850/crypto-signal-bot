"""
ربات سیگنال‌دهی بر پایه پرایس‌اکشن خالص (بدون اندیکاتور)
شامل: ساختار بازار (BOS/CHoCH)، الگوهای کندلی، مناطق عرضه/تقاضا، Liquidity Grab
+ محاسبه دقیق حجم معامله، اهرم و ریسک بر اساس سرمایه مشخص‌شده

منبع قیمت: Binance Public Data Mirror (بدون نیاز به API Key)
ارسال: تلگرام
"""

import os
import pandas as pd
import numpy as np
import requests

# ---------------------------------------------------------------
# تنظیمات
# ---------------------------------------------------------------
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT",
    "DOGEUSDT", "TRXUSDT", "ADAUSDT", "LINKUSDT", "AVAXUSDT",
]

CAPITAL_USD = float(os.environ.get("CAPITAL_USD", "100"))
RISK_PERCENT = float(os.environ.get("RISK_PERCENT", "1.5"))          # درصد سرمایه در معرض ریسک در هر معامله
MARGIN_ALLOCATION_PERCENT = float(os.environ.get("MARGIN_ALLOCATION_PERCENT", "20"))  # درصد سرمایه به‌عنوان مارجین هر معامله
MAX_LEVERAGE = float(os.environ.get("MAX_LEVERAGE", "10"))           # سقف اهرم مجاز (برای امنیت)
RISK_REWARD = 2.0
PA_SCORE_THRESHOLD = 65

TELEGRAM_TOKEN = os.environ.get("PA_TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("PA_TELEGRAM_CHAT_ID")

BINANCE_KLINES_URL = "https://data-api.binance.vision/api/v3/klines"


# ---------------------------------------------------------------
# دریافت داده
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


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ---------------------------------------------------------------
# ابزارهای پرایس‌اکشن
# ---------------------------------------------------------------
def find_swing_points(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    df = df.copy()
    highs, lows = df["high"], df["low"]
    swing_high = pd.Series(False, index=df.index)
    swing_low = pd.Series(False, index=df.index)
    for i in range(window, len(df) - window):
        if highs.iloc[i] == highs.iloc[i - window:i + window + 1].max():
            swing_high.iloc[i] = True
        if lows.iloc[i] == lows.iloc[i - window:i + window + 1].min():
            swing_low.iloc[i] = True
    df["swing_high"] = swing_high
    df["swing_low"] = swing_low
    return df


def get_last_swings(df: pd.DataFrame, n: int = 3):
    highs = df.loc[df["swing_high"], "high"].tail(n).tolist()
    lows = df.loc[df["swing_low"], "low"].tail(n).tolist()
    return highs, lows


def market_structure_trend(df: pd.DataFrame):
    df = find_swing_points(df)
    highs, lows = get_last_swings(df, 3)
    trend = "range"
    if len(highs) >= 2 and len(lows) >= 2:
        if highs[-1] > highs[-2] and lows[-1] > lows[-2]:
            trend = "bull"
        elif highs[-1] < highs[-2] and lows[-1] < lows[-2]:
            trend = "bear"
    return trend, highs, lows


def detect_bos_choch(df: pd.DataFrame, trend: str):
    df = find_swing_points(df)
    highs, lows = get_last_swings(df, 2)
    last_close = df["close"].iloc[-1]
    if trend == "bull" and highs and last_close > highs[-1]:
        return "bull_bos"
    if trend == "bear" and lows and last_close < lows[-1]:
        return "bear_bos"
    if trend == "bull" and lows and last_close < lows[-1]:
        return "bear_choch"
    if trend == "bear" and highs and last_close > highs[-1]:
        return "bull_choch"
    return None


def detect_candle_pattern(df: pd.DataFrame):
    c, p = df.iloc[-1], df.iloc[-2]
    body = abs(c["close"] - c["open"])
    rng = c["high"] - c["low"]
    upper_wick = c["high"] - max(c["close"], c["open"])
    lower_wick = min(c["close"], c["open"]) - c["low"]

    if c["close"] > c["open"] and p["close"] < p["open"] and c["close"] >= p["open"] and c["open"] <= p["close"]:
        return "bullish_engulfing"
    if c["close"] < c["open"] and p["close"] > p["open"] and c["open"] >= p["close"] and c["close"] <= p["open"]:
        return "bearish_engulfing"
    if rng > 0 and lower_wick > 2 * body and upper_wick < body:
        return "bullish_pin"
    if rng > 0 and upper_wick > 2 * body and lower_wick < body:
        return "bearish_pin"
    return None


def find_supply_demand_zone(df: pd.DataFrame, lookback: int = 40):
    recent = df.tail(lookback).copy()
    recent["body"] = (recent["close"] - recent["open"]).abs()
    avg_body = recent["body"].mean()
    strong = recent[recent["body"] > 1.5 * avg_body]
    if strong.empty:
        return None
    last_strong = strong.iloc[-1]
    if last_strong["close"] > last_strong["open"]:
        return ("demand", last_strong["low"], last_strong["high"])
    return ("supply", last_strong["low"], last_strong["high"])


def price_in_zone(price: float, zone) -> bool:
    if zone is None:
        return False
    _, lo, hi = zone
    return lo <= price <= hi


def detect_liquidity_grab(df: pd.DataFrame):
    df = find_swing_points(df)
    highs, lows = get_last_swings(df, 2)
    c = df.iloc[-1]
    if lows and c["low"] < lows[-1] and c["close"] > lows[-1]:
        return "bullish_grab"
    if highs and c["high"] > highs[-1] and c["close"] < highs[-1]:
        return "bearish_grab"
    return None


# ---------------------------------------------------------------
# تحلیل اصلی پرایس‌اکشن (چندتایم‌فریمی)
# ---------------------------------------------------------------
def analyze_price_action(symbol: str):
    df4h = fetch_klines(symbol, "4h", 150)
    df1h = fetch_klines(symbol, "1h", 200)
    df15m = fetch_klines(symbol, "15m", 200)

    bull, bear = 0.0, 0.0
    reasons = []

    # ۱. ساختار بازار در تایم‌فریم ۴ ساعته (وزن ۳۰)
    trend4h, _, _ = market_structure_trend(df4h)
    if trend4h == "bull":
        bull += 30
        reasons.append("ساختار بازار (۴ساعته): سقف/کف‌های بالاتر (روند صعودی)")
    elif trend4h == "bear":
        bear += 30
        reasons.append("ساختار بازار (۴ساعته): سقف/کف‌های پایین‌تر (روند نزولی)")

    # ۲. تایید شکست ساختار (BOS/CHoCH) در ۱ ساعته (وزن ۲۵)
    trend1h, _, _ = market_structure_trend(df1h)
    bos = detect_bos_choch(df1h, trend1h if trend1h != "range" else trend4h)
    if bos == "bull_bos":
        bull += 25
        reasons.append("شکست ساختار صعودی (BOS) در ۱ساعته - ادامه روند")
    elif bos == "bear_bos":
        bear += 25
        reasons.append("شکست ساختار نزولی (BOS) در ۱ساعته - ادامه روند")
    elif bos == "bull_choch":
        bull += 15
        reasons.append("نشانه تغییر روند به صعودی (CHoCH) در ۱ساعته")
    elif bos == "bear_choch":
        bear += 15
        reasons.append("نشانه تغییر روند به نزولی (CHoCH) در ۱ساعته")

    # ۳. الگوی کندلی در ۱ساعته یا ۱۵دقیقه (وزن ۲۰)
    candle_1h = detect_candle_pattern(df1h)
    candle_15m = detect_candle_pattern(df15m)
    if candle_1h in ("bullish_engulfing", "bullish_pin") or candle_15m in ("bullish_engulfing", "bullish_pin"):
        bull += 20
        pat = candle_15m if candle_15m in ("bullish_engulfing", "bullish_pin") else candle_1h
        reasons.append(f"الگوی کندلی صعودی شناسایی شد ({pat})")
    elif candle_1h in ("bearish_engulfing", "bearish_pin") or candle_15m in ("bearish_engulfing", "bearish_pin"):
        bear += 20
        pat = candle_15m if candle_15m in ("bearish_engulfing", "bearish_pin") else candle_1h
        reasons.append(f"الگوی کندلی نزولی شناسایی شد ({pat})")

    # ۴. موقعیت قیمت نسبت به منطقه عرضه/تقاضا (وزن ۱۵)
    zone = find_supply_demand_zone(df1h)
    price = df15m["close"].iloc[-1]
    if zone and price_in_zone(price, zone):
        if zone[0] == "demand":
            bull += 15
            reasons.append("قیمت داخل منطقه تقاضا (Demand Zone)")
        else:
            bear += 15
            reasons.append("قیمت داخل منطقه عرضه (Supply Zone)")

    # ۵. Liquidity Grab / شکار استاپ (وزن ۱۰)
    grab_1h = detect_liquidity_grab(df1h)
    grab_15m = detect_liquidity_grab(df15m)
    if grab_1h == "bullish_grab" or grab_15m == "bullish_grab":
        bull += 10
        reasons.append("جمع‌آوری نقدینگی زیر کف قبلی و بازگشت (Liquidity Grab صعودی)")
    elif grab_1h == "bearish_grab" or grab_15m == "bearish_grab":
        bear += 10
        reasons.append("جمع‌آوری نقدینگی بالای سقف قبلی و بازگشت (Liquidity Grab نزولی)")

    confidence = max(bull, bear)
    direction = "BUY" if bull > bear else "SELL" if bear > bull else "NEUTRAL"

    if confidence < PA_SCORE_THRESHOLD:
        return None

    # ---------------------------------------------------------------
    # تعیین حد ضرر بر پایه ساختار قیمت (نه عدد دلخواه)
    # ---------------------------------------------------------------
    df1h_sw = find_swing_points(df1h)
    highs1h, lows1h = get_last_swings(df1h_sw, 3)
    atr1h = atr(df1h).iloc[-1]
    entry = price

    if direction == "BUY":
        structure_sl = lows1h[-1] if lows1h else entry - 1.5 * atr1h
        sl = min(structure_sl, entry - 0.5 * atr1h) - 0.25 * atr1h
        tp = entry + (entry - sl) * RISK_REWARD
    else:
        structure_sl = highs1h[-1] if highs1h else entry + 1.5 * atr1h
        sl = max(structure_sl, entry + 0.5 * atr1h) + 0.25 * atr1h
        tp = entry - (sl - entry) * RISK_REWARD

    stop_distance_pct = abs(entry - sl) / entry
    if stop_distance_pct <= 0:
        return None

    # ---------------------------------------------------------------
    # محاسبه حجم معامله، اهرم و ریسک بر اساس سرمایه
    # ---------------------------------------------------------------
    risk_amount_usd = CAPITAL_USD * RISK_PERCENT / 100
    required_position_usd = risk_amount_usd / stop_distance_pct
    margin_usd = CAPITAL_USD * MARGIN_ALLOCATION_PERCENT / 100
    required_leverage = required_position_usd / margin_usd
    leverage = max(1, min(required_leverage, MAX_LEVERAGE))
    actual_position_usd = leverage * margin_usd
    actual_risk_usd = actual_position_usd * stop_distance_pct
    tp_distance_pct = abs(tp - entry) / entry
    potential_profit_usd = actual_position_usd * tp_distance_pct

    return {
        "symbol": symbol,
        "direction": direction,
        "confidence": round(confidence),
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "reasons": reasons,
        "margin_usd": margin_usd,
        "leverage": round(leverage, 1),
        "position_usd": actual_position_usd,
        "risk_usd": actual_risk_usd,
        "profit_usd": potential_profit_usd,
    }


# ---------------------------------------------------------------
# ارسال به تلگرام
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


def format_message(r) -> str:
    direction_fa = "خرید (BUY) 🟢" if r["direction"] == "BUY" else "فروش (SELL) 🔴"
    lines = [
        f"<b>🕯️ سیگنال پرایس‌اکشن | {r['symbol']}</b>",
        f"امتیاز اطمینان: <b>{r['confidence']} / 100</b>",
        f"جهت: <b>{direction_fa}</b>",
        "",
        f"قیمت ورود: {r['entry']:.5f}",
        f"🎯 حد سود (TP): {r['tp']:.5f}",
        f"🛑 حد ضرر (SL): {r['sl']:.5f}",
        "",
        "<b>💵 مدیریت سرمایه (بر اساس سرمایه $" + f"{CAPITAL_USD:.0f})</b>",
        f"مارجین این معامله: <b>${r['margin_usd']:.2f}</b>",
        f"اهرم پیشنهادی: <b>{r['leverage']}x</b>",
        f"حجم کل پوزیشن: ${r['position_usd']:.2f}",
        f"ریسک این معامله (اگر استاپ بخوره): <b>${r['risk_usd']:.2f}</b>",
        f"سود هدف (اگر تی‌پی بخوره): <b>${r['profit_usd']:.2f}</b>",
        "",
        "<i>دلایل تحلیل پرایس‌اکشن (۴ساعته → ۱ساعته → ۱۵دقیقه):</i>",
    ]
    lines += [f"• {x}" for x in r["reasons"]]
    lines.append(
        "\n⚠️ فقط با همین حجم و اهرم پیشنهادی وارد شو، نه بیشتر. "
        "این عدد جوری حساب شده که با چند بار استاپ خوردن پشت‌سرهم، سرمایه‌ات از بین نره. "
        "صرفاً تحلیل خودکار است، نه تضمین سود."
    )
    return "\n".join(lines)


def main():
    any_signal = False
    for symbol in SYMBOLS:
        try:
            result = analyze_price_action(symbol)
            if result:
                send_telegram_message(format_message(result))
                any_signal = True
                print(f"سیگنال پرایس‌اکشن {symbol} ارسال شد (امتیاز {result['confidence']}).")
            else:
                print(f"{symbol}: شرایط پرایس‌اکشن با کیفیت کافی پیدا نشد.")
        except Exception as e:
            print(f"خطا در تحلیل {symbol}: {e}")

    if not any_signal:
        print("در این اجرا هیچ ارزی ستاپ پرایس‌اکشن با کیفیت نداشت.")


if __name__ == "__main__":
    main()
