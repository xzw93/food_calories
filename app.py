# -*- coding: utf-8 -*-
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import threading
import re
from datetime import datetime
from flask import Flask, request, abort

from linebot import LineBotApi, WebhookHandler
from linebot.models import MessageEvent, TextMessage, ImageMessage, TextSendMessage
from linebot.exceptions import InvalidSignatureError

from model_100 import predict_food, load_model  # ✅ 建議你 model_100.py 提供 load_model
from calories import get_calorie, parse_kcal_range  # ✅ calories.py 需新增 parse_kcal_range

app = Flask(__name__)

LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# ===== 使用者飲食紀錄（暫存於記憶體）=====
# 結構: user_records[user_id][date] = [ {food, calorie_text, kcal_min, kcal_max, source, confidence?}, ...]
user_records = {}

# ✅ 重要：模型只載入一次（避免每次圖片都 load，超慢 + timeout）
MODEL = None
MODEL_LOCK = threading.Lock()

def get_model_once():
    global MODEL
    if MODEL is None:
        with MODEL_LOCK:
            if MODEL is None:
                MODEL = load_model()  # model_100.py 的 load_model()
    return MODEL

@app.get("/")
def health():
    return "OK", 200

@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers.get("X-Line-Signature")
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return "OK"

# =================================================
# 文字訊息
# =================================================
@handler.add(MessageEvent, message=TextMessage)
def handle_text(event):
    user_id = event.source.user_id
    text = event.message.text.strip()

    today = datetime.now().strftime("%Y-%m-%d")
    user_data = user_records.setdefault(user_id, {})

    if text == "說明":
        reply = (
            "📸 傳送食物照片即可辨識餐點\n"
            "🔥 系統會估算每 100g 熱量區間\n\n"
            "可用指令：\n"
            "▪ 說明\n"
            "▪ 新增 食物名稱（模型無法辨識時）\n"
            "▪ 今日紀錄\n"
            "▪ 熱量統計\n"
            "▪ 查詢日期 YYYY-MM-DD\n"
            "▪ 刪除上一筆\n"
            "▪ 清除今日"
        )

    elif text.startswith("新增"):
        try:
            _, food_zh = text.split(" ", 1)
            food_zh = food_zh.strip()
            if not food_zh:
                raise ValueError

            calorie_text = get_calorie(food_zh)
            kcal_min, kcal_max = parse_kcal_range(calorie_text)

            user_data.setdefault(today, []).append({
                "food": food_zh,
                "calorie_text": calorie_text,
                "kcal_min": kcal_min,
                "kcal_max": kcal_max,
                "source": "manual",
            })

            reply = (
                f"✍️ 手動紀錄成功\n"
                f"🍽 食物：{food_zh}\n"
                f"🔥 熱量估計：{calorie_text}"
            )
        except ValueError:
            reply = "❌ 格式錯誤，請輸入：新增 食物名稱"

    elif text == "今日紀錄":
        records = user_data.get(today, [])
        if not records:
            reply = "📭 今天尚未紀錄任何飲食"
        else:
            lines = [f"{i+1}. {r['food']}" for i, r in enumerate(records)]
            reply = f"📋 {today} 飲食紀錄：\n" + "\n".join(lines)

    elif text == "刪除上一筆":
        records = user_data.get(today, [])
        if not records:
            reply = "📭 今天沒有任何紀錄可刪除"
        else:
            last = records.pop()
            reply = (
                "🗑 已刪除上一筆紀錄\n"
                f"🍽 食物：{last['food']}\n"
                f"🔥 熱量：{last.get('calorie_text', 'N/A')}"
            )

    elif text == "熱量統計":
        records = user_data.get(today, [])
        if not records:
            reply = "📭 今天尚未紀錄任何飲食"
        else:
            lines = []
            total_min, total_max = 0, 0
            for r in records:
                lines.append(f"{r['food']}：{r['calorie_text']}")
                # ✅ 只加合理的數字，不要把字串亂 parse 成超大數
                if r["kcal_min"] is not None and r["kcal_max"] is not None:
                    total_min += r["kcal_min"]
                    total_max += r["kcal_max"]

            if total_min == 0 and total_max == 0:
                total_line = "📊 今日總熱量：無法計算（缺少可解析的數值）"
            else:
                total_line = f"📊 今日總熱量：約 {total_min}–{total_max} kcal（每 100 g 估算）"

            reply = f"🔥 {today} 熱量估計：\n" + "\n".join(lines) + "\n\n" + total_line

    elif text.startswith("查詢日期"):
        try:
            _, date_str = text.split()
            records = user_data.get(date_str, [])
            if not records:
                reply = f"📭 {date_str} 沒有紀錄"
            else:
                lines = [f"{i+1}. {r['food']}（{r['calorie_text']}）" for i, r in enumerate(records)]
                reply = f"📅 {date_str} 飲食紀錄：\n" + "\n".join(lines)
        except ValueError:
            reply = "❌ 格式錯誤，請輸入：查詢日期 YYYY-MM-DD"

    elif text == "清除今日":
        user_data.pop(today, None)
        reply = f"🧹 已清除 {today} 的飲食紀錄"

    else:
        reply = "請傳送食物照片，或輸入「說明」查看指令"

    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply))

# =================================================
# 圖片訊息（✅ 秒回 + 背景推論，避免 webhook timeout）
# =================================================
@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    # 1) 先秒回，避免 LINE timeout
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text="✅ 已收到圖片，辨識中（約數秒）...")
    )

    try:
        message_content = line_bot_api.get_message_content(event.message.id)
        image_bytes = b"".join(message_content.iter_content())

        food_en, food_zh, food_idx, confidence = predict_food(image_bytes)

        calorie_text = get_calorie(food_zh)
        kcal_min, kcal_max = parse_kcal_range(calorie_text)

        reply = (
            f"🍽 食物判斷：{food_zh}\n"
            f"🎯 信心分數：{confidence:.3f}\n"
            f"🔥 熱量估計：{calorie_text}"
        )

        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=reply)
        )

    except Exception as e:
        print("❌ 圖片處理錯誤:", e)
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="❌ 圖片辨識失敗，請再試一次")
        )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
