import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from datetime import datetime
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.models import (
    MessageEvent, TextMessage, ImageMessage, TextSendMessage
)
from linebot.exceptions import InvalidSignatureError
import re

from model_100 import predict_food
from calories import get_calorie


# ===== 使用者飲食紀錄（暫存於記憶體）=====
user_records = {}

app = Flask(__name__)

# ⚠️ 建議使用環境變數（本機 / 雲端皆可）
LINE_CHANNEL_ACCESS_TOKEN = os.environ["iEpxYFHcpO7OgBpD2zW/rN0ZkjKGtyQ3ILF9GUsHsvxWhKIC1dFAPZZPXaYoCm+WB1rg2odk1SwO9rMdWgOxCoUMCnv2BNCDM4lvhV+1gFObVYAK/unc4uqsd+0p0ycn1gHY0emgY8ge0q4GW3LD4QdB04t89/1O/w1cDnyilFU="]
LINE_CHANNEL_SECRET = os.environ["256d5a2b375807389c34bc5c9b65cbb7"]

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)


@app.route("/callback", methods=["POST"])
def callback():
    signature = request.headers.get("X-Line-Signature")
    body = request.get_data(as_text=True)

    # 👉 LINE Verify / 健康檢查
    if not signature:
        return "OK", 200

    # 👉 先立刻回 200（超重要）
    try:
        handler.handle(body, signature)
    except Exception as e:
        print("Handler error:", e)

    return "OK", 200


def parse_kcal_range(text: str) -> int | None:
    """
    從 '約 180–280 kcal（每 100 g）' 解析出代表性 kcal
    規則：
    - 有區間 → 取平均
    - 只有一個數字 → 直接用
    - 解析失敗 → None
    """
    numbers = list(map(int, re.findall(r"\d+", text)))

    if not numbers:
        return None

    if len(numbers) >= 2:
        return int(sum(numbers[:2]) / 2)

    return numbers[0]

# =================================================
# 處理文字訊息
# =================================================
@handler.add(MessageEvent, message=TextMessage)
def handle_text(event):
    user_id = event.source.user_id
    text = event.message.text.strip()

    today = datetime.now().strftime("%Y-%m-%d")
    user_data = user_records.get(user_id, {})

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
            "▪ 刪除今日"
        )

    elif text.startswith("新增"):
        try:
            _, food_zh = text.split(" ", 1)
            food_zh = food_zh.strip()
            if not food_zh:
                raise ValueError

            calorie_text = get_calorie(food_zh)

            user_records.setdefault(user_id, {}).setdefault(today, []).append({
                "food": food_zh,
                "calorie": calorie_text,
                "source": "manual"
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

    elif text == "熱量統計":
        records = user_data.get(today, [])

        if not records:
            reply = "📭 今天尚未紀錄任何飲食"
        else:
            lines = []
            total_kcal = 0

            for r in records:
                lines.append(f"{r['food']}：{r['calorie']}")

                # 嘗試從文字中抓 kcal 數字
                kcal_val = parse_kcal_range(r["calorie"])
                if kcal_val is not None:
                    total_kcal += kcal_val

            reply = (
                f"🔥 {today} 熱量估計：\n"
                + "\n".join(lines)
                + "\n\n"
                + f"📊 今日總熱量：約 {total_kcal} kcal"
            )

    elif text.startswith("查詢日期"):
        try:
            _, date_str = text.split()
            records = user_data.get(date_str, [])
            if not records:
                reply = f"📭 {date_str} 沒有紀錄"
            else:
                lines = [
                    f"{i+1}. {r['food']}（{r['calorie']}）"
                    for i, r in enumerate(records)
                ]
                reply = f"📅 {date_str} 飲食紀錄：\n" + "\n".join(lines)
        except ValueError:
            reply = "❌ 格式錯誤，請輸入：查詢日期 YYYY-MM-DD"
    elif text == "刪除上一筆":
        records = user_data.get(today, [])

        if not records:
            reply = "📭 今天尚無任何紀錄可刪除"
        else:
            removed = records.pop()
            reply = (
                "🗑 已刪除上一筆紀錄\n"
                f"🍽 食物：{removed['food']}\n"
                f"🔥 熱量：{removed['calorie']}"
            )

    elif text == "刪除今日":
        user_records.get(user_id, {}).pop(today, None)
        reply = f"🧹 已清除 {today} 的飲食紀錄"

    else:
        reply = "請傳送食物照片，或輸入「說明」查看指令"

    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply)
    )


# =================================================
# 處理圖片訊息
# =================================================
@handler.add(MessageEvent, message=ImageMessage)
def handle_image(event):
    try:
        # 1️⃣ 取得圖片內容（正確做法）
        message_content = line_bot_api.get_message_content(event.message.id)

        image_bytes = b""
        for chunk in message_content.iter_content():
            image_bytes += chunk

        # 2️⃣ 模型推論（中文類別 + 信心分數）
        food_en, food_zh, food_idx, confidence = predict_food(image_bytes)

        # 3️⃣ 查熱量（只用中文）
        calorie_text = get_calorie(food_zh)

        user_id = event.source.user_id
        today = datetime.now().strftime("%Y-%m-%d")

        user_records.setdefault(user_id, {}).setdefault(today, []).append({
            "food": food_zh,
            "calorie": calorie_text,
            "confidence": round(confidence, 3),
            "source": "model"
        })

        # 4️⃣ 回覆 LINE
        reply = (
            f"🍽 食物判斷：{food_zh}\n"
            f"🔥 熱量估計：{calorie_text}"
        )

        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=reply)
        )

    except Exception as e:
        print("❌ 圖片處理錯誤：", e)
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="圖片辨識失敗，請再試一次 🙏")
        )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

