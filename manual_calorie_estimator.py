# =================================================
# 人工熱量補估模組
# 用於：官方營養資料庫無法對應的品項
# 單位：每 100 g
# =================================================

from typing import Dict


# =================================================
# 1. 人工補估規則表（Rule-based）
# =================================================
# keywords：判斷關鍵字
# kcal_range：估算熱量區間
# reason：估算依據（可寫進報告）

MANUAL_CALORIE_RULES = [
    {
        "keywords": ["炸"],
        "kcal_range": (250, 350),
        "reason": "高溫油炸食品平均熱量"
    },
    {
        "keywords": ["麵"],
        "kcal_range": (130, 180),
        "reason": "一般熟麵條類熱量區間"
    },
    {
        "keywords": ["飯"],
        "kcal_range": (130, 160),
        "reason": "白飯與拌飯類平均熱量"
    },
    {
        "keywords": ["湯"],
        "kcal_range": (30, 80),
        "reason": "清湯或非濃湯類食品"
    },
    {
        "keywords": ["蛋"],
        "kcal_range": (140, 180),
        "reason": "蛋類料理常見熱量"
    },
    {
        "keywords": ["雞", "豬", "牛", "肉"],
        "kcal_range": (180, 280),
        "reason": "一般熟肉類平均熱量"
    },
    {
        "keywords": ["甜", "蛋糕", "甜點"],
        "kcal_range": (300, 450),
        "reason": "甜點類高糖高脂食品"
    },
    {
        "keywords": ["飲料", "奶茶", "茶"],
        "kcal_range": (40, 80),
        "reason": "含糖飲料常見熱量"
    },
]


# =================================================
# 2. 預設補估（完全無法判斷時）
# =================================================
DEFAULT_ESTIMATE = {
    "kcal_range": (150, 250),
    "reason": "以一般熟食平均值進行保守估算"
}


# =================================================
# 3. 對外使用的估算函式
# =================================================
def manual_calorie_estimate(food_zh: str) -> Dict[str, str]:
    """
    傳入中文食物名稱，回傳人工補估結果

    return format:
    {
        "kcal_min": int,
        "kcal_max": int,
        "source": str
    }
    """

    for rule in MANUAL_CALORIE_RULES:
        for kw in rule["keywords"]:
            if kw in food_zh:
                return {
                    "kcal_min": rule["kcal_range"][0],
                    "kcal_max": rule["kcal_range"][1],
                    "source": f"人工補估（{rule['reason']}）"
                }

    # 若完全無法比對任何規則
    return {
        "kcal_min": DEFAULT_ESTIMATE["kcal_range"][0],
        "kcal_max": DEFAULT_ESTIMATE["kcal_range"][1],
        "source": f"人工補估（{DEFAULT_ESTIMATE['reason']}）"
    }


# =================================================
# 4. 測試用（可直接 python manual_calorie_estimator.py）
# =================================================
if __name__ == "__main__":
    test_foods = ["炸豬排", "牛肉麵", "海鮮湯", "未知料理"]

    for food in test_foods:
        est = manual_calorie_estimate(food)
        print(
            f"{food} → {est['kcal_min']}–{est['kcal_max']} kcal "
            f"({est['source']})"
        )
