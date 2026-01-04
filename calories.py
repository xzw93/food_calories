import pandas as pd
from pathlib import Path
from manual_calorie_estimator import manual_calorie_estimate
import re
from typing import Optional, Tuple

# =================================================
# 1. 讀取食品營養成分資料庫（2024 UPDATE2）
# =================================================
DATA_PATH = Path(__file__).parent / "食品營養成分資料庫2024UPDATE2.xlsx"

df_raw = pd.read_excel(DATA_PATH, header=1)

df = df_raw.rename(columns={
    "食品分類": "category",
    "樣品名稱": "name",
    "內容物描述": "desc",
    "熱量(kcal)": "kcal"
})

df = df.dropna(subset=["kcal"])
df["kcal"] = pd.to_numeric(df["kcal"], errors="coerce")
df = df.dropna(subset=["kcal"])

# =================================================
# 2. 僅保留「Top100 且有營養資料」的品項
# （由前一步自動產生）
# =================================================
TOP100_OK_PATH = Path(__file__).parent / "outputs_calorie_check" / "top100_with_nutrition.csv"

df_ok = pd.read_csv(TOP100_OK_PATH)

# 建立：中文品項 → 搜尋關鍵字
# 這裡策略：直接用中文名稱本身
TAIWAN_FOOD_TO_KEYWORDS = {
    row["zh_name"]: [row["zh_name"]]
    for _, row in df_ok.iterrows()
}

# =================================================
# 3. 查詢熱量（每 100 g）
# =================================================
def get_calorie(food_zh: str) -> str:
    # 1️⃣ 先查官方資料
    keywords = TAIWAN_FOOD_TO_KEYWORDS.get(food_zh)

    if keywords:
        matched = pd.DataFrame()
        for kw in keywords:
            matched = pd.concat([
                matched,
                df[df["name"].str.contains(kw, na=False)]
            ])

        if not matched.empty:
            kcal_min = int(matched["kcal"].min())
            kcal_max = int(matched["kcal"].max())
            return f"約 {kcal_min}–{kcal_max} kcal（每 100 g，官方資料）"

    # 2️⃣ 官方查不到 → 人工補估
    est = manual_calorie_estimate(food_zh)
    return (
        f"約 {est['kcal_min']}–{est['kcal_max']} kcal（每 100 g）\n"
        f"※ {est['source']}"
    )

def parse_kcal_range(text: str) -> Tuple[Optional[int], Optional[int]]:
    """
    從 '約 249–321 kcal（每 100 g ...）' 抓出 (249, 321)
    抓不到就回 (None, None)
    """
    if not text:
        return None, None

    # 支援 249–321 / 249-321 / 249~321
    m = re.search(r"(\d+)\s*[–\-~]\s*(\d+)", text)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return (min(a, b), max(a, b))

    # 只有單一數字：例如 '約 300 kcal（每 100 g）'
    m2 = re.search(r"(\d+)\s*kcal", text, flags=re.IGNORECASE)
    if m2:
        v = int(m2.group(1))
        return v, v

    return None, None