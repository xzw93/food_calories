import io
from pathlib import Path
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image


# =================================================
# 1. UECFOOD 100 類
# =================================================
UEC100_CLASSES = [
    "miso_soup","rice","ramen_noodle","green_salad","beef_curry","omelet",
    "hamburger","egg_sunny-side_up","toast","grilled_pacific_saury",
    "fried_rice","beef_bowl","jiaozi","chinese_soup","sandwiches",
    "soba_noodle","cold_tofu","fried_chicken","french_fries","sushi",
    "udon_noodle","spaghetti","pork_cutlet_on_rice","chip_butty","natto",
    "sashimi_bowl","cutlet_curry","sirloin_cutlet","beef_noodle","mixed_rice",
    "Japanese-style_pancake","tempura_bowl","hambarg_steak","pizza",
    "takoyaki","fried_noodle","eels_on_rice","potato_salad","croquette",
    "dipping_noodles","omelet_with_fried_rice","spaghetti_meat_sauce",
    "sashimi","fish-shaped_pancake_with_bean_jam","sukiyaki",
    "chicken-'n'-egg_on_rice","fried_fish","croissant","Pork_Sticky_Noodles",
    "sauteed_vegetables","spicy_chili-flavored_tofu",
    "dish_consisting_of_stir-fried_potato,_eggplant_and_green_pepper",
    "fish_ball_soup","oxtail_soup",
    "salt_&_pepper_fried_shrimp_with_shell","Thai_papaya_salad",
    "beef_noodle_soup","braised_pork_meat_ball_with_napa_cabbage",
    "fine_white_noodles","khao_soi","sausage","shrimp_with_chill_source",
    "tempura","winter_melon_soup","Wonton_soup","yudofu","chilled_noodle",
    "ginger_pork_saute","hot_&_sour_soup","nasi_campur","oatmeal","pho",
    "pork_miso_soup","Rice_crispy_pork","three_cup_chicken","zoni",
    "chow_mein","fried_pork_dumplings_served_in_soup","grilled_salmon",
    "popcorn","rice_gruel","roast_duck","seasoned_beef_with_potatoes",
    "steamed_spareribs","broiled_eel_bowl","charcoal-boiled_pork_neck",
    "churro","clear_soup","eight_treasure_rice","fried_shrimp","gratin",
    "Japanese_tofu_and_vegetable_chowder","lamb_kebabs","pilaf",
    "pork_fillet_cutlet","pork_loin_cutlet","steamed_meat_dumpling",
    "vegetable_tempura","adobo","ayam_bakar"
]
NUM_CLASSES = len(UEC100_CLASSES)
IDX_TO_CLASS = {i: name for i, name in enumerate(UEC100_CLASSES)}


# =================================================
# 2. 英文 → 中文顯示（顯示層，不影響訓練）
# =================================================
UEC_TO_ZH_OVERRIDE = {
    "miso_soup": "味噌湯",
    "rice": "白飯",
    "ramen_noodle": "拉麵",
    "green_salad": "生菜沙拉",
    "beef_curry": "牛肉咖哩",
    "omelet": "歐姆蛋",
    "hamburger": "漢堡",
    "egg_sunny-side_up": "荷包蛋",
    "toast": "吐司",
    "grilled_pacific_saury": "烤秋刀魚",
    "fried_rice": "炒飯",
    "beef_bowl": "牛丼",
    "jiaozi": "餃子",
    "chinese_soup": "中式湯",
    "sandwiches": "三明治",
    "soba_noodle": "蕎麥麵",
    "cold_tofu": "冷豆腐",
    "fried_chicken": "炸雞",
    "french_fries": "薯條",
    "sushi": "壽司",
    "udon_noodle": "烏龍麵",
    "spaghetti": "義大利麵",
    "pork_cutlet_on_rice": "豬排丼",
    "chip_butty": "薯條三明治",
    "natto": "納豆",
    "sashimi_bowl": "生魚片丼",
    "cutlet_curry": "豬排咖哩飯",
    "sirloin_cutlet": "沙朗炸排",
    "beef_noodle": "牛肉麵",
    "mixed_rice": "什錦拌飯",
    "Japanese-style_pancake": "日式鬆餅",
    "tempura_bowl": "天婦羅丼",
    "hambarg_steak": "漢堡排",
    "pizza": "披薩",
    "takoyaki": "章魚燒",
    "fried_noodle": "炒麵",
    "eels_on_rice": "鰻魚飯",
    "potato_salad": "馬鈴薯沙拉",
    "croquette": "可樂餅",
    "dipping_noodles": "沾麵",
    "omelet_with_fried_rice": "蛋包炒飯",
    "spaghetti_meat_sauce": "肉醬義大利麵",
    "sashimi": "生魚片",
    "fish-shaped_pancake_with_bean_jam": "鯛魚燒",
    "sukiyaki": "壽喜燒",
    "chicken-'n'-egg_on_rice": "親子丼",
    "fried_fish": "炸魚",
    "croissant": "可頌",
    "Pork_Sticky_Noodles": "豬肉拌麵",
    "sauteed_vegetables": "炒蔬菜",
    "spicy_chili-flavored_tofu": "麻辣豆腐",
    "dish_consisting_of_stir-fried_potato,_eggplant_and_green_pepper": "地三鮮",
    "fish_ball_soup": "魚丸湯",
    "oxtail_soup": "牛尾湯",
    "salt_&_pepper_fried_shrimp_with_shell": "椒鹽帶殼炸蝦",
    "Thai_papaya_salad": "泰式青木瓜沙拉",
    "beef_noodle_soup": "牛肉湯麵",
    "braised_pork_meat_ball_with_napa_cabbage": "白菜滷豬肉丸",
    "fine_white_noodles": "細白麵線",
    "khao_soi": "泰北咖哩麵",
    "sausage": "香腸",
    "shrimp_with_chill_source": "辣醬蝦",
    "tempura": "天婦羅",
    "winter_melon_soup": "冬瓜湯",
    "Wonton_soup": "餛飩湯",
    "yudofu": "湯豆腐",
    "chilled_noodle": "冷麵",
    "ginger_pork_saute": "薑汁豬肉",
    "hot_&_sour_soup": "酸辣湯",
    "nasi_campur": "印尼什錦飯",
    "oatmeal": "燕麥粥",
    "pho": "越南河粉",
    "pork_miso_soup": "豬肉味噌湯",
    "Rice_crispy_pork": "脆皮豬飯",
    "three_cup_chicken": "三杯雞",
    "zoni": "雜煮湯",
    "chow_mein": "廣式炒麵",
    "fried_pork_dumplings_served_in_soup": "豬肉餃子湯",
    "grilled_salmon": "烤鮭魚",
    "popcorn": "爆米花",
    "rice_gruel": "稀飯",
    "roast_duck": "烤鴨",
    "seasoned_beef_with_potatoes": "馬鈴薯燉牛肉",
    "steamed_spareribs": "蒸排骨",
    "broiled_eel_bowl": "蒲燒鰻魚丼",
    "charcoal-boiled_pork_neck": "炭煮豬頸肉",
    "churro": "吉拿棒",
    "clear_soup": "清湯",
    "eight_treasure_rice": "八寶飯",
    "fried_shrimp": "炸蝦",
    "gratin": "焗烤",
    "Japanese_tofu_and_vegetable_chowder": "日式豆腐蔬菜雜燴湯",
    "lamb_kebabs": "羊肉串",
    "pilaf": "皮拉夫飯",
    "pork_fillet_cutlet": "豬菲力炸排",
    "pork_loin_cutlet": "豬里肌炸排",
    "steamed_meat_dumpling": "蒸肉餃",
    "vegetable_tempura": "蔬菜天婦羅",
    "adobo": "阿多波",
    "ayam_bakar": "印尼烤雞",
}
def uec_name_to_zh(name: str) -> str:
    if name in UEC_TO_ZH_OVERRIDE:
        return UEC_TO_ZH_OVERRIDE[name]
    s = name.replace("_", " ").replace("-", " ").replace("&", " ").replace(",", " ").lower()
    replaces = [
        ("noodle", "麵"),
        ("soup", "湯"),
        ("rice", "飯"),
        ("chicken", "雞"),
        ("pork", "豬"),
        ("beef", "牛"),
        ("fish", "魚"),
        ("shrimp", "蝦"),
        ("egg", "蛋"),
        ("salad", "沙拉"),
        ("curry", "咖哩"),
        ("fried", "炸"),
        ("grilled", "烤"),
        ("steamed", "蒸"),
        ("tofu", "豆腐"),
    ]
    for a, b in replaces:
        s = s.replace(a, b)
    return s.strip()


# =================================================
# 3. 自建 ResNet-like
# =================================================
def conv3x3(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)

class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(identity)
        return self.relu(out + identity)


class FoodCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
        )
        self.in_ch = 64
        self.layer1 = self._make_layer(64, 2, 1)
        self.layer2 = self._make_layer(128, 2, 2)
        self.layer3 = self._make_layer(256, 2, 2)
        self.layer4 = self._make_layer(512, 2, 2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_ch, blocks, stride):
        layers = [BasicBlock(self.in_ch, out_ch, stride)]
        self.in_ch = out_ch
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


# =================================================
# 4. 推論前處理
# =================================================
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# =================================================
# 5. 🔥 Lazy-load 模型（重點）
# =================================================
_MODEL = None

def load_model(device=None):
    global _MODEL
    if _MODEL is None:
        device = device or torch.device("cpu")  # Render 一律用 CPU
        weight_path = Path(__file__).parent / "outputs_100" / "uec100_model.pth"
        model = FoodCNN().to(device)
        model.load_state_dict(torch.load(weight_path, map_location=device))
        model.eval()
        _MODEL = model
    return _MODEL


@torch.no_grad()
def predict_food(image_bytes: bytes):
    device = torch.device("cpu")
    model = load_model(device)

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    x = inference_transform(img).unsqueeze(0)

    logits = model(x)
    prob = torch.softmax(logits, dim=1)
    conf, idx = prob.max(dim=1)

    idx = int(idx.item())
    food_en = IDX_TO_CLASS[idx]
    food_zh = uec_name_to_zh(food_en)
    return food_en, food_zh, idx, float(conf.item())
