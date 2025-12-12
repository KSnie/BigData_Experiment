# ==========================
# ID3 Decision Tree on Titanic
# ==========================

import pandas as pd          # ใช้จัดการตารางข้อมูล (DataFrame)
import numpy as np           # ใช้คำนวณเชิงตัวเลขทั่วไป
from collections import Counter   # ใช้นับความถี่ของแต่ละคลาส / ค่า
from math import log2             # ใช้คำนวณ log2 ในสูตร entropy
from sklearn.model_selection import train_test_split  # ใช้แบ่ง train/test

# --------------------------
# 1) อ่านและเตรียมข้อมูล
# --------------------------

# อ่านไฟล์ titanic.csv (ต้องอยู่โฟลเดอร์เดียวกับไฟล์ .py)
data = pd.read_csv("titanic.csv")

# ลบคอลัมน์ที่ไม่ใช้ตามที่โจทย์กำหนด
data = data.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

# เติมค่า missing ของ Age ด้วยค่าเฉลี่ย
data["Age"] = data["Age"].fillna(data["Age"].mean())

# เติมค่า missing ของ Embarked ด้วยค่า mode (ค่าที่พบมากที่สุด)
data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])

# แปลงเพศจาก string เป็นตัวเลข: male=0, female=1
data["Sex"] = data["Sex"].map({"male": 0, "female": 1})

# แปลง Embarked จากตัวอักษรเป็นเลข category (0,1,2,...) เพื่อให้แบ่งตามค่าได้ง่าย
embarked_mapping = {v: i for i, v in enumerate(data["Embarked"].unique())}
data["Embarked"] = data["Embarked"].map(embarked_mapping)

# แยก feature X และ label y  (label คือ Survived ตามโจทย์)
X = data.drop(columns=["Survived"])
y = data["Survived"]

# แบ่งข้อมูลเป็น training set 80% และ test set 20%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------
# 2) ฟังก์ชัน Entropy และ Information Gain
# --------------------------

def entropy(labels):
    """
    คำนวณค่าเอนโทรปีของชุด label (y)
    สูตร: -sum(p_i * log2(p_i))  สำหรับทุกคลาส i
    """
    total = len(labels)                # จำนวนตัวอย่างทั้งหมด
    counts = Counter(labels)           # นับจำนวนของแต่ละคลาส
    ent = 0.0
    for c in counts.values():
        p = c / total                  # ความน่าจะเป็นของคลาสนั้น
        ent -= p * log2(p)             # บวกสะสมค่า -p*log2(p)
    return ent

def information_gain(data, labels, feature):
    """
    คำนวณ Information Gain ของ feature หนึ่งตัว
    data   : DataFrame เฉพาะ feature
    labels : series ของ y ที่สอดคล้องกับ data
    feature: ชื่อคอลัมน์ที่จะใช้เป็นตัวแบ่ง
    """
    # เอนโทรปีของชุดข้อมูลเดิม (ก่อนแบ่ง)
    base_entropy = entropy(labels)

    # รายการค่าที่เป็นไปได้ของ feature นี้
    values = data[feature].unique()
    total = len(data)

    # เอนโทรปีเฉลี่ยแบบถ่วงน้ำหนักหลังจากแบ่งตาม feature
    weighted_entropy = 0.0
    for v in values:
        # เลือก subset ที่ feature == v
        idx = data[feature] == v
        subset_labels = labels[idx]
        # เอนโทรปีของ subset * สัดส่วนของ subset
        weighted_entropy += (len(subset_labels) / total) * entropy(subset_labels)

    # IG = entropy ก่อนแบ่ง - entropy หลังแบ่งถ่วงน้ำหนัก
    return base_entropy - weighted_entropy

# --------------------------
# 3) โครงสร้าง Node และฟังก์ชันสร้างต้นไม้ ID3
# --------------------------

class TreeNode:
    """
    โครงสร้าง node ของต้นไม้ตัดสินใจ (Decision Tree)
    - ถ้าเป็น internal node: มี feature ที่ใช้แบ่ง + children
    - ถ้าเป็น leaf node: มี label (class) อย่างเดียว
    """
    def __init__(self, feature=None, children=None, label=None):
        self.feature = feature                  # ชื่อ feature ที่ใช้ split (None ถ้า leaf)
        self.children = children if children is not None else {}  # dict: value -> child node
        self.label = label                      # class label ถ้าเป็น leaf

def majority_label(labels):
    """
    คืนค่าคลาสที่มีจำนวนมากที่สุดใน labels
    ใช้ในกรณีต้องตัดสินใจด้วย majority vote
    """
    return Counter(labels).most_common(1)[0][0]

def id3(data, labels, features):
    """
    สร้าง decision tree ด้วยอัลกอริทึม ID3 แบบ recursion
    data    : DataFrame ของ feature
    labels  : series ของ y
    features: list ชื่อ feature ที่ยังเหลือให้เลือก
    """
    # 1) ถ้า label ทั้งหมดเป็นคลาสเดียวกัน -> สร้าง leaf ทันที
    if len(set(labels)) == 1:
        return TreeNode(label=list(set(labels))[0])

    # 2) ถ้าไม่มี feature ให้แบ่งต่อแล้ว -> leaf ด้วย majority vote
    if len(features) == 0:
        return TreeNode(label=majority_label(labels))

    # 3) คำนวณ IG ของ feature ทุกตัวแล้วเลือกตัวที่ดีที่สุด
    gains = {f: information_gain(data, labels, f) for f in features}
    best_feature = max(gains, key=gains.get)

    # 4) สร้าง node ใหม่ที่ใช้ best_feature เป็นตัว split
    node = TreeNode(feature=best_feature)

    # 5) แบ่งข้อมูลออกเป็น subset ตามค่าต่าง ๆ ของ best_feature
    for v in data[best_feature].unique():
        idx = data[best_feature] == v
        sub_data = data[idx]
        sub_labels = labels[idx]

        # กรณี subset ว่าง (ป้องกัน error) -> ใช้ majority label ของ node ปัจจุบันแทน
        if len(sub_data) == 0:
            child = TreeNode(label=majority_label(labels))
        else:
            # เรียก id3 ซ้ำ โดยเอา feature นี้ออกจากรายการ (ห้ามใช้ซ้ำ)
            new_features = [f for f in features if f != best_feature]
            child = id3(sub_data, sub_labels, new_features)

        # ผูก child node กับค่าของ feature v
        node.children[v] = child

    return node

# --------------------------
# 4) ฟังก์ชันทำนายจากต้นไม้
# --------------------------

def predict_one(node, x):
    """
    ทำนายผลของตัวอย่างเดี่ยว (หนึ่งแถว) จากต้นไม้
    node: รากของต้นไม้
    x   : แถวข้อมูลของผู้โดยสาร (pandas Series)
    """
    # ถ้า node เป็น leaf -> คืน label ได้เลย
    if node.label is not None:
        return node.label

    # ถ้าเป็น internal node -> ดูค่า feature ของ node นี้
    v = x[node.feature]

    # ถ้าไม่มี child สำหรับค่านี้ (กรณีเจอค่าใหม่ในชุดทดสอบ) -> ให้ default เป็น majority แบบง่าย ๆ
    if v not in node.children:
        # ตรงนี้จะเลือกเป็น 0 หรือ majority ของ train ก็ได้
        return 0

    # เดินลงไปที่ child node ต่อ (recursion)
    return predict_one(node.children[v], x)

def predict(node, X):
    """
    ทำนายทั้ง DataFrame X
    node: รากของต้นไม้
    X   : DataFrame ของชุดข้อมูลที่ต้องการทำนาย
    """
    preds = []
    for i in range(len(X)):
        preds.append(predict_one(node, X.iloc[i]))
    return preds

# --------------------------
# 5) สร้างต้นไม้ ฝึก และประเมินผล
# --------------------------

# รายชื่อ feature ที่ใช้สร้างต้นไม้
feature_names = list(X_train.columns)

# สร้างต้นไม้ด้วย ID3 จาก training set
tree_root = id3(X_train, y_train, feature_names)

# ทำนายผลบนชุดทดสอบ
y_pred = predict(tree_root, X_test)

# คำนวณค่า accuracy
accuracy = np.mean(np.array(y_pred) == np.array(y_test))
print("Accuracy on test set:", accuracy)

# (เลือกทำ) บันทึกผลจริงกับผลทำนายออกเป็นไฟล์ csv เผื่อเอาไปวิเคราะห์ต่อ
pd.DataFrame({"y_true": y_test.values, "y_pred": y_pred}).to_csv(
    "id3_titanic_predictions.csv", index=False
)
print("Saved predictions to id3_titanic_predictions.csv")
