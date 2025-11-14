import pandas as pd

# --- จุดแก้ 1: อ่านไฟล์แบบ binary แล้วแปลงเป็น list ของชื่อสินค้า ---
df = pd.read_csv('ShopCT.csv')
transactions = []

for _, row in df.iterrows():
    # แก้จาก 'transaction = list(row)' เป็น 'basket = [col for col in df.columns if row[col] == 1]'
    # เพราะเราต้องการชื่อสินค้า ไม่ใช่เลข 0/1
    basket = [col for col in df.columns if row[col] == 1]  # <--- จุดแก้สำคัญ
    transactions.append(basket)

def create_c1(dataset):
    c1=[]
    for transaction in dataset:
        for item in transaction:
            if not {item} in c1:
                c1.append({item})
    c1.sort()
    return list(map(frozenset, c1))

def create_freq_transaction(dataset, ck, min_support=0.5):
    sscnt = {}
    for transaction in dataset:
        for can in ck:
            if can.issubset(transaction):
                sscnt[can] = sscnt.get(can, 0) + 1
    num_transactions = float(len(dataset))
    freq_transaction = []
    support_data = {}
    for key in sscnt:
        support = sscnt[key]/num_transactions
        support_data[key] = support
        if support >= min_support:
            freq_transaction.append(key)
    return support_data, freq_transaction

def create_ck(freq_k_transaction):
    ck = []
    k = len(freq_k_transaction)
    for i in range(k):
        for j in range(i + 1, k):
            t1 = freq_k_transaction[i]
            t2 = freq_k_transaction[j]
            t = t1 | t2
            if (not t in ck) and (len(t) == len(freq_k_transaction[0]) + 1):
                ck.append(t)
    return ck

def apriori(dataset, min_support=0.5):
    c1 = create_c1(dataset)
    support_data, freq_transaction_1 = create_freq_transaction(dataset, c1, min_support=min_support)
    all_freq_transaction = [freq_transaction_1]
    k = 2
    while len(all_freq_transaction[-1]) > 0:
        ck = create_ck(all_freq_transaction[-1])
        support_data_k, freq_transaction_k = create_freq_transaction(dataset, ck, min_support=min_support)
        support_data.update(support_data_k)
        all_freq_transaction.append(freq_transaction_k)
        k = k + 1
    return support_data, all_freq_transaction

def create_subset(fromlist, tolist):
    for i in range(len(fromlist)):
        t = [fromlist[i]]
        tt = frozenset(set(fromlist) - set(t))
        if not tt in tolist:
            tolist.append(tt)
        tt = list(t)
        if len(tt) > 1:
            create_subset(tt, tolist)
    return None

def cal_conf(fre_set, h, support_data, rulelist, min_conf):
    for after in h:
        conf = support_data[fre_set]/support_data[fre_set-after]
        lift = support_data[fre_set]/(support_data[fre_set-after] * support_data[after])
        if conf >= min_conf and lift > 1:
            print(fre_set-after, '-->', after, '\n',
                  'before support:', round(support_data[fre_set-after],3), ',',
                  'after support:', round(support_data[after],3), ',',
                  'together support:', round(support_data[fre_set],3), ',',
                  'conf:', round(conf,3), ',',
                  'lift:', round(lift,3))
            rulelist.append((fre_set-after, after, round(conf,3)))
    return None

def create_rules(support_data, all_freq_transaction, min_conf=0.8):
    all_rulelist = []
    for i in range(1, len(all_freq_transaction)):
        for fre_set in all_freq_transaction[i]:
            fre_list = list(fre_set)
            all_subset = []
            create_subset(fre_list, all_subset)
            cal_conf(fre_set, all_subset, support_data, all_rulelist, min_conf)
    return all_rulelist

support_data, all_freq_transaction = apriori(transactions, min_support=0.05)  # หรือปรับค่า min_support ตามต้องการ
x  = create_rules(support_data, all_freq_transaction, min_conf=0.3)  # ปรับ confidence threshold ตามที่อยากเห็น rule

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from matplotlib import rcParams

# เลือกชื่อฟอนต์ที่รองรับภาษาไทยในเครื่องเรา เช่น 'Tahoma' หรือ 'Thonburi'
rcParams['font.family'] = 'Tahoma'      # หรือ 'Thonburi' / ฟอนต์ไทยอื่น
rcParams['axes.unicode_minus'] = False  # กันปัญหาเครื่องหมายลบเป็นกล่อง

# ดึงเฉพาะ frequent 1-itemset (ขนาด 1)
freq_1 = all_freq_transaction[0]
items = []
supports = []
for f in freq_1:
    # f คือ frozenset({'item_name'})
    item_name = list(f)[0]
    items.append(item_name)
    supports.append(support_data[f])

plt.figure(figsize=(10, 5))
plt.bar(items, supports)
plt.xticks(rotation=45, ha='right')
plt.ylabel('Support')
plt.title('Support ของสินค้าแต่ละตัว')
plt.tight_layout()
plt.show()
