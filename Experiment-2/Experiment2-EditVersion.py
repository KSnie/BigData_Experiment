import pandas as pd

# -----------------------------
# 1) อ่านไฟล์และสร้าง transactions
# -----------------------------

# --- จุดแก้ 1: อ่านไฟล์แบบ binary แล้วแปลงเป็น list ของชื่อสินค้า ---
df = pd.read_csv('ShopCT.csv')
transactions = []
for _, row in df.iterrows():
    # แก้จาก 'transaction = list(row)' เป็น 'basket = [col for col in df.columns if row[col] == 1]'
    # เพราะเราต้องการชื่อสินค้า ไม่ใช่เลข 0/1
    basket = [col for col in df.columns if row[col] == 1]  # <--- จุดแก้สำคัญ
    transactions.append(basket)

# เพิ่ม: list สำหรับเก็บผลลัพธ์กฎ ในรูปแบบตาราง
rule_rows = []  # <--- ใหม่: ใช้เก็บข้อมูลทุก rule แล้วค่อยสร้าง DataFrame ทีเดียว

# -----------------------------
# 2) ฟังก์ชัน Apriori
# -----------------------------
# --- ด้านล่างคือฟังก์ชัน Apriori ใช้กับ transactions ที่แก้แล้ว ---
def create_c1(dataset):
    c1 = []
    for transaction in dataset:
        for item in transaction:
            # ต้องใช้ frozenset([item]) แทน {item} เพื่อให้ Hash ได้และเปรียบเทียบในชุดได้ถูกต้อง
            if not frozenset([item]) in c1:  # <--- จุดแก้สำคัญ (จาก {item} เป็น frozenset([item]))
                c1.append(frozenset([item]))
    c1.sort()
    return c1

def create_freq_transaction(dataset, ck, min_support=0.5):
    sscnt = {}
    for transaction in dataset:
        tran_set = set(transaction)  # <--- จุดแก้ใหม่ (convert ให้เป็น set สำหรับ issubset)
        for can in ck:
            if can.issubset(tran_set):  # <--- เปรียบเทียบด้วย set ไม่ใช่ list
                sscnt[can] = sscnt.get(can, 0) + 1
    num_transactions = float(len(dataset))
    freq_transaction = []
    support_data = {}
    for key in sscnt:
        support = sscnt[key] / num_transactions
        support_data[key] = support
        if support >= min_support:
            freq_transaction.append(key)
    return support_data, freq_transaction

def create_ck(freq_k_transaction):
    ck = []
    k = len(freq_k_transaction)
    l = len(list(freq_k_transaction[0])) + 1  # <--- เพิ่มความถูกต้องในการนับ item ใน candidate set
    for i in range(k):
        for j in range(i+1, k):
            t1 = freq_k_transaction[i]
            t2 = freq_k_transaction[j]
            t = t1 | t2
            if len(t) == l and t not in ck:  # <--- เงื่อนไขเปรียบเทียบผู้สมัครถูกต้อง
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
        k += 1
    return support_data, all_freq_transaction

# -----------------------------
# 3) สร้าง subset สำหรับกฎ
# -----------------------------

def create_subset(fromlist, tolist):
    n = len(fromlist)
    for i in range(n):
        t = [fromlist[i]]
        tt = frozenset(set(fromlist) - set(t))  # <--- ใช้ frozenset สำหรับ subset (แก้จาก list -> frozenset)
        if len(tt) > 0 and not tt in tolist:
            tolist.append(tt)
            if len(tt) > 1:
                create_subset(list(tt), tolist)
    return None

# -----------------------------
# 4) คำนวณ confidence & lift และเก็บเป็นตาราง
# -----------------------------
def cal_conf(fre_set, h, support_data, rulelist, min_conf):
    for after in h:
        # แก้: เงื่อนไขป้องกัน division by zero
        if support_data.get(fre_set - after, 0) == 0 or support_data.get(after, 0) == 0:
            continue  # <--- จุดแก้
        conf = support_data[fre_set] / support_data[fre_set - after]
        lift = support_data[fre_set] / (support_data[fre_set - after] * support_data[after])

        # เดิม: print เป็นบรรทัดข้อความ
        # ตอนนี้เปลี่ยนเป็นเก็บลง rule_rows เพื่อเอาไปสร้างตารางทีเดียว
        if conf >= min_conf and lift > 1:
            # print(f"{fre_set - after} --> {after}\n before support: {support_data[fre_set - after]:.3f}, after support: {support_data[after]:.3f}, together support: {support_data[fre_set]:.3f}, conf: {conf:.3f}, lift: {lift:.3f}")
            # เก็บข้อมูลสำหรับสร้างตาราง
            lhs_str = str(fre_set - after)          # <--- ใหม่: แปลง LHS เป็น string
            rhs_str = str(after)                    # <--- ใหม่: แปลง RHS เป็น string
            rule_rows.append({                      # <--- ใหม่: เก็บข้อมูลทุกคอลัมน์ของตาราง
                "LHS": lhs_str,
                "RHS": rhs_str,
                "support_before": round(support_data[fre_set - after], 3),
                "support_after": round(support_data[after], 3),
                "support_together": round(support_data[fre_set], 3),
                "confidence": round(conf, 3),
                "lift": round(lift, 3)
            })
            rulelist.append((fre_set - after, after, round(conf,3)))
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

# ตอนเรียกใช้งานให้ใช้:
if __name__ == "__main__":
    support_data, all_freq_transaction = apriori(transactions, min_support=0.05)  # หรือปรับค่า min_support ตามต้องการ
    create_rules(support_data, all_freq_transaction, min_conf=0.3)  # ปรับ confidence threshold ตามที่อยากเห็น rule

    # ใหม่: สร้าง DataFrame แล้ว print เป็นตาราง
    rules_df = pd.DataFrame(rule_rows, columns=[
        "LHS", "RHS",
        "support_before", "support_after", "support_together",
        "confidence", "lift"
    ])  # <--- ใหม่: กำหนดคอลัมน์ให้ชัดเจน

    # แสดงเป็นตารางใน console
    rules_df = rules_df.sort_values(by="lift", ascending=False)  # <--- sort ตาม lift

    print(rules_df.to_string(index=False))  # <--- ใหม่: print ตารางสวยๆ

    # ถ้าอยากเซฟเป็นไฟล์ CSV ด้วย
    # rules_df.to_csv("association_rules_table.csv", index=False)  # <--- ใหม่: export เป็น CSV (เปิดใน Excel ได้)
