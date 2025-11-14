import pandas as pd

# --- (1) ขั้นตอนเตรียมข้อมูล (INPUT: Dataset D) ---
# อ่านไฟล์ ShopCT.csv แล้วแปลงแต่ละแถวเป็น basket (รายการชื่อสินค้าในแต่ละ transaction)
df = pd.read_csv('ShopCT.csv')
transactions = []
for _, row in df.iterrows():
    basket = [col for col in df.columns if row[col] == 1]  # → "For all transactions t∈D" (จาก pseudocode 1)
    transactions.append(basket)

# (ใช้เก็บผลลัพธ์ rule สำหรับแสดงเป็นตารางท้ายโปรแกรม)
rule_rows = []

# --- (2) create_c1 สร้าง candidate 1-itemsets (Clark: L1 = {large 1-itemsets}) ---
def create_c1(dataset):
    c1 = []
    for transaction in dataset:
        for item in transaction:
            # สร้างชุดที่มี 1 item (frozenset ทำให้ set นี้มาใช้เทียบและ hash ได้)
            if not frozenset([item]) in c1:
                c1.append(frozenset([item]))
    c1.sort()
    return c1

# --- (1) & (2) สแกน transaction ทั้งหมด/นับ support ---
def create_freq_transaction(dataset, ck, min_support=0.5):
    # ck = candidate set of k-itemsets (Ck)
    sscnt = {}
    for transaction in dataset:
        tran_set = set(transaction)
        for can in ck:
            if can.issubset(tran_set):
                sscnt[can] = sscnt.get(can, 0) + 1  # "for all candidates c ∈Ct do c.count++;"
    num_transactions = float(len(dataset))
    freq_transaction = []
    support_data = {}
    for key in sscnt:
        support = sscnt[key] / num_transactions
        support_data[key] = support
        if support >= min_support:
            freq_transaction.append(key)    # Lk = {c ∈Ck| c.count ≧ minsup_count }
    return support_data, freq_transaction

# --- (2) ขั้นตอน apriori-gen (สร้าง candidate k-itemsets) ---
def create_ck(freq_k_transaction):
    ck = []
    k = len(freq_k_transaction)
    l = len(list(freq_k_transaction[0])) + 1  # จำนวน items ของ candidate ชุดใหม่
    for i in range(k):
        for j in range(i+1, k):
            t1 = freq_k_transaction[i]
            t2 = freq_k_transaction[j]
            t = t1 | t2
            # เงื่อนไขแบบ join ใน pseudocode "If p.item1=q.item1,...,p.itemk-2=q.itemk-2, p.itemk-1<q.itemk-1"
            if len(t) == l and t not in ck:
                # (ตรงนี้ใน pseudocode จะมีการ prune ด้วย has_infrequent_subset ก่อน add)
                ck.append(t)
    return ck

# --- (2)-(3) Apriori main loop ---
def apriori(dataset, min_support=0.5):
    c1 = create_c1(dataset)                   # L1 = create_c1(D)
    support_data, freq_transaction_1 = create_freq_transaction(dataset, c1, min_support=min_support)
    all_freq_transaction = [freq_transaction_1] # L = L1
    k = 2
    while len(all_freq_transaction[-1]) > 0:   # "For (k=2; Lk-1≠Φ; k++)"
        ck = create_ck(all_freq_transaction[-1]) # Ck = apriori-gen (Lk-1)
        support_data_k, freq_transaction_k = create_freq_transaction(dataset, ck, min_support=min_support)
        support_data.update(support_data_k)
        all_freq_transaction.append(freq_transaction_k) # Lk = ...; L = L ∪ Lk
        k += 1
    return support_data, all_freq_transaction  # L

# --- (3) ตรวจสอบว่ามี subset ไหนใน candidate ที่ไม่ frequent หรือไม่ (prune, optional ใน code นี้) ---
def has_infrequent_subset(candidate, prev_freq_sets):
    # สำหรับ candidate k-itemset c และ (k-1)-frequent itemset Lk-1
    from itertools import combinations
    k = len(candidate)
    for subset in combinations(candidate, k-1):
        if frozenset(subset) not in prev_freq_sets:
            return True    # "If Not(S∈Lk-1) THEN return TRUE;"
    return False           # "Return FALSE;"

# --- (4) เรียกใช้ stage สร้าง rule จาก frequent itemsets ---
def create_subset(fromlist, tolist):
    # สร้างทุก subset ของ fromlist ใช้สำหรับ association rule generation
    n = len(fromlist)
    for i in range(n):
        t = [fromlist[i]]
        tt = frozenset(set(fromlist) - set(t))
        if len(tt) > 0 and not tt in tolist:
            tolist.append(tt)
            if len(tt) > 1:
                create_subset(list(tt), tolist)
    return None

# --- (4)-(5) สร้าง rule, คำนวณ confidence/lift และเก็บลงตาราง ---
def cal_conf(fre_set, h, support_data, rulelist, min_conf):
    for after in h:
        # conf = support(lk)/support(xm-1);
        if support_data.get(fre_set - after, 0) == 0 or support_data.get(after, 0) == 0:
            continue
        conf = support_data[fre_set] / support_data[fre_set - after]
        lift = support_data[fre_set] / (support_data[fre_set - after] * support_data[after])
        if conf >= min_conf and lift > 1:
            lhs_str = str(fre_set - after)
            rhs_str = str(after)
            rule_rows.append({
                "LHS": lhs_str,              # Output rules: xm-1->(lk-xm-1)
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
    # "FOR each frequent itemset lk in L; generules(lk,lk);"
    all_rulelist = []
    for i in range(1, len(all_freq_transaction)):
        for fre_set in all_freq_transaction[i]:
            fre_list = list(fre_set)
            all_subset = []
            create_subset(fre_list, all_subset)
            cal_conf(fre_set, all_subset, support_data, all_rulelist, min_conf)
    return all_rulelist

# --- (1)-(5): กระบวนการหลัก ---
if __name__ == "__main__":
    # ขั้นตอน Apriori: หา frequent itemsets
    support_data, all_freq_transaction = apriori(transactions, min_support=0.05)  # Input: D, min_supp_count
    # สร้าง strong association rules
    create_rules(support_data, all_freq_transaction, min_conf=0.3)                # Input: L, min_conf

    # post-processing: ทำเป็นตารางและ sort (ส่วนนี้ไม่ได้อยู่ใน pseudocode ดั้งเดิม แต่อ่านผลง่ายขึ้น)
    rules_df = pd.DataFrame(rule_rows, columns=[
        "LHS", "RHS",
        "support_before", "support_after", "support_together",
        "confidence", "lift"
    ])
    rules_df = rules_df.sort_values(by="lift", ascending=False)
    print(rules_df.to_string(index=False))
