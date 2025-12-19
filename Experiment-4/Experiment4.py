# ======================================================================
# Experiment 4 : Implementation of K-Means Clustering algorithm on Iris
# งานนี้เป็นการทดลอง unsupervised learning แบบ K-Means
# - ใช้ข้อมูล Iris dataset (เลือก 2 คุณลักษณะ)
# - แบ่งข้อมูลออกเป็น k กลุ่ม (ที่นี่ใช้ k = 3 ตามจำนวน species จริง)
# - แสดงขั้นตอนการทำงานของอัลกอริทึมในแต่ละ iteration ด้วยรูปภาพ
# - คำนวณ evaluation function (inertia / SSE) เพื่อดูคุณภาพของการจัดกลุ่ม
# - บันทึกรูปและผลลัพธ์ทั้งหมดลงโฟลเดอร์ result
# ======================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy

# ------------------------------
# 1) เตรียมโฟลเดอร์สำหรับเก็บผลลัพธ์
# ------------------------------

# __file__ คือ path ของไฟล์ Python ปัจจุบัน (Experiment4.py)
# os.path.abspath(__file__) แปลงเป็น path แบบเต็ม
# os.path.dirname(...) เอาเฉพาะโฟลเดอร์ที่หุ้มไฟล์นี้ (เช่น .../Experiment-4)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ต้องการให้เก็บผลลัพธ์ไว้ในโฟลเดอร์ย่อยชื่อ "result"
# path ที่ได้ก็จะประมาณ .../Experiment-4/result
RESULT_DIR = os.path.join(BASE_DIR, "result")

# ถ้าโฟลเดอร์ result ยังไม่มี ให้สร้างขึ้นมา (exist_ok=True = ถ้ามีอยู่แล้วจะไม่ error)
os.makedirs(RESULT_DIR, exist_ok=True)

# ------------------------------
# 2) โหลดข้อมูล Iris และเลือก feature
# ------------------------------

# อ่านไฟล์ iris.csv ที่อยู่ในโฟลเดอร์เดียวกันกับไฟล์ Python
# ในไฟล์นี้มีคอลัมน์: Sepal.Length, Sepal.Width, Petal.Length, Petal.Width, Species
csv_path = os.path.join(BASE_DIR, "iris.csv")
df = pd.read_csv(csv_path)

# เลือกเฉพาะ 2 คอลัมน์เป็น attribute vector ของแต่ละ sample
# ตัวอย่างนี้ใช้ Petal.Length และ Petal.Width เพราะแยกคลัสเตอร์ได้ชัดเจน
# ได้ X เป็น numpy array รูป (จำนวนแถวข้อมูล, 2)
X = df[["Petal.Length", "Petal.Width"]].values

print("จำนวนตัวอย่าง = ", X.shape[0])
print("จำนวนคุณลักษณะต่อจุดข้อมูล = ", X.shape[1])


# ------------------------------
# 3) ฟังก์ชันระยะทาง (Distance function)
# ------------------------------

def euclidean_distance(a, b):
    """
    ฟังก์ชันคำนวณระยะทางแบบ Euclidean ระหว่างเวกเตอร์ a และ b
    - ถ้า a เป็นจุดเดียวและ b เป็นจุดเดียว => คืนค่าระยะทางจริงๆ 1 ค่า
    - ถ้า a เป็นชุดของหลายจุด และ b เป็น 1 จุด => ใช้ axis=-1 ให้ได้ระยะทางของทุกจุด
    สูตร: sqrt( (x1 - c1)^2 + (x2 - c2)^2 + ... + (xd - cd)^2 )
    ซึ่งเป็นระยะทางมาตรฐานที่ K-Means นิยมใช้ [web:39][web:40]
    """
    return np.linalg.norm(a - b, axis=-1)


# ------------------------------
# 4) การสุ่มจุดเริ่มต้นของ centroid
# ------------------------------

def initialize_centroids(X, k, random_state=42):
    """
    เลือกจุดเริ่มต้นของ centroid จำนวน k จุดจากข้อมูล X แบบสุ่ม
    - X : ข้อมูลทั้งหมด รูป (n_samples, n_features)
    - k : จำนวนคลัสเตอร์
    - random_state : seed เพื่อให้สุ่มแล้วได้ผลเหมือนเดิมทุกครั้ง (ทำซ้ำได้)
    วิธีนี้เรียกว่า random initialization (แบบง่ายที่สุดของ K-Means) [web:42][web:48]
    """
    rng = np.random.RandomState(random_state)   # สร้างตัวสุ่มด้วย seed ที่กำหนด
    indices = rng.choice(len(X), size=k, replace=False)  # สุ่ม index k ตัว ไม่ซ้ำกัน
    return X[indices]  # เอาค่า feature ของจุดที่สุ่มได้มาเป็น centroid เริ่มต้น


# ------------------------------
# 5) ขั้นตอน assign จุดข้อมูลเข้าคลัสเตอร์ที่ใกล้ที่สุด
# ------------------------------

def assign_clusters(X, centroids):
    """
    รับข้อมูล X และ centroid ปัจจุบัน
    คืนค่า labels ของแต่ละจุดว่าอยู่ cluster ไหน (0 .. k-1)

    หลักการ:
    - คำนวณระยะทางจากทุกจุด ไปยัง centroid ทุกตัว
    - เลือก centroid ที่มีระยะทางน้อยที่สุด => นั่นคือ cluster ของจุดนั้น
    """
    n_samples = X.shape[0]
    k = centroids.shape[0]

    # สร้างเมทริกซ์เก็บระยะทาง (n_samples, k)
    distance_matrix = np.zeros((n_samples, k))

    # วนลูปทุก centroid แล้วคำนวณระยะทางจากทุกจุดไปยัง centroid นั้น
    for j, c in enumerate(centroids):
        # euclidean_distance(X, c) จะคืน array ยาว n_samples
        distance_matrix[:, j] = euclidean_distance(X, c)

    # หา index ของ centroid ที่ใกล้ที่สุดในแต่ละแถว (axis=1)
    labels = np.argmin(distance_matrix, axis=1)
    return labels


# ------------------------------
# 6) ขั้นตอนอัปเดต centroid จากค่าเฉลี่ยของจุดในคลัสเตอร์
# ------------------------------

def update_centroids(X, labels, k):
    """
    รับข้อมูล X และ labels ของแต่ละจุด
    คำนวณ centroid ใหม่ของคลัสเตอร์ทั้ง k คลัสเตอร์
    - centroid ของคลัสเตอร์ = ค่าเฉลี่ย (mean) ของจุดทั้งหมดในคลัสเตอร์นั้น [web:39][web:46]
    """
    # เตรียม array ว่างสำหรับเก็บ centroid ใหม่ (k แถว, n_features คอลัมน์)
    new_centroids = np.zeros((k, X.shape[1]))

    for cluster_id in range(k):
        # เลือกเฉพาะจุดที่มี label เท่ากับ cluster_id
        points_in_cluster = X[labels == cluster_id]

        if len(points_in_cluster) == 0:
            # กรณีที่คลัสเตอร์ว่าง (ไม่มีจุดเลย)
            # จะสุ่มจุดจากข้อมูลมาหนึ่งจุดเพื่อใช้เป็น centroid ใหม่
            # ป้องกันไม่ให้ centroid หายไปจากระบบ
            new_centroids[cluster_id] = X[np.random.randint(0, len(X))]
        else:
            # ถ้ามีจุดในคลัสเตอร์ ให้ใช้ค่าเฉลี่ยทุกมิติเป็น centroid
            new_centroids[cluster_id] = points_in_cluster.mean(axis=0)

    return new_centroids


# ------------------------------
# 7) Evaluation function: Inertia / SSE
# ------------------------------

def compute_inertia(X, centroids, labels):
    """
    คำนวณ Within-cluster Sum of Squares (WCSS) หรือ inertia [web:31][web:44][web:47][web:50]
    นิยาม: ผลรวมของระยะทางกำลังสองระหว่างจุดทุกจุดกับ centroid ของคลัสเตอร์ตัวเอง
    - ยิ่งค่านี้น้อย แสดงว่าแต่ละคลัสเตอร์มีจุดกระจุกตัวใกล้ centroid มาก (cluster แน่น)
    - K-Means พยายามหาค่า centroid และการแบ่งกลุ่มที่ทำให้ inertia ต่ำที่สุด
    """
    sse = 0.0  # เก็บผลรวม

    k = centroids.shape[0]
    for cluster_id in range(k):
        # ดึงจุดทั้งหมดที่อยู่ในคลัสเตอร์นี้
        points = X[labels == cluster_id]
        if len(points) > 0:
            # คำนวณ (points - centroid)^2 ทุกมิติ แล้ว sum รวม
            sse += np.sum((points - centroids[cluster_id]) ** 2)

    return sse


# ------------------------------
# 8) ฟังก์ชันวาดกราฟ cluster + centroid และเซฟรูป
# ------------------------------

def plot_clusters(X, labels, centroids, iteration):
    """
    วาดกราฟ scatter (2 มิติ) ของข้อมูล:
    - จุดสีต่างกันตามคลัสเตอร์
    - centroid แสดงเป็นเครื่องหมาย X สีแดง
    - เซฟรูปลงโฟลเดอร์ result ตามชื่อ iteration เช่น clusters_iter_1.png
    """
    plt.figure(figsize=(6, 5))

    k = centroids.shape[0]

    # วาดจุดของแต่ละคลัสเตอร์ด้วยสีแตกต่างกัน
    for cluster_id in range(k):
        cluster_points = X[labels == cluster_id]
        plt.scatter(
            cluster_points[:, 0],          # แกน X = Petal.Length
            cluster_points[:, 1],          # แกน Y = Petal.Width
            s=40,                          # ขนาดจุด
            alpha=0.7,                     # ความโปร่ง
            label=f"Cluster {cluster_id}"  # ชื่อ legend
        )

    # วาด centroid ทั้งหมดด้วยสีแดง marker รูป X
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        s=200,
        c="red",
        marker="X",
        edgecolor="black",
        linewidth=1.5,
        label="Centroids"
    )

    # ใส่ title / label / grid
    plt.title(f"K-Means clustering (Iteration {iteration})")
    plt.xlabel("Petal Length")
    plt.ylabel("Petal Width")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # สร้างชื่อไฟล์รูปภาพ เช่น clusters_iter_1.png
    filename = os.path.join(RESULT_DIR, f"clusters_iter_{iteration}.png")

    # บันทึกรูปลงไฟล์ (ไม่ใช้ plt.show() เพราะต้องการเซฟอย่างเดียว)
    plt.savefig(filename)

    # ปิดรูปเพื่อคืนหน่วยความจำ ป้องกันไม่ให้ figure สะสมเยอะเกินไป
    plt.close()


# ------------------------------
# 9) ฟังก์ชันหลักของ K-Means (วนลูป iteration)
# ------------------------------

def kmeans(X, k=3, max_iters=10, tol=1e-4, random_state=42, visualize=True):
    """
    ขั้นตอนหลักของอัลกอริทึม K-Means [web:3][web:39][web:46]
    พารามิเตอร์:
    - X           : ข้อมูล (n_samples, n_features)
    - k           : จำนวนคลัสเตอร์ที่ต้องการแบ่ง
    - max_iters   : จำนวนรอบ iteration สูงสุด ถ้าคอนเวอร์จเร็วกว่านี้จะหยุดก่อน
    - tol         : เกณฑ์ความเปลี่ยนแปลงของ centroid ถ้าน้อยกว่าค่านี้ถือว่าคงที่แล้ว
    - random_state: seed สำหรับการสุ่ม centroid เริ่มต้น
    - visualize   : ถ้า True จะวาดรูปแต่ละ iteration แล้วเซฟลงโฟลเดอร์ result

    ลูปหลักจะทำตาม step:
    1) เลือก centroid เริ่มต้น
    2) assign จุดข้อมูลเข้าคลัสเตอร์ใกล้ที่สุด
    3) คำนวณ centroid ใหม่จากค่าเฉลี่ยของจุดในแต่ละคลัสเตอร์
    4) คำนวณ inertia เพื่อประเมินคุณภาพของการจัดกลุ่ม
    5) เช็คว่าศูนย์กลาง (centroid) ขยับน้อยมากหรือไม่ ถ้าน้อยกว่า tol ให้หยุด (คอนเวอร์จ)
    """
    # 1) สุ่ม centroid เริ่มต้น
    centroids = initialize_centroids(X, k, random_state=random_state)

    # เก็บประวัติ inertia ของแต่ละ iteration เพื่อเอาไป plot ภายหลัง
    inertia_history = []

    for it in range(max_iters):
        # เก็บ centroid รอบก่อนหน้าไว้เพื่อใช้วัดการขยับ
        old_centroids = deepcopy(centroids)

        # 2) assign จุดทั้งหมดไปยัง cluster ที่ใกล้ centroid ที่สุด
        labels = assign_clusters(X, centroids)

        # 3) อัปเดต centroid จากค่าเฉลี่ยของจุดในแต่ละคลัสเตอร์
        centroids = update_centroids(X, labels, k)

        # 4) คำนวณ inertia รอบนี้
        inertia = compute_inertia(X, centroids, labels)
        inertia_history.append(inertia)

        print(f"Iteration {it + 1}, inertia = {inertia:.4f}")

        # 5) วาดและเซฟกราฟการจัดกลุ่มของ iteration ปัจจุบัน
        if visualize:
            plot_clusters(X, labels, centroids, iteration=it + 1)

        # 6) เช็คการขยับของ centroid ว่าลดลงมากไหม
        centroid_shift = np.linalg.norm(centroids - old_centroids)
        print(f"  centroid shift = {centroid_shift:.6f}")

        # ถ้า centroid แทบไม่ขยับแล้ว (< tol) แปลว่าคลัสเตอร์นิ่ง => หยุดลูป
        if centroid_shift < tol:
            print("Centroids converged. Stop iteration.")
            break

    # คืนค่า centroid สุดท้าย, label ของแต่ละจุด และประวัติ inertia
    return centroids, labels, inertia_history


# ------------------------------
# 10) ส่วน main: เรียกใช้งานฟังก์ชัน และบันทึกผลลัพธ์
# ------------------------------

if __name__ == "__main__":
    # กำหนดจำนวนคลัสเตอร์ k = 3 (อิงจาก Iris มี 3 species)
    k = 3

    # จำนวนรอบสูงสุดของ K-Means (ส่วนใหญ่คอนเวอร์จเร็วกว่านี้)
    max_iters = 10

    # เรียกใช้ K-Means บนข้อมูล X
    final_centroids, final_labels, inertia_history = kmeans(
        X,
        k=k,
        max_iters=max_iters,
        tol=1e-4,
        random_state=0,
        visualize=True
    )

    # แสดงค่า centroid สุดท้ายใน console เพื่อดูคร่าวๆ
    print("Final centroids (Petal.Length, Petal.Width):")
    print(final_centroids)

    # --------------------------
    # บันทึกประวัติ inertia ลงไฟล์ text
    # --------------------------
    result_txt = os.path.join(RESULT_DIR, "inertia_history.txt")

    # เขียนไฟล์ผลลัพธ์แบบง่าย ๆ: iteration \t inertia
    with open(result_txt, "w", encoding="utf-8") as f:
        f.write("Iteration\tInertia (Within-cluster SSE)\n")
        for i, val in enumerate(inertia_history, start=1):
            f.write(f"{i}\t{val}\n")

    # --------------------------
    # วาดกราฟ inertia ต่อ iteration แล้วบันทึกรูป
    # --------------------------
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(inertia_history) + 1), inertia_history, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Inertia (Within-cluster SSE)")
    plt.title("K-Means Inertia per Iteration")
    plt.grid(True)
    plt.tight_layout()

    # เซฟรูปกราฟ inertia ลงโฟลเดอร์ result
    inertia_plot_path = os.path.join(RESULT_DIR, "inertia_vs_iteration.png")
    plt.savefig(inertia_plot_path)
    plt.close()

    print("All result files are saved in:", RESULT_DIR)
