from pathlib import Path
import json
import textwrap

import nbformat as nbf
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
for folder in ["docs", "data/input", "data/output", "notebooks", "src"]:
    (ROOT / folder).mkdir(parents=True, exist_ok=True)


# Du lieu mau co cau truc tuong tu Mall Customers, tao co seed de tai lap.
rng = np.random.default_rng(42)
profiles = [
    (35, 35, 35, 45),   # tuoi, thu nhap, spending, so luong
    (28, 30, 75, 40),
    (42, 75, 20, 38),
    (32, 78, 78, 42),
    (58, 55, 45, 35),
]
rows = []
customer_id = 1
for age_mu, income_mu, score_mu, count in profiles:
    for _ in range(count):
        rows.append({
            "CustomerID": customer_id,
            "Gender": rng.choice(["Male", "Female"]),
            "Age": int(np.clip(rng.normal(age_mu, 6), 18, 70)),
            "Annual Income (k$)": int(np.clip(rng.normal(income_mu, 8), 15, 140)),
            "Spending Score (1-100)": int(np.clip(rng.normal(score_mu, 9), 1, 100)),
        })
        customer_id += 1
df = pd.DataFrame(rows)
df.to_csv(ROOT / "data/input/mall_customers.csv", index=False)


def md(text):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


cells = [
    md("""
    # Lab 2 - Phan khuc khach hang bang Clustering

    **Muc tieu:** phan nhom khach hang theo tuoi, thu nhap hang nam va diem chi tieu; danh gia bang Silhouette, Davies-Bouldin va Calinski-Harabasz theo Chapter 3.

    > De bai la **clustering (hoc khong giam sat)**, khong phai classification. Phan classification o cuoi notebook chi la mo rong: hoc mo hinh du doan nhan cum do K-Means tao ra, khong duoc xem la nhan that.
    """),
    md("""
    ## Step 1 - Import va cau hinh

    Notebook tu dong tim thu muc goc du an, nen co the chay tu Jupyter hoac VS Code.
    """),
    code("""
    from pathlib import Path
    import warnings
    warnings.filterwarnings('ignore')

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                                 calinski_harabasz_score, classification_report,
                                 ConfusionMatrixDisplay, accuracy_score)
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.decomposition import PCA

    sns.set_theme(style='whitegrid')
    candidates = [Path.cwd(), Path.cwd().parent]
    ROOT = next(p for p in candidates if (p / 'data/input/mall_customers.csv').exists())
    INPUT = ROOT / 'data/input'
    OUTPUT = ROOT / 'data/output'
    OUTPUT.mkdir(parents=True, exist_ok=True)
    print('Project root:', ROOT)
    """),
    md("""
    ## Step 2 - Doc va kham pha du lieu

    Kiem tra kich thuoc, kieu du lieu, gia tri thieu, trung lap, thong ke mo ta va outlier theo quy tac IQR. CustomerID chi la dinh danh nen khong dung lam dac trung.
    """),
    code("""
    df = pd.read_csv(INPUT / 'mall_customers.csv')
    display(df.head())
    print('Shape:', df.shape)
    print('Missing values:\\n', df.isna().sum())
    print('Duplicated rows:', df.duplicated().sum())
    display(df.describe(include='all').T)
    """),
    code("""
    numeric = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    q1, q3 = df[numeric].quantile(.25), df[numeric].quantile(.75)
    iqr = q3 - q1
    outlier_count = ((df[numeric] < q1 - 1.5*iqr) | (df[numeric] > q3 + 1.5*iqr)).sum()
    print('Outliers by IQR rule:\\n', outlier_count)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for col, ax in zip(numeric, axes):
        sns.histplot(df[col], kde=True, ax=ax, color='#2878B5')
        ax.set_title(f'Distribution: {col}')
    fig.tight_layout()
    fig.savefig(OUTPUT / '01_feature_distributions.png', dpi=160, bbox_inches='tight')
    plt.show()
    """),
    md("""
    ## Step 3 - Tien xu ly va chuan hoa

    StandardScaler dua ba bien so ve cung thang do. Gender duoc giu de mo ta cum; K-Means chinh dung ba dac trung so dung nhu de bai.
    """),
    code("""
    X = df[numeric].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print('Mean after scaling:', np.round(X_scaled.mean(axis=0), 4))
    print('Std after scaling:', np.round(X_scaled.std(axis=0), 4))
    """),
    md("""
    ## Step 4 - Thu nghiem K-Means voi k = 2..10

    Chon k dua tren nhieu chi so: Silhouette cao, Davies-Bouldin thap, Calinski-Harabasz cao. Inertia duoc dung cho Elbow.
    """),
    code("""
    rows = []
    models = {}
    for k in range(2, 11):
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = model.fit_predict(X_scaled)
        models[k] = model
        rows.append({
            'k': k,
            'inertia': model.inertia_,
            'silhouette': silhouette_score(X_scaled, labels),
            'davies_bouldin': davies_bouldin_score(X_scaled, labels),
            'calinski_harabasz': calinski_harabasz_score(X_scaled, labels),
        })
    metrics = pd.DataFrame(rows)
    display(metrics.round(4))
    metrics.to_csv(OUTPUT / 'kmeans_metrics.csv', index=False)
    best_k = int(metrics.loc[metrics['silhouette'].idxmax(), 'k'])
    print('Best k by Silhouette:', best_k)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(metrics.k, metrics.inertia, marker='o')
    axes[0].set(title='Elbow method', xlabel='k', ylabel='Inertia')
    axes[1].plot(metrics.k, metrics.silhouette, marker='o', color='#D95319')
    axes[1].axvline(best_k, ls='--', color='gray')
    axes[1].set(title='Silhouette by k', xlabel='k', ylabel='Silhouette')
    fig.tight_layout()
    fig.savefig(OUTPUT / '02_k_selection.png', dpi=160, bbox_inches='tight')
    plt.show()
    """),
    md("""
    ## Step 5 - So sanh thuat toan

    So sanh K-Means, Agglomerative (Ward) va DBSCAN. DBSCAN co the gan nhan -1 cho noise; chi tinh metric khi co it nhat 2 cum hop le.
    """),
    code("""
    algorithms = {
        f'KMeans(k={best_k})': KMeans(n_clusters=best_k, random_state=42, n_init=20),
        f'Agglomerative(k={best_k})': AgglomerativeClustering(n_clusters=best_k, linkage='ward'),
        'DBSCAN(eps=0.55,min_samples=6)': DBSCAN(eps=.55, min_samples=6),
    }
    comparison = []
    labels_by_model = {}
    for name, model in algorithms.items():
        labels = model.fit_predict(X_scaled)
        labels_by_model[name] = labels
        valid = labels != -1
        cluster_count = len(set(labels[valid]))
        if cluster_count >= 2 and valid.sum() > cluster_count:
            comparison.append({
                'model': name, 'clusters': cluster_count,
                'noise_points': int((labels == -1).sum()),
                'silhouette': silhouette_score(X_scaled[valid], labels[valid]),
                'davies_bouldin': davies_bouldin_score(X_scaled[valid], labels[valid]),
                'calinski_harabasz': calinski_harabasz_score(X_scaled[valid], labels[valid]),
            })
    comparison = pd.DataFrame(comparison).sort_values('silhouette', ascending=False)
    display(comparison.round(4))
    comparison.to_csv(OUTPUT / 'algorithm_comparison.csv', index=False)
    """),
    md("""
    ## Step 6 - Mo hinh cuoi, truc quan hoa va ho so tung cum
    """),
    code("""
    final_model = models[best_k]
    df['Cluster'] = final_model.labels_
    pca = PCA(n_components=2)
    points_2d = pca.fit_transform(X_scaled)
    plot_df = pd.DataFrame(points_2d, columns=['PC1', 'PC2']).assign(Cluster=df.Cluster.astype(str))

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=plot_df, x='PC1', y='PC2', hue='Cluster', palette='tab10', s=65)
    plt.title(f'Customer segments - KMeans k={best_k} (PCA 2D)')
    plt.tight_layout()
    plt.savefig(OUTPUT / '03_final_clusters_pca.png', dpi=160, bbox_inches='tight')
    plt.show()

    profile = df.groupby('Cluster').agg(
        Customers=('CustomerID', 'count'),
        Mean_Age=('Age', 'mean'),
        Mean_Income=('Annual Income (k$)', 'mean'),
        Mean_Spending=('Spending Score (1-100)', 'mean'),
        Female_Rate=('Gender', lambda s: (s == 'Female').mean())
    ).round(2)
    display(profile)
    profile.to_csv(OUTPUT / 'cluster_profiles.csv')
    df.to_csv(OUTPUT / 'customers_with_clusters.csv', index=False)
    """),
    code("""
    def segment_name(row):
        income = row['Mean_Income']; spend = row['Mean_Spending']; age = row['Mean_Age']
        if income >= 60 and spend >= 60: return 'Gia tri cao - uu tien cham soc'
        if income >= 60 and spend < 40: return 'Thu nhap cao - tiem nang kich hoat'
        if income < 45 and spend >= 60: return 'Chi tieu cao - nhay voi uu dai'
        if age >= 50: return 'Truong thanh - chi tieu on dinh'
        return 'Pho thong - can nuoi duong'

    profile['Segment_Name'] = profile.apply(segment_name, axis=1)
    display(profile)
    profile.to_csv(OUTPUT / 'cluster_profiles_named.csv')
    """),
    md("""
    ## Step 7 - Classification mo rong

    Random Forest du doan **nhan cum K-Means** cho khach hang moi. Day la pseudo-label classification, phu hop trien khai nhung accuracy khong chung minh cac cum la nhan that.
    """),
    code("""
    feature_cols = ['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    X_cls = df[feature_cols]
    y_cls = df['Cluster']
    X_train, X_test, y_train, y_test = train_test_split(
        X_cls, y_cls, test_size=.25, random_state=42, stratify=y_cls)

    preprocess = ColumnTransformer([
        ('num', StandardScaler(), numeric),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Gender'])
    ])
    classifier = Pipeline([
        ('preprocess', preprocess),
        ('model', RandomForestClassifier(n_estimators=250, random_state=42, class_weight='balanced'))
    ])
    classifier.fit(X_train, y_train)
    pred = classifier.predict(X_test)
    print('Pseudo-label classification accuracy:', round(accuracy_score(y_test, pred), 4))
    print(classification_report(y_test, pred, zero_division=0))

    ConfusionMatrixDisplay.from_predictions(y_test, pred, cmap='Blues')
    plt.title('Confusion matrix - du doan nhan cum')
    plt.tight_layout()
    plt.savefig(OUTPUT / '04_classification_confusion_matrix.png', dpi=160, bbox_inches='tight')
    plt.show()
    """),
    md("""
    ## Step 8 - Ket luan

    - Da hoan thanh EDA, kiem tra missing/outlier, scaling, thu nghiem tham so, so sanh thuat toan va danh gia noi bo.
    - Chon K-Means theo Silhouette; ket qua cu the nam trong `data/output/kmeans_metrics.csv`.
    - Ho so cum ho tro de xuat marketing; ten cum la dien giai nghiep vu, khong phai nhan co san.
    - Classification chi la phan mo rong de gan khach hang moi vao cac segment da hoc.
    """),
]

nb = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
)
nbf.write(nb, ROOT / "notebooks/lab2_customer_segmentation.ipynb")


(ROOT / "README.md").write_text("""# Lab 2 - Machine Learning\n\nBai thuc hanh phan khuc khach hang theo Chapter 3 - Unsupervised Learning.\n\nXem huong dan tai `docs/HUONG_DAN.md` va notebook tai `notebooks/lab2_customer_segmentation.ipynb`.\n""", encoding="utf-8")

(ROOT / "docs/HUONG_DAN.md").write_text("""# Huong dan Lab 2\n\n## Phan biet yeu cau\n\nDe trong anh la **clustering**: du lieu khong co nhan dich va can tim cac nhom khach hang tu nhien. Classification can nhan co san, vi vay khong the thay the buoc phan cum. Notebook van co mot buoc classification mo rong de du doan nhan cum (pseudo-label) cho khach hang moi.\n\n## Cac step\n\n1. Doc du lieu va kham pha: missing, duplicate, thong ke, outlier IQR, histogram.\n2. Chon Age, Annual Income, Spending Score; loai CustomerID; chuan hoa StandardScaler.\n3. Thu K-Means voi k tu 2 den 10.\n4. Danh gia bang Silhouette (cao tot), Davies-Bouldin (thap tot), Calinski-Harabasz (cao tot) va Elbow.\n5. So sanh K-Means voi Agglomerative va DBSCAN.\n6. Truc quan PCA 2D, tao ho so va dat ten cac phan khuc.\n7. Mo rong classification: Random Forest hoc nhan cum de gan segment cho khach hang moi.\n8. Luu bang va bieu do vao `data/output`.\n\n## Cach chay\n\nMo `notebooks/lab2_customer_segmentation.ipynb`, chon Python 3 va Run All. Notebook tu tim thu muc goc du an.\n\n## Cau truc\n\n- `docs/`: huong dan va bao cao.\n- `data/input/`: CSV dau vao, anh de, file bai giang.\n- `data/output/`: ket qua CSV va PNG.\n- `notebooks/`: notebook da chay san.\n- `src/`: ma Python tai su dung.\n""", encoding="utf-8")

(ROOT / "requirements.txt").write_text("""numpy\npandas\nmatplotlib\nseaborn\nscikit-learn\nnbformat\nnbclient\nipykernel\n""", encoding="utf-8")

(ROOT / "src/predict_segment.py").write_text("""\"\"\"Ham tien ich du doan cluster tu mot pipeline sklearn da huan luyen.\"\"\"\n\ndef predict_segment(model, gender, age, annual_income, spending_score):\n    import pandas as pd\n    sample = pd.DataFrame([{\n        'Gender': gender,\n        'Age': age,\n        'Annual Income (k$)': annual_income,\n        'Spending Score (1-100)': spending_score,\n    }])\n    return int(model.predict(sample)[0])\n""", encoding="utf-8")

print(json.dumps({"rows": len(df), "notebook": str(ROOT / 'notebooks/lab2_customer_segmentation.ipynb')}, ensure_ascii=False))
