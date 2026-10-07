import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from skimage.feature import local_binary_pattern
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Paramètres globaux

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ANNOTATION_FILE = os.path.join(BASE_DIR, "labels", "labels_joint_anno.txt")

R = 1
P = 8 * R
METHOD = "uniform"
IMG_SIZE = (128, 128)
N_BINS = P + 2

np.random.seed(42)



def load_gray(path):
    """Lit une image en niveaux de gris et la redimensionne.
    np.fromfile + imdecode évite les problèmes de chemins avec accents sous Windows."""
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return cv2.resize(img, IMG_SIZE)


def lbp_histogram(img_gray):
    """Image grise (déjà redimensionnée) -> histogramme LBP normalisé."""
    lbp = local_binary_pattern(img_gray, P, R, METHOD)
    hist, _ = np.histogram(lbp.ravel(), bins=N_BINS, range=(0, N_BINS))
    hist = hist.astype("float")
    s = hist.sum()
    if s > 0:
        hist /= s
    return hist



# ETAPE 1 :

image_paths, labels_text = [], []

with open(ANNOTATION_FILE, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            img_path = os.path.join(BASE_DIR, parts[0])
            primary_label = parts[1]       
            if os.path.exists(img_path):
                image_paths.append(img_path)
                labels_text.append(primary_label)
            else:
                print(f"Image introuvable : {img_path}")

le = LabelEncoder()
labels = le.fit_transform(labels_text)
classes = le.classes_

print(f"Total d'images chargées : {len(image_paths)}")
print(f"Nombre de classes uniques : {len(classes)}")

unique_labels, counts = np.unique(labels, return_counts=True)
for u, c in zip(unique_labels, counts):
    print(f"Classe '{classes[u]}' : {c} images")

ratio = counts.min() / counts.max()
print(f"Rapport min/max des effectifs : {ratio:.2f} -> "
      f"{'classes équilibrées' if ratio > 0.8 else 'classes déséquilibrées'}")


N_EX = 4
plt.figure(figsize=(3 * N_EX, 3 * len(classes)))
for ci, cname in enumerate(classes):
    idxs = np.where(labels == ci)[0][:N_EX]
    for j, idx in enumerate(idxs):
        img = cv2.imdecode(np.fromfile(image_paths[idx], dtype=np.uint8), cv2.IMREAD_COLOR)
        plt.subplot(len(classes), N_EX, ci * N_EX + j + 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(cname)
        plt.axis("off")
plt.suptitle("Exemples de chaque classe")
plt.tight_layout()
plt.show()


# ETAPES 2 et 3 : 

sample_idx = [int(np.where(labels == ci)[0][0]) for ci in range(len(classes))]

plt.figure(figsize=(8, 3.5 * len(sample_idx)))
for i, idx in enumerate(sample_idx):
    img_resized = load_gray(image_paths[idx])
    lbp = local_binary_pattern(img_resized, P, R, METHOD)

    plt.subplot(len(sample_idx), 2, 2 * i + 1)
    plt.imshow(img_resized, cmap="gray")
    plt.title(f"Image grise 128x128 ({labels_text[idx]})")
    plt.axis("off")

    plt.subplot(len(sample_idx), 2, 2 * i + 2)
    plt.imshow(lbp, cmap="gray")
    plt.title(f"LBP (P={P}, R={R})")
    plt.axis("off")
plt.tight_layout()
plt.show()


# ETAPE 4 :

plt.figure(figsize=(12, 3.5))
for i, idx in enumerate(sample_idx):
    hist = lbp_histogram(load_gray(image_paths[idx]))
    if i == 0:
        print("Vecteur LBP (première image) :", np.round(hist, 4))
    plt.subplot(1, len(sample_idx), i + 1)
    plt.bar(range(N_BINS), hist, color="steelblue")
    plt.title(f"Histogramme LBP ({labels_text[idx]})")
    plt.xlabel("Motifs LBP")
    plt.ylabel("Fréquence normalisée")
plt.tight_layout()
plt.show()


# ETAPE 5 :

X_list, y_list, valid_paths = [], [], []
for i, path in enumerate(image_paths):
    img = load_gray(path)
    if img is None:
        print(f"Lecture impossible : {path}")
        continue
    X_list.append(lbp_histogram(img))
    y_list.append(labels[i])
    valid_paths.append(path)

X = np.vstack(X_list)
y = np.array(y_list)

print(f"Dimension de la matrice X : {X.shape}")
print(f"Dimension du vecteur y : {y.shape}")
print("Effectif par classe :", dict(zip(classes, np.bincount(y))))


# ETAPE 6 : 

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
print(f"Variance expliquée par les 2 composantes : {pca.explained_variance_ratio_.sum():.2%}")

plt.figure(figsize=(7, 6))
for ci, cname in enumerate(classes):
    m = y == ci
    plt.scatter(X_pca[m, 0], X_pca[m, 1], label=cname, alpha=0.7)
plt.title("PCA des histogrammes LBP")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.show()


# ETAPE 7 : 

indices = np.arange(len(y))
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, indices, test_size=0.3, random_state=42, stratify=y
)
print(f"Train : {X_train.shape[0]} images | Test : {X_test.shape[0]} images")


# ETAPES 8 et 9 : 

models = {}

for k in (3, 5, 7):
    models[f"LBP + k-NN (k={k})"] = KNeighborsClassifier(n_neighbors=k)

models["LBP + SVM (rbf)"] = SVC(kernel="rbf", C=10, gamma="scale")
# Les SVM sont sensibles à l'échelle des variables : version avec standardisation
models["LBP + SVM (rbf, standardisé)"] = make_pipeline(
    StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale")
)


# ETAPE 10 : 

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = accuracy_score(y_test, y_pred)
    print(f"\n=== {name} ===")
    print(f"Accuracy : {results[name]:.4f}")
    print(classification_report(y_test, y_pred, target_names=classes, zero_division=0))


# ETAPE 11 : 

best_name = max(results, key=results.get)
best_model = models[best_name]
y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)

print(f"\nMeilleur modèle : {best_name} (accuracy = {results[best_name]:.4f})")
print("Matrice de confusion :\n", cm)

plt.figure(figsize=(5.5, 5))
plt.imshow(cm, cmap="Blues")
plt.title(f"Matrice de confusion\n{best_name}")
plt.xticks(range(len(classes)), classes, rotation=45)
plt.yticks(range(len(classes)), classes)
for i in range(len(classes)):
    for j in range(len(classes)):
        plt.text(j, i, cm[i, j], ha="center", va="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.xlabel("Classe prédite")
plt.ylabel("Classe réelle")
plt.colorbar()
plt.tight_layout()
plt.show()

recall_per_class = cm.diagonal() / cm.sum(axis=1)
print(f"Classe la mieux reconnue    : {classes[recall_per_class.argmax()]} "
      f"({recall_per_class.max():.2%})")
print(f"Classe la moins bien reconnue : {classes[recall_per_class.argmin()]} "
      f"({recall_per_class.min():.2%})")

cm_off = cm.copy()
np.fill_diagonal(cm_off, 0)
i, j = np.unravel_index(cm_off.argmax(), cm_off.shape)
print(f"Confusion la plus fréquente : '{classes[i]}' prédit comme '{classes[j]}' "
      f"({cm_off[i, j]} fois)")


# ETAPE 12 : 

def make_darker(img):
    return cv2.convertScaleAbs(img, alpha=1.0, beta=-60)

def make_contrast(img):
    return cv2.convertScaleAbs(img, alpha=1.8, beta=0)

def make_noisy(img):
    noise = np.random.normal(0, 20, img.shape)
    return np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)

transforms = {
    "Originale": lambda im: im,
    "Plus sombre": make_darker,
    "Contraste augmenté": make_contrast,
    "Bruitée": make_noisy,
}

img0 = load_gray(valid_paths[idx_test[0]])
plt.figure(figsize=(14, 6))
for k, (tname, fn) in enumerate(transforms.items()):
    im = fn(img0)
    plt.subplot(2, 4, k + 1)
    plt.imshow(im, cmap="gray", vmin=0, vmax=255)
    plt.title(tname)
    plt.axis("off")
    plt.subplot(2, 4, k + 5)
    plt.bar(range(N_BINS), lbp_histogram(im), color="darkorange")
    plt.title("Histogramme LBP")
plt.tight_layout()
plt.show()

robust = {}
for tname, fn in transforms.items():
    Xt = np.vstack([lbp_histogram(fn(load_gray(valid_paths[i]))) for i in idx_test])
    robust[tname] = accuracy_score(y_test, best_model.predict(Xt))
    print(f"Robustesse - {tname:20s}: accuracy = {robust[tname]:.4f}")

plt.figure(figsize=(7, 4))
plt.bar(robust.keys(), robust.values(), color="seagreen")
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title(f"Robustesse de LBP ({best_name})")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()


# ETAPE 13 : 

print("\n" + "=" * 50)
print("COMPARAISON FINALE")
print("=" * 50)
for name, acc in sorted(results.items(), key=lambda kv: -kv[1]):
    print(f"{name:35s} accuracy = {acc:.4f}")

plt.figure(figsize=(9, 4.5))
plt.barh(list(results.keys()), list(results.values()), color="slateblue")
plt.xlim(0, 1)
plt.xlabel("Accuracy (jeu de test)")
plt.title("Comparaison des modèles LBP")
plt.tight_layout()
plt.show()