"""
Decision Tree example — animal classification (mammal vs non-mammal).

This is a small, textbook-style dataset. We encode categorical features
and train a DecisionTreeClassifier.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Columns: name, body_temperature, skin_cover, live_birth, aquatic, flies, has_legs, hibernates, class_label
DATA = [
    ["Human", "Warm-blooded", "Hair", "Yes", "No", "No", "Yes", "No", "Mammal"],
    ["Salmon", "Cold-blooded", "Scales", "No", "Yes", "No", "No", "No", "Fish"],
    ["Whale", "Warm-blooded", "Hair", "Yes", "Yes", "No", "No", "No", "Mammal"],
    ["Frog", "Cold-blooded", "None", "No", "Semi", "No", "Yes", "Yes", "Amphibian"],
    ["Komodo dragon", "Cold-blooded", "Scales", "No", "No", "No", "Yes", "Yes", "Reptile"],
    ["Bat", "Warm-blooded", "Hair", "Yes", "No", "Yes", "Yes", "No", "Mammal"],
    ["Pigeon", "Warm-blooded", "Feathers", "No", "No", "Yes", "Yes", "No", "Bird"],
    ["Cat", "Warm-blooded", "Fur", "Yes", "No", "No", "Yes", "No", "Mammal"],
    ["Leopard shark", "Cold-blooded", "Scales", "Yes", "Yes", "No", "No", "No", "Fish"],
    ["Sea turtle", "Cold-blooded", "Scales", "No", "Semi", "No", "Yes", "Yes", "Reptile"],
    ["Penguin", "Warm-blooded", "Feathers", "No", "Yes", "No", "Yes", "No", "Bird"],
    ["Porcupine", "Warm-blooded", "Bristles", "Yes", "No", "No", "Yes", "No", "Mammal"],
    ["Eel", "Cold-blooded", "Scales", "No", "Yes", "No", "No", "No", "Fish"],
    ["Salamander", "Cold-blooded", "None", "No", "Semi", "No", "Yes", "Yes", "Amphibian"],
]

COLUMNS = [
    "name",
    "body_temperature",
    "skin_cover",
    "live_birth",
    "aquatic",
    "flies",
    "has_legs",
    "hibernates",
    "class_label",
]


def load_and_encode():
    df = pd.DataFrame(DATA, columns=COLUMNS)

    # Target: Mammal=1, Non-mammal=0
    y = (df["class_label"] == "Mammal").astype(int)

    X = pd.DataFrame(
        {
            "is_warm_blooded": (df["body_temperature"] == "Warm-blooded").astype(int),
            "skin_cover_code": df["skin_cover"].map(
                {"None": 0, "Scales": 1, "Hair": 2, "Fur": 2, "Bristles": 2, "Feathers": 3}
            ),
            "live_birth": (df["live_birth"] == "Yes").astype(int),
            "aquatic_level": df["aquatic"].map({"No": 0, "Semi": 1, "Yes": 2}),
            "flies": (df["flies"] == "Yes").astype(int),
            "has_legs": (df["has_legs"] == "Yes").astype(int),
            "hibernates": (df["hibernates"] == "Yes").astype(int),
        }
    )

    feature_names = list(X.columns)
    return X, y, feature_names


def main() -> None:
    X, y, feature_names = load_and_encode()

    clf = DecisionTreeClassifier(criterion="entropy", random_state=42)
    clf.fit(X, y)

    print("Training accuracy:", clf.score(X, y))
    preds = clf.predict(X)
    print("Predictions (mammal=1):", preds.tolist())

    plt.figure(figsize=(14, 8))
    plot_tree(
        clf,
        feature_names=feature_names,
        class_names=["Non-mammal", "Mammal"],
        filled=True,
        rounded=True,
    )
    plt.title("Decision Tree: Mammal vs Non-mammal")
    plt.tight_layout()

    images_dir = os.path.join(os.path.dirname(__file__), "..", "images")
    out_path = os.path.join(images_dir, "decision_tree_animals.png")
    plt.savefig(out_path, dpi=120)
    print("Saved tree image to:", os.path.abspath(out_path))
    plt.show()


if __name__ == "__main__":
    main()

