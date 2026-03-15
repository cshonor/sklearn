"""
决策树示例：动物分类（哺乳类 vs 非哺乳类）

对应教材中「决策树」章节的示例数据与二分类目标：
根据体温、表皮覆盖、胎生、水生、飞行、有腿、冬眠等特征，将动物分为哺乳类与非哺乳类。
"""

import os
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# 教材表格数据：名字, 体温, 表皮覆盖, 胎生, 水生动物, 飞行动物, 有腿, 冬眠, 类标号
DATA = [
    ["人类", "恒温", "毛发", "是", "否", "否", "是", "否", "哺乳类"],
    ["鮭魚", "冷血", "鳞片", "否", "是", "否", "否", "否", "鱼类"],
    ["鯨", "恒温", "毛发", "是", "是", "否", "否", "否", "哺乳类"],
    ["青蛙", "冷血", "无", "否", "半", "否", "是", "是", "两栖类"],
    ["巨蜥", "冷血", "鳞片", "否", "否", "否", "是", "是", "爬行类"],
    ["蝙蝠", "恒温", "毛发", "是", "否", "是", "是", "否", "哺乳类"],
    ["鸽子", "恒温", "羽毛", "否", "否", "是", "是", "否", "鸟类"],
    ["猫", "恒温", "软毛", "是", "否", "否", "是", "否", "哺乳类"],
    ["豹纹鲨", "冷血", "鳞片", "是", "是", "否", "否", "否", "鱼类"],
    ["海龟", "冷血", "鳞片", "否", "半", "否", "是", "是", "爬行类"],
    ["企鹅", "恒温", "羽毛", "否", "是", "否", "是", "否", "鸟类"],
    ["豪猪", "恒温", "刚毛", "是", "否", "否", "是", "否", "哺乳类"],
    ["鰻", "冷血", "鳞片", "否", "是", "否", "否", "否", "鱼类"],
    ["蝾螈", "冷血", "无", "否", "半", "否", "是", "是", "两栖类"],
]

COLUMNS = ["名字", "体温", "表皮覆盖", "胎生", "水生动物", "飞行动物", "有腿", "冬眠", "类标号"]


def load_and_encode():
    """构建 DataFrame 并编码为数值特征。"""
    df = pd.DataFrame(DATA, columns=COLUMNS)

    # 二分类目标：哺乳类=1，非哺乳类=0
    df["目标"] = (df["类标号"] == "哺乳类").astype(int)

    # 二值特征
    df["体温_恒温"] = (df["体温"] == "恒温").astype(int)
    df["胎生"] = (df["胎生"] == "是").astype(int)
    df["水生动物"] = df["水生动物"].map({"否": 0, "半": 1, "是": 2})
    df["飞行动物"] = (df["飞行动物"] == "是").astype(int)
    df["有腿"] = (df["有腿"] == "是").astype(int)
    df["冬眠"] = (df["冬眠"] == "是").astype(int)

    # 表皮覆盖：无序类别，用 LabelEncoder 或映射（此处用映射便于解读）
    cover_map = {"无": 0, "鳞片": 1, "毛发": 2, "软毛": 2, "刚毛": 2, "羽毛": 3}
    df["表皮覆盖_编码"] = df["表皮覆盖"].map(cover_map)

    feature_cols = ["体温_恒温", "表皮覆盖_编码", "胎生", "水生动物", "飞行动物", "有腿", "冬眠"]
    X = df[feature_cols]
    y = df["目标"]

    return df, X, y, feature_cols


def main():
    df, X, y, feature_cols = load_and_encode()

    print("数据集（编码后）：")
    print(df[["名字"] + feature_cols + ["类标号", "目标"]].to_string(index=False))
    print()

    clf = DecisionTreeClassifier(criterion="entropy", random_state=42)
    clf.fit(X, y)

    print("在训练集上的准确率:", clf.score(X, y))
    print("预测（哺乳类=1）：", clf.predict(X).tolist())
    print()

    # 绘制决策树
    plt.figure(figsize=(14, 8))
    plot_tree(
        clf,
        feature_names=feature_cols,
        class_names=["非哺乳类", "哺乳类"],
        filled=True,
        rounded=True,
    )
    plt.title("决策树：哺乳类 vs 非哺乳类")
    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "decision_tree_animals.png")
    plt.savefig(out_path, dpi=120)
    print("决策树图已保存:", out_path)
    plt.show()


if __name__ == "__main__":
    main()
