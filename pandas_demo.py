import pandas as pd
import matplotlib.pyplot as plt

# 1. 创建一个简单的示例数据（如果没有CSV文件，先用这个测试）
data = {
    "日期": ["2026-06-01", "2026-06-02", "2026-06-03", "2026-06-04", "2026-06-05"],
    "销售额": [1200, 1500, 1300, 1800, 1600],
    "客流量": [100, 130, 110, 160, 140]
}
df = pd.DataFrame(data)

# 2. 查看数据基本信息
print("数据前几行：")
print(df.head())
print("\n数据信息：")
df.info()

# 3. 简单的统计分析
print("\n销售额统计：")
print(df["销售额"].describe())

# 4. 绘制折线图
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 解决中文乱码
plt.rcParams["axes.unicode_minus"] = False

plt.figure(figsize=(8, 4))
plt.plot(df["日期"], df["销售额"], marker='o', label="销售额")
plt.plot(df["日期"], df["客流量"], marker='s', label="客流量")
plt.title("每日销售额与客流量趋势")
plt.xlabel("日期")
plt.ylabel("数值")
plt.legend()
plt.tight_layout()
plt.savefig("trend.png")
plt.show()

# 5. 保存数据
df.to_csv("sales_data.csv", index=False, encoding="utf-8-sig")
print("\n数据已保存为 sales_data.csv，图表已保存为 trend.png")