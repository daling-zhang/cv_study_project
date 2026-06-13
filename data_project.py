import pandas as pd
import matplotlib.pyplot as plt
import os

# ---------------------- 配置中文显示 ----------------------
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ---------------------- 确保文件夹存在 ----------------------
os.makedirs("data", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# ---------------------- 1. 读取数据 ----------------------
df = pd.read_csv("data/raw_data.csv", encoding="latin-1")

print("===== 数据基本信息 =====")
print(df.info())
print("\n===== 前5行数据 =====")
print(df.head())
print("\n===== 缺失值统计 =====")
print(df.isnull().sum())

# ---------------------- 2. 清洗数据 ----------------------
# 处理缺失值（这里的数据集主要是 Postal Code 有缺失，直接删除即可）
df_clean = df.dropna()

print("\n===== 清洗后缺失值统计 =====")
print(df_clean.isnull().sum())

# ---------------------- 3. 条件筛选 ----------------------
# 示例：筛选销售额大于 500 的订单
df_filtered = df_clean[df_clean["Sales"] > 500]
print("\n===== 销售额大于 500 的订单 =====")
print(df_filtered[["Order Date", "Region", "Category", "Sales", "Profit"]].head())

# ---------------------- 4. 分组统计 ----------------------
# 按地区分组，计算销售额和利润的总和与平均值
df_grouped_region = df_clean.groupby("Region").agg(
    总销售额=("Sales", "sum"),
    平均销售额=("Sales", "mean"),
    总利润=("Profit", "sum"),
    平均利润=("Profit", "mean")
).reset_index()

# 按商品类别分组
df_grouped_category = df_clean.groupby("Category").agg(
    总销量=("Quantity", "sum"),
    总销售额=("Sales", "sum")
).reset_index()

print("\n===== 按地区分组统计 =====")
print(df_grouped_region)

# ---------------------- 5. 保存清洗后的数据 ----------------------
df_clean.to_csv("data/clean_data.csv", index=False, encoding="utf-8-sig")
print("\n✅ 清洗后数据已保存到 data/clean_data.csv")

# ---------------------- 6. 生成10张图表 ----------------------

# 图1：每日销售额趋势
df_clean["Order Date"] = pd.to_datetime(df_clean["Order Date"])
df_clean["month"] = df_clean["Order Date"].dt.to_period("M")
monthly_sales = df_clean.groupby("month")["Sales"].sum()

plt.figure(figsize=(10, 4))
monthly_sales.plot(kind='line', marker='o', color='b')
plt.title("月度销售额趋势")
plt.xlabel("月份")
plt.ylabel("销售额")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plots/01_monthly_sales.png")
plt.close()

# 图2：各地区总销售额柱状图
plt.figure(figsize=(6, 4))
plt.bar(df_grouped_region["Region"], df_grouped_region["总销售额"], color=['r', 'g', 'b', 'orange'])
plt.title("各地区总销售额对比")
plt.xlabel("地区")
plt.ylabel("总销售额")
plt.tight_layout()
plt.savefig("plots/02_region_sales.png")
plt.close()

# 图3：各商品类别销售额占比饼图
plt.figure(figsize=(6, 6))
plt.pie(df_grouped_category["总销售额"], labels=df_grouped_category["Category"], autopct='%1.1f%%')
plt.title("各商品类别销售额占比")
plt.tight_layout()
plt.savefig("plots/03_category_pie.png")
plt.close()

# 图4：销售额分布箱线图
plt.figure(figsize=(6, 4))
plt.boxplot(df_clean["Sales"], vert=False)
plt.title("销售额分布箱线图")
plt.xlabel("销售额")
plt.tight_layout()
plt.savefig("plots/04_sales_boxplot.png")
plt.close()

# 图5：利润 vs 销售额散点图
plt.figure(figsize=(6, 4))
plt.scatter(df_clean["Sales"], df_clean["Profit"], alpha=0.6, color='green')
plt.title("销售额与利润关系")
plt.xlabel("销售额")
plt.ylabel("利润")
plt.tight_layout()
plt.savefig("plots/05_sales_profit_scatter.png")
plt.close()

# 图6：各地区平均利润柱状图
plt.figure(figsize=(6, 4))
plt.bar(df_grouped_region["Region"], df_grouped_region["平均利润"], color='purple')
plt.title("各地区平均利润对比")
plt.xlabel("地区")
plt.ylabel("平均利润")
plt.tight_layout()
plt.savefig("plots/06_region_profit.png")
plt.close()

# 图7：订单数量前10的产品
top_products = df_clean.groupby("Product Name")["Quantity"].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 4))
top_products.plot(kind='barh', color='teal')
plt.title("订单数量前10的产品")
plt.xlabel("总销量")
plt.tight_layout()
plt.savefig("plots/07_top_products.png")
plt.close()

# 图8：不同细分市场的销售额对比
segment_sales = df_clean.groupby("Segment")["Sales"].sum()
plt.figure(figsize=(6, 4))
segment_sales.plot(kind='bar', color=['#ff9999','#66b3ff','#99ff99'])
plt.title("不同细分市场的销售额")
plt.xlabel("细分市场")
plt.ylabel("总销售额")
plt.tight_layout()
plt.savefig("plots/08_segment_sales.png")
plt.close()

# 图9：折扣与利润的关系
plt.figure(figsize=(6, 4))
plt.scatter(df_clean["Discount"], df_clean["Profit"], alpha=0.5, color='red')
plt.title("折扣与利润的关系")
plt.xlabel("折扣")
plt.ylabel("利润")
plt.tight_layout()
plt.savefig("plots/09_discount_profit.png")
plt.close()

# 图10：各商品子类别销售额堆叠柱状图
subcat_sales = df_clean.groupby(["Category", "Sub-Category"])["Sales"].sum().unstack()
subcat_sales.plot(kind='bar', stacked=True, figsize=(10, 5))
plt.title("各商品子类别销售额")
plt.xlabel("商品类别")
plt.ylabel("总销售额")
plt.legend(title="子类别", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("plots/10_subcategory_sales.png")
plt.close()

print("\n✅ 10张图表已全部保存到 plots/ 文件夹")