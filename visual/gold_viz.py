import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

df = pd.read_csv(r"D:\EXE112\anhthuongemnhatma\GOLD_PRICE.csv", encoding="utf-8-sig")

df["Ngày"] = pd.to_datetime(df["Ngày"])
df["Giá mua"] = pd.to_numeric(df["Giá mua"])
df["Giá bán"] = pd.to_numeric(df["Giá bán"])

df = df.sort_values("Ngày")

df["Giá mua"] /= 1e6
df["Giá bán"] /= 1e6

df_daily = df.groupby("Ngày")[["Giá mua", "Giá bán"]].mean().reset_index()

plt.figure(figsize=(10,5))
plt.plot(df_daily["Ngày"], df_daily["Giá mua"], label="Buy Price")
plt.plot(df_daily["Ngày"], df_daily["Giá bán"], label="Sell Price")
plt.legend()
plt.title("Gold Price Trend (Million VND)")
plt.show()

df_plot = df[df["Mã vàng"].isin(["BTSJC", "BT9999NTT"])]

fig = px.line(
    df_plot,
    x="Ngày",
    y="Giá mua",
    color="Mã vàng",
    title="Gold Price Comparison"
)

fig.show()