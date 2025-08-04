import os
import yfinance as yf

# === 获取项目根目录 ===
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_path = os.path.join(project_root, "data", "stocks")
os.makedirs(data_path, exist_ok=True)

# === 输入参数 ===
company = input("Enter the company ticker symbol (e.g., NVDA): ").strip().upper()
start = input("Enter the start date (YYYY-MM-DD): ").strip()
end = input("Enter the end date (YYYY-MM-DD): ").strip()

# === 下载数据 ===
print(f"📥 正在下载 {company} 从 {start} 到 {end} 的数据...")
data = yf.download(company, start=start, end=end)

# === 保存文件 ===
filename = f"{company}_{start}_{end}.csv"
csv_path = os.path.join(data_path, filename)
data.to_csv(csv_path)

print(f"✅ 数据已保存至：{csv_path}")
