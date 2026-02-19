# crypto-analysis

crypto-analysis multi trendline — TradingView 风格的加密货币技术分析网页版：K 线、多周期、多趋势线/形态识别、手绘线、回测。

## 运行

1. **安装依赖**：
   ```bash
   pip install -r requirements.txt
   ```

2. **启动服务**：
   ```bash
   python run.py
   ```

3. **打开浏览器**：  
   会自动打开 `http://127.0.0.1:8001`（若 8001 被占用会尝试 8002、8003…）。未自动打开时请手动输入终端里显示的地址。

## 功能概览

- **K 线 + 成交量**：多交易对、多周期（5m / 15m / 1h / 4h / 1D）
- **数据**：OKX API 直连，最新数据，无需本地 CSV（可配置为 API-only）
- **Recognizing**：自动识别形态（三角形、头肩、通道等）并绘制
- **Assist**：支撑/阻力趋势线（延伸）
- **手绘线**：趋势线、水平线
- **Backtest**：MFI/MA 策略回测、参数优化
- **形态统计**：当前形态 vs 历史同类成功率、线相似度

## 技术栈

- 前端：HTML / CSS / JavaScript，LightweightCharts
- 后端：FastAPI，Python（形态识别、回测、OKX 数据）
