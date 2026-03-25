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


### Windows 一键启动

在 Windows 上可直接双击运行：

```bat
scripts\start_windows.bat
```

这个脚本会切到项目根目录并执行 `python run.py`。服务启动后会自动打开浏览器，地址默认是 `http://127.0.0.1:8001`（端口占用会自动递增）。

### 生成公网访问链接（可选）

如果你需要把本机服务临时分享给别人，可用内网穿透工具把本机端口映射到公网：

```bash
ngrok http 8001
```

如果 run.py 最终使用的是 `8002/8003...`，请把命令里的端口改成终端实际打印的端口。

4. **快速可用性自检（推荐）**：
   ```bash
   ./scripts/smoke_test.sh
   ```
   这个脚本会自动启动服务并检查 health / symbols / ohlcv 三个关键路径，确保“服务真实可跑可用”。

## 功能概览

- **K 线 + 成交量**：多交易对、多周期（5m / 15m / 1h / 4h / 1D）
- **数据**：OKX API 直连，最新数据，无需本地 CSV（可配置为 API-only）
- **数据模式**：默认混合模式（本地 CSV + API 增量），API 被限制时仍可用；可配置为 API-only
- **Recognizing**：自动识别形态（三角形、头肩、通道等）并绘制
- **Assist**：支撑/阻力趋势线（延伸）
- **手绘线**：趋势线、水平线
- **Backtest**：MFI/MA 策略回测、参数优化
- **形态统计**：当前形态 vs 历史同类成功率、线相似度

## 技术栈

- 前端：HTML / CSS / JavaScript，LightweightCharts
- 后端：FastAPI，Python（形态识别、回测、OKX 数据）
