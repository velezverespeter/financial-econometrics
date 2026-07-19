# Financial Econometrics

A collection of independent time-series and econometric models applied to financial market data, each in its own subfolder with a dedicated README, dataset, and source code. Built with public market data and open-source tools for applied learning and demonstration purposes.

## Analyses

| Analysis | Description | Status |
|---|---|---|
| [`garch-volatility-model`](./garch-volatility-model) | GARCH(1,1) conditional volatility modeling and forecasting — captures volatility clustering in daily equity index returns | Complete |
| `cointegration-pairs` | Engle-Granger cointegration testing for statistical arbitrage pair selection | Planned |
| `event-studies` | Abnormal return analysis around corporate events (earnings, M&A announcements) | Planned |

Each subfolder is self-contained: its own `README.md`, `requirements.txt`, and `src/` directory, so any analysis can be run independently without needing the others installed.

## Repository Structure

```
financial-econometrics/
├── README.md                    (this file)
├── LICENSE
├── .gitignore
└── garch-volatility-model/
    ├── README.md
    ├── requirements.txt
    ├── src/
    ├── notebooks/
    ├── data/
    └── outputs/
```

## License

MIT (see LICENSE) — applies repo-wide unless a subfolder specifies otherwise.

---

<sub>This repository contains independent academic and demonstration work using publicly available data. It does not constitute investment research, financial advice, or a recommendation to buy, hold, or sell any security.</sub>
