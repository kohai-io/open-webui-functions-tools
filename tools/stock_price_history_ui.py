import os
import requests
from datetime import datetime, timedelta
from pydantic import Field, BaseModel
from fastapi.responses import HTMLResponse


class Tools:
    class Valves(BaseModel):
        ALPHA_VANTAGE_API_KEY: str = Field(
            default="",
            description="API key for Alpha Vantage (get a free key at https://www.alphavantage.co/support/#api-key).",
        )

    def __init__(self):
        self.valves = self.Valves()
        pass

    # ---- internal: shared renderer with basic CSS + JS ----
    def _render_html(
        self, title: str, body_html: str, error: bool = False, chart_js: str = ""
    ) -> HTMLResponse:
        style = """
        <style>
            :root {
                --bg: #0b1020;
                --card: #111a33;
                --text: #e7ecff;
                --muted: #9fb2ffcc;
                --accent: #82a0ff;
                --danger: #ff6b6b;
                --ok: #8bd6a5;
                --border: #233058;
                --shadow: 0 10px 30px rgba(0,0,0,0.35);
                --radius: 16px;
            }
            * { box-sizing: border-box; }
            body {
                margin: 0; padding: 24px;
                font: 15px/1.55 ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, "Helvetica Neue", Arial, "Apple Color Emoji", "Segoe UI Emoji";
                background: radial-gradient(1200px 700px at 80% -10%, #14204a 0%, transparent 60%),
                            radial-gradient(1000px 600px at -10% 0%, #1b2a63 0%, transparent 55%),
                            var(--bg);
                color: var(--text);
            }
            .wrap {
                max-width: 920px; margin: 0 auto;
            }
            .card {
                background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
                border: 1px solid var(--border);
                border-radius: var(--radius);
                box-shadow: var(--shadow);
                padding: 20px 22px;
                backdrop-filter: blur(6px);
            }
            h1 {
                font-size: 20px; margin: 0 0 12px;
                display: flex; align-items: center; gap: 10px;
            }
            h1 .dot {
                width: 10px; height: 10px; border-radius: 999px;
                background: var(--accent);
                box-shadow: 0 0 12px var(--accent);
            }
            .error h1 .dot { background: var(--danger); box-shadow: 0 0 12px var(--danger); }
            .meta {
                display: grid; grid-template-columns: 1fr 1fr; gap: 10px 16px; margin-top: 12px;
            }
            .kv {
                background: rgba(255,255,255,0.03);
                border: 1px solid var(--border);
                border-radius: 12px; padding: 10px 12px;
            }
            .kv .k { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }
            .kv .v { font-size: 16px; margin-top: 2px; }
            .lead { color: var(--muted); margin: 8px 0 0; }
            .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; }
            .ok { color: var(--ok); }
            .error .lead { color: #ffd1d1; }
            @media (max-width:640px){ .meta { grid-template-columns: 1fr; } }
        </style>
        """
        html = f"""
        <html>
        <head>
            <meta charset="utf-8">
            {style}
        </head>
        <body>
            <div class="wrap">
                <div class="card {'error' if error else ''}">
                    <h1><span class="dot"></span>{title}</h1>
                    {body_html}
                </div>
            </div>
            {chart_js}
        </body>
        </html>
        """
        response = None
        headers = {"Content-Disposition": "inline"}
        if error:
            response = HTMLResponse(content=html, status_code=404, headers=headers)
        else:
            response = HTMLResponse(content=html, headers=headers)
        return response

    # ---- public method with chart ----
    def get_stock_price_history(
        self,
        symbol: str = Field(
            "AAPL", description="Stock symbol (e.g., AAPL, GOOGL, MSFT, TSLA)"
        ),
        period: str = Field(
            "1month", description="Time period: 1week, 1month, 3months, 6months, 1year"
        ),
    ) -> HTMLResponse:
        api_key = self.valves.ALPHA_VANTAGE_API_KEY
        if not api_key:
            body = """<p class="lead">Missing Alpha Vantage API key.</p>"""
            return self._render_html("Configuration Error", body, error=True)

        function_map = {
            "1week": "TIME_SERIES_DAILY",
            "1month": "TIME_SERIES_DAILY",
            "3months": "TIME_SERIES_DAILY",
            "6months": "TIME_SERIES_DAILY",
            "1year": "TIME_SERIES_WEEKLY",
        }
        function = function_map.get(period, "TIME_SERIES_DAILY")
        base_url = "https://www.alphavantage.co/query"
        params = {
            "function": function,
            "symbol": symbol.upper(),
            "apikey": api_key,
            "outputsize": "compact" if period in ["1week", "1month"] else "full",
        }

        try:
            response = requests.get(base_url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()

            time_series_key = next((k for k in data.keys() if "Time Series" in k), None)
            if not time_series_key or not data.get(time_series_key):
                return self._render_html(
                    "No Data", "<p>No stock data found.</p>", error=True
                )

            time_series = data[time_series_key]

            now = datetime.now()
            period_days = {
                "1week": 7,
                "1month": 30,
                "3months": 90,
                "6months": 180,
                "1year": 365,
            }
            cutoff_date = now - timedelta(days=period_days.get(period, 30))
            filtered_data = {
                d: v
                for d, v in time_series.items()
                if datetime.strptime(d, "%Y-%m-%d") >= cutoff_date
            }
            if not filtered_data:
                filtered_data = dict(
                    list(time_series.items())[: period_days.get(period, 30)]
                )

            sorted_dates = sorted(filtered_data.keys())
            prices = [float(filtered_data[d]["4. close"]) for d in sorted_dates]
            labels = sorted_dates

            current_price = prices[-1]
            start_price = prices[0]
            change = current_price - start_price
            change_pct = (change / start_price * 100) if start_price else 0

            high_price = max(prices)
            low_price = min(prices)
            avg_price = sum(prices) / len(prices)

            change_class = "ok" if change >= 0 else "error"
            change_symbol = "+" if change >= 0 else ""

            # Chart.js integration
            chart_js = f"""
            <script>
            const ctx = document.createElement('canvas');
            ctx.id = "stockChart";
            document.querySelector('.card').appendChild(ctx);
            new Chart(ctx, {{
                type: 'line',
                data: {{
                    labels: {labels},
                    datasets: [{{
                        label: '{symbol.upper()} closing price',
                        data: {prices},
                        fill: true,
                        borderColor: '#82a0ff',
                        backgroundColor: 'rgba(130,160,255,0.1)',
                        tension: 0.15,
                        pointRadius: 2
                    }}]
                }},
                options: {{
                    responsive: true,
                    scales: {{
                        x: {{
                            ticks: {{ color: '#9fb2ff' }},
                            grid: {{ color: 'rgba(255,255,255,0.05)' }}
                        }},
                        y: {{
                            ticks: {{ color: '#9fb2ff' }},
                            grid: {{ color: 'rgba(255,255,255,0.05)' }}
                        }}
                    }},
                    plugins: {{
                        legend: {{ labels: {{ color: '#e7ecff' }} }}
                    }}
                }}
            }});
            </script>
            """

            body = f"""
                <p class="lead">Stock price data from Alpha Vantage API</p>
                <div class="meta">
                    <div class="kv"><div class="k">Symbol</div><div class="v mono">{symbol.upper()}</div></div>
                    <div class="kv"><div class="k">Current</div><div class="v mono">${current_price:.2f}</div></div>
                    <div class="kv"><div class="k">{period} Change</div>
                        <div class="v mono" style="color:var(--{change_class});">{change_symbol}${change:.2f} ({change_symbol}{change_pct:.2f}%)</div>
                    </div>
                    <div class="kv"><div class="k">High</div><div class="v mono">${high_price:.2f}</div></div>
                    <div class="kv"><div class="k">Low</div><div class="v mono">${low_price:.2f}</div></div>
                    <div class="kv"><div class="k">Average</div><div class="v mono">${avg_price:.2f}</div></div>
                </div>
            """
            return self._render_html(
                f"Stock Analysis: {symbol.upper()}", body, chart_js=chart_js
            )

        except Exception as e:
            return self._render_html(
                "Error", f"<p class='lead'>Error: {e}</p>", error=True
            )
