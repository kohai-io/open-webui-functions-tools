"""
title: Weather & Environment Data
author: open-webui
version: 1.6.3
description: Provides comprehensive weather and environment data for content production contexts. Includes current conditions, forecasts, Fire Weather Index (FWI), weather alerts, and air quality data. Works worldwide using OpenWeather API.
required_open_webui_version: 0.3.9
requirements: cryptography, aiohttp
"""

from pydantic import BaseModel, Field, GetCoreSchemaHandler, field_validator
from pydantic_core import core_schema
from typing import Optional, Callable, Awaitable, Any
from fastapi.responses import HTMLResponse
import aiohttp
import asyncio
import logging
import hashlib
import base64
import os
import json
from datetime import datetime, timezone
from cryptography.fernet import Fernet, InvalidToken

log = logging.getLogger(__name__)


class EncryptedStr(str):
    """A string type that automatically handles encryption/decryption"""

    @classmethod
    def _get_encryption_key(cls) -> Optional[bytes]:
        secret = os.getenv("WEBUI_SECRET_KEY")
        if not secret:
            return None
        hashed_key = hashlib.sha256(secret.encode()).digest()
        return base64.urlsafe_b64encode(hashed_key)

    @classmethod
    def encrypt(cls, value: str) -> str:
        if not value or value.startswith("encrypted:"):
            return value
        key = cls._get_encryption_key()
        if not key:
            return value
        f = Fernet(key)
        encrypted = f.encrypt(value.encode())
        return f"encrypted:{encrypted.decode()}"

    @classmethod
    def decrypt(cls, value: str) -> str:
        if not value or not value.startswith("encrypted:"):
            return value
        key = cls._get_encryption_key()
        if not key:
            return value[len("encrypted:"):]
        try:
            encrypted_part = value[len("encrypted:"):]
            f = Fernet(key)
            decrypted = f.decrypt(encrypted_part.encode())
            return decrypted.decode()
        except (InvalidToken, Exception):
            return value

    def get_decrypted(self) -> str:
        return self.decrypt(self)

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.str_schema()


class Tools:
    class Valves(BaseModel):
        OPENWEATHER_API_KEY: EncryptedStr = Field(
            default="",
            description="OpenWeather API key (get free key at https://openweathermap.org/api). Will be encrypted for security."
        )
        
        @field_validator('OPENWEATHER_API_KEY', mode='before')
        @classmethod
        def encrypt_api_key(cls, v):
            """Encrypt API key on save"""
            if v and isinstance(v, str) and not v.startswith("encrypted:"):
                return EncryptedStr.encrypt(v)
            return v
        
        DEFAULT_UNITS: str = Field(
            default="metric",
            description="Default units: 'metric' (Celsius), 'imperial' (Fahrenheit), or 'standard' (Kelvin)"
        )
        ENABLE_FIRE_INDEX: bool = Field(
            default=False,
            description="Enable Fire Weather Index (FWI) data for fire danger assessment. ⚠️ Requires paid OpenWeather subscription - contact sales at https://openweathermap.org/price"
        )
        ENABLE_AIR_QUALITY: bool = Field(
            default=True,
            description="Enable air quality index and pollution data"
        )
        ENABLE_ALERTS: bool = Field(
            default=True,
            description="Enable weather alerts and warnings"
        )
        FORECAST_DAYS: int = Field(
            default=5,
            description="Number of days for weather forecast (1-5)"
        )

    def __init__(self):
        self.valves = self.Valves()
        self.base_url = "https://api.openweathermap.org/data/2.5"
        self.onecall_url = "https://api.openweathermap.org/data/3.0/onecall"
        self.geo_url = "https://api.openweathermap.org/geo/1.0"

    def _get_api_key(self) -> str:
        """Get decrypted API key"""
        return EncryptedStr.decrypt(self.valves.OPENWEATHER_API_KEY)

    async def _geocode_location(self, location: str) -> Optional[dict]:
        """Convert location name to coordinates using OpenWeather Geocoding API"""
        api_key = self._get_api_key()
        if not api_key:
            log.error("[WEATHER] No API key available for geocoding")
            return None

        # Try multiple location format variations
        location_variants = [location]
        
        # If location has comma, try without state/country code
        if "," in location:
            parts = [p.strip() for p in location.split(",")]
            # Try just the city name
            location_variants.append(parts[0])
            # Try city + full country name for common abbreviations
            if len(parts) == 2:
                country_map = {
                    "SA": "South Africa",
                    "UK": "United Kingdom", 
                    "US": "United States",
                    "USA": "United States"
                }
                if parts[1].strip().upper() in country_map:
                    full_country = country_map[parts[1].strip().upper()]
                    location_variants.append(f"{parts[0]}, {full_country}")
        
        log.info(f"[WEATHER] Trying geocoding variants: {location_variants}")
        
        for variant in location_variants:
            try:
                url = f"{self.geo_url}/direct"
                params = {
                    "q": variant,
                    "limit": 1,
                    "appid": api_key
                }
                
                log.info(f"[WEATHER] Geocoding request: {url} with q={variant}")
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as response:
                        log.info(f"[WEATHER] Geocoding response status: {response.status}")
                        
                        if response.status == 200:
                            try:
                                results = await response.json()
                                log.info(f"[WEATHER] Geocoding results: {results}")
                                if results and len(results) > 0:
                                    result = results[0]
                                    coords = {
                                        "lat": result["lat"],
                                        "lon": result["lon"],
                                        "name": result.get("name", location),
                                        "country": result.get("country", ""),
                                        "state": result.get("state", "")
                                    }
                                    log.info(f"[WEATHER] Geocoding success: {coords}")
                                    return coords
                                else:
                                    log.warning(f"[WEATHER] Geocoding returned empty results for: {variant}")
                            except Exception as json_error:
                                response_text = await response.text()
                                log.error(f"[WEATHER] Failed to parse geocoding JSON: {json_error}, response: {response_text[:200]}")
                        elif response.status == 401:
                            log.error(f"[WEATHER] Geocoding API key invalid (401)")
                            return None
                        else:
                            response_text = await response.text()
                            log.warning(f"[WEATHER] Geocoding failed for '{variant}': HTTP {response.status}, response: {response_text[:200]}")
            except asyncio.TimeoutError:
                log.error(f"[WEATHER] Geocoding timeout for: {variant}")
            except Exception as e:
                log.error(f"[WEATHER] Geocoding error for '{variant}': {e}", exc_info=True)
        
        log.error(f"[WEATHER] All geocoding attempts failed for: {location}")
        return None

    def _get_weather_icon_emoji(self, icon_code: str) -> str:
        """Convert OpenWeather icon codes to emojis"""
        icon_map = {
            "01d": "☀️", "01n": "🌙",
            "02d": "⛅", "02n": "☁️",
            "03d": "☁️", "03n": "☁️",
            "04d": "☁️", "04n": "☁️",
            "09d": "🌧️", "09n": "🌧️",
            "10d": "🌦️", "10n": "🌧️",
            "11d": "⛈️", "11n": "⛈️",
            "13d": "❄️", "13n": "❄️",
            "50d": "🌫️", "50n": "🌫️"
        }
        return icon_map.get(icon_code, "🌤️")

    def _get_aqi_info(self, aqi: int) -> dict:
        """Get air quality index information with color and description"""
        aqi_levels = {
            1: {"label": "Good", "color": "#00e400", "description": "Air quality is satisfactory"},
            2: {"label": "Fair", "color": "#ffff00", "description": "Air quality is acceptable"},
            3: {"label": "Moderate", "color": "#ff7e00", "description": "Sensitive groups may experience effects"},
            4: {"label": "Poor", "color": "#ff0000", "description": "Health effects for everyone"},
            5: {"label": "Very Poor", "color": "#8f3f97", "description": "Serious health effects"}
        }
        return aqi_levels.get(aqi, {"label": "Unknown", "color": "#808080", "description": "No data"})

    def _get_fwi_info(self, fwi: float) -> dict:
        """Get Fire Weather Index information with color and risk level"""
        if fwi < 5.2:
            return {"level": "Low", "color": "#00ff00", "description": "Minimal fire danger"}
        elif fwi < 11.2:
            return {"level": "Moderate", "color": "#ffff00", "description": "Moderate fire danger"}
        elif fwi < 21.3:
            return {"level": "High", "color": "#ff9900", "description": "High fire danger"}
        elif fwi < 38.0:
            return {"level": "Very High", "color": "#ff0000", "description": "Very high fire danger"}
        else:
            return {"level": "Extreme", "color": "#cc0000", "description": "Extreme fire danger - high risk"}

    def _format_timestamp(self, timestamp: int) -> str:
        """Format Unix timestamp to readable date/time"""
        return datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    async def get_capabilities(
        self,
        __user__=None,
    ):
        """
        Explain what this weather tool can do.
        
        WHEN TO USE:
        - User asks: "what can you do?", "help", "what are your capabilities?"
        - User wants to know what features are available
        - User is unsure how to use the weather tool
        
        :return: Description of available features
        """
        
        return """## 🌤️ Weather & Environment Tool Capabilities

I can provide **comprehensive weather and environmental data** for any location worldwide with both visual dashboards and data analysis.

### What I Can Show You:

**🌡️ Current Conditions**
- Real-time temperature, humidity, pressure, wind speed/direction
- Weather description and conditions
- "Feels like" temperature

**📊 Visual Forecasts**
- **24-Hour Temperature Chart** - Hourly trends with feels-like overlay
- **5-Day Forecast Chart** - Daily high/day/low temperatures with weather icons
- Interactive Chart.js visualizations with tooltips

**🌬️ Air Quality (Optional)**
- AQI (Air Quality Index) with health descriptions
- Pollutant levels: PM2.5, PM10, O₃, NO₂, SO₂, CO
- Color-coded indicators (Good → Hazardous)

**🔥 Fire Weather Index (Optional)**
- Estimated fire danger rating (Low → Extreme)
- Based on temperature, humidity, and wind conditions
- Critical for wildfire-prone areas

**🚨 Weather Alerts (Optional)**
- Active weather warnings and advisories
- Alert descriptions, timing, and issuing authority
- Requires One Call API 3.0 subscription

### How to Use:

**Visual Dashboard:**
- **"show weather for [location]"** - Interactive dashboard with charts
- **"weather dashboard for London"** - Full visual display
- **"display Cape Town weather"** - Charts and metrics

**Data Analysis:**
- **"what's the temperature in Paris?"** - Get specific data points
- **"will it rain tomorrow in Tokyo?"** - Forecast details
- **"hourly breakdown for New York"** - Detailed hourly data
- **"air quality in Beijing"** - Pollution levels

**Text Summary:**
- **"quick weather check for Sydney"** - Brief text summary
- **"weather status in Dubai"** - Concise overview

### Features:

✅ **Dark/Light Mode** - Theme toggle with localStorage persistence
✅ **Worldwide Coverage** - Any location via OpenWeather API
✅ **Unit Options** - Metric (°C), Imperial (°F), or Kelvin
✅ **Real-time Data** - Updated current conditions
✅ **Multi-day Forecasts** - Up to 8 days (configurable)
✅ **Hourly Forecasts** - 48-hour detailed predictions

### Configuration:

- **DEFAULT_UNITS**: Choose 'metric', 'imperial', or 'standard'
- **FORECAST_DAYS**: Number of forecast days (1-8)
- **ENABLE_AIR_QUALITY**: Toggle air quality display
- **ENABLE_FIRE_INDEX**: Toggle fire danger index
- **ENABLE_ALERTS**: Toggle weather alerts (requires One Call API 3.0)

### API Requirements:

- Free OpenWeather API key required
- Optional: One Call API 3.0 subscription for hourly forecasts and alerts
- Get your key at: https://openweathermap.org/api

**Pro Tip:** The visual dashboard is perfect for planning outdoor activities, content production (filming locations), travel planning, or simply staying informed about conditions worldwide! 🌍"""

    async def get_weather(
        self,
        location: str,
        units: str = None,
        __user__=None,
        __event_emitter__=None,
    ):
        """
        Get comprehensive weather and environment data with rich VISUAL dashboard.
        Returns interactive HTML with Chart.js visualizations.
        
        Provides current conditions, forecasts, fire danger index, weather alerts, and air quality.
        Perfect for content production contexts where location weather matters (e.g., filming locations).
        
        WHEN TO USE:
        - User asks: "show me weather for [location]", "weather dashboard", "visual weather"
        - User wants: "see the forecast", "chart the temperature", "display weather"
        - Content production: "weather during filming", "check weather for shoot location"
        - User wants visual charts and interactive display
        
        OUTPUT: Interactive HTML dashboard with Chart.js charts (visual display)
        For raw data/metrics, use get_weather_data() instead.
        
        :param location: Location name (e.g., "Cape Town", "Los Angeles", "London, UK") or coordinates "lat,lon"
        :param units: Temperature units - 'metric' (°C), 'imperial' (°F), or 'standard' (K). Defaults to valve setting.
        :param __user__: User information
        :param __event_emitter__: Event emitter for status updates
        :return: Interactive HTML dashboard with weather visualizations
        """
        
        api_key = self._get_api_key()
        if not api_key:
            return "❌ **OpenWeather API key not configured.** Get a free key at https://openweathermap.org/api and add it to the tool settings."
        
        if not location:
            return "❌ **Please provide a location.** Example: 'get weather for Cape Town' or 'weather in Los Angeles'"
        
        units = units or self.valves.DEFAULT_UNITS
        
        try:
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": f"Fetching weather data for {location}...", "done": False}
                })
            
            # Parse location - check if it's coordinates or a place name
            coords = None
            if "," in location and len(location.split(",")) == 2:
                try:
                    lat, lon = location.split(",")
                    coords = {"lat": float(lat.strip()), "lon": float(lon.strip()), "name": "Custom Location"}
                except ValueError:
                    pass
            
            # If not coordinates, geocode the location
            if not coords:
                coords = await self._geocode_location(location)
                if not coords:
                    return f"❌ **Location not found:** {location}\n\n**Suggestions:**\n- Use full country name: 'Cape Town, South Africa' (not 'SA')\n- Try coordinates: '-33.9249,18.4241'\n- Check API key is valid at https://openweathermap.org/api\n\n**Common formats:**\n- 'London, United Kingdom'\n- 'Los Angeles, United States'\n- 'Tokyo, Japan'"
            
            location_name = coords.get("name", location)
            country = coords.get("country", "")
            state = coords.get("state", "")
            display_location = f"{location_name}"
            if state:
                display_location += f", {state}"
            if country:
                display_location += f", {country}"
            
            lat = coords["lat"]
            lon = coords["lon"]
            
            # Fetch data using One Call API 3.0 (current, forecast, alerts in one call)
            onecall_data = await self._fetch_onecall_data(lat, lon, units, api_key)
            
            # Parse One Call API response
            results = {
                "current": onecall_data.get("current") if onecall_data else None,
                "forecast": onecall_data.get("daily") if onecall_data else None,
                "hourly": onecall_data.get("hourly") if onecall_data else None,
                "alerts": onecall_data.get("alerts", []) if onecall_data else [],
            }
            
            # Fetch optional data in parallel
            tasks = {}
            
            if self.valves.ENABLE_AIR_QUALITY:
                tasks["air_quality"] = self._fetch_air_quality(lat, lon, api_key)
            
            if self.valves.ENABLE_FIRE_INDEX:
                tasks["fire_index"] = self._fetch_fire_weather_index(lat, lon, api_key)
            
            for key, task in tasks.items():
                try:
                    results[key] = await task
                except Exception as e:
                    log.error(f"[WEATHER] Failed to fetch {key}: {e}")
                    results[key] = None
            
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "Generating weather dashboard...", "done": False}
                })
            
            # Generate HTML dashboard
            html = self._generate_weather_dashboard(
                location=display_location,
                lat=lat,
                lon=lon,
                current=results.get("current"),
                forecast=results.get("forecast"),
                hourly=results.get("hourly"),
                air_quality=results.get("air_quality"),
                fire_index=results.get("fire_index"),
                alerts=results.get("alerts"),
                units=units
            )
            
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "Weather dashboard ready!", "done": True}
                })
            
            headers = {"Content-Disposition": "inline"}
            return HTMLResponse(content=html, headers=headers)
            
        except Exception as e:
            log.exception(f"[WEATHER] Error: {e}")
            return f"❌ **Error fetching weather data:** {str(e)}"

    async def _fetch_onecall_data(self, lat: float, lon: float, units: str, api_key: str) -> dict:
        """Fetch comprehensive weather data using One Call API 3.0
        
        Includes: current, minutely (48h), hourly (48h), daily (8 days), alerts
        Free tier: 1,000 calls/day
        """
        url = self.onecall_url
        params = {
            "lat": lat,
            "lon": lon,
            "units": units,
            "appid": api_key,
            "exclude": "minutely"  # Exclude minutely to reduce payload size
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    data = await response.json()
                    log.info(f"[WEATHER] One Call API success: current={bool(data.get('current'))}, daily={len(data.get('daily', []))}, alerts={len(data.get('alerts', []))}")
                    return data
                else:
                    error_text = await response.text()
                    log.error(f"[WEATHER] One Call API error ({response.status}): {error_text[:200]}")
                    return None


    async def _fetch_air_quality(self, lat: float, lon: float, api_key: str) -> dict:
        """Fetch air quality data"""
        url = f"{self.base_url}/air_pollution"
        params = {
            "lat": lat,
            "lon": lon,
            "appid": api_key
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    log.error(f"[WEATHER] Air quality API error: {response.status}")
                    return None

    async def _fetch_fire_weather_index(self, lat: float, lon: float, api_key: str) -> dict:
        """Calculate Fire Weather Index (FWI) - estimated from weather conditions
        
        NOTE: This is a simplified estimation. For official FWI data, a paid subscription is required.
        """
        try:
            # Get current weather data from One Call API for FWI calculation
            onecall_data = await self._fetch_onecall_data(lat, lon, "metric", api_key)
            if not onecall_data or not onecall_data.get("current"):
                return None
            
            current = onecall_data["current"]
            
            # Extract weather parameters (One Call API format)
            temp = current.get("temp", 20)
            humidity = current.get("humidity", 50)
            wind_speed = current.get("wind_speed", 0)
            
            # Simple fire danger calculation (not official FWI, but useful indicator)
            # High temp, low humidity, high wind = high fire danger
            fire_score = (temp / 10) + ((100 - humidity) / 10) + (wind_speed / 2)
            
            return {
                "fwi": fire_score,
                "temp": temp,
                "humidity": humidity,
                "wind_speed": wind_speed,
                "calculated": True  # Indicates this is estimated, not official FWI
            }
        except Exception as e:
            log.error(f"[WEATHER] Fire index calculation error: {e}")
        
        return None


    def _create_line_chart(self, chart_id: str, labels: list, datasets: list, title: str, y_label: str, height: int = 300) -> str:
        """Generate Chart.js line chart for temperature trends
        
        Args:
            chart_id: Unique ID for the chart
            labels: X-axis labels (dates/times)
            datasets: List of dataset dicts with format:
                {"label": "Temp", "data": [values], "borderColor": "#color", "backgroundColor": "rgba(...)"}
            title: Chart title
            y_label: Y-axis label
            height: Chart height in pixels
        """
        import base64
        import json
        import random
        
        chart_id = f"chart_{random.randint(10000, 99999)}"
        
        # Base64 encode to avoid HTML/markdown processing
        labels_b64 = base64.b64encode(json.dumps(labels).encode()).decode()
        datasets_b64 = base64.b64encode(json.dumps(datasets).encode()).decode()
        
        return f'''
            <div id="container_{chart_id}" style="height: {height}px; position: relative;"
                 data-labels="{labels_b64}"
                 data-datasets="{datasets_b64}"></div>
            <script>
                (function() {{
                    try {{
                        const container = document.getElementById('container_{chart_id}');
                        const canvas = document.createElement('canvas');
                        canvas.id = '{chart_id}';
                        container.appendChild(canvas);
                        
                        const chartLabels = JSON.parse(atob(container.dataset.labels));
                        const chartDatasets = JSON.parse(atob(container.dataset.datasets));
                        
                        new Chart(canvas, {{
                            type: 'line',
                            data: {{
                                labels: chartLabels,
                                datasets: chartDatasets
                            }},
                            options: {{
                                responsive: true,
                                maintainAspectRatio: false,
                                plugins: {{
                                    legend: {{
                                        position: 'top',
                                        labels: {{ color: '#333', font: {{ size: 12 }} }}
                                    }},
                                    title: {{
                                        display: true,
                                        text: '{title}',
                                        color: '#667eea',
                                        font: {{ size: 16, weight: 'bold' }}
                                    }},
                                    tooltip: {{
                                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                                        padding: 12,
                                        cornerRadius: 8
                                    }}
                                }},
                                scales: {{
                                    y: {{
                                        beginAtZero: false,
                                        title: {{
                                            display: true,
                                            text: '{y_label}',
                                            color: '#666'
                                        }},
                                        grid: {{ color: 'rgba(0, 0, 0, 0.1)' }}
                                    }},
                                    x: {{
                                        grid: {{ display: false }},
                                        ticks: {{ color: '#666' }}
                                    }}
                                }}
                            }}
                        }});
                    }} catch(e) {{
                        console.error('Line chart error ({chart_id}):', e);
                        container.innerHTML = '<p style="color: red; padding: 20px;">Chart error: ' + e.message + '</p>';
                    }}
                }})();
            </script>
        '''
    
    def _extract_weather_data(
        self,
        location: str,
        lat: float,
        lon: float,
        current: dict,
        forecast: list,
        hourly: list,
        air_quality: dict,
        fire_index: dict,
        alerts: list,
        units: str
    ) -> dict:
        """Extract structured weather data for chat-to-data capability"""
        
        unit_symbol = "°C" if units == "metric" else ("°F" if units == "imperial" else "K")
        wind_unit = "m/s" if units == "metric" else "mph"
        
        data = {
            "location": {
                "name": location,
                "latitude": lat,
                "longitude": lon
            },
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        }
        
        # Current weather
        if current:
            weather_info = current.get("weather", [{}])[0]
            data["current"] = {
                "temperature": current.get("temp"),
                "feels_like": current.get("feels_like"),
                "humidity": current.get("humidity"),
                "pressure": current.get("pressure"),
                "wind_speed": current.get("wind_speed"),
                "wind_direction": current.get("wind_deg"),
                "description": weather_info.get("description", "Unknown"),
                "icon_code": weather_info.get("icon", "01d"),
                "units": {
                    "temperature": unit_symbol,
                    "wind_speed": wind_unit,
                    "pressure": "hPa"
                }
            }
        
        # Forecast (daily)
        if forecast and isinstance(forecast, list):
            data["forecast"] = {
                "days": self.valves.FORECAST_DAYS,
                "daily": []
            }
            for item in forecast[:self.valves.FORECAST_DAYS]:
                dt = datetime.fromtimestamp(item.get("dt", 0), tz=timezone.utc)
                temp_day = item.get("temp", {})
                weather_info = item.get("weather", [{}])[0]
                data["forecast"]["daily"].append({
                    "date": dt.strftime("%Y-%m-%d"),
                    "day_name": dt.strftime("%A"),
                    "temp_day": temp_day.get("day"),
                    "temp_min": temp_day.get("min"),
                    "temp_max": temp_day.get("max"),
                    "description": weather_info.get("description", "Unknown"),
                    "humidity": item.get("humidity"),
                    "wind_speed": item.get("wind_speed"),
                    "rain_probability": item.get("pop", 0) * 100  # Probability of precipitation
                })
        
        # Hourly forecast
        if hourly and isinstance(hourly, list):
            data["hourly"] = []
            for hour in hourly[:24]:  # Next 24 hours
                dt = datetime.fromtimestamp(hour.get("dt", 0), tz=timezone.utc)
                data["hourly"].append({
                    "time": dt.strftime("%Y-%m-%d %H:%M"),
                    "temperature": hour.get("temp"),
                    "feels_like": hour.get("feels_like"),
                    "humidity": hour.get("humidity"),
                    "wind_speed": hour.get("wind_speed"),
                    "rain_probability": hour.get("pop", 0) * 100
                })
        
        # Air quality
        if air_quality and air_quality.get("list"):
            aqi_data = air_quality["list"][0]
            aqi = aqi_data["main"]["aqi"]
            aqi_info = self._get_aqi_info(aqi)
            components = aqi_data.get("components", {})
            data["air_quality"] = {
                "aqi": aqi,
                "level": aqi_info["label"],
                "description": aqi_info["description"],
                "pollutants": {
                    "pm2_5": components.get("pm2_5", 0),
                    "pm10": components.get("pm10", 0),
                    "o3": components.get("o3", 0),
                    "no2": components.get("no2", 0),
                    "so2": components.get("so2", 0),
                    "co": components.get("co", 0)
                },
                "unit": "μg/m³"
            }
        
        # Fire danger index
        if fire_index and fire_index.get("fwi") is not None:
            fwi_value = fire_index["fwi"]
            fwi_info = self._get_fwi_info(fwi_value)
            data["fire_danger"] = {
                "index": fwi_value,
                "level": fwi_info["level"],
                "description": fwi_info["description"],
                "is_estimated": fire_index.get("calculated", False),
                "factors": {
                    "temperature": fire_index.get("temp"),
                    "humidity": fire_index.get("humidity"),
                    "wind_speed": fire_index.get("wind_speed")
                }
            }
        
        # Weather alerts
        if alerts:
            data["alerts"] = []
            for alert in alerts:
                data["alerts"].append({
                    "event": alert.get("event", "Weather Alert"),
                    "description": alert.get("description", "No details available"),
                    "start": self._format_timestamp(alert.get("start", 0)),
                    "end": self._format_timestamp(alert.get("end", 0)),
                    "sender": alert.get("sender_name", "Unknown")
                })
        
        return data
    
    def _generate_weather_dashboard(
        self,
        location: str,
        lat: float,
        lon: float,
        current: dict,
        forecast: dict,
        hourly: list,
        air_quality: dict,
        fire_index: dict,
        alerts: list,
        units: str
    ) -> str:
        """Generate comprehensive weather dashboard HTML"""
        
        unit_symbol = "°C" if units == "metric" else ("°F" if units == "imperial" else "K")
        wind_unit = "m/s" if units == "metric" else "mph"
        
        # Current weather card (One Call API 3.0 format)
        current_html = ""
        if current:
            temp = current.get("temp", 0)
            feels_like = current.get("feels_like", temp)
            humidity = current.get("humidity", 0)
            pressure = current.get("pressure", 0)
            wind_speed = current.get("wind_speed", 0)
            wind_deg = current.get("wind_deg", 0)
            weather = current.get("weather", [{}])[0]
            description = weather.get("description", "Unknown").capitalize()
            icon = self._get_weather_icon_emoji(weather.get("icon", "01d"))
            
            current_html = f'''
            <div class="weather-card current-weather">
                <div class="weather-icon-large">{icon}</div>
                <div class="weather-details">
                    <div class="temperature">{temp:.1f}{unit_symbol}</div>
                    <div class="condition">{description}</div>
                    <div class="feels-like">Feels like {feels_like:.1f}{unit_symbol}</div>
                </div>
                <div class="weather-metrics">
                    <div class="metric">
                        <span class="metric-label">💧 Humidity</span>
                        <span class="metric-value">{humidity}%</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">💨 Wind</span>
                        <span class="metric-value">{wind_speed:.1f} {wind_unit}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">🌡️ Pressure</span>
                        <span class="metric-value">{pressure} hPa</span>
                    </div>
                </div>
            </div>
            '''
        
        # Forecast chart (One Call API 3.0 format - daily array)
        forecast_html = ""
        if forecast and isinstance(forecast, list):
            # forecast is already an array of daily forecasts from One Call API
            daily_forecasts = []
            for item in forecast[:self.valves.FORECAST_DAYS]:
                dt = datetime.fromtimestamp(item.get("dt", 0), tz=timezone.utc)
                temp_day = item.get("temp", {})
                daily_forecasts.append({
                    "date": dt.strftime("%a %d"),
                    "temp": temp_day.get("day", 0),
                    "temp_min": temp_day.get("min", 0),
                    "temp_max": temp_day.get("max", 0),
                    "description": item.get("weather", [{}])[0].get("description", "Unknown"),
                    "icon": self._get_weather_icon_emoji(item.get("weather", [{}])[0].get("icon", "01d"))
                })
            
            forecast_labels = [f["date"] for f in daily_forecasts]
            forecast_temps = [f["temp"] for f in daily_forecasts]
            forecast_temps_min = [f["temp_min"] for f in daily_forecasts]
            forecast_temps_max = [f["temp_max"] for f in daily_forecasts]
            
            # Create daily forecast chart
            forecast_chart = self._create_line_chart(
                chart_id="daily_forecast",
                labels=forecast_labels,
                datasets=[
                    {
                        "label": "High Temp",
                        "data": forecast_temps_max,
                        "borderColor": "#f97316",
                        "backgroundColor": "rgba(249, 115, 22, 0.1)",
                        "fill": True,
                        "tension": 0.4
                    },
                    {
                        "label": "Day Temp",
                        "data": forecast_temps,
                        "borderColor": "#667eea",
                        "backgroundColor": "rgba(102, 126, 234, 0.1)",
                        "fill": True,
                        "tension": 0.4
                    },
                    {
                        "label": "Low Temp",
                        "data": forecast_temps_min,
                        "borderColor": "#3b82f6",
                        "backgroundColor": "rgba(59, 130, 246, 0.1)",
                        "fill": True,
                        "tension": 0.4
                    }
                ],
                title=f"{self.valves.FORECAST_DAYS}-Day Temperature Forecast",
                y_label=f"Temperature ({unit_symbol})",
                height=300
            )
            
            forecast_html = f'''
            <div class="weather-card forecast-card">
                <h3>📅 {self.valves.FORECAST_DAYS}-Day Forecast</h3>
                {forecast_chart}
                <div class="forecast-grid" style="margin-top: 20px;">
                    {"".join([f'''
                    <div class="forecast-day">
                        <div class="forecast-date">{f["date"]}</div>
                        <div class="forecast-icon">{f["icon"]}</div>
                        <div class="forecast-temp">{f["temp"]:.0f}{unit_symbol}</div>
                        <div class="forecast-range">{f["temp_min"]:.0f}° - {f["temp_max"]:.0f}°</div>
                    </div>
                    ''' for f in daily_forecasts])}
                </div>
            </div>
            '''
        
        # Hourly forecast chart (48 hours)
        hourly_html = ""
        if hourly and isinstance(hourly, list):
            hourly_data = hourly[:24]  # Show 24 hours for readability
            hourly_labels = []
            hourly_temps = []
            hourly_feels = []
            
            for hour in hourly_data:
                dt = datetime.fromtimestamp(hour.get("dt", 0), tz=timezone.utc)
                hourly_labels.append(dt.strftime("%H:%M"))
                hourly_temps.append(hour.get("temp", 0))
                hourly_feels.append(hour.get("feels_like", 0))
            
            hourly_chart = self._create_line_chart(
                chart_id="hourly_forecast",
                labels=hourly_labels,
                datasets=[
                    {
                        "label": "Temperature",
                        "data": hourly_temps,
                        "borderColor": "#667eea",
                        "backgroundColor": "rgba(102, 126, 234, 0.2)",
                        "fill": True,
                        "tension": 0.4,
                        "pointRadius": 3
                    },
                    {
                        "label": "Feels Like",
                        "data": hourly_feels,
                        "borderColor": "#f59e0b",
                        "backgroundColor": "rgba(245, 158, 11, 0.1)",
                        "fill": True,
                        "tension": 0.4,
                        "pointRadius": 2,
                        "borderDash": [5, 5]
                    }
                ],
                title="24-Hour Temperature Forecast",
                y_label=f"Temperature ({unit_symbol})",
                height=250
            )
            
            hourly_html = f'''
            <div class="weather-card hourly-card">
                <h3>⏰ Hourly Forecast</h3>
                {hourly_chart}
            </div>
            '''
        
        # Air quality card
        air_quality_html = ""
        if air_quality and air_quality.get("list"):
            aqi_data = air_quality["list"][0]
            aqi = aqi_data["main"]["aqi"]
            aqi_info = self._get_aqi_info(aqi)
            components = aqi_data.get("components", {})
            
            air_quality_html = f'''
            <div class="weather-card air-quality-card">
                <h3>🌬️ Air Quality</h3>
                <div class="aqi-indicator" style="background-color: {aqi_info['color']};">
                    <div class="aqi-value">{aqi}</div>
                    <div class="aqi-label">{aqi_info['label']}</div>
                </div>
                <div class="aqi-description">{aqi_info['description']}</div>
                <div class="pollutants">
                    <div class="pollutant">
                        <span>PM2.5</span>
                        <strong>{components.get('pm2_5', 0):.1f}</strong>
                    </div>
                    <div class="pollutant">
                        <span>PM10</span>
                        <strong>{components.get('pm10', 0):.1f}</strong>
                    </div>
                    <div class="pollutant">
                        <span>O₃</span>
                        <strong>{components.get('o3', 0):.1f}</strong>
                    </div>
                    <div class="pollutant">
                        <span>NO₂</span>
                        <strong>{components.get('no2', 0):.1f}</strong>
                    </div>
                </div>
            </div>
            '''
        
        # Fire danger card
        fire_html = ""
        if fire_index and fire_index.get("fwi") is not None:
            fwi_value = fire_index["fwi"]
            fwi_info = self._get_fwi_info(fwi_value)
            is_calculated = fire_index.get("calculated", False)
            
            fire_html = f'''
            <div class="weather-card fire-card">
                <h3>🔥 Fire Danger Index</h3>
                <div class="fwi-indicator" style="background-color: {fwi_info['color']};">
                    <div class="fwi-value">{fwi_value:.1f}</div>
                    <div class="fwi-label">{fwi_info['level']}</div>
                </div>
                <div class="fwi-description">{fwi_info['description']}</div>
                {'<div class="fwi-note">⚠️ Estimated index based on temperature, humidity, and wind speed</div>' if is_calculated else ''}
                <div class="fire-factors">
                    <div class="factor">
                        <span>🌡️ Temperature</span>
                        <strong>{fire_index.get('temp', 0):.1f}°C</strong>
                    </div>
                    <div class="factor">
                        <span>💧 Humidity</span>
                        <strong>{fire_index.get('humidity', 0)}%</strong>
                    </div>
                    <div class="factor">
                        <span>💨 Wind Speed</span>
                        <strong>{fire_index.get('wind_speed', 0):.1f} m/s</strong>
                    </div>
                </div>
            </div>
            '''
        
        # Alerts card
        alerts_html = ""
        if alerts:
            alerts_list = "".join([f'''
            <div class="alert-item">
                <div class="alert-header">
                    <span class="alert-icon">⚠️</span>
                    <strong>{alert.get('event', 'Weather Alert')}</strong>
                </div>
                <div class="alert-description">{alert.get('description', 'No details available')[:200]}...</div>
                <div class="alert-time">
                    {self._format_timestamp(alert.get('start', 0))} - {self._format_timestamp(alert.get('end', 0))}
                </div>
            </div>
            ''' for alert in alerts[:3]])
            
            alerts_html = f'''
            <div class="weather-card alerts-card">
                <h3>🚨 Active Weather Alerts</h3>
                {alerts_list}
            </div>
            '''
        
        # Complete HTML with CSS
        html = f'''
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Weather Dashboard - {location}</title>
            <style>
                /* Light Theme (Default) */
                :root {{
                    --bg-primary: #667eea;
                    --bg-secondary: #764ba2;
                    --card-bg: rgba(255, 255, 255, 0.95);
                    --card-gradient-start: rgba(255, 255, 255, 0.98);
                    --card-gradient-end: rgba(255, 255, 255, 0.95);
                    --card-border: rgba(102, 126, 234, 0.1);
                    --text-primary: #2d3748;
                    --text-secondary: #666;
                    --text-muted: #999;
                    --accent: #667eea;
                    --accent-glow: rgba(102, 126, 234, 0.4);
                    --shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
                    --shadow-hover: 0 15px 40px rgba(0, 0, 0, 0.2);
                    --radius: 16px;
                    --radius-sm: 12px;
                    --radial-1: rgba(118, 75, 162, 0.3);
                    --radial-2: rgba(102, 126, 234, 0.3);
                    --header-text: white;
                }}
                
                /* Dark Theme */
                [data-theme="dark"] {{
                    --bg-primary: #0b1020;
                    --bg-secondary: #14204a;
                    --card-bg: rgba(17, 26, 51, 0.95);
                    --card-gradient-start: rgba(17, 26, 51, 0.98);
                    --card-gradient-end: rgba(17, 26, 51, 0.95);
                    --card-border: rgba(35, 48, 88, 0.6);
                    --text-primary: #e7ecff;
                    --text-secondary: #9fb2ffcc;
                    --text-muted: #9fb2ff99;
                    --accent: #82a0ff;
                    --accent-glow: rgba(130, 160, 255, 0.5);
                    --shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
                    --shadow-hover: 0 15px 40px rgba(0, 0, 0, 0.5);
                    --radial-1: rgba(20, 32, 74, 0.8);
                    --radial-2: rgba(27, 42, 99, 0.7);
                    --header-text: #e7ecff;
                }}
                
                * {{
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }}
                
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    background: radial-gradient(1200px 700px at 80% -10%, var(--radial-1) 0%, transparent 60%),
                                radial-gradient(1000px 600px at -10% 0%, var(--radial-2) 0%, transparent 55%),
                                linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
                    color: var(--text-primary);
                    padding: 20px;
                    min-height: 100vh;
                    transition: background 0.3s ease;
                }}
                
                .dashboard-container {{
                    max-width: 1200px;
                    margin: 0 auto;
                }}
                
                .dashboard-header {{
                    background: linear-gradient(180deg, var(--card-gradient-start), var(--card-gradient-end));
                    border: 1px solid var(--card-border);
                    border-radius: var(--radius);
                    padding: 30px;
                    margin-bottom: 30px;
                    box-shadow: var(--shadow);
                    backdrop-filter: blur(10px);
                    text-align: center;
                }}
                
                .dashboard-header h1 {{
                    color: var(--accent);
                    font-size: 2.5em;
                    margin-bottom: 10px;
                    font-weight: 700;
                }}
                
                .dashboard-header .coordinates,
                .dashboard-header .timestamp {{
                    color: var(--text-secondary);
                    font-size: 0.9em;
                    margin: 5px 0;
                }}
                
                .weather-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 20px;
                    margin-bottom: 20px;
                }}
                
                .weather-card {{
                    background: linear-gradient(180deg, var(--card-gradient-start), var(--card-gradient-end));
                    border: 1px solid var(--card-border);
                    border-radius: var(--radius-sm);
                    padding: 24px;
                    box-shadow: var(--shadow);
                    backdrop-filter: blur(10px);
                    transition: all 0.3s ease;
                }}
                
                .weather-card:hover {{
                    transform: translateY(-5px);
                    box-shadow: var(--shadow-hover);
                }}
                
                .current-weather {{
                    grid-column: span 2;
                    display: flex;
                    align-items: center;
                    gap: 30px;
                }}
                
                .weather-icon-large {{
                    font-size: 120px;
                    line-height: 1;
                }}
                
                .weather-details {{
                    flex: 1;
                }}
                
                .temperature {{
                    font-size: 4em;
                    font-weight: bold;
                    color: var(--accent);
                }}
                
                .condition {{
                    font-size: 1.5em;
                    color: var(--text-secondary);
                    margin: 10px 0;
                }}
                
                .feels-like {{
                    color: var(--text-muted);
                    font-size: 1.1em;
                }}
                
                .weather-metrics {{
                    display: grid;
                    grid-template-columns: repeat(3, 1fr);
                    gap: 15px;
                    margin-top: 20px;
                }}
                
                .metric {{
                    display: flex;
                    flex-direction: column;
                    text-align: center;
                    padding: 10px;
                    background: rgba(139, 148, 184, 0.08);
                    border-radius: 8px;
                }}
                
                .metric-label {{
                    font-size: 0.9em;
                    color: var(--text-secondary);
                    margin-bottom: 5px;
                }}
                
                .metric-value {{
                    font-size: 1.2em;
                    font-weight: bold;
                    color: var(--text-primary);
                }}
                
                .forecast-card h3,
                .hourly-card h3,
                .air-quality-card h3,
                .fire-card h3,
                .alerts-card h3 {{
                    font-size: 1.3em;
                    margin-bottom: 20px;
                    color: var(--accent);
                    font-weight: 600;
                }}
                
                .forecast-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
                    gap: 15px;
                }}
                
                .forecast-day {{
                    text-align: center;
                    padding: 15px;
                    background: rgba(139, 148, 184, 0.08);
                    border-radius: 10px;
                }}
                
                .forecast-date {{
                    font-weight: bold;
                    color: var(--accent);
                    margin-bottom: 10px;
                }}
                
                .forecast-icon {{
                    font-size: 2.5em;
                    margin: 10px 0;
                }}
                
                .forecast-temp {{
                    font-size: 1.5em;
                    font-weight: bold;
                    color: var(--text-primary);
                }}
                
                .forecast-range {{
                    font-size: 0.85em;
                    color: var(--text-muted);
                    margin-top: 5px;
                }}
                
                .aqi-indicator, .fwi-indicator {{
                    text-align: center;
                    padding: 20px;
                    border-radius: 12px;
                    color: white;
                    margin-bottom: 15px;
                }}
                
                .aqi-value, .fwi-value {{
                    font-size: 3em;
                    font-weight: bold;
                }}
                
                .aqi-label, .fwi-label {{
                    font-size: 1.2em;
                    margin-top: 5px;
                }}
                
                .aqi-description, .fwi-description {{
                    text-align: center;
                    color: var(--text-secondary);
                    margin-bottom: 15px;
                }}
                
                .fwi-note {{
                    text-align: center;
                    color: var(--text-muted);
                    font-size: 0.85em;
                    margin-bottom: 15px;
                    font-style: italic;
                }}
                
                .pollutants, .fire-factors {{
                    display: grid;
                    grid-template-columns: repeat(2, 1fr);
                    gap: 10px;
                }}
                
                .pollutant, .factor {{
                    display: flex;
                    justify-content: space-between;
                    padding: 10px;
                    background: rgba(139, 148, 184, 0.08);
                    border-radius: 8px;
                    color: var(--text-primary);
                }}
                
                .alert-item {{
                    background: #fff3cd;
                    border-left: 4px solid #ff9800;
                    padding: 15px;
                    margin-bottom: 10px;
                    border-radius: 8px;
                }}
                
                .alert-header {{
                    display: flex;
                    align-items: center;
                    gap: 10px;
                    margin-bottom: 8px;
                }}
                
                .alert-icon {{
                    font-size: 1.5em;
                }}
                
                .alert-description {{
                    color: var(--text-secondary);
                    margin-bottom: 8px;
                    line-height: 1.4;
                }}
                
                .alert-time {{
                    font-size: 0.85em;
                    color: var(--text-muted);
                }}
                
                .footer {{
                    text-align: center;
                    color: var(--text-secondary);
                    margin-top: 30px;
                    padding: 20px;
                    background: linear-gradient(180deg, var(--card-gradient-start), var(--card-gradient-end));
                    border: 1px solid var(--card-border);
                    border-radius: var(--radius-sm);
                    backdrop-filter: blur(10px);
                    font-size: 0.9em;
                }}
                
                @media (max-width: 768px) {{
                    .current-weather {{
                        grid-column: span 1;
                        flex-direction: column;
                    }}
                    
                    .temperature {{
                        font-size: 3em;
                    }}
                    
                    .weather-metrics {{
                        grid-template-columns: 1fr;
                    }}
                    
                    .theme-toggle {{
                        top: 10px;
                        right: 10px;
                        padding: 8px 16px;
                        font-size: 12px;
                    }}
                }}
                
                .theme-toggle {{
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    z-index: 1000;
                    background: var(--card-bg);
                    border: 1px solid var(--card-border);
                    border-radius: 50px;
                    padding: 10px 20px;
                    box-shadow: var(--shadow);
                    backdrop-filter: blur(10px);
                    cursor: pointer;
                    display: flex;
                    align-items: center;
                    gap: 10px;
                    font-size: 14px;
                    font-weight: 600;
                    color: var(--text-primary);
                    transition: all 0.3s ease;
                }}
                .theme-toggle:hover {{
                    transform: translateY(-2px);
                    box-shadow: var(--shadow-hover);
                }}
                .theme-toggle .icon {{
                    font-size: 18px;
                }}
            </style>
            <script>
                // Theme toggle functionality with localStorage persistence
                function initTheme() {{
                    const savedTheme = localStorage.getItem('weather-dashboard-theme') || 'light';
                    document.documentElement.setAttribute('data-theme', savedTheme);
                    updateToggleButton(savedTheme);
                }}
                
                function toggleTheme() {{
                    const currentTheme = document.documentElement.getAttribute('data-theme') || 'light';
                    const newTheme = currentTheme === 'light' ? 'dark' : 'light';
                    document.documentElement.setAttribute('data-theme', newTheme);
                    localStorage.setItem('weather-dashboard-theme', newTheme);
                    updateToggleButton(newTheme);
                }}
                
                function updateToggleButton(theme) {{
                    const btn = document.getElementById('theme-toggle-btn');
                    if (btn) {{
                        if (theme === 'dark') {{
                            btn.innerHTML = '<span class="icon">☀️</span><span>Light Mode</span>';
                        }} else {{
                            btn.innerHTML = '<span class="icon">🌙</span><span>Dark Mode</span>';
                        }}
                    }}
                }}
                
                // Initialize theme on page load
                document.addEventListener('DOMContentLoaded', initTheme);
            </script>
        </head>
        <body>
            <!-- Theme Toggle Button -->
            <button id="theme-toggle-btn" class="theme-toggle" onclick="toggleTheme()">
                <span class="icon">🌙</span>
                <span>Dark Mode</span>
            </button>
            
            <div class="dashboard-container">
                <div class="dashboard-header">
                    <h1>🌍 {location}</h1>
                    <div class="coordinates">📍 {lat:.4f}, {lon:.4f}</div>
                    <div class="timestamp">Updated: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}</div>
                </div>
                
                <div class="weather-grid">
                    {current_html}
                </div>
                
                <div class="weather-grid">
                    {hourly_html}
                </div>
                
                <div class="weather-grid">
                    {forecast_html}
                </div>
                
                {air_quality_html}
                {fire_html}
                
                {f'<div class="weather-grid">{alerts_html}</div>' if alerts_html else ''}
                
                <div class="footer">
                    Powered by OpenWeather API • Data updated in real-time
                </div>
            </div>
        </body>
        </html>
        '''
        
        return html


    async def get_weather_summary(
        self,
        location: str,
        units: str = None,
        __user__=None,
    ):
        """
        Get concise text summary of weather and environment conditions.
        Returns plain text/markdown without rich UI visualizations.
        
        WHEN TO USE:
        - User asks: "quick weather check", "brief weather summary", "weather status"
        - Need weather info without full dashboard
        - API/programmatic access to weather data
        - Embedded in other workflows
        
        :param location: Location name or coordinates "lat,lon"
        :param units: Temperature units - 'metric', 'imperial', or 'standard'
        :param __user__: User information
        :return: Markdown-formatted text summary
        """
        
        api_key = self._get_api_key()
        if not api_key:
            return "❌ **OpenWeather API key not configured.**"
        
        units = units or self.valves.DEFAULT_UNITS
        unit_symbol = "°C" if units == "metric" else ("°F" if units == "imperial" else "K")
        
        try:
            # Parse location
            coords = None
            if "," in location and len(location.split(",")) == 2:
                try:
                    lat, lon = location.split(",")
                    coords = {"lat": float(lat.strip()), "lon": float(lon.strip()), "name": "Custom Location"}
                except ValueError:
                    pass
            
            if not coords:
                coords = await self._geocode_location(location)
                if not coords:
                    return f"❌ **Location not found:** {location}"
            
            location_name = coords.get("name", location)
            lat = coords["lat"]
            lon = coords["lon"]
            
            # Fetch weather data using One Call API
            onecall_data = await self._fetch_onecall_data(lat, lon, units, api_key)
            if not onecall_data or not onecall_data.get("current"):
                return f"❌ **Failed to fetch weather data for {location_name}**"
            
            current = onecall_data["current"]
            temp = current.get("temp", 0)
            feels_like = current.get("feels_like", temp)
            humidity = current.get("humidity", 0)
            wind_speed = current.get("wind_speed", 0)
            weather = current.get("weather", [{}])[0]
            description = weather.get("description", "Unknown").capitalize()
            icon = self._get_weather_icon_emoji(weather.get("icon", "01d"))
            
            wind_unit = "m/s" if units == "metric" else "mph"
            
            summary = f"""## {icon} Weather for {location_name}

**Current Conditions:**
- 🌡️ Temperature: {temp:.1f}{unit_symbol} (feels like {feels_like:.1f}{unit_symbol})
- ☁️ Conditions: {description}
- 💧 Humidity: {humidity}%
- 💨 Wind: {wind_speed:.1f} {wind_unit}
"""
            
            # Add fire danger if enabled
            if self.valves.ENABLE_FIRE_INDEX:
                fire_index = await self._fetch_fire_weather_index(lat, lon, api_key)
                if fire_index:
                    fwi_info = self._get_fwi_info(fire_index["fwi"])
                    summary += f"\n**Fire Danger:** 🔥 {fwi_info['level']} ({fwi_info['description']})"
            
            # Add air quality if enabled
            if self.valves.ENABLE_AIR_QUALITY:
                air_quality = await self._fetch_air_quality(lat, lon, api_key)
                if air_quality and air_quality.get("list"):
                    aqi = air_quality["list"][0]["main"]["aqi"]
                    aqi_info = self._get_aqi_info(aqi)
                    summary += f"\n**Air Quality:** 🌬️ {aqi_info['label']} ({aqi_info['description']})"
            
            summary += f"\n\n📍 Coordinates: {lat:.4f}, {lon:.4f}"
            summary += f"\n⏰ Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
            
            return summary
            
        except Exception as e:
            log.exception(f"[WEATHER] Error: {e}")
            return f"❌ **Error:** {str(e)}"

    async def get_weather_data(
        self,
        location: str,
        units: str = None,
        __user__=None,
    ):
        """
        Get raw weather data as structured JSON for programmatic analysis and chat.
        Returns numbers, metrics, and data WITHOUT any charts or visual display.
        
        WHEN TO USE:
        - User asks: "what's the temperature", "how humid is it", "wind speed", "will it rain"
        - User wants to ANALYZE, COMPARE, or CALCULATE with specific metrics
        - Questions like: "what's the forecast for tomorrow?", "hourly breakdown", "air quality levels?"
        - User needs: specific stats, numerical data, hourly details, forecast data
        - Follow-up queries: "show me just the temperature", "what about humidity?"
        
        OUTPUT: Pure JSON dictionary with structured weather metrics (no charts)
        For visual dashboards with charts, use get_weather() instead.
        
        :param location: Location name (e.g., "Cape Town", "Los Angeles", "London, UK") or coordinates "lat,lon"
        :param units: Temperature units - 'metric' (°C), 'imperial' (°F), or 'standard' (K). Defaults to valve setting.
        :param __user__: User information
        :return: Dictionary with structured weather data
        """
        
        api_key = self._get_api_key()
        if not api_key:
            return {"error": "OpenWeather API key not configured"}
        
        if not location:
            return {"error": "Please provide a location"}
        
        units = units or self.valves.DEFAULT_UNITS
        
        try:
            # Parse location - check if it's coordinates or a place name
            coords = None
            if "," in location and len(location.split(",")) == 2:
                try:
                    lat, lon = location.split(",")
                    coords = {"lat": float(lat.strip()), "lon": float(lon.strip()), "name": "Custom Location"}
                except ValueError:
                    pass
            
            # If not coordinates, geocode the location
            if not coords:
                coords = await self._geocode_location(location)
                if not coords:
                    return {"error": f"Location not found: {location}"}
            
            location_name = coords.get("name", location)
            country = coords.get("country", "")
            state = coords.get("state", "")
            display_location = f"{location_name}"
            if state:
                display_location += f", {state}"
            if country:
                display_location += f", {country}"
            
            lat = coords["lat"]
            lon = coords["lon"]
            
            # Fetch data using One Call API 3.0
            onecall_data = await self._fetch_onecall_data(lat, lon, units, api_key)
            
            # Parse One Call API response
            results = {
                "current": onecall_data.get("current") if onecall_data else None,
                "forecast": onecall_data.get("daily") if onecall_data else None,
                "hourly": onecall_data.get("hourly") if onecall_data else None,
                "alerts": onecall_data.get("alerts", []) if onecall_data else [],
            }
            
            # Fetch optional data in parallel
            tasks = {}
            
            if self.valves.ENABLE_AIR_QUALITY:
                tasks["air_quality"] = self._fetch_air_quality(lat, lon, api_key)
            
            if self.valves.ENABLE_FIRE_INDEX:
                tasks["fire_index"] = self._fetch_fire_weather_index(lat, lon, api_key)
            
            for key, task in tasks.items():
                try:
                    results[key] = await task
                except Exception as e:
                    log.error(f"[WEATHER] Failed to fetch {key}: {e}")
                    results[key] = None
            
            # Extract structured data
            data = self._extract_weather_data(
                location=display_location,
                lat=lat,
                lon=lon,
                current=results.get("current"),
                forecast=results.get("forecast"),
                hourly=results.get("hourly"),
                air_quality=results.get("air_quality"),
                fire_index=results.get("fire_index"),
                alerts=results.get("alerts"),
                units=units
            )
            
            return data
            
        except Exception as e:
            log.exception(f"[WEATHER] Error: {e}")
            return {"error": str(e)}
