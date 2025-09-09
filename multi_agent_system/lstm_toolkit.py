"""LSTM Toolkit for numerical and economic data analysis."""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import re
from datetime import datetime
import io
import base64

# Optional matplotlib import for visualization
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not installed. Visualization features will be disabled.")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from .logging_config import SystemLogger


class LSTMToolkit:
    """Toolkit for LSTM-based time series analysis and forecasting."""
    
    def __init__(self):
        self.logger = SystemLogger("lstm_toolkit")
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        
    def _detect_numerical_data(self, text: str) -> Optional[List[float]]:
        """Detect and extract numerical data from text."""
        # Look for numerical patterns: lists, arrays, or sequences of numbers
        patterns = [
            r'\[([\d\.,\s]+)\]',  # [1, 2, 3, 4]
            r'array\(\[([\d\.,\s]+)\]\)',  # array([1, 2, 3, 4])
            r'(\d+\.?\d*)[,\s]+(\d+\.?\d*)',  # 1, 2, 3, 4 or 1 2 3 4
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                # Flatten matches and convert to numbers
                numbers = []
                for match in matches:
                    if isinstance(match, tuple):
                        numbers.extend([float(x.replace(',', '').strip()) for x in match if x.strip()])
                    else:
                        numbers.extend([float(x.replace(',', '').strip()) for x in match.split() if x.strip()])
                
                if numbers:
                    self.logger.info("Numerical data detected", data_points=len(numbers))
                    return numbers
        
        return None
    
    def _prepare_data(self, data: List[float], look_back: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for LSTM training."""
        # Normalize the data
        data_array = np.array(data).reshape(-1, 1)
        data_normalized = self.scaler.fit_transform(data_array)
        
        # Create sequences for LSTM
        X, y = [], []
        for i in range(len(data_normalized) - look_back):
            X.append(data_normalized[i:(i + look_back), 0])
            y.append(data_normalized[i + look_back, 0])
        
        return np.array(X), np.array(y)
    
    def build_model(self, look_back: int, units: int = 50) -> Sequential:
        """Build LSTM model architecture."""
        model = Sequential()
        model.add(LSTM(units, return_sequences=True, input_shape=(look_back, 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(25))
        model.add(Dense(1))
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        return model
    
    async def analyze_time_series(self, text: str, look_back: int = 10, epochs: int = 50) -> Dict[str, Any]:
        """Analyze time series data using LSTM."""
        try:
            # Detect numerical data
            data = self._detect_numerical_data(text)
            if not data or len(data) < look_back * 2:
                return {
                    "success": False,
                    "error": "Insufficient numerical data detected for analysis",
                    "data_points": len(data) if data else 0,
                    "minimum_required": look_back * 2
                }
            
            self.logger.info("Starting time series analysis", data_points=len(data), look_back=look_back)
            
            # Prepare data
            X, y = self._prepare_data(data, look_back)
            
            # Reshape for LSTM [samples, time steps, features]
            X = X.reshape(X.shape[0], X.shape[1], 1)
            
            # Build and train model
            self.model = self.build_model(look_back)
            history = self.model.fit(X, y, epochs=epochs, batch_size=32, verbose=0, validation_split=0.2)
            
            # Make predictions
            predictions = self.model.predict(X, verbose=0)
            predictions = self.scaler.inverse_transform(predictions).flatten()
            actual = self.scaler.inverse_transform(y.reshape(-1, 1)).flatten()
            
            # Calculate metrics
            mse = mean_squared_error(actual, predictions)
            mae = mean_absolute_error(actual, predictions)
            rmse = np.sqrt(mse)
            
            # Generate forecast
            last_sequence = X[-1].reshape(1, look_back, 1)
            forecast = self.model.predict(last_sequence, verbose=0)
            forecast = self.scaler.inverse_transform(forecast)[0][0]
            
            # Create visualization
            plot_url = self._create_visualization(actual, predictions, data)
            
            return {
                "success": True,
                "data_points": len(data),
                "metrics": {
                    "mse": float(mse),
                    "mae": float(mae),
                    "rmse": float(rmse)
                },
                "forecast": float(forecast),
                "trend": self._analyze_trend(data),
                "visualization": plot_url,
                "analysis_summary": self._generate_analysis_summary(data, predictions, forecast)
            }
            
        except Exception as e:
            self.logger.error("Time series analysis failed", error=str(e))
            return {
                "success": False,
                "error": str(e)
            }
    
    def _analyze_trend(self, data: List[float]) -> str:
        """Analyze overall trend of the data."""
        if len(data) < 2:
            return "Insufficient data for trend analysis"
        
        # Simple trend analysis
        first_half = data[:len(data)//2]
        second_half = data[len(data)//2:]
        
        avg_first = sum(first_half) / len(first_half)
        avg_second = sum(second_half) / len(second_half)
        
        if avg_second > avg_first * 1.1:
            return "Strong upward trend"
        elif avg_second > avg_first:
            return "Moderate upward trend"
        elif avg_second < avg_first * 0.9:
            return "Strong downward trend"
        elif avg_second < avg_first:
            return "Moderate downward trend"
        else:
            return "Relatively stable"
    
    def _create_visualization(self, actual: np.ndarray, predictions: np.ndarray, full_data: List[float]) -> str:
        """Create matplotlib visualization and return as base64 encoded image."""
        if not MATPLOTLIB_AVAILABLE:
            return "Visualization disabled: matplotlib not installed"
        
        try:
            plt.figure(figsize=(12, 6))
            
            # Plot full data
            plt.plot(range(len(full_data)), full_data, label='Full Data', alpha=0.7)
            
            # Plot predictions (aligned with the end of the data)
            pred_start = len(full_data) - len(actual)
            plt.plot(range(pred_start, pred_start + len(actual)), actual, label='Actual', linewidth=2)
            plt.plot(range(pred_start, pred_start + len(predictions)), predictions, label='Predicted', linewidth=2)
            
            plt.title('Time Series Analysis with LSTM')
            plt.xlabel('Time Steps')
            plt.ylabel('Values')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            plt.close()
            
            # Encode as base64
            img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
            return f"data:image/png;base64,{img_str}"
            
        except Exception as e:
            return f"Visualization failed: {str(e)}"
    
    def _generate_analysis_summary(self, data: List[float], predictions: np.ndarray, forecast: float) -> str:
        """Generate human-readable analysis summary."""
        trend = self._analyze_trend(data)
        volatility = np.std(data) / np.mean(data) if np.mean(data) != 0 else 0
        
        summary = f"""
Time Series Analysis Summary:
- Data Points: {len(data)}
- Overall Trend: {trend}
- Volatility: {volatility:.3f}
- Next Period Forecast: {forecast:.2f}
- Forecast Confidence: {'High' if len(data) >= 50 else 'Medium' if len(data) >= 20 else 'Low'}

Key Insights:
"""
        
        if trend.lower().startswith('strong upward'):
            summary += "- Strong positive momentum detected\n"
            summary += "- Consider potential for continued growth\n"
        elif trend.lower().startswith('strong downward'):
            summary += "- Significant decline observed\n"
            summary += "- Caution advised for future projections\n"
        
        if volatility > 0.2:
            summary += "- High volatility suggests unstable patterns\n"
        elif volatility > 0.1:
            summary += "- Moderate volatility indicates some instability\n"
        else:
            summary += "- Low volatility suggests stable patterns\n"
        
        return summary
    
    async def economic_analysis(self, text: str) -> Dict[str, Any]:
        """Specialized economic data analysis using LSTM."""
        result = await self.analyze_time_series(text, look_back=12, epochs=100)
        
        if result["success"]:
            # Add economic-specific insights
            result["economic_insights"] = self._generate_economic_insights(result)
        
        return result
    
    def _generate_economic_insights(self, analysis_result: Dict[str, Any]) -> Dict[str, str]:
        """Generate economic-specific insights from analysis."""
        insights = {}
        
        trend = analysis_result.get("trend", "")
        forecast = analysis_result.get("forecast", 0)
        metrics = analysis_result.get("metrics", {})
        
        insights["market_outlook"] = self._get_market_outlook(trend, forecast)
        insights["investment_advice"] = self._get_investment_advice(trend, metrics.get("rmse", 0))
        insights["risk_assessment"] = self._get_risk_assessment(metrics.get("volatility", 0))
        
        return insights
    
    def _get_market_outlook(self, trend: str, forecast: float) -> str:
        """Generate market outlook based on trend and forecast."""
        if "upward" in trend.lower():
            return f"Bullish outlook with expected continuation to {forecast:.2f}"
        elif "downward" in trend.lower():
            return f"Bearish outlook with potential decline to {forecast:.2f}"
        else:
            return "Neutral market outlook with stable expectations"
    
    def _get_investment_advice(self, trend: str, error: float) -> str:
        """Generate investment advice based on trend and error."""
        advice = []
        
        if "upward" in trend.lower() and error < 0.1:
            advice.append("Consider increasing investment exposure")
        elif "downward" in trend.lower():
            advice.append("Recommend reducing risk exposure")
        
        if error > 0.2:
            advice.append("High prediction error suggests cautious approach")
        elif error < 0.05:
            advice.append("Low prediction error indicates reliable forecasts")
        
        return "; ".join(advice) if advice else "Maintain current investment strategy"
    
    def _get_risk_assessment(self, volatility: float) -> str:
        """Generate risk assessment based on volatility."""
        if volatility > 0.3:
            return "High risk - Significant price fluctuations expected"
        elif volatility > 0.15:
            return "Medium risk - Moderate price movements anticipated"
        else:
            return "Low risk - Stable price environment expected"


# Global instance for easy access
lstm_toolkit = LSTMToolkit()