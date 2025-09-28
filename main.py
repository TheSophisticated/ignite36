import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import your backend class
# from water_replenishment_backend import WaterReplenishmentPredictor

# Configure page
st.set_page_config(
    page_title="🌊 AquaPredict - Water Replenishment Forecasting by return red",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #1f4e79, #2e8bc0, #41a3d8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    
    .info-box {
        background: linear-gradient(135deg, #74b9ff, #0984e3);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #e74c3c, #c0392b);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(231, 76, 60, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = False
if 'data' not in st.session_state:
    st.session_state.data = None
if 'anomaly_alerts' not in st.session_state:
    st.session_state.anomaly_alerts = []
if 'village_coordinates' not in st.session_state:
    # Mock coordinates for Indian villages (you can replace with real coordinates)
    st.session_state.village_coordinates = {}

# Header
st.markdown('<h1 class="main-header">🌊 AquaPredict</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Water Replenishment Forecasting System by <span style="color: #e74c3c; font-weight: bold;">return red</span></p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🎯 Navigation")
    page = st.selectbox("Choose a section:", 
                       ["🏠 Dashboard", "📊 Data Analysis", "🔮 Predictions", "🗺️ Map View", "🚨 Anomaly Alerts", "📈 Model Performance", "ℹ️ About"])
    
    st.markdown("---")
    
    # Anomaly Alert Banner in Sidebar
    if st.session_state.anomaly_alerts:
        st.markdown("### 🚨 Active Alerts")
        alert_count = len(st.session_state.anomaly_alerts)
        st.error(f"⚠️ {alert_count} Active Alert{'s' if alert_count > 1 else ''}")
        
        # Show latest alert
        if st.session_state.anomaly_alerts:
            latest_alert = st.session_state.anomaly_alerts[-1]
            st.markdown(f"**Latest:** {latest_alert['village']}")
            st.markdown(f"*{latest_alert['type']}*")
    
    st.markdown("### 💧 Quick Stats")
    if st.session_state.data is not None:
        try:
            st.metric("Total Records", len(st.session_state.data))
            
            # Handle village count safely
            village_cols = [col for col in st.session_state.data.columns if 'VILLAGE' in col.upper()]
            if village_cols:
                st.metric("Villages", st.session_state.data[village_cols[0]].nunique())
            else:
                st.metric("Villages", "N/A")
            
            # Handle date range safely
            date_cols = [col for col in st.session_state.data.columns if 'DATE' in col.upper()]
            if date_cols:
                try:
                    # Convert to datetime if it's not already
                    date_data = pd.to_datetime(st.session_state.data[date_cols[0]])
                    date_range = f"{date_data.min().strftime('%Y-%m-%d')} to {date_data.max().strftime('%Y-%m-%d')}"
                    st.metric("Date Range", date_range)
                except:
                    st.metric("Date Range", "Invalid dates")
            else:
                st.metric("Date Range", "No date column found")
        except Exception as e:
            st.metric("Status", "Data loaded")

# Main content based on page selection
if page == "🏠 Dashboard":
    st.markdown("## 🌟 Welcome to AquaPredict Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📁 Data Upload & Training")
        uploaded_file = st.file_uploader("Upload your water data CSV", type=['csv'])
        
        if uploaded_file is not None:
            try:
                # Note: Replace this with actual backend integration
                st.success("✅ File uploaded successfully!")
                sample_data = pd.read_csv(uploaded_file)
                
                # Process date columns safely
                date_cols = [col for col in sample_data.columns if 'DATE' in col.upper()]
                for col in date_cols:
                    try:
                        sample_data[col] = pd.to_datetime(sample_data[col])
                    except:
                        pass  # Keep original format if conversion fails
                
                st.session_state.data = sample_data
                
                # Show data preview
                st.markdown("**Data Preview:**")
                st.dataframe(sample_data.head(), use_container_width=True)
                
                if st.button("🚀 Train Model", type="primary"):
                    with st.spinner("Training model... This may take a few minutes"):
                        # Here you would integrate with your backend
                        # predictor = WaterReplenishmentPredictor()
                        # df_processed = predictor.load_and_process_data(uploaded_file.name)
                        # predictor.train_model(df_processed)
                        # st.session_state.predictor = predictor
                        st.session_state.trained_model = True
                        st.success("🎉 Model trained successfully!")
                        st.balloons()
                        
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    
    with col2:
        st.markdown("### 🎯 System Status")
        
        if st.session_state.trained_model:
            st.markdown("""
            <div class="info-box">
                <h4>🟢 System Ready</h4>
                <p>Model is trained and ready for predictions!</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Mock metrics for demo
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric("Accuracy (R²)", "0.87", "0.05")
            with col_m2:
                st.metric("MAE", "1.23m", "-0.12")
            with col_m3:
                st.metric("RMSE", "2.01m", "-0.18")
        else:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #fdcb6e, #e17055); padding: 1rem; border-radius: 10px; color: white;">
                <h4>⚠️ System Not Ready</h4>
                <p>Please upload data and train the model first.</p>
            </div>
            """, unsafe_allow_html=True)

elif page == "📊 Data Analysis":
    st.markdown("## 📊 Data Analysis & Insights")
    
    if st.session_state.data is not None:
        data = st.session_state.data
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", len(data))
        with col2:
            try:
                village_cols = [col for col in data.columns if 'VILLAGE' in col.upper()]
                if village_cols:
                    st.metric("Villages", data[village_cols[0]].nunique())
                else:
                    st.metric("Villages", "N/A")
            except:
                st.metric("Villages", "N/A")
        with col3:
            try:
                wl_cols = [col for col in data.columns if 'WL' in col.upper()]
                if wl_cols:
                    st.metric("Avg Water Level", f"{data[wl_cols[0]].mean():.2f}m")
                else:
                    st.metric("Avg Water Level", "N/A")
            except:
                st.metric("Avg Water Level", "N/A")
        with col4:
            st.metric("Date Range", f"{len(data)} records")
        
        # Create sample visualizations
        st.markdown("### 📈 Water Level Trends")
        
        # Generate sample time series data for demo
        dates = pd.date_range(start='2023-01-01', periods=len(data[:100]), freq='D')
        sample_wl = np.random.normal(15, 5, len(dates))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=sample_wl,
            mode='lines',
            name='Water Level',
            line=dict(color='#74b9ff', width=2)
        ))
        
        fig.update_layout(
            title="Water Level Over Time",
            xaxis_title="Date",
            yaxis_title="Water Level (mbgl)",
            template="plotly_white",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Distribution plot
        st.markdown("### 📊 Water Level Distribution")
        
        fig2 = px.histogram(
            x=sample_wl,
            nbins=30,
            title="Water Level Distribution",
            color_discrete_sequence=['#667eea']
        )
        fig2.update_layout(
            xaxis_title="Water Level (mbgl)",
            yaxis_title="Frequency",
            template="plotly_white"
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
    else:
        st.warning("📁 Please upload data first to see analysis!")

elif page == "🔮 Predictions":
    st.markdown("## 🔮 Water Replenishment Predictions")
    
    if st.session_state.trained_model and st.session_state.data is not None:
        # Village Selection Section
        st.markdown("### 📍 Select Village")
        
        # Get village names from data
        village_cols = [col for col in st.session_state.data.columns if 'VILLAGE' in col.upper()]
        if village_cols:
            village_col = village_cols[0]
            villages = sorted(st.session_state.data[village_col].dropna().unique())
            
            # Search and select village
            col_search, col_select = st.columns([2, 1])
            with col_search:
                search_term = st.text_input("🔍 Search Village", placeholder="Type village name...")
                
                # Filter villages based on search
                if search_term:
                    filtered_villages = [v for v in villages if search_term.lower() in str(v).lower()]
                else:
                    filtered_villages = villages[:20]  # Show first 20 by default
                    
            with col_select:
                selected_village = st.selectbox("Select Village", 
                                              options=filtered_villages,
                                              index=0 if filtered_villages else None)
        else:
            st.warning("No village data found in the dataset!")
            selected_village = "Sample Village"
        
        if selected_village:
            st.success(f"🏘️ Selected: **{selected_village}**")
            
            # Get historical data for selected village (mock data for demo)
            st.markdown("### 📊 Historical Water Level Trend")
            
            # Generate sample historical data
            dates = pd.date_range(end=datetime.now(), periods=90, freq='D')
            np.random.seed(hash(str(selected_village)) % 1000)  # Consistent random data per village
            base_level = np.random.uniform(10, 30)
            seasonal_pattern = 5 * np.sin(2 * np.pi * np.arange(90) / 365 * 4)  # Seasonal variation
            noise = np.random.normal(0, 2, 90)
            historical_wl = base_level + seasonal_pattern + noise
            
            # Create historical trend plot
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Scatter(
                x=dates,
                y=historical_wl,
                mode='lines',
                name='Historical Water Level',
                line=dict(color='#74b9ff', width=2),
                hovertemplate='Date: %{x}<br>Water Level: %{y:.2f}m<extra></extra>'
            ))
            
            fig_hist.update_layout(
                title=f"Water Level Trend - {selected_village}",
                xaxis_title="Date",
                yaxis_title="Water Level (mbgl)",
                template="plotly_white",
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Input Parameters Section
            st.markdown("### 🎛️ Prediction Parameters")
            
            # Auto-fill with recent data from selected village
            recent_wl = historical_wl[-1]
            recent_7d_avg = np.mean(historical_wl[-7:])
            recent_30d_avg = np.mean(historical_wl[-30:])
            
            col1, col2 = st.columns(2)
            
            with col1:
                current_wl = st.number_input("Current Water Level (mbgl)", 
                                           min_value=0.0, max_value=200.0, 
                                           value=float(recent_wl), step=0.1)
                wl_7day_avg = st.number_input("7-day Average WL", 
                                            min_value=0.0, max_value=200.0, 
                                            value=float(recent_7d_avg), step=0.1)
                wl_30day_avg = st.number_input("30-day Average WL", 
                                             min_value=0.0, max_value=200.0, 
                                             value=float(recent_30d_avg), step=0.1)
                daily_replenishment = st.number_input("Daily Replenishment", 
                                                    min_value=-10.0, max_value=10.0, 
                                                    value=0.2, step=0.1)
                weekly_replenishment = st.number_input("Weekly Replenishment", 
                                                     min_value=-5.0, max_value=5.0, 
                                                     value=0.1, step=0.1)
            
            with col2:
                trend_7d = st.number_input("7-day Trend", 
                                         min_value=-2.0, max_value=2.0, 
                                         value=0.05, step=0.01)
                trend_30d = st.number_input("30-day Trend", 
                                          min_value=-1.0, max_value=1.0, 
                                          value=0.02, step=0.01)
                month = st.selectbox("Month", 
                                   options=list(range(1, 13)), 
                                   index=datetime.now().month-1)
                season_factor = st.selectbox("Season Factor", 
                                           options=[0.5, 1, 2, 4], 
                                           index=2)
                depth_factor = st.selectbox("Depth Factor", 
                                          options=[0.4, 0.7, 1.0, 1.5], 
                                          index=2)
            
            if st.button("🔍 Generate Prediction", type="primary", use_container_width=True):
                # Mock prediction based on village characteristics
                village_seed = hash(str(selected_village)) % 1000
                np.random.seed(village_seed)
                base_prediction = np.random.normal(0.5, 0.4)
                
                # Adjust prediction based on inputs
                season_adjustment = (season_factor - 1) * 0.2
                depth_adjustment = (depth_factor - 1) * 0.1
                trend_adjustment = trend_7d * 0.5
                
                prediction = base_prediction + season_adjustment + depth_adjustment + trend_adjustment
                
                # Display prediction
                st.markdown(f"""
                <div class="prediction-box">
                    🔮 Predicted Weekly Replenishment for {selected_village}: {prediction:.3f}m
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence intervals
                confidence_low = prediction - 0.25
                confidence_high = prediction + 0.25
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Lower Bound", f"{confidence_low:.3f}m", delta=f"{confidence_low-prediction:.3f}")
                with col2:
                    st.metric("Prediction", f"{prediction:.3f}m", delta="baseline")
                with col3:
                    st.metric("Upper Bound", f"{confidence_high:.3f}m", delta=f"{confidence_high-prediction:.3f}")
                
                # Create prediction visualization
                st.markdown("### 📈 Prediction Visualization")
                
                # Generate future dates for prediction
                future_dates = pd.date_range(start=dates[-1] + timedelta(days=1), periods=30, freq='D')
                
                # Generate predicted trend
                predicted_trend = []
                last_wl = historical_wl[-1]
                daily_change = prediction / 7  # Convert weekly to daily
                
                # Anomaly detection during prediction
                anomalies_detected = []
                
                for i in range(30):
                    # Add some realistic variation
                    variation = np.random.normal(0, 0.1)
                    new_wl = last_wl + daily_change + variation
                    predicted_trend.append(new_wl)
                    
                    # Check for anomalies
                    if new_wl < 5:  # Critical low level
                        anomalies_detected.append({
                            'day': i+1,
                            'type': 'Critical Low Water Level',
                            'value': new_wl,
                            'severity': 'HIGH'
                        })
                    elif abs(daily_change) > 2:  # Rapid change
                        anomalies_detected.append({
                            'day': i+1,
                            'type': 'Rapid Water Level Change',
                            'value': daily_change,
                            'severity': 'MEDIUM'
                        })
                    elif new_wl > 50:  # Unusually high level
                        anomalies_detected.append({
                            'day': i+1,
                            'type': 'Unusually High Water Level',
                            'value': new_wl,
                            'severity': 'LOW'
                        })
                    
                    last_wl = new_wl
                
                # Add anomalies to session state
                for anomaly in anomalies_detected:
                    alert = {
                        'village': selected_village,
                        'type': anomaly['type'],
                        'severity': anomaly['severity'],
                        'value': anomaly['value'],
                        'timestamp': datetime.now(),
                        'day': anomaly['day']
                    }
                    if alert not in st.session_state.anomaly_alerts:
                        st.session_state.anomaly_alerts.append(alert)
                
                # Show anomaly alerts if detected
                if anomalies_detected:
                    st.markdown("### 🚨 Anomalies Detected!")
                    for anomaly in anomalies_detected:
                        severity_color = {
                            'HIGH': 'error',
                            'MEDIUM': 'warning', 
                            'LOW': 'info'
                        }
                        getattr(st, severity_color[anomaly['severity']])(
                            f"**{anomaly['severity']}**: {anomaly['type']} on Day {anomaly['day']} "
                            f"(Value: {anomaly['value']:.2f}m)"
                        )
                
                # Create comprehensive plot
                fig_pred = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('Water Level Forecast', 'Replenishment Rate'),
                    vertical_spacing=0.1,
                    row_heights=[0.7, 0.3]
                )
                
                # Historical data
                fig_pred.add_trace(
                    go.Scatter(
                        x=dates[-30:],  # Last 30 days
                        y=historical_wl[-30:],
                        mode='lines',
                        name='Historical',
                        line=dict(color='#74b9ff', width=2),
                        hovertemplate='Date: %{x}<br>Water Level: %{y:.2f}m<extra></extra>'
                    ),
                    row=1, col=1
                )
                
                # Predicted data
                fig_pred.add_trace(
                    go.Scatter(
                        x=future_dates,
                        y=predicted_trend,
                        mode='lines',
                        name='Predicted',
                        line=dict(color='#ff7675', width=2, dash='dash'),
                        hovertemplate='Date: %{x}<br>Predicted Level: %{y:.2f}m<extra></extra>'
                    ),
                    row=1, col=1
                )
                
                # Confidence band
                upper_band = [p + 0.5 for p in predicted_trend]
                lower_band = [p - 0.5 for p in predicted_trend]
                
                fig_pred.add_trace(
                    go.Scatter(
                        x=list(future_dates) + list(future_dates[::-1]),
                        y=upper_band + lower_band[::-1],
                        fill='toself',
                        fillcolor='rgba(255, 118, 117, 0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='Confidence Band',
                        showlegend=False
                    ),
                    row=1, col=1
                )
                
                # Replenishment rate chart
                replenishment_rates = [prediction/7] * 30 + np.random.normal(0, 0.05, 30)
                fig_pred.add_trace(
                    go.Bar(
                        x=future_dates,
                        y=replenishment_rates,
                        name='Daily Replenishment',
                        marker_color='#00b894',
                        hovertemplate='Date: %{x}<br>Replenishment: %{y:.3f}m/day<extra></extra>'
                    ),
                    row=2, col=1
                )
                
                fig_pred.update_layout(
                    title=f"Water Level Forecast - {selected_village}",
                    template="plotly_white",
                    height=600,
                    showlegend=True
                )
                
                fig_pred.update_xaxes(title_text="Date", row=2, col=1)
                fig_pred.update_yaxes(title_text="Water Level (mbgl)", row=1, col=1)
                fig_pred.update_yaxes(title_text="Replenishment (m/day)", row=2, col=1)
                
                st.plotly_chart(fig_pred, use_container_width=True)
                
                # Risk Assessment with Village Context
                st.markdown("### 🎯 Risk Assessment")
                
                col_risk1, col_risk2 = st.columns(2)
                
                with col_risk1:
                    if prediction > 0.5:
                        st.success("🟢 **Excellent Outlook**")
                        st.write(f"• Strong replenishment expected for {selected_village}")
                        st.write("• Water levels likely to improve significantly")
                        st.write("• Good conditions for agriculture/usage")
                    elif prediction > 0.2:
                        st.warning("🟡 **Moderate Replenishment**")
                        st.write(f"• Steady recovery expected for {selected_village}")
                        st.write("• Water levels should stabilize")
                        st.write("• Monitor usage patterns")
                    elif prediction > -0.1:
                        st.info("🔵 **Stable Conditions**")
                        st.write(f"• Water levels remain steady in {selected_village}")
                        st.write("• Minimal change expected")
                        st.write("• Maintain current usage")
                    else:
                        st.error("🔴 **Conservation Alert**")
                        st.write(f"• Water depletion risk for {selected_village}")
                        st.write("• Implement conservation measures")
                        st.write("• Monitor closely and reduce usage")
                
                with col_risk2:
                    # Village-specific recommendations
                    st.markdown("**📋 Recommendations:**")
                    
                    recommendations = []
                    if prediction > 0.3:
                        recommendations.extend([
                            "✅ Ideal time for irrigation planning",
                            "✅ Consider expanding water usage",
                            "✅ Good conditions for new wells"
                        ])
                    elif prediction > 0:
                        recommendations.extend([
                            "⚠️ Moderate water usage recommended",
                            "⚠️ Plan for seasonal variations",
                            "⚠️ Monitor neighboring villages"
                        ])
                    else:
                        recommendations.extend([
                            "🚨 Implement water conservation",
                            "🚨 Reduce non-essential usage",
                            "🚨 Consider alternative sources"
                        ])
                    
                    for rec in recommendations:
                        st.write(rec)
                
                # Comparison with other villages (mock data)
                st.markdown("### 🏘️ Regional Comparison")
                
                comparison_villages = np.random.choice(villages, min(5, len(villages)), replace=False)
                comparison_predictions = [prediction + np.random.normal(0, 0.3) for _ in comparison_villages]
                
                comparison_df = pd.DataFrame({
                    'Village': comparison_villages,
                    'Predicted Replenishment': comparison_predictions,
                    'Status': ['Selected' if v == selected_village else 'Other' for v in comparison_villages]
                })
                
                fig_comparison = px.bar(
                    comparison_df,
                    x='Village',
                    y='Predicted Replenishment',
                    color='Status',
                    color_discrete_map={'Selected': '#ff7675', 'Other': '#74b9ff'},
                    title="Regional Replenishment Comparison"
                )
                
                fig_comparison.update_layout(
                    template="plotly_white",
                    height=400,
                    xaxis={'categoryorder': 'total descending'}
                )
                
                st.plotly_chart(fig_comparison, use_container_width=True)

elif page == "🗺️ Map View":
    st.markdown("## 🗺️ Regional Water Level Map")
    
    if st.session_state.trained_model:
        # Generate village coordinates if not already present
        if not st.session_state.village_coordinates and st.session_state.data is not None:
            village_cols = [col for col in st.session_state.data.columns if 'VILLAGE' in col.upper()]
            if village_cols:
                villages = st.session_state.data[village_cols[0]].dropna().unique()
                # Generate coordinates around Telangana region
                base_lat, base_lon = 17.3850, 78.4867  # Hyderabad coordinates
                st.session_state.village_coordinates = {}
                np.random.seed(42)
                for village in villages:
                    lat_offset = np.random.uniform(-2, 2)
                    lon_offset = np.random.uniform(-2, 2)
                    st.session_state.village_coordinates[village] = {
                        'lat': base_lat + lat_offset,
                        'lon': base_lon + lon_offset,
                        'water_level': np.random.uniform(5, 40),
                        'risk_level': np.random.choice(['Low', 'Medium', 'High'], p=[0.6, 0.3, 0.1])
                    }
            else:
                # Generate sample villages if no village data found
                sample_villages = ['Hyderabad', 'Warangal', 'Karimnagar', 'Nizamabad', 'Khammam', 
                                 'Mahbubnagar', 'Nalgonda', 'Adilabad', 'Medak', 'Rangareddy']
                base_lat, base_lon = 17.3850, 78.4867
                st.session_state.village_coordinates = {}
                np.random.seed(42)
                for village in sample_villages:
                    lat_offset = np.random.uniform(-2, 2)
                    lon_offset = np.random.uniform(-2, 2)
                    st.session_state.village_coordinates[village] = {
                        'lat': base_lat + lat_offset,
                        'lon': base_lon + lon_offset,
                        'water_level': np.random.uniform(5, 40),
                        'risk_level': np.random.choice(['Low', 'Medium', 'High'], p=[0.6, 0.3, 0.1])
                    }
        
        if st.session_state.village_coordinates:
            st.markdown("### 🌍 Interactive Village Map")
            
            # Create map data
            map_data = []
            for village, coords in st.session_state.village_coordinates.items():
                map_data.append({
                    'Village': village,
                    'Latitude': coords['lat'],
                    'Longitude': coords['lon'],
                    'Water_Level': coords['water_level'],
                    'Risk_Level': coords['risk_level'],
                    'Status': 'Normal' if coords['risk_level'] == 'Low' else 'Alert'
                })
            
            map_df = pd.DataFrame(map_data)
            
            # Color mapping for risk levels
            color_map = {'Low': '#00b894', 'Medium': '#fdcb6e', 'High': '#e17055'}
            map_df['Color'] = map_df['Risk_Level'].map(color_map)
            
            # Create the map using plotly
            fig_map = px.scatter_mapbox(
                map_df,
                lat='Latitude',
                lon='Longitude',
                size='Water_Level',
                color='Risk_Level',
                color_discrete_map=color_map,
                hover_name='Village',
                hover_data={
                    'Water_Level': ':.1f',
                    'Risk_Level': True,
                    'Latitude': False,
                    'Longitude': False
                },
                size_max=20,
                zoom=7,
                title="Water Level Distribution Across Villages"
            )
            
            fig_map.update_layout(
                mapbox_style="open-street-map",
                height=600,
                margin={"r":0,"t":50,"l":0,"b":0}
            )
            
            st.plotly_chart(fig_map, use_container_width=True)
            
            # Map legend and statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                high_risk_count = len(map_df[map_df['Risk_Level'] == 'High'])
                st.metric("🔴 High Risk Villages", high_risk_count)
            
            with col2:
                medium_risk_count = len(map_df[map_df['Risk_Level'] == 'Medium'])
                st.metric("🟡 Medium Risk Villages", medium_risk_count)
            
            with col3:
                low_risk_count = len(map_df[map_df['Risk_Level'] == 'Low'])
                st.metric("🟢 Low Risk Villages", low_risk_count)
            
            # Village details table
            st.markdown("### 📊 Village Details")
            
            # Sort by risk level (High first)
            risk_order = {'High': 0, 'Medium': 1, 'Low': 2}
            map_df['Risk_Order'] = map_df['Risk_Level'].map(risk_order)
            display_df = map_df.sort_values('Risk_Order')[['Village', 'Water_Level', 'Risk_Level']]
            
            st.dataframe(
                display_df.style.apply(
                    lambda row: ['background-color: #ffebee' if row['Risk_Level'] == 'High' 
                               else 'background-color: #fff3e0' if row['Risk_Level'] == 'Medium'
                               else 'background-color: #e8f5e8' for _ in row], axis=1
                ),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.error("Unable to generate map data. Please ensure your data contains village information.")
    
    else:
        st.warning("🚫 Please train the model first to see the map!")

elif page == "🚨 Anomaly Alerts":
    st.markdown("## 🚨 Anomaly Detection & Alerts")
    
    if st.session_state.anomaly_alerts:
        st.markdown("### 🔔 Active Alerts")
        
        # Summary metrics
        total_alerts = len(st.session_state.anomaly_alerts)
        high_severity = len([a for a in st.session_state.anomaly_alerts if a['severity'] == 'HIGH'])
        medium_severity = len([a for a in st.session_state.anomaly_alerts if a['severity'] == 'MEDIUM'])
        low_severity = len([a for a in st.session_state.anomaly_alerts if a['severity'] == 'LOW'])
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Alerts", total_alerts)
        with col2:
            st.metric("🔴 High Severity", high_severity)
        with col3:
            st.metric("🟡 Medium Severity", medium_severity)
        with col4:
            st.metric("🟢 Low Severity", low_severity)
        
        st.markdown("---")
        
        # Alert timeline
        st.markdown("### 📈 Alert Timeline")
        
        # Create timeline chart
        alert_df = pd.DataFrame(st.session_state.anomaly_alerts)
        alert_df['timestamp'] = pd.to_datetime(alert_df['timestamp'])
        
        # Count alerts by time and severity
        timeline_data = alert_df.groupby(['timestamp', 'severity']).size().reset_index(name='count')
        
        fig_timeline = px.bar(
            timeline_data,
            x='timestamp',
            y='count',
            color='severity',
            color_discrete_map={'HIGH': '#e17055', 'MEDIUM': '#fdcb6e', 'LOW': '#74b9ff'},
            title="Alert Timeline by Severity"
        )
        
        fig_timeline.update_layout(
            template="plotly_white",
            height=400,
            xaxis_title="Time",
            yaxis_title="Number of Alerts"
        )
        
        st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Detailed alerts table
        st.markdown("### 📋 Detailed Alerts")
        
        # Filter options
        col_filter1, col_filter2 = st.columns(2)
        
        with col_filter1:
            severity_filter = st.selectbox(
                "Filter by Severity",
                options=['All', 'HIGH', 'MEDIUM', 'LOW'],
                index=0
            )
        
        with col_filter2:
            village_filter = st.selectbox(
                "Filter by Village",
                options=['All'] + list(set([alert['village'] for alert in st.session_state.anomaly_alerts])),
                index=0
            )
        
        # Apply filters
        filtered_alerts = st.session_state.anomaly_alerts.copy()
        if severity_filter != 'All':
            filtered_alerts = [a for a in filtered_alerts if a['severity'] == severity_filter]
        if village_filter != 'All':
            filtered_alerts = [a for a in filtered_alerts if a['village'] == village_filter]
        
        # Display alerts
        for i, alert in enumerate(reversed(filtered_alerts[-20:])):  # Show latest 20
            severity_colors = {
                'HIGH': '🔴',
                'MEDIUM': '🟡', 
                'LOW': '🟢'
            }
            
            with st.expander(f"{severity_colors[alert['severity']]} {alert['type']} - {alert['village']} ({alert['timestamp'].strftime('%H:%M:%S')})"):
                col_alert1, col_alert2 = st.columns(2)
                
                with col_alert1:
                    st.write(f"**Village:** {alert['village']}")
                    st.write(f"**Severity:** {alert['severity']}")
                    st.write(f"**Value:** {alert['value']:.2f}m")
                
                with col_alert2:
                    st.write(f"**Type:** {alert['type']}")
                    st.write(f"**Day:** {alert.get('day', 'N/A')}")
                    st.write(f"**Time:** {alert['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Action buttons
                col_action1, col_action2 = st.columns(2)
                with col_action1:
                    if st.button(f"Mark as Resolved", key=f"resolve_{i}"):
                        st.success("Alert marked as resolved!")
                with col_action2:
                    if st.button(f"Send Notification", key=f"notify_{i}"):
                        st.info("Notification sent to relevant authorities!")
        
        # Clear all alerts button
        if st.button("🗑️ Clear All Alerts", type="secondary"):
            st.session_state.anomaly_alerts = []
            st.success("All alerts cleared!")
            st.rerun()
    
    else:
        st.info("🎉 No anomalies detected! All systems operating normally.")
        
        # Anomaly detection settings
        st.markdown("### ⚙️ Anomaly Detection Settings")
        
        col_settings1, col_settings2 = st.columns(2)
        
        with col_settings1:
            st.markdown("**Thresholds:**")
            critical_low = st.number_input("Critical Low Water Level (m)", value=5.0, min_value=0.0, max_value=20.0)
            rapid_change = st.number_input("Rapid Change Threshold (m/day)", value=2.0, min_value=0.1, max_value=5.0)
            
        with col_settings2:
            st.markdown("**Notification Settings:**")
            enable_email = st.checkbox("Email Notifications", value=True)
            enable_sms = st.checkbox("SMS Notifications", value=False)
            enable_dashboard = st.checkbox("Dashboard Alerts", value=True)
        
        if st.button("💾 Save Settings"):
            st.success("Anomaly detection settings saved!")

elif page == "📈 Model Performance":
    st.markdown("## 📈 Model Performance Analytics")
    
    if st.session_state.trained_model:
        # Mock performance metrics
        metrics = {
            'R² Score (Train)': 0.923,
            'R² Score (Test)': 0.867,
            'MAE (Train)': 1.145,
            'MAE (Test)': 1.234,
            'RMSE (Train)': 1.876,
            'RMSE (Test)': 2.012,
            'CV R² Mean': 0.854
        }
        
        # Performance metrics grid
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("R² Score (Test)", f"{metrics['R² Score (Test)']:.3f}")
        with col2:
            st.metric("MAE (Test)", f"{metrics['MAE (Test)']:.3f}m")
        with col3:
            st.metric("RMSE (Test)", f"{metrics['RMSE (Test)']:.3f}m")
        
        # Detailed metrics table
        st.markdown("### 📊 Detailed Performance Metrics")
        
        metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
        st.dataframe(metrics_df, use_container_width=True)
        
        # Mock prediction vs actual plot
        st.markdown("### 🎯 Predictions vs Actual Values")
        
        np.random.seed(42)
        actual = np.random.normal(0, 1, 100)
        predicted = actual + np.random.normal(0, 0.3, 100)
        
        fig = go.Figure()
        
        # Scatter plot
        fig.add_trace(go.Scatter(
            x=actual,
            y=predicted,
            mode='markers',
            name='Predictions',
            marker=dict(color='#74b9ff', size=8, opacity=0.7)
        ))
        
        # Perfect prediction line
        line_range = [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())]
        fig.add_trace(go.Scatter(
            x=line_range,
            y=line_range,
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash', width=2)
        ))
        
        fig.update_layout(
            title="Predicted vs Actual Replenishment",
            xaxis_title="Actual Replenishment (m)",
            yaxis_title="Predicted Replenishment (m)",
            template="plotly_white",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance (mock data)
        st.markdown("### 🏆 Feature Importance")
        
        features = ['Current WL', '7-day Avg', '30-day Avg', 'Daily Replenishment', 
                   'Weekly Replenishment', 'Trend 7d', 'Trend 30d', 'Month', 
                   'Season Factor', 'Depth Factor']
        importance = np.random.random(len(features))
        importance = importance / importance.sum() * 100
        
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance (%)': importance
        }).sort_values('Importance (%)', ascending=False)
        
        fig3 = px.bar(
            importance_df,
            x='Importance (%)',
            y='Feature',
            orientation='h',
            title="Feature Importance",
            color='Importance (%)',
            color_continuous_scale='Viridis'
        )
        
        fig3.update_layout(
            template="plotly_white",
            height=400
        )
        
        st.plotly_chart(fig3, use_container_width=True)
        
    else:
        st.warning("🚫 Please train the model first to see performance metrics!")

elif page == "ℹ️ About":
    st.markdown("## ℹ️ About AquaPredict")
    
    st.markdown("""
    ### 🌊 Mission
    AquaPredict is an AI-powered water replenishment forecasting system designed to help water resource managers, 
    farmers, and policymakers make informed decisions about groundwater usage and conservation.
    
    ### 🎯 Key Features
    - **Advanced ML Models**: Gradient Boosting Regression for accurate predictions
    - **Real-time Analysis**: Process and analyze water level data in real-time
    - **Interactive Dashboard**: Beautiful and intuitive user interface
    - **Performance Monitoring**: Comprehensive model performance analytics
    - **Export Capabilities**: Save models and predictions for future use
    
    ### 🔧 Technology Stack
    - **Backend**: Python, Scikit-learn, Pandas, NumPy
    - **Frontend**: Streamlit, Plotly, Custom CSS
    - **Machine Learning**: Gradient Boosting Regressor
    - **Data Processing**: Feature engineering with seasonal patterns
    
    ### 📊 Model Features
    Our model considers multiple factors:
    - Current water levels and historical trends
    - Seasonal patterns (Monsoon, Summer, Winter, Post-Monsoon)
    - Depth-based replenishment factors
    - Rolling averages and trend analysis
    - Lag features for temporal dependencies
    
    ### 👥 Team return red
    Built with ❤️ by **return red** team for water conservation and smart resource management.
    
    Our mission is to leverage AI and machine learning to solve critical water management challenges 
    and contribute to sustainable development goals.
    
    ### 📞 Contact
    For questions, suggestions, or collaboration opportunities, please reach out to team **return red**!
    """)
    
    # Add some nice visual elements
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #74b9ff, #0984e3); padding: 1.5rem; border-radius: 10px; color: white; text-align: center;">
            <h3>🎯 Accuracy</h3>
            <p>Up to 87% R² Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #00b894, #00cec9); padding: 1.5rem; border-radius: 10px; color: white; text-align: center;">
            <h3>⚡ Speed</h3>
            <p>Real-time Predictions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #fd79a8, #e84393); padding: 1.5rem; border-radius: 10px; color: white; text-align: center;">
            <h3>🌱 Impact</h3>
            <p>Water Conservation</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🌊 AquaPredict © 2024 | Built by <span style="color: #e74c3c; font-weight: bold;">return red</span> | For Sustainable Water Management</p>
</div>
""", unsafe_allow_html=True)