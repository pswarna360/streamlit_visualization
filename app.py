import streamlit as st
import pandas as pd
import plotly.express as px

# 1. Load and clean the data
@st.cache_data
def load_data():
    df = pd.read_csv('monthly_deaths.csv')
    
    # Convert 'date' column to datetime objects
    df['date'] = pd.to_datetime(df['date'])
    
    # Clean the 'Likelihood of Death per Birth (%)' column (remove '%' and convert to float)
    df['Likelihood of Death per Birth (%)'] = df['Likelihood of Death per Birth (%)'].astype(str).str.rstrip('%').astype(float)
    
    return df

df = load_data()

# 2. App Header & Description
st.title("Doctors: A 30s Handwash Cuts Patient Mortality by 9%")
st.write("Explore the data below. The time-series clearly demonstrates the sustained and drastic decline in patient mortality following the implementation of handwashing.")

# 3. Interactive Sidebar / Controls
st.sidebar.header("Data Filters & Controls")

min_date = df['date'].min().to_pydatetime().date()
max_date = df['date'].max().to_pydatetime().date()

# Callback function to reset the date range in session state
def reset_date_range():
    st.session_state.date_filter_key = (min_date, max_date)

# Initialize session state for the date filter if it doesn't exist
if 'date_filter_key' not in st.session_state:
    st.session_state.date_filter_key = (min_date, max_date)

# Reset Button
st.sidebar.button("Reset Date Range", on_click=reset_date_range)

# Date Range Filter (now tied to session state)
# Removed the 'value=' parameter to resolve the Session State API conflict warning
date_range = st.sidebar.date_input(
    "Filter by Date Range", 
    min_value=min_date, 
    max_value=max_date,
    key='date_filter_key'
)

# Hardcoded Intervention Date
intervention_datetime = pd.to_datetime('1847-06-01')

# Apply date range filter to a new dataframe for plotting
if len(date_range) == 2:
    filtered_df = df[(df['date'].dt.date >= date_range[0]) & (df['date'].dt.date <= date_range[1])]
else:
    filtered_df = df.copy()

# 4. Calculate Dynamic Metrics based on the filtered view
before_intervention = filtered_df[filtered_df['date'] < intervention_datetime]
after_intervention = filtered_df[filtered_df['date'] >= intervention_datetime]

avg_before = before_intervention['Likelihood of Death per Birth (%)'].mean() if not before_intervention.empty else 0
avg_after = after_intervention['Likelihood of Death per Birth (%)'].mean() if not after_intervention.empty else 0
absolute_drop = avg_before - avg_after

# 5. Display Metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.write("Avg Death Rate (Before)")
    st.markdown(f"<h2 style='color: #ff4b4b; margin-top: -10px;'>{avg_before:.1f}%</h2>", unsafe_allow_html=True)

with col2:
    st.write("Avg Death Rate (After)")
    st.markdown(f"<h2 style='color: #09ab3b; margin-top: -10px;'>{avg_after:.1f}%</h2>", unsafe_allow_html=True)

with col3:
    st.write("Mortality Drop")
    # Using HTML with the down arrow to perfectly mimic Streamlit's native delta, 
    # without a background color, while keeping it aligned and sized as an h2.
    st.markdown(f"<h2 style='color: #09ab3b; margin-top: -10px;'>↓ {absolute_drop:.1f}%</h2>", unsafe_allow_html=True)

# 6. Plotly Interactive Chart (Original Visual)
fig = px.line(
    filtered_df, 
    x='date', 
    y='Likelihood of Death per Birth (%)', 
    title='Likelihood of Death per Birth (%) From 1841 to 1849'
)

# Update line styling to match the blue from the slide
fig.update_traces(line=dict(color='#1E90FF', width=3))

# Add a vertical dashed line for the hardcoded intervention date
fig.add_vline(
    x=intervention_datetime.timestamp() * 1000, 
    line_dash="dash", 
    line_color="red", 
    annotation_text="Handwashing Required", 
    annotation_position="top right"
)

# Clean up layout
fig.update_layout(
    xaxis_title="Year", 
    yaxis_title="Likelihood of Death per Birth (%)",
    hovermode="x unified" 
)

# Render the chart (updated syntax to resolve use_container_width deprecation warning)
st.plotly_chart(fig, width='stretch')

# 7. Findings Explanation
st.markdown("---")
st.markdown("### Findings")
st.write("The data reveals a stark contrast in patient outcomes before and after the mid-1847 handwashing mandate. While mortality rates were highly volatile before the mandate, they plummeted and stabilized almost immediately following the intervention. The sustained and drastic decline in likelihood of death following implementation of handwashing shown by the time-series is exactly what Dr. Semmelweis could have used to better convince his audience.")