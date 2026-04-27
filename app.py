import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="ICE Enforcement Outcomes Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional KPI cards and layout
st.markdown("""
<style>
div[data-testid="metric-container"] {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    padding: 20px;
    border-radius: 8px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    border-left: 5px solid #1f77b4;
}
</style>
""", unsafe_allow_html=True)

st.title("From Arrests to Detention: Evaluating ICE Enforcement Outcomes")
st.markdown("""
**Narrative Alignment:** This dashboard provides transparency into ICE enforcement operations, highlighting operational bottlenecks, capacity drains, and potential demographic disparities in detention times.
""")

# ==========================================
# 2. DATA LOADING & CACHING (Performance)
# ==========================================
@st.cache_data
def load_data():
    # Load ERO Arrests Data (using dummy generation fallback if missing for robustness)
    try:
        arrests_df = pd.read_csv('ERO_Admin_Arrests.csv', parse_dates=['Apprehension Date', 'Departed Date'])
    except FileNotFoundError:
        st.warning("Data file 'ERO_Admin_Arrests.csv' not found. Please ensure the file is in the directory. Halting execution.")
        st.stop()

    # Load Budget Data
    try:
        budget_df = pd.read_csv('ice_budget_2021_2023_clean.csv')
        # Extract ERO specific budget
        ero_budget = budget_df[budget_df['Category'] == 'Enforcement and Removal Operations']
        # Simplified average for KPI purposes (using FY22 PB as a baseline proxy)
        annual_ero_budget = ero_budget['FY2022_PB'].values[0] * 1000 if not ero_budget.empty else 4000000000
    except:
        annual_ero_budget = 4000000000 # Fallback 4 Billion

    # Preprocessing
    # Handle missing departed dates (Active detentions)
    max_date = arrests_df['Apprehension Date'].max()
    if pd.isna(max_date): 
        max_date = pd.to_datetime('today')
        
    arrests_df['Is_Active'] = arrests_df['Departed Date'].isna()
    
    # Calculate Days in Detention
    # If departed, Diff(Departed, Apprehension). If active, Diff(Max_Date, Apprehension)
    arrests_df['Calc_End_Date'] = arrests_df['Departed Date'].fillna(max_date)
    arrests_df['Days_in_Detention'] = (arrests_df['Calc_End_Date'] - arrests_df['Apprehension Date']).dt.days
    
    # Clean up negatives/zeros from data entry errors
    arrests_df['Days_in_Detention'] = arrests_df['Days_in_Detention'].clip(lower=0)

    # Fiscal Year Calculation
    arrests_df['FY'] = arrests_df['Apprehension Date'].dt.year + (arrests_df['Apprehension Date'].dt.month >= 10).astype(int)

    return arrests_df, annual_ero_budget

# Load the data
df_raw, annual_ero_budget = load_data()

# ==========================================
# 3. SIDEBAR FILTERS (Usability)
# ==========================================
st.sidebar.header("Global Filters")

# Date Filter
min_date = df_raw['Apprehension Date'].min().date()
max_date = df_raw['Apprehension Date'].max().date()
start_date, end_date = st.sidebar.date_input("Apprehension Date Range", [min_date, max_date])

# Categorical Filters
selected_aor = st.sidebar.multiselect("Apprehension AOR", options=df_raw['Apprehension AOR'].dropna().unique(), default=None)
selected_crim = st.sidebar.multiselect("Criminality", options=df_raw['Apprehension Criminality'].dropna().unique(), default=None)

# Apply Filters
mask = (df_raw['Apprehension Date'].dt.date >= start_date) & (df_raw['Apprehension Date'].dt.date <= end_date)
if selected_aor:
    mask &= df_raw['Apprehension AOR'].isin(selected_aor)
if selected_crim:
    mask &= df_raw['Apprehension Criminality'].isin(selected_crim)

df_filtered = df_raw[mask]

# ==========================================
# 4. TOP KPI BANNER (Functionality)
# ==========================================
st.markdown("### Key Performance Indicators")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

total_arrests = len(df_filtered)
active_detentions = df_filtered['Is_Active'].sum()
median_days = df_filtered['Days_in_Detention'].median()

# Prevent division by zero
cost_per_arrest = annual_ero_budget / total_arrests if total_arrests > 0 else 0

kpi1.metric("Total Arrests", f"{total_arrests:,}")
kpi2.metric("Active Detentions", f"{active_detentions:,}", help="Apprehensions with no Departed Date")
kpi3.metric("Median Days in Detention", f"{median_days:.1f} Days")
kpi4.metric("Est. ERO Budget per Arrest", f"${cost_per_arrest:,.0f}", help="Based on aggregate FY budget divided by filtered arrests.")

st.divider()

# ==========================================
# 5. MAIN DASHBOARD TABS
# ==========================================
tab1, tab2 = st.tabs(["Operational Efficiency & Bottlenecks", "Fairness & Demographic Bias Check"])

# ------------------------------------------
# TAB 1: OPERATIONAL EFFICIENCY (Decision Support)
# ------------------------------------------
with tab1:
    st.markdown("""
    **Insight Goal:** Identify resource drains and processing bottlenecks. High arrest volume combined with high detention times flags a severe capacity strain in specific Areas of Responsibility (AORs).
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Bottleneck Analysis by AOR")
        # Aggregate data by AOR
        aor_stats = df_filtered.groupby('Apprehension AOR').agg(
            Total_Arrests=('Apprehension Date', 'count'),
            Avg_Detention=('Days_in_Detention', 'mean')
        ).reset_index()
        
        fig_scatter = px.scatter(
            aor_stats, x='Total_Arrests', y='Avg_Detention', 
            color='Apprehension AOR', size='Total_Arrests',
            labels={'Total_Arrests': 'Volume of Arrests', 'Avg_Detention': 'Average Days in Detention'},
            title="Arrest Volume vs. Detention Duration (Bottlenecks)",
            template="plotly_white",
            color_discrete_sequence=px.colors.qualitative.Prism
        )
        fig_scatter.update_layout(margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
        # Add quadrants or lines to show decision support
        fig_scatter.add_hline(y=aor_stats['Avg_Detention'].mean(), line_dash="dot", annotation_text="Avg Detention Time", line_color="#8c8c8c")
        fig_scatter.add_vline(x=aor_stats['Total_Arrests'].mean(), line_dash="dot", annotation_text="Avg Arrest Volume", line_color="#8c8c8c")
        
        # Updated to use width='stretch' per Streamlit deprecation warning
        st.plotly_chart(fig_scatter, width='stretch')

    with col2:
        st.subheader("Resource Drain by Criminality")
        # Boxplot to show distribution of detention days by criminality
        fig_box = px.box(
            df_filtered, x='Apprehension Criminality', y='Days_in_Detention',
            color='Apprehension Criminality',
            title="Detention Duration Distribution by Offense Level",
            labels={'Days_in_Detention': 'Days in Detention', 'Apprehension Criminality': 'Criminality Level'},
            template="plotly_white",
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        fig_box.update_layout(showlegend=False, xaxis_title=None, margin=dict(l=20, r=20, t=40, b=20))
        
        # Updated to use width='stretch' per Streamlit deprecation warning
        st.plotly_chart(fig_box, width='stretch')


# ------------------------------------------
# TAB 2: FAIRNESS & BIAS CHECK (Narrative Alignment)
# ------------------------------------------
with tab2:
    st.markdown("""
    **Insight Goal:** Evaluate systemic fairness. Do individuals from certain countries spend significantly longer in detention for the *same level of criminality*? This directly addresses the narrative of accountability and transparency.
    """)
    
    # We only want to look at the top N countries to avoid visual clutter
    top_n = st.slider("Select Top N Countries by Arrest Volume", min_value=5, max_value=20, value=10)
    top_countries = df_filtered['Citizenship Country'].value_counts().nlargest(top_n).index
    df_fairness = df_filtered[df_filtered['Citizenship Country'].isin(top_countries)]
    
    # Filter out active cases for fairness check to only look at completed process times
    df_completed = df_fairness[~df_fairness['Is_Active']]
    
    if not df_completed.empty:
        # Heatmap or Bar chart showing Avg Days by Country AND Criminality
        fairness_agg = df_completed.groupby(['Citizenship Country', 'Apprehension Criminality'])['Days_in_Detention'].mean().reset_index()
        
        fig_fairness = px.bar(
            fairness_agg, 
            x='Citizenship Country', y='Days_in_Detention', 
            color='Apprehension Criminality', barmode='group',
            title="Average Detention Days by Country (Grouped by Criminality)",
            labels={'Days_in_Detention': 'Avg Completed Detention Days', 'Citizenship Country': ''},
            template="plotly_white",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_fairness.update_layout(legend_title_text='Criminality Level', margin=dict(l=20, r=20, t=40, b=20))
        
        # Updated to use width='stretch' per Streamlit deprecation warning
        st.plotly_chart(fig_fairness, width='stretch')
    else:
        st.info("Not enough completed detention data for the selected filters.")

    st.markdown("""
    **How to use this view for action:** If a specific country bar (e.g., non-criminal apprehensions) is significantly taller than others in the same criminality grouping, it indicates a processing delay specific to that demographic (e.g., difficulty acquiring travel documents, or potential systemic bias) requiring policy review.
    """)