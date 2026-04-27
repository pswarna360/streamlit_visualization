import os
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Force headless mode for macOS
import matplotlib.pyplot as plt
import seaborn as sns

def assign_fiscal_year(date_series):
    return date_series.dt.year + (date_series.dt.month >= 10).astype(int)

def run_pipeline():
    # --- 1. SETUP WORKSPACE ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("outputs", f"comprehensive_analysis_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    txt_path = os.path.join(output_dir, "analytical_report.txt")

    def log_output(text):
        print(text)
        with open(txt_path, "a") as f:
            f.write(text + "\n")

    log_output(f"Initializing Redesigned Analysis. Outputs routing to: {output_dir}\n")
    log_output("1. Loading and Parsing High-Granularity Datasets...")

    # Load data with more categorical columns
    budget_df = pd.read_csv('ice_budget_2021_2023_clean.csv')
    
    arrests_cols = ['Apprehension Date', 'Departed Date', 'Apprehension AOR', 'Citizenship Country', 'Apprehension Criminality', 'Gender']
    arrests_df = pd.read_csv('ERO_Admin_Arrests.csv', usecols=lambda c: c in arrests_cols)
    
    detentions_cols = ['Book In Date Time', 'Detention Facility', 'Citizenship Country', 'Case Threat Level', 'Gender']
    detentions_df = pd.read_csv('ICE Detentions.csv', usecols=lambda c: c in detentions_cols)

    # Convert dates robustly (coercing bad string formats to NaT)
    arrests_df['Apprehension Date'] = pd.to_datetime(arrests_df['Apprehension Date'], errors='coerce')
    arrests_df['Departed Date'] = pd.to_datetime(arrests_df['Departed Date'], errors='coerce')
    detentions_df['Book In Date Time'] = pd.to_datetime(detentions_df['Book In Date Time'], errors='coerce')

    log_output("2. Cleaning and Time-Bounding Data (Oct 2019 - Present)...")
    # Drop rows without primary dates and restrict to FY20+ to remove historical garbage
    arrests_df.dropna(subset=['Apprehension Date'], inplace=True)
    detentions_df.dropna(subset=['Book In Date Time'], inplace=True)
    
    arrests_df = arrests_df[arrests_df['Apprehension Date'] >= '2019-10-01'].copy()
    detentions_df = detentions_df[detentions_df['Book In Date Time'] >= '2019-10-01'].copy()

    # Create plotting timelines
    arrests_df['YearMonth'] = arrests_df['Apprehension Date'].dt.to_period('M').dt.to_timestamp()
    arrests_df['Departed_YearMonth'] = arrests_df['Departed Date'].dt.to_period('M').dt.to_timestamp()
    detentions_df['YearMonth'] = detentions_df['Book In Date Time'].dt.to_period('M').dt.to_timestamp()

    # --- OBJECTIVE 1: PROCESS FLOW TIMELINES ---
    log_output("3. Generating Process Flow Timelines (Arrests -> Detentions -> Deportations)...")
    monthly_arrests = arrests_df.groupby('YearMonth').size().rename('Arrests')
    monthly_detentions = detentions_df.groupby('YearMonth').size().rename('Detentions')
    # Count deportations based on when they actually departed
    monthly_departures = arrests_df.dropna(subset=['Departed Date']).groupby('Departed_YearMonth').size().rename('Departures')

    timeline_df = pd.concat([monthly_arrests, monthly_detentions, monthly_departures], axis=1).fillna(0)
    timeline_df = timeline_df[timeline_df.index >= '2019-10-01'] # Re-enforce bound after concat
    
    # Smooth data with 3-Month Moving Average
    timeline_df_smooth = timeline_df.rolling(window=3, min_periods=1).mean()

    plt.figure(figsize=(14, 6))
    plt.plot(timeline_df_smooth.index, timeline_df_smooth['Arrests'], label='Arrests (3MA)', color='tab:red', linewidth=2.5)
    plt.plot(timeline_df_smooth.index, timeline_df_smooth['Detentions'], label='Detentions (3MA)', color='tab:orange', linewidth=2.5)
    plt.plot(timeline_df_smooth.index, timeline_df_smooth['Departures'], label='Deportations/Departures (3MA)', color='tab:green', linewidth=2.5)
    
    plt.title('ICE Enforcement Process Pipeline (Oct 2019 - Present)')
    plt.ylabel('Volume of Individuals')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_Process_Pipeline_Trends.png"), dpi=300)
    plt.close()

    # --- OBJECTIVE 2: GEOSPATIAL / LOCATION TRENDS ---
    log_output("4. Analyzing Geospatial Concentration...")
    top_aors = arrests_df['Apprehension AOR'].value_counts().head(10)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(y=top_aors.index, x=top_aors.values, palette='viridis')
    plt.title('Top 10 Areas of Responsibility (AOR) for ICE Arrests')
    plt.xlabel('Total Arrests')
    plt.ylabel('AOR')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_Geospatial_AOR_Trends.png"), dpi=300)
    plt.close()

    # --- OBJECTIVE 3: DISPARITIES & CASE DETAILS ---
    log_output("5. Analyzing Disparities and Case Demographics...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Subplot 1: Top 10 Citizenship Countries (Disparities)
    top_countries = arrests_df['Citizenship Country'].value_counts().head(10)
    sns.barplot(x=top_countries.values, y=top_countries.index, ax=ax1, palette='mako')
    ax1.set_title('Top 10 Citizenship Countries (Arrest Volume)')
    ax1.set_xlabel('Total Arrests')

    # Subplot 2: Threat Levels / Criminality (Case Details)
    # Cleaning the criminality text if needed, taking top 5
    top_criminality = arrests_df['Apprehension Criminality'].value_counts().head(5)
    sns.barplot(x=top_criminality.values, y=top_criminality.index, ax=ax2, palette='rocket')
    ax2.set_title('Primary Criminality / Threat Levels')
    ax2.set_xlabel('Total Arrests')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "03_Demographics_and_Threat_Levels.png"), dpi=300)
    plt.close()

    # --- OBJECTIVE 4: FUNDING VS OUTCOMES ---
    log_output("6. Assessing Funding Impact on Outcomes...")
    
    # Re-calculate cleanly by Fiscal Year
    arrests_df['FY'] = assign_fiscal_year(arrests_df['Apprehension Date'])
    detentions_df['FY'] = assign_fiscal_year(detentions_df['Book In Date Time'])
    
    fy_outcomes = pd.DataFrame({
        'Total_Arrests': arrests_df.groupby('FY').size(),
        'Total_Detentions': detentions_df.groupby('FY').size()
    }).reset_index()

    budget_melted = budget_df.melt(id_vars='Category', var_name='FY_String', value_name='Budget_Thousands')
    budget_filtered = budget_melted[budget_melted['FY_String'].isin(['FY2021', 'FY2022_PB', 'FY2023_PB'])].copy()
    budget_filtered['FY'] = budget_filtered['FY_String'].str.extract(r'(\d{4})').astype(int)
    budget_pivot = budget_filtered.pivot(index='FY', columns='Category', values='Budget_Thousands').reset_index()

    financial_df = pd.merge(budget_pivot, fy_outcomes, on='FY', how='inner')

    # Create a dual-axis bar/line chart for direct FY comparison
    fig, ax1 = plt.subplots(figsize=(10, 6))
    x = np.arange(len(financial_df['FY']))
    width = 0.35

    ax1.bar(x - width/2, financial_df['Total_Arrests'], width, label='Annual Arrests', color='tab:red', alpha=0.7)
    ax1.bar(x + width/2, financial_df['Total_Detentions'], width, label='Annual Detentions', color='tab:orange', alpha=0.7)
    ax1.set_xlabel('Fiscal Year')
    ax1.set_ylabel('Total Individuals', color='black')
    ax1.set_xticks(x)
    ax1.set_xticklabels(financial_df['FY'])

    ax2 = ax1.twinx()
    ax2.plot(x, financial_df['Enforcement and Removal Operations'], color='tab:blue', marker='o', linewidth=3, label='ERO Budget ($ Thousands)')
    ax2.set_ylabel('ERO Budget (Thousands $)', color='tab:blue')
    
    # Combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

    plt.title('Annual ERO Budget vs. Total Enforcement Operations')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "04_Financial_Impact.png"), dpi=300)
    plt.close()

    # --- WRITING STATISTICAL REPORT ---
    log_output("\n--- STATISTICAL & INSIGHT SUMMARY ---")
    log_output("\nTop 5 AORs for Arrests:")
    log_output(top_aors.head(5).to_string())
    
    log_output("\nTop 5 Citizenship Countries Targeted:")
    log_output(top_countries.head(5).to_string())
    
    try:
        # Calculate process funnel efficiency (Arrests to Departures)
        total_arrests = timeline_df['Arrests'].sum()
        total_departures = timeline_df['Departures'].sum()
        conversion_rate = (total_departures / total_arrests) * 100
        log_output(f"\nOverall Deportation Rate (Departures per Arrest): {conversion_rate:.2f}%")
    except Exception as e:
        pass

    log_output(f"\nPipeline successfully generated 4 analytical images in: {output_dir}")

if __name__ == "__main__":
    run_pipeline()