# river-pollution-in-india

 "Dataset not included due to size. Original data available on request."

 [NWMP_full.sql](https://github.com/user-attachments/files/29882842/NWMP_full.sql)

 [river_eda.py](https://github.com/user-attachments/files/29882850/river_eda.py)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 150)
sns.set_style('whitegrid')

# ------------------------------------------------------------
# 1. LOAD + MERGE  (equivalent to SQL JOIN stations + readings)
# ------------------------------------------------------------
stations = pd.read_csv('stations.csv')
readings = pd.read_csv('readings.csv')

df = readings.merge(stations, on='station_code', how='inner')

print("="*70)
print("1. SHAPE & DTYPES")
print("="*70)
print(f"stations: {stations.shape} | readings: {readings.shape} | merged df: {df.shape}")
print(df.dtypes)

# ------------------------------------------------------------
# 2. MISSING VALUES / DATA QUALITY
# ------------------------------------------------------------
print("\n" + "="*70)
print("2. MISSING VALUES")
print("="*70)
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "No missing values found.")

print("\nData_Completeness value counts:")
print(df['data_completeness'].value_counts())

# ------------------------------------------------------------
# 3. DESCRIPTIVE STATISTICS
# ------------------------------------------------------------
print("\n" + "="*70)
print("3. DESCRIPTIVE STATS — key pollution metrics")
print("="*70)
key_cols = ['water_quality_score', 'dissolved_oxygen_mgl', 'ph', 'bod_mgl',
            'fecal_coliform_mpn', 'total_coliform_mpn', 'nitrate_mgl']
print(df[key_cols].describe().round(2))

# ------------------------------------------------------------
# 4. CATEGORICAL DISTRIBUTIONS
# ------------------------------------------------------------
print("\n" + "="*70)
print("4. CATEGORICAL DISTRIBUTIONS")
print("="*70)
print("\nWater Quality Grade:")
print(df['water_quality_grade'].value_counts().sort_index())
print("\nRisk Tier:")
print(df['risk_tier'].value_counts())
print("\nPrimary Pollution Type:")
print(df['primary_pollution_type'].value_counts())

# ------------------------------------------------------------
# 5. STATE-LEVEL AGGREGATION  (equivalent to SQL GROUP BY state)
# ------------------------------------------------------------
print("\n" + "="*70)
print("5. AVERAGE WQS BY STATE (worst -> best)")
print("="*70)
state_summary = df.groupby('state_name').agg(
    avg_wqs=('water_quality_score', 'mean'),
    avg_bod=('bod_mgl', 'mean'),
    avg_do=('dissolved_oxygen_mgl', 'mean'),
    n_readings=('reading_id', 'count')
).round(2).sort_values('avg_wqs')
print(state_summary)

# ------------------------------------------------------------
# 6. YEARLY TREND  (equivalent to SQL GROUP BY year + LAG window fn)
# ------------------------------------------------------------
print("\n" + "="*70)
print("6. YEARLY TREND — Avg WQS with YoY change")
print("="*70)
yearly = df.groupby('year')['water_quality_score'].mean().round(2)
yoy = yearly.diff().round(2)
trend = pd.DataFrame({'avg_wqs': yearly, 'yoy_change': yoy})
print(trend)

# ------------------------------------------------------------
# 7. CORRELATION MATRIX
# ------------------------------------------------------------
print("\n" + "="*70)
print("7. CORRELATION MATRIX (key numeric metrics)")
print("="*70)
corr_cols = ['water_quality_score', 'dissolved_oxygen_mgl', 'ph', 'bod_mgl',
             'fecal_coliform_mpn', 'total_coliform_mpn', 'nitrate_mgl', 'conductivity_umho']
corr = df[corr_cols].corr().round(2)
print(corr)

# ------------------------------------------------------------
# 8. OUTLIER / HOTSPOT DETECTION
# ------------------------------------------------------------
print("\n" + "="*70)
print("8. TOP 10 WORST READINGS (lowest WQS)")
print("="*70)
worst = df.nsmallest(10, 'water_quality_score')[
    ['reading_id', 'location_name', 'river_name', 'state_name', 'year', 'water_quality_score', 'risk_tier']
]
print(worst.to_string(index=False))

print("\n" + "="*70)
print("9. STATIONS APPEARING MOST IN BOTTOM-10-WORST ACROSS ALL YEARS (chronic hotspots)")
print("="*70)
chronic = df.sort_values('water_quality_score').groupby('station_code').size()
worst_readings_all = df.nsmallest(50, 'water_quality_score')
hotspot_counts = worst_readings_all['location_name'].value_counts().head(10)
print(hotspot_counts)

# ------------------------------------------------------------
# 10. GROUP-WISE t-test style comparison: DO vs Pollution Status
# ------------------------------------------------------------
print("\n" + "="*70)
print("10. DISSOLVED OXYGEN BY POLLUTION STATUS")
print("="*70)
print(df.groupby('pollution_status')['dissolved_oxygen_mgl'].agg(['mean','median','std','count']).round(2))

# ------------------------------------------------------------
# 11. VISUALIZATIONS
# ------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# a. Yearly WQS trend
axes[0,0].plot(trend.index, trend['avg_wqs'], marker='o', color='steelblue')
axes[0,0].set_title('National Avg Water Quality Score by Year')
axes[0,0].set_xlabel('Year'); axes[0,0].set_ylabel('Avg WQS')

# b. State comparison (bottom 10)
worst_states = state_summary.head(10)
axes[0,1].barh(worst_states.index, worst_states['avg_wqs'], color='indianred')
axes[0,1].set_title('10 Worst States by Avg WQS')
axes[0,1].invert_yaxis()

# c. Grade distribution
grade_counts = df['water_quality_grade'].value_counts().sort_index()
axes[0,2].bar(grade_counts.index, grade_counts.values, color='seagreen')
axes[0,2].set_title('Water Quality Grade Distribution')

# d. DO vs BOD scatter colored by risk tier
for tier, color in zip(['Low Risk','Moderate Risk','High Risk'], ['green','orange','red']):
    subset = df[df['risk_tier']==tier]
    axes[1,0].scatter(subset['dissolved_oxygen_mgl'], subset['bod_mgl'], label=tier, alpha=0.5, s=15, color=color)
axes[1,0].set_xlabel('Dissolved Oxygen (mg/L)'); axes[1,0].set_ylabel('BOD (mg/L)')
axes[1,0].set_title('DO vs BOD by Risk Tier'); axes[1,0].legend()

# e. Correlation heatmap
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=axes[1,1], cbar=False)
axes[1,1].set_title('Correlation Heatmap')

# f. Risk tier distribution
risk_counts = df['risk_tier'].value_counts()
axes[1,2].pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%',
              colors=['green','orange','red'])
axes[1,2].set_title('Risk Tier Distribution')

plt.tight_layout()
plt.savefig('eda_charts/nwmp_eda_overview.png', dpi=120)
print("\nSaved chart: eda_charts/nwmp_eda_overview.png")


<img width="622" height="356" alt="Screenshot 2026-07-02 185534" src="https://github.com/user-attachments/assets/e4a313a3-b5bd-490e-a1c1-5a1062399258" />

<img width="635" height="358" alt="Screenshot 2026-07-02 185603" src="https://github.com/user-attachments/assets/31949fe5-42f7-4321-8919-1a123a5bdfd8" />

<img width="623" height="357" alt="Screenshot 2026-07-02 185624" src="https://github.com/user-attachments/assets/bd8e1f78-dc4a-456b-b123-dfdf3b1f9615" />

[River_Pollution_Capstone_Report.pdf](https://github.com/user-attachments/files/29882882/River_Pollution_Capstone_Report.pdf)
