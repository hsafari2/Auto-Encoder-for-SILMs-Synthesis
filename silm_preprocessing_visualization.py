import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

# Load data
ProcessedDataAddress = r'/work/bavarian/hsafari2/CO2Absorption/Processed_Sorption_Data.xlsx'
OriginalDataAddress = r'/work/bavarian/hsafari2/CO2Absorption/Raw Sorption Data for ML modeling.xlsx'
ProcessedData = pd.read_excel(ProcessedDataAddress)
OriginalData = pd.read_excel(OriginalDataAddress)

# Data processing steps
OriginalData = OriginalData.drop(columns=['Ionic Liquid'])
OriginalData['Mol. Wt.'] = OriginalData['Mol. Wt.'].astype(str).str.strip().replace('kynar', '2000000')
OriginalData['Mol. Wt.'] = pd.to_numeric(OriginalData['Mol. Wt.'], errors='coerce')
ProcessedData['Mol. Wt.'] = pd.to_numeric(ProcessedData['Mol. Wt.'], errors='coerce')

for df in [ProcessedData, OriginalData]:
    if 'moles of CO2/kg sorbent' in df.columns:
        df.drop(columns=['moles of CO2/kg sorbent'], inplace=True)
    df['moles of CO2/kg sorbent'] = df['moles of CO2/kg IL'] / 2
    df.drop(columns=['moles of CO2/kg IL'], inplace=True)

# Set style for better aesthetics
plt.style.use('default')
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 12
#plt.rcParams['axes.titleweight'] = 'bold'
#plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'
plt.rcParams['text.color'] = 'black'

# Create a figure with subplots - 2 rows and 3 columns
fig, axes = plt.subplots(2, 3, figsize=(8.27 * 1.2, 5.8 * 1.2))  # A4 width, about 1/3 of A4 height

def format_col_name(col):
    return col.replace('moles of CO2/kg sorbent', 'moles of CO$_2$/kg sorbent')

datasets = [OriginalData, ProcessedData]
labels = ['Raw Data', 'Processed Data']
col = 'moles of CO2/kg sorbent'

# ADJUSTABLE: Colors for raw and processed data
raw_color = 'red'
processed_color = '#1f77b4'

# ADJUSTABLE: Position for skewness and kurtosis information
sk_x, sk_y = 0.95, 0.95
sk_ha, sk_va = 'right', 'top'

for i, (data, label) in enumerate(zip(datasets, labels)):
    formatted_col = format_col_name(col)
    color = raw_color if i == 0 else processed_color
    
    # Add row title
    fig.text(-0.02, 0.75 - i * 0.50, label, ha='left', va='center', fontsize=14, rotation=90, color='black')
    
    # Distribution plot (histogram with KDE)
    sns.histplot(data[col], kde=True, ax=axes[i, 0], bins=30, color=color, edgecolor='black', alpha=0.7)
    if i == 0:  # Only set title for the first row
        axes[i, 0].set_title(f'Distribution of {formatted_col}')
    axes[i, 0].set_xlabel(formatted_col)
    axes[i, 0].set_ylabel('Density')
    
    # Add skewness and kurtosis information inside the plot
    skewness = data[col].skew()
    kurtosis = data[col].kurtosis()
    axes[i, 0].text(sk_x, sk_y, f'Skewness: {skewness:.2f}\nKurtosis: {kurtosis:.2f}', 
                    transform=axes[i, 0].transAxes, 
                    verticalalignment=sk_va, 
                    horizontalalignment=sk_ha,
                    fontsize=10, 
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    # Q-Q plot
    stats.probplot(data[col], dist="norm", plot=axes[i, 1])
    axes[i, 1].get_lines()[0].set_markerfacecolor(color)
    axes[i, 1].get_lines()[0].set_markeredgecolor('black')
    axes[i, 1].get_lines()[0].set_alpha(0.7)
    axes[i, 1].get_lines()[1].set_color('red')
    if i == 0:  # Only set title for the first row
        axes[i, 1].set_title(f'Q-Q Plot of {formatted_col}')
    elif i == 1:
        axes[i, 1].set_title('')  # Remove the title for the second row
    axes[i, 1].set_xlabel('Theoretical Quantiles')
    axes[i, 1].set_ylabel('Sample Quantiles')
    
    # Box plot
    sns.boxplot(x=data[col], ax=axes[i, 2], color=color, width=0.5)
    sns.stripplot(x=data[col], ax=axes[i, 2], color='black', size=2, alpha=0.3)
    if i == 0:  # Only set title for the first row
        axes[i, 2].set_title(f'Box Plot of {formatted_col}')
    axes[i, 2].set_xlabel(formatted_col)
    axes[i, 2].set_ylabel('moles of CO$_2$/kg sorbent')  # Add y-axis label

# Add A, B, C, D, E, F labels inside the figures
for n, ax in enumerate(axes.flatten()):
    ax.text(0.03, 0.97, chr(65 + n), transform=ax.transAxes, 
            size=14, va='top', ha='left', color='black')

# ADJUSTABLE: Spacing between subplots
row_spacing = 0.3  # Increase this value to add more space between rows
column_spacing = 0.4  # Increase this value to add more space between columns

plt.tight_layout()
plt.subplots_adjust(left=0.07, right=0.98, top=0.94, bottom=0.08, 
                    hspace=row_spacing, wspace=column_spacing)

# Add a main title
fig.suptitle('\n\n\n', fontsize=12, fontweight='bold', y=0.98)

plt.savefig('CO2_Absorption_Analysis.png', dpi=600, bbox_inches='tight')
#plt.savefig('CO2_Absorption_Analysis.pdf', bbox_inches='tight')
plt.show()
