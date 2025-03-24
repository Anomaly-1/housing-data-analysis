import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

median_sale_price = pd.read_csv('median_sale_price.csv')
construction_median_sale_price = pd.read_csv('construction_median_sale_price.csv')
zori_data = pd.read_csv('zori.csv')
inventory_data = pd.read_csv('inventory.csv')

# melt data
def melt_data(df, value_name):
    id_vars = ['RegionID', 'SizeRank', 'RegionName', 'RegionType', 'StateName']
    df = df.melt(id_vars=id_vars, var_name='Date', value_name=value_name)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

median_sale_price = melt_data(median_sale_price, 'MedianSalePrice')
construction_median_sale_price = melt_data(construction_median_sale_price, 'MedianSalePricePerSqft')
zori_melted = melt_data(zori_data, 'ZORI')
inventory_melted = melt_data(inventory_data, 'Inventory')

# Holes & zeroes
if (construction_median_sale_price['MedianSalePricePerSqft'] == 0).any():
    print("Warning: 'MedianSalePricePerSqft' in construction_median_sale_price contains zero values. Replacing zeros with NaN.")
    construction_median_sale_price['MedianSalePricePerSqft'].replace(0, np.nan, inplace=True)

if zori_melted['ZORI'].isnull().any():
    print("Warning: 'ZORI' contains missing values.")

merged_data = pd.merge(
    median_sale_price, 
    construction_median_sale_price, 
    on=['RegionID', 'Date', 'RegionName'],
    suffixes=('_existing', '_new')
)

# merge p1
merged_data = pd.merge(
    merged_data,
    zori_melted,
    on=['RegionID', 'Date', 'RegionName'],
    how='left'
)

# merge
merged_data = pd.merge(
    merged_data,
    inventory_melted,
    on=['RegionID', 'Date', 'RegionName'],
    how='left'
)

# Holes & zeroes
if (merged_data['MedianSalePricePerSqft'] == 0).any():
    print("Warning: 'MedianSalePricePerSqft' contains zero values. Replacing zeros with NaN.")
    merged_data['MedianSalePricePerSqft'].replace(0, np.nan, inplace=True)

# avg median sale price over time
avg_prices_existing = median_sale_price.groupby('Date')['MedianSalePrice'].mean()
avg_prices_new = construction_median_sale_price.groupby('Date')['MedianSalePricePerSqft'].mean()
avg_zori = zori_melted.groupby('Date')['ZORI'].mean()
avg_inventory = inventory_melted.groupby('Date')['Inventory'].mean()  # avg inventory over time

# price increase percentage over time
price_increase_existing = avg_prices_existing.pct_change(fill_method=None).dropna() * 100
price_increase_new = avg_prices_new.pct_change(fill_method=None).dropna() * 100
zori_increase = avg_zori.pct_change(fill_method=None).dropna() * 100

# Create visualizations
with PdfPages('housing_affordability_analysis.pdf') as pdf:
    # Plot 1: Average Median Sale Price Over Time (Existing Homes)
    plt.figure(figsize=(12, 6))
    plt.plot(avg_prices_existing.index, avg_prices_existing, label='Existing Homes', color='blue')
    plt.title('Average Median Sale Price Over Time (Existing Homes)', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Median Sale Price ($)', fontsize=12)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # Plot 2: Average Median Sale Price Over Time (New Construction Homes)
    plt.figure(figsize=(12, 6))
    plt.plot(avg_prices_new.index, avg_prices_new, label='New Construction Median Sale Price Per Sqft', color='green')
    plt.title('Average Median Sale Price Over Time (New Construction Homes)', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Median Sale Price Per Sqft ($)', fontsize=12)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # Plot 3: National ZORI Trend Over Time
    plt.figure(figsize=(12, 6))
    plt.plot(avg_zori.index, avg_zori, label='ZORI', color='orange')
    plt.title('National ZORI Trend Over Time', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('ZORI', fontsize=12)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # Plot 4: Correlation Between ZORI and Median Sale Price
    correlation_data = merged_data[['MedianSalePrice', 'MedianSalePricePerSqft', 'ZORI']].dropna()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_data.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Between ZORI and Median Sale Price', fontsize=14)
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # Plot 5: Regional Price Changes (Top 10 Regions with Highest Median Sale Prices)
    top_regions_by_price = merged_data.groupby('RegionName')['MedianSalePrice'].max().nlargest(5)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_regions_by_price.index, y=top_regions_by_price.values, hue=top_regions_by_price.index, palette='viridis', legend=False)
    plt.title('Top 5 Regions with Highest Median Sale Prices', fontsize=14)
    plt.xlabel('Region', fontsize=12)
    plt.ylabel('Median Sale Price ($)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # Plot 6: Regional ZORI Trends for Top 5 Regions (Based on Highest ZORI Values)
    top_regions_by_zori = merged_data.groupby('RegionName')['ZORI'].max().nlargest(5)
    top_5_regions_zori = top_regions_by_zori.index.tolist()[:5]
    top_5_data_zori = merged_data[merged_data['RegionName'].isin(top_5_regions_zori)]
    plt.figure(figsize=(12, 6))
    for region in top_5_regions_zori:
        region_data = top_5_data_zori[top_5_data_zori['RegionName'] == region]
        plt.plot(region_data['Date'], region_data['ZORI'], label=region)
    plt.title('Top 5 Regional ZORI Trends (Based on Highest ZORI Values)', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('ZORI', fontsize=12)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # Plot 7: Pie Chart of Inventory Distribution Across Regions (excluding "THE US")
    inventory_by_region = merged_data[merged_data['RegionName'] != "United States"].groupby('RegionName')['Inventory'].sum().nlargest(10)
    plt.figure(figsize=(8, 8))
    plt.pie(inventory_by_region, labels=inventory_by_region.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('viridis'))
    plt.title('Inventory Distribution Across Top 10 Regions', fontsize=14)
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # Plot 8: Box Plot of Median Sale Prices by Top 30 Regions
    top_30_regions = merged_data.groupby('RegionName')['MedianSalePrice'].max().nlargest(30).index
    filtered_data = merged_data[merged_data['RegionName'].isin(top_30_regions)]
    plt.figure(figsize=(12, 6))
    # virdis color palette just cus it looks nice
    sns.boxplot(data=filtered_data, x='RegionName', y='MedianSalePrice', palette='viridis')
    plt.title('Distribution of Median Sale Prices by Top 30 Regions', fontsize=14)
    plt.xlabel('Region', fontsize=12)
    plt.ylabel('Median Sale Price ($)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # Plot 9: Heatmap of Median Sale Prices Over Time by Region (shortened date format)
    heatmap_data = merged_data.pivot_table(index='RegionName', columns='Date', values='MedianSalePrice', aggfunc='mean')
    heatmap_data.columns = pd.to_datetime(heatmap_data.columns).strftime('%Y-%m-%d')
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, cmap='viridis', cbar_kws={'label': 'Median Sale Price ($)'})
    plt.title('Median Sale Prices Over Time by Region', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Region', fontsize=12)
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # Plot 10: Trend of Inventory Over Time
    plt.figure(figsize=(12, 6))
    plt.plot(avg_inventory.index, avg_inventory, label='Inventory', color='purple')
    plt.title('Trend of Inventory Over Time', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Inventory', fontsize=12)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # Plot 11: Time Series of Median Sale Price and Inventory Over Time
    plt.figure(figsize=(12, 6))
    ax1 = plt.gca()
    ax1.plot(avg_prices_existing.index, avg_prices_existing, label='Median Sale Price', color='blue')
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Median Sale Price ($)', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='blue')
    ax2 = ax1.twinx()
    ax2.plot(avg_inventory.index, avg_inventory, label='Inventory', color='purple')
    ax2.set_ylabel('Inventory', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='purple')
    plt.title('Time Series of Median Sale Price and Inventory Over Time', fontsize=14)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax1.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # Plot 12: Inventory vs. Price Increase
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=merged_data, x='Inventory', y='MedianSalePrice', hue='Date', palette='viridis', alpha=0.6)
    plt.title('Inventory vs. Median Sale Price', fontsize=14)
    plt.xlabel('Inventory', fontsize=12)
    plt.legend([],[],frameon=False)
    plt.ylabel('Median Sale Price ($)', fontsize=12)
    plt.grid()
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # Summary Statistics Page
    summary_stats = pd.DataFrame({
        'Dataset': ['Median Sale Price', 'Construction Median Sale Price', 'ZORI', 'Inventory'],
        'Mean': [median_sale_price['MedianSalePrice'].mean(), construction_median_sale_price['MedianSalePricePerSqft'].mean(), zori_melted['ZORI'].mean(), inventory_melted['Inventory'].mean()],
        'Median': [median_sale_price['MedianSalePrice'].median(), construction_median_sale_price['MedianSalePricePerSqft'].median(), zori_melted['ZORI'].median(), inventory_melted['Inventory'].median()],
        'Range': [median_sale_price['MedianSalePrice'].max() - median_sale_price['MedianSalePrice'].min(), construction_median_sale_price['MedianSalePricePerSqft'].max() - construction_median_sale_price['MedianSalePricePerSqft'].min(), zori_melted['ZORI'].max() - zori_melted['ZORI'].min(), inventory_melted['Inventory'].max() - inventory_melted['Inventory'].min()],
        'Std Dev': [median_sale_price['MedianSalePrice'].std(), construction_median_sale_price['MedianSalePricePerSqft'].std(), zori_melted['ZORI'].std(), inventory_melted['Inventory'].std()]
    })
    plt.figure(figsize=(10, 6))
    plt.table(cellText=summary_stats.values, colLabels=summary_stats.columns, loc='center')
    plt.axis('off')
    plt.title('Summary Statistics of Datasets', fontsize=14)
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # Observations
    observations = """
    Observations:
    1. The average median sale price for existing homes has been increasing over time.
    2. The New Construction Median Sale Price Per Sqft shows a steady increase, indicating rising costs for newly constructed homes.
    3. The ZORI (Zillow Observed Rent Index) shows a steady increase, indicating rising rental prices.
    4. There is a strong correlation between ZORI and median sale prices, suggesting that rising rents may be driving housing affordability challenges.
    5. Certain regions have experienced significantly higher price increases compared to others, making them less affordable.
    6. The scatter plot reveals a relationship between existing home prices and inventory levels.
    7. The trend of inventory over time shows how the availability of homes has changed.
    8. The time series plot highlights the relationship between median sale prices and inventory over time.
    """
    plt.figure(figsize=(10, 6))
    plt.text(0.1, 0.5, observations, fontsize=12, wrap=True)
    plt.axis('off')
    plt.tight_layout()
    pdf.savefig()
    plt.close()