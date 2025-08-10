import pandas as pd
import matplotlib.pyplot as plt

# Load the dataframe from the CSV file.
try:
    df = pd.read_csv('thermal_sim_results_grid_search.csv')

    # Omit the specified columns from the analysis.
    columns_to_omit = ['Successful Flights', 'Avg Failed Distance (m)', 'Avg Total Time (s)']
    df_amended_4 = df.drop(columns=columns_to_omit)

    # Calculate the correlation matrix for the amended data.
    correlation_matrix = df_amended_4.corr()

    # Get the correlation of 'Probability' with other columns.
    probability_correlations = correlation_matrix['Probability'].drop('Probability')
    # Get the correlation of 'Avg Speed Made Good (km/h)' with other columns.
    speed_correlations = correlation_matrix['Avg Speed Made Good (km/h)'].drop('Avg Speed Made Good (km/h)')

    # Sort and select the top 8 correlations for each.
    top_8_prob_correlations = probability_correlations.abs().sort_values(ascending=False).head(8)
    top_8_speed_correlations = speed_correlations.abs().sort_values(ascending=False).head(8)

    # Convert to DataFrames and sort for plotting.
    df_prob_corr = probability_correlations[top_8_prob_correlations.index].to_frame().sort_values(by='Probability', ascending=True)
    df_speed_corr = speed_correlations[top_8_speed_correlations.index].to_frame().sort_values(by='Avg Speed Made Good (km/h)', ascending=True)

    # Create and configure the plot.
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Plot for Probability correlations with the new title.
    df_prob_corr.plot(kind='barh', ax=axes[0], legend=False)
    axes[0].set_title('Probability Flight Completed', fontsize=14)
    axes[0].set_xlabel('Correlation Coefficient', fontsize=12)
    axes[0].set_ylabel('Variable', fontsize=12)
    axes[0].tick_params(axis='x', rotation=45, labelsize=10)
    axes[0].tick_params(axis='y', labelsize=10)
    axes[0].grid(axis='x', linestyle='--', alpha=0.7)

    # Plot for Average Speed Made Good correlations with the new title.
    df_speed_corr.plot(kind='barh', ax=axes[1], color='orange', legend=False)
    axes[1].set_title('Average Speed Made Good', fontsize=14)
    axes[1].set_xlabel('Correlation Coefficient', fontsize=12)
    axes[1].set_ylabel('Variable', fontsize=12)
    axes[1].tick_params(axis='x', rotation=45, labelsize=10)
    axes[1].tick_params(axis='y', labelsize=10)
    axes[1].grid(axis='x', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('dependency_analysis.png')
    print("Plot saved as dependency_analysis.png")

except FileNotFoundError:
    print("Error: The file 'nested_loop_simulation_results.csv' was not found.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")