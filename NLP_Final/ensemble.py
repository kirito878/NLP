import pandas as pd

# Load the five CSV files
df1 = pd.read_csv('data/submission-6.csv')
df2 = pd.read_csv('data/submission_all6.csv')
df3 = pd.read_csv('data/submission_085.csv')
df4 = pd.read_csv('data/submission-3.csv')  # replace with your actual filename
df5 = pd.read_csv('data/submission_all-3.csv')  # replace with your actual filename

# Set 'index' as the index for correct row-wise operations
df1.set_index('index', inplace=True)
df2.set_index('index', inplace=True)
df3.set_index('index', inplace=True)
df4.set_index('index', inplace=True)
df5.set_index('index', inplace=True)

# Sum the response_quality columns
sum_df = df1['response_quality'] + df2['response_quality'] + df3['response_quality'] + df4['response_quality'] + df5['response_quality']

# Create a new DataFrame for the result
result_df = pd.DataFrame(sum_df, columns=['response_quality'])

# If the sum is greater than or equal to 3, set to 1; otherwise, set to 0
result_df['response_quality'] = result_df['response_quality'].apply(lambda x: 1 if x >= 3 else 0)

# Reset the index to turn it back into a column
result_df.reset_index(inplace=True)

# Save the result to a new CSV file
result_df.to_csv('data/submission_result.csv', index=False)
