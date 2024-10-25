import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Load the dataset (assuming it's in CSV format)
df = pd.read_csv('data.csv')

# Open the text file to save the analysis output
with open("analysis_output.txt", "w") as file:
    # Preview the dataset (first 5 rows)
    file.write("Dataset Preview:\n")
    file.write(str(df.head()) + "\n\n")

    # Convert Likert scale responses to numerical values for easier analysis (if necessary)
    likert_scale = {
        'Strongly Disagree': 1,
        'Disagree': 2,
        'Neutral': 3,
        'Agree': 4,
        'Strongly Agree': 5
    }

    # Apply this mapping to the relevant columns
    df['Job Satisfaction'] = df['Job Satisfaction'].map(likert_scale)
    df['Overall Engagement'] = df['Overall Engagement'].map(likert_scale)
    df['Work-Life Balance'] = df['Work-Life Balance'].map(likert_scale)

    # 1. Descriptive Statistics for Overall Engagement and Job Satisfaction
    engagement_stats = df['Overall Engagement'].describe()
    job_satisfaction_stats = df['Job Satisfaction'].describe()

    # Write Descriptive statistics to file
    file.write("Overall Engagement Stats:\n")
    file.write(str(engagement_stats) + "\n\n")

    file.write("Job Satisfaction Stats:\n")
    file.write(str(job_satisfaction_stats) + "\n\n")

    # Visualize the distributions (still displayed)
    sns.histplot(df['Overall Engagement'], kde=True, bins=5)
    plt.title('Distribution of Overall Engagement')
    plt.show()

    sns.histplot(df['Job Satisfaction'], kde=True, bins=5)
    plt.title('Distribution of Job Satisfaction')
    plt.show()

    # 2. Trends Analysis by Demographics
    dept_satisfaction = df.groupby('Department')['Job Satisfaction'].mean()
    age_satisfaction = df.groupby('Age Bracket')['Job Satisfaction'].mean()

    # Write Trends Analysis to file
    file.write("Average Job Satisfaction by Department:\n")
    file.write(str(dept_satisfaction) + "\n\n")

    file.write("Average Job Satisfaction by Age Bracket:\n")
    file.write(str(age_satisfaction) + "\n\n")

    # Visualize the trends (still displayed)
    dept_satisfaction.plot(kind='bar', title='Average Job Satisfaction by Department')
    plt.ylabel('Job Satisfaction')
    plt.show()

    age_satisfaction.plot(kind='bar', title='Average Job Satisfaction by Age Bracket')
    plt.ylabel('Job Satisfaction')
    plt.show()

    # 3. Inferential Statistics: Compare Job Satisfaction between IT and HR
    it_satisfaction = df[df['Department'] == 'IT']['Job Satisfaction']
    hr_satisfaction = df[df['Department'] == 'HR']['Job Satisfaction']

    t_stat, p_value = stats.ttest_ind(it_satisfaction, hr_satisfaction)

    # Write Hypothesis Test results to file
    file.write(f"T-Statistic: {t_stat}, P-Value: {p_value}\n")
    if p_value < 0.05:
        file.write(
            "Reject the null hypothesis: There is a significant difference in Job Satisfaction between IT and HR.\n\n")
    else:
        file.write(
            "Fail to reject the null hypothesis: No significant difference in Job Satisfaction between IT and HR.\n\n")

    # 4. Correlation Analysis: Work-Life Balance vs Overall Engagement
    corr_coef, corr_p_value = stats.pearsonr(df['Work-Life Balance'], df['Overall Engagement'])

    # Write Correlation results to file
    file.write(f"Correlation Coefficient: {corr_coef}, P-Value: {corr_p_value}\n")
    if corr_coef > 0:
        file.write(
            "There is a positive correlation: As Work-Life Balance improves, Overall Engagement tends to increase.\n")
    else:
        file.write(
            "There is a negative correlation: As Work-Life Balance improves, Overall Engagement tends to decrease.\n")

    # Visualize the correlation (still displayed)
    sns.scatterplot(x='Work-Life Balance', y='Overall Engagement', data=df)
    plt.title('Work-Life Balance vs Overall Engagement')
    plt.show()









