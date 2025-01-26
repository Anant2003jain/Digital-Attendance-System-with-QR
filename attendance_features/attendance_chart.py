import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(r'StudentData\attendancelist.csv')

# Assuming the first column is 'nameID', you can set it as the index
df.set_index('nameID', inplace=True)

# Count the number of present days for each student
df['Total_Present'] = df.apply(lambda row: row.str.count('Present').sum(), axis=1)

# Count the total number of days (excluding the 'nameID' column)
total_days = len(df.columns) - 1

# Calculate the percentage of attendance for each student
df['Attendance_Percentage'] = (df['Total_Present'] / total_days) * 100

# Print the DataFrame with total present days and attendance percentage
print(df[['Total_Present', 'Attendance_Percentage']])

# Create a pie chart
for student, attendance_percentage in df['Attendance_Percentage'].items():
    plt.figure(figsize=(6, 6))
    plt.pie([attendance_percentage, 100 - attendance_percentage], labels=['Present', 'Absent'], autopct='%1.1f%%', startangle=140)
    plt.title(f'Attendance Ratio for {student}')
    plt.savefig(f'StudentData\\Attendance_chart\\{student}_attendance_chart.png')
    #plt.close()

print("Pie charts generated and saved for each student.")
