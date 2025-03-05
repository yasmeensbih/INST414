import pandas as pd   # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns   # type: ignore


#reading the dataset 
df = pd.read_csv("jobs_in_data.csv")



print(df.info())
print(df.head())


#cleaning the data
#Removing any duplicates within the dataset
df = df.drop_duplicates()

#summarizing the data
#converting the salary 
df['salary'] = pd.to_numeric(df['salary'], errors = 'coerce')
#dropping that are missing salary values
df = df.dropna(subset = ['salary']) 

#Grouping by job title
job_salary = df.groupby('experience_level')['salary'].mean().sort_values(ascending= False).head(10)


#Experience level salary summary
experience_salary = df.groupby('experience_level')['salary'].mean()



#Visualizing the data 

#Salary by Experience Level

plt.figure(figsize=(8, 5))
sns.barplot(x=experience_salary.index, y= experience_salary.values, palette= 'viridis')
plt.xlabel("Experience Level")
plt.ylabel("Average Salary (USD)")
plt.title("Average Salary by Experience Level")
plt.show()


print("Top 10 Highest-Paying Jobs:")
print(job_salary)


print("\nSalary Differences by Experience Level:")
print(experience_salary)