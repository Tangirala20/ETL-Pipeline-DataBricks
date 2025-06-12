# Databricks notebook source
import requests
import pandas as pd
from io import StringIO

# URL of the CSV file in the Git repository
url = 'https://raw.githubusercontent.com/Tangirala20/ETL-Pipeline-DataBricks/main/ai_job_dataset.csv'

# Download the file
response = requests.get(url)
data = StringIO(response.text)

# Load the CSV file into a Pandas DataFrame
pdf = pd.read_csv(data)

# Convert the Pandas DataFrame to a Spark DataFrame
df = spark.createDataFrame(pdf)

# Display the DataFrame
display(df)

# COMMAND ----------

df.columns

# COMMAND ----------

# MAGIC %md
# MAGIC Removing Unwanted Columns

# COMMAND ----------

df = df.drop('job_description_length', 'experience_level', 'benfits_score')
df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC Removing Null Values

# COMMAND ----------

df = df.na.drop(subset=['job_title', 'company_location', 'industry', 'employee_residence', 'company_name'])
if not df.isEmpty(): df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC Displaying the data for analysing the essence of the dataset

# COMMAND ----------

df.select('job_title','years_experience','salary_usd').display()

# COMMAND ----------

# MAGIC %md
# MAGIC Filtering the columns W.R.T salary, Jop title & YOE

# COMMAND ----------

from pyspark.sql.functions import col
df.filter((col('salary_usd') > 100000) & (col('years_experience') < 4) & (col('job_title') == 'Data Engineer')).display()

# COMMAND ----------

# MAGIC %md
# MAGIC Grouping by Job title and displaying the results

# COMMAND ----------


df.groupBy('job_title').count().display()
df.groupBy('job_title')

# COMMAND ----------

# MAGIC %md
# MAGIC Counting & extracting the duplicates if any

# COMMAND ----------

from pyspark.sql import functions as F

dupes_df = df.groupBy(df.columns) \
              .count() \
              .filter(F.col('count') > 1)
dupes_count = dupes_df.selectExpr("sum(count - 1) as total_duplicates").collect()[0]['total_duplicates']

print(f"Total number of duplicate rows: {dupes_count}")

# COMMAND ----------

# MAGIC %md
# MAGIC Sorting the entire dataset based on Company Name(ASC) & Salary_usd(DESC)

# COMMAND ----------

df_sorted = df.orderBy(['company_name',df['salary_usd'].desc()])

# COMMAND ----------

# MAGIC %md
# MAGIC Creating new columns to check if the candidate should have Python & AWS as the skills

# COMMAND ----------

from pyspark.sql.functions import split, explode, trim, array_contains
df_sorted = df_sorted.withColumn('has_python', array_contains(split(trim(df_sorted['required_skills']), ','),'Python'))
df_sorted = df_sorted.withColumn('has_AWS', array_contains(split(trim(df_sorted['required_skills']), ','),'AWS'))
df_skillset = df_sorted.filter(df_sorted['has_python'] & df_sorted['has_AWS']).select('job_id','job_title','salary_usd','company_name')

# COMMAND ----------

# MAGIC %md
# MAGIC Replacing Row data

# COMMAND ----------

from pyspark.sql.functions import col, when

df_sorted = df_sorted.replace({'FL':'Full Time', 'PT':'Part Time', 'NA':'Not Applicable', 'CT': 'Contract'}, subset=['employment_type'])



# COMMAND ----------

# MAGIC %md
# MAGIC Cleaning column text

# COMMAND ----------

from pyspark.sql.functions import col, lower, trim, initcap
cols = ['job_title','company_location', 'industry', 'employee_residence', 'company_name']
for i in cols:
    df_sorted = df_sorted.withColumn(i, trim(initcap(col(i))))

# COMMAND ----------

from pyspark.sql.functions import unix_timestamp, col, datediff

df_sorted = df_sorted.withColumn(
    "posting_date_ts", 
    unix_timestamp(col("posting_date"), "yyyy-MM-dd").cast("timestamp")
).withColumn(
    "deadline_ts", 
    unix_timestamp(col("application_deadline"), "yyyy-MM-dd").cast("timestamp")
).withColumn(
    "days_until_deadline", 
    datediff(col("deadline_ts"), col("posting_date_ts"))
)

display(df_sorted)

# COMMAND ----------

df_sorted = df_sorted.drop('posting_date_ts','deadline_ts','benifit_score')

# COMMAND ----------

# MAGIC %md
# MAGIC Creating a new column salary bucket

# COMMAND ----------

from pyspark.sql.functions import when

df_sorted = df_sorted.withColumn('salary_bucket', when(col('salary_usd') < 50000, 'Low').when(col('salary_usd').between(50000, 100000), 'Medium').otherwise('High'))

# COMMAND ----------

# MAGIC %md
# MAGIC Performing operations using SQL

# COMMAND ----------

df_sorted.write.mode("overwrite").saveAsTable("ai_jobdataset")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM ai_jobdataset;

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Get jobs in cities where the average salary is above 150K
# MAGIC SELECT *
# MAGIC FROM ai_jobdataset
# MAGIC WHERE employee_residence IN (
# MAGIC     SELECT employee_residence
# MAGIC     FROM processed_jobs_table
# MAGIC     GROUP BY employee_residence
# MAGIC     HAVING AVG(salary_usd) > 150000
# MAGIC );
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT job_id,
# MAGIC        posting_date,
# MAGIC        CASE 
# MAGIC            WHEN MONTH(posting_date) = 1 THEN 'Posted in January'
# MAGIC            WHEN MONTH(posting_date) = 12 THEN 'Posted in December'
# MAGIC            ELSE 'Posted in another month'
# MAGIC        END AS month_status
# MAGIC FROM ai_jobdataset;
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC Storing the TRANSFORMED .CSV to Workspace

# COMMAND ----------

df_sorted.toPandas().to_csv("processed_jobs.csv", index=False)

# COMMAND ----------

# MAGIC %md
# MAGIC LOADING the processed data to Hugging Face

# COMMAND ----------

%pip install huggingface_hub

# COMMAND ----------

# MAGIC %restart_python
# MAGIC

# COMMAND ----------

from huggingface_hub import HfApi

api = HfApi()

api.upload_file(
    path_or_fileobj="processed_jobs.csv",        # your existing file path
    path_in_repo="Pyspark_processed.csv",                     # how it should be named on Hugging Face
    repo_id="Gamerfleet/Pyspark_processed",      # replace with your HF dataset repo
    repo_type="dataset",
    token=""                         # your token here
)


# COMMAND ----------

