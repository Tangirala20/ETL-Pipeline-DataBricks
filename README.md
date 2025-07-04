This project showcases a complete data engineering pipeline implemented in Databricks using PySpark, Hugging face, SQL to process, clean, analyze, and store an AI job listings dataset sourced from a public GitHub repository.

Key Steps and Features:

Data Ingestion:
Fetched raw .csv data from GitHub using Python's requests and loaded it into a Spark DataFrame via Pandas.

Data Cleaning & Preprocessing:

Removed unwanted columns and null values.

Standardized text fields using trimming and capitalization.

Replaced encoded employment types (e.g., 'FL', 'PT') with meaningful labels.

Filtering & Transformation:

Extracted relevant records (e.g., Data Engineer jobs with high salary and <4 years of experience).

Created boolean flags (has_python, has_AWS) by parsing skillsets.

Engineered a salary bucket column to classify jobs into Low, Medium, and High income groups.

Analytical Operations:

Detected duplicate rows using groupings and counts.

Sorted data by company_name (ASC) and salary_usd (DESC).

Computed days remaining till job application deadlines using datediff.

SQL-Based Insights (via Spark SQL):

Created a permanent Spark table ai_jobdataset.

Queried job postings by salary averages across locations.

Analyzed seasonal trends using SQL CASE statements.

Output & Storage:

Exported the transformed dataset to a .csv file.

Uploaded the processed data to Hugging Face Hub for public sharing and future model usage.

Tech Stack:
Databricks, PySpark, Pandas, Python, Spark SQL, Hugging Face Hub
