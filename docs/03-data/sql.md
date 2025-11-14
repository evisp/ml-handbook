# SQL Introduction: Working with Relational Databases

This tutorial introduces SQL (Structured Query Language) and relational databases. You'll learn to create databases and tables, query data with SELECT statements, modify records with INSERT/UPDATE/DELETE, and use functions and subqueries to solve complex data problems.

**Estimated time:** 40 minutes

## Why This Matters

**Problem statement:** 

> Data without structure becomes unusable at scale.

**Real-world data needs organization**. When you have thousands of customer records, product inventories, or transaction logs, spreadsheets break down. You need a system that stores data efficiently, enforces relationships between tables, handles concurrent access, and retrieves specific records instantly.

**Databases solve these problems**. They provide structure, speed, and reliability that flat files can't match.

**Practical benefits:** SQL skills let you extract insights from production databases, build data pipelines that feed analytics tools, and communicate with backend engineers about data requirements. Every data role involves querying databases.

**Professional context:** SQL ranks among the most requested skills in data job postings. Companies store operational data in relational databases, and analysts spend significant time writing queries to access it. Understanding SQL fundamentals is non-negotiable for data work.

> SQL is the language of data. Master it! 

## Core Concepts

### Understanding Databases

**Database:** An organized collection of data stored electronically. Databases manage how data is stored, accessed, updated, and protected.

**Database Management System (DBMS):** Software that creates and manages databases. Examples include MySQL, PostgreSQL, Oracle, SQL Server, and SQLite.

**Why databases matter:** They provide structured storage with built-in rules (no duplicate IDs, required fields, data type enforcement), fast retrieval using indexes, concurrent access allowing multiple users simultaneously, and data integrity through relationships and constraints.

### What Is a Relational Database?

**Relational databases** organize data into tables (also called relations) with rows and columns. Tables connect to each other through shared key values, forming relationships.

**Key characteristics:**

- **Tables:** Data organized in rows (records) and columns (fields)
- **Relationships:** Tables link via foreign keys referencing primary keys
- **Schema:** Predefined structure defining table layouts and data types
- **ACID properties:** Atomicity, Consistency, Isolation, Durability guarantee reliable transactions

**Example structure:**

```
Customers Table          Orders Table
----------------        ----------------
customer_id (PK)        order_id (PK)
name                    customer_id (FK) → links to Customers
email                   order_date
                        total_amount
```

The `customer_id` in Orders references `customer_id` in Customers, creating a relationship.

### What Does SQL Stand For?

**SQL = Structured Query Language**

SQL is the standard language for interacting with relational databases. It lets you define structure (CREATE tables), manipulate data (INSERT, UPDATE, DELETE), query data (SELECT), and control access (GRANT permissions).

**SQL is declarative:** You describe *what* you want, not *how* to get it. The database figures out the optimal execution plan.

### What Is MySQL?

**MySQL** is one of the most popular open-source relational database management systems. It's fast, reliable, widely supported, and free for most use cases.

**When MySQL is used:** Web applications, content management systems, e-commerce platforms, data warehousing, and logging systems commonly use MySQL.

**Alternatives:** PostgreSQL (more features), SQLite (lightweight, no server), SQL Server (Microsoft), Oracle (enterprise).

**For this tutorial:** Examples use MySQL syntax, but core concepts apply to all SQL databases with minor syntax variations.

## Step-by-Step Guide

###   1: Creating a Database

> Databases contain tables; tables contain data. Start by creating the container.

**Creating a database:**

```sql
-- Create a new database
CREATE DATABASE company_data;

-- View all databases
SHOW DATABASES;

-- Select database to use
USE company_data;
```

**Best practice:** Use lowercase with underscores for database names. Avoid spaces and special characters.

**Checking current database:**

```sql
SELECT DATABASE();
```

**Dropping (deleting) a database:**

```sql
-- WARNING: This permanently deletes all data
DROP DATABASE company_data;
```

###   2: Creating Tables

Tables define structure with columns, data types, and constraints.

**Basic table creation:**

```sql
CREATE TABLE employees (
    employee_id INT PRIMARY KEY AUTO_INCREMENT,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    email VARCHAR(100) UNIQUE,
    hire_date DATE,
    salary DECIMAL(10, 2),
    department_id INT
);
```

**Key components:**

- **Column name:** `first_name`, `salary`
- **Data type:** `INT` (integer), `VARCHAR(50)` (text up to 50 chars), `DATE`, `DECIMAL(10,2)` (numbers with 2 decimal places)
- **Constraints:** 
  - `PRIMARY KEY` - unique identifier for each row
  - `AUTO_INCREMENT` - automatically generates sequential numbers
  - `NOT NULL` - value required
  - `UNIQUE` - no duplicates allowed

**Common data types:**

| Type | Description | Example |
|------|-------------|---------|
| INT | Whole numbers | 42, -100, 0 |
| VARCHAR(n) | Variable-length text | 'John', 'New York' |
| TEXT | Large text blocks | Product descriptions |
| DATE | Date (YYYY-MM-DD) | '2025-01-15' |
| DATETIME | Date and time | '2025-01-15 14:30:00' |
| DECIMAL(p,s) | Precise numbers | 99.99, 12345.67 |
| BOOLEAN | True/False | TRUE, FALSE |

**Viewing table structure:**

```sql
DESCRIBE employees;
-- or
SHOW COLUMNS FROM employees;
```

**Altering existing tables:**

```sql
-- Add a new column
ALTER TABLE employees 
ADD COLUMN phone_number VARCHAR(15);

-- Modify column definition
ALTER TABLE employees 
MODIFY COLUMN salary DECIMAL(12, 2);

-- Rename column
ALTER TABLE employees 
CHANGE COLUMN phone_number contact_number VARCHAR(15);

-- Drop column
ALTER TABLE employees 
DROP COLUMN contact_number;
```

###   3: Selecting Data

**SELECT** retrieves data from tables. It's the most common SQL operation.

**Basic SELECT syntax:**

```sql
-- Get all columns and rows
SELECT * FROM employees;

-- Get specific columns
SELECT first_name, last_name, salary FROM employees;

-- Get unique values
SELECT DISTINCT department_id FROM employees;
```

**Filtering with WHERE:**

```sql
-- Single condition
SELECT * FROM employees 
WHERE department_id = 5;

-- Multiple conditions
SELECT first_name, last_name, salary 
FROM employees 
WHERE salary > 50000 AND department_id = 5;

-- Pattern matching
SELECT * FROM employees 
WHERE email LIKE '%@company.com';

-- Range
SELECT * FROM employees 
WHERE salary BETWEEN 40000 AND 80000;

-- List matching
SELECT * FROM employees 
WHERE department_id IN (1, 3, 5);
```

**Sorting results:**

```sql
-- Sort ascending (default)
SELECT first_name, last_name, salary 
FROM employees 
ORDER BY salary;

-- Sort descending
SELECT first_name, last_name, salary 
FROM employees 
ORDER BY salary DESC;

-- Multiple columns
SELECT * FROM employees 
ORDER BY department_id, last_name;
```

**Limiting results:**

```sql
-- Get first 10 rows
SELECT * FROM employees 
LIMIT 10;

-- Skip first 20, then get 10
SELECT * FROM employees 
LIMIT 10 OFFSET 20;
```

###   4: Inserting Data

**INSERT** adds new rows to tables.

**Inserting single row:**

```sql
INSERT INTO employees (first_name, last_name, email, hire_date, salary, department_id)
VALUES ('John', 'Doe', 'john.doe@company.com', '2025-01-15', 65000.00, 3);
```

**Inserting multiple rows:**

```sql
INSERT INTO employees (first_name, last_name, email, hire_date, salary, department_id)
VALUES 
    ('Jane', 'Smith', 'jane.smith@company.com', '2025-02-01', 70000.00, 2),
    ('Bob', 'Johnson', 'bob.j@company.com', '2025-02-15', 55000.00, 3),
    ('Alice', 'Williams', 'alice.w@company.com', '2025-03-01', 80000.00, 1);
```

**Tip:** If you include all columns in order, you can omit column names:

```sql
INSERT INTO employees 
VALUES (NULL, 'Mike', 'Brown', 'mike@company.com', '2025-03-15', 60000.00, 2);
-- NULL for AUTO_INCREMENT primary key
```

###   5: Updating Data

**UPDATE** modifies existing rows.

**Basic update:**

```sql
-- Update single record
UPDATE employees 
SET salary = 75000.00 
WHERE employee_id = 5;

-- Update multiple columns
UPDATE employees 
SET salary = 68000.00, department_id = 4 
WHERE employee_id = 3;

-- Update multiple rows
UPDATE employees 
SET salary = salary * 1.05 
WHERE department_id = 2;
```

**Warning:** Always include WHERE clause unless you want to update ALL rows.

```sql
-- DANGER: Updates every row in table
UPDATE employees SET salary = 50000.00;

-- SAFE: Updates only matching rows
UPDATE employees SET salary = 50000.00 WHERE employee_id = 10;
```

###   6: Deleting Data

**DELETE** removes rows from tables.

**Basic deletion:**

```sql
-- Delete specific row
DELETE FROM employees 
WHERE employee_id = 15;

-- Delete multiple rows
DELETE FROM employees 
WHERE hire_date < '2020-01-01';

-- Delete with conditions
DELETE FROM employees 
WHERE department_id = 7 AND salary < 40000;
```

**Warning:** DELETE without WHERE removes all rows.

```sql
-- DANGER: Deletes entire table contents
DELETE FROM employees;

-- SAFER: Use WHERE to target specific rows
DELETE FROM employees WHERE employee_id = 20;
```

**Difference from DROP:**

- **DELETE:** Removes rows, keeps table structure
- **DROP TABLE:** Deletes entire table and structure
- **TRUNCATE:** Removes all rows faster than DELETE (but can't undo)

###   7: Using Subqueries

**Subqueries** are queries nested inside other queries, useful for complex filtering and calculations.

**Subquery in WHERE clause:**

```sql
-- Find employees earning above average
SELECT first_name, last_name, salary 
FROM employees 
WHERE salary > (SELECT AVG(salary) FROM employees);
```

**Subquery with IN:**

```sql
-- Find employees in departments located in 'New York'
SELECT first_name, last_name 
FROM employees 
WHERE department_id IN (
    SELECT department_id 
    FROM departments 
    WHERE location = 'New York'
);
```

**Subquery in SELECT (correlated):**

```sql
-- Show each employee with their department's average salary
SELECT 
    first_name, 
    last_name, 
    salary,
    (SELECT AVG(salary) 
     FROM employees e2 
     WHERE e2.department_id = e1.department_id) AS dept_avg_salary
FROM employees e1;
```

**When to use subqueries:**

- Filter based on aggregated data
- Compare values across tables
- Perform calculations for each row based on related data

**Alternative:** JOINs often perform better than subqueries for relating tables.

###   8: Using MySQL Functions

MySQL provides built-in functions for calculations, text manipulation, and data transformation.

**Aggregate functions:**

```sql
-- Count rows
SELECT COUNT(*) FROM employees;

-- Count non-null values
SELECT COUNT(email) FROM employees;

-- Sum values
SELECT SUM(salary) FROM employees;

-- Average
SELECT AVG(salary) FROM employees;

-- Min and Max
SELECT MIN(salary), MAX(salary) FROM employees;

-- Aggregate with grouping
SELECT department_id, AVG(salary) AS avg_salary, COUNT(*) AS employee_count
FROM employees
GROUP BY department_id;
```

**String functions:**

```sql
-- Concatenate
SELECT CONCAT(first_name, ' ', last_name) AS full_name 
FROM employees;

-- Uppercase/lowercase
SELECT UPPER(first_name), LOWER(email) 
FROM employees;

-- Substring
SELECT SUBSTRING(email, 1, 5) AS email_prefix 
FROM employees;

-- Length
SELECT first_name, LENGTH(first_name) AS name_length 
FROM employees;
```

**Date functions:**

```sql
-- Current date and time
SELECT NOW(), CURDATE(), CURTIME();

-- Extract parts
SELECT 
    hire_date,
    YEAR(hire_date) AS hire_year,
    MONTH(hire_date) AS hire_month,
    DAY(hire_date) AS hire_day
FROM employees;

-- Date arithmetic
SELECT 
    hire_date,
    DATE_ADD(hire_date, INTERVAL 90 DAY) AS probation_end
FROM employees;

-- Date difference
SELECT 
    first_name,
    DATEDIFF(CURDATE(), hire_date) AS days_employed
FROM employees;
```

**Mathematical functions:**

```sql
-- Rounding
SELECT salary, ROUND(salary, -3) AS rounded_salary 
FROM employees;

-- Absolute value
SELECT ABS(-42);

-- Power
SELECT POWER(2, 3);  -- Returns 8
```

## Common SQL Challenges

### NULL Values

**Problem:** NULL represents missing data and behaves differently than regular values.

```sql
-- Wrong: This doesn't work as expected
SELECT * FROM employees WHERE email = NULL;

-- Correct: Use IS NULL / IS NOT NULL
SELECT * FROM employees WHERE email IS NULL;
SELECT * FROM employees WHERE email IS NOT NULL;

-- Handling NULLs in calculations
SELECT first_name, COALESCE(phone_number, 'No phone') AS contact
FROM employees;
```

### Avoiding Accidental Data Loss

**Problem:** DELETE or UPDATE without WHERE affects all rows.

**Solution:** Always test with SELECT first:

```sql
-- Step 1: Test your WHERE clause
SELECT * FROM employees WHERE department_id = 7;

-- Step 2: If results look correct, change to DELETE
DELETE FROM employees WHERE department_id = 7;
```

### Understanding DISTINCT vs GROUP BY

Both reduce duplicates but serve different purposes:

```sql
-- DISTINCT: Returns unique combinations
SELECT DISTINCT department_id FROM employees;

-- GROUP BY: Aggregates data
SELECT department_id, COUNT(*) 
FROM employees 
GROUP BY department_id;
```

Use `DISTINCT` for simple deduplication, `GROUP BY` when calculating aggregates.

## Best Practices

**Always use WHERE with UPDATE and DELETE:** Protect against accidentally modifying all rows. Test your WHERE clause with SELECT first.

**Name things clearly:** Use descriptive names like `customer_email` instead of `ce`. Future you will thank past you.

**Use consistent naming conventions:** Pick a style (snake_case or camelCase) and stick with it throughout your database.

**Index frequently queried columns:** Add indexes to columns used in `WHERE` clauses and `JOIN`s to speed up queries. Primary keys are automatically indexed.

```sql
CREATE INDEX idx_employee_department ON employees(department_id);
```

**Back up before structural changes:** Always back up data before running ALTER TABLE or DROP statements.

**Comment complex queries:** Add comments to explain business logic:

```sql
-- Calculate bonus for employees hired in 2024 with sales > $100k
SELECT 
    employee_id,
    salary * 0.10 AS bonus  -- 10% of salary
FROM employees 
WHERE YEAR(hire_date) = 2024;
```

**Use transactions for multi-step operations:** Ensure all-or-nothing execution:

```sql
START TRANSACTION;

UPDATE accounts SET balance = balance - 100 WHERE account_id = 1;
UPDATE accounts SET balance = balance + 100 WHERE account_id = 2;

COMMIT;  -- Or ROLLBACK if something went wrong
```

## Quick Reference

### Essential SQL Commands

| Command | Purpose | Example |
|---------|---------|---------|
| `CREATE DATABASE` | Create new database | `CREATE DATABASE company;` |
| `USE` | Select database | `USE company;` |
| `CREATE TABLE` | Define new table | `CREATE TABLE users (...);` |
| `SELECT` | Query data | `SELECT * FROM users;` |
| `WHERE` | Filter rows | `WHERE age > 25` |
| `ORDER BY` | Sort results | `ORDER BY name DESC` |
| `INSERT INTO` | Add rows | `INSERT INTO users VALUES (...);` |
| `UPDATE` | Modify rows | `UPDATE users SET age = 30 WHERE id = 5;` |
| `DELETE` | Remove rows | `DELETE FROM users WHERE id = 10;` |
| `ALTER TABLE` | Modify table | `ALTER TABLE users ADD COLUMN email VARCHAR(100);` |

### Common Operators

| Operator | Meaning | Example |
|----------|---------|---------|
| `=` | Equals | `WHERE status = 'active'` |
| `!=` or `<>` | Not equals | `WHERE status != 'inactive'` |
| `>`, `<`, `>=`, `<=` | Comparison | `WHERE salary > 50000` |
| `BETWEEN` | Range | `WHERE age BETWEEN 25 AND 35` |
| `IN` | Match list | `WHERE dept IN (1, 2, 3)` |
| `LIKE` | Pattern match | `WHERE name LIKE 'J%'` |
| `IS NULL` | Check for NULL | `WHERE email IS NULL` |
| `AND`, `OR`, `NOT` | Logical | `WHERE age > 25 AND dept = 5` |

### Basic Query Template

```sql
SELECT column1, column2, aggregate_function(column3)
FROM table_name
WHERE condition
GROUP BY column1
HAVING aggregate_condition
ORDER BY column1 DESC
LIMIT 10;
```

## Summary & Next Steps

**Key accomplishments:** You've learned what databases and relational databases are, how to create databases and tables with proper structure, how to query data with SELECT and filter with WHERE, how to insert, update, and delete records safely, how subqueries enable complex filtering, and how to use built-in MySQL functions.

**Critical insights:**

- **SQL is declarative:** Describe what you want, not how to get it
- **Structure matters:** Well-designed tables with proper data types prevent problems
- **Test before executing:** Always verify WHERE clauses with SELECT before UPDATE/DELETE
- **Functions extend capabilities:** Built-in functions handle common tasks efficiently

**What's next:**

With SQL fundamentals mastered, you're ready to explore JOINs (combining multiple tables), indexes (optimizing query performance), views (saving complex queries), and stored procedures (reusable SQL logic). You can also start integrating SQL queries into Python workflows using libraries like `pandas.read_sql()` or SQLAlchemy.

**Practice resources:**

- [SQLBolt](https://sqlbolt.com/) - interactive SQL lessons with instant feedback
- [Mode SQL Tutorial](https://mode.com/sql-tutorial/) - real datasets for practice
- [LeetCode Database Problems](https://leetcode.com/problemset/database/) - SQL challenges

**External resources:**

- [MySQL Official Documentation](https://dev.mysql.com/doc/) - comprehensive reference
- [W3Schools SQL Tutorial](https://www.w3schools.com/sql/) - quick examples and testing
- [SQL Style Guide](https://www.sqlstyle.guide/) - formatting conventions

> **Remember:** SQL is the foundation of data work. Master these basics, and you'll access, manipulate, and analyze data from any relational database with confidence.
