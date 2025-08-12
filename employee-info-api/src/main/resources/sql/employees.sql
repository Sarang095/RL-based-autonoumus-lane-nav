-- Create database and table, then seed with example data
CREATE DATABASE IF NOT EXISTS employees_db;
USE employees_db;

CREATE TABLE IF NOT EXISTS employees (
  id INT PRIMARY KEY,
  name VARCHAR(100) NOT NULL,
  department VARCHAR(100) NOT NULL,
  salary DECIMAL(10,2) NOT NULL
);

INSERT INTO employees (id, name, department, salary) VALUES
  (1, 'Alice Johnson', 'Engineering', 105000.00),
  (2, 'Bob Martinez', 'Finance', 92000.00),
  (3, 'Cara Singh', 'HR', 78000.00)
ON DUPLICATE KEY UPDATE
  name = VALUES(name),
  department = VALUES(department),
  salary = VALUES(salary);