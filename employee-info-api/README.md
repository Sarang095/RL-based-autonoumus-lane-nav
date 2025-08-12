# employee-info-api

A tiny MuleSoft API that returns basic employee info from a MySQL database. It exposes one endpoint: `GET /employees/{id}`. Good starter project if you want to learn how HTTP + Database + DataWeave fit together in Mule 4.

## What this does (and why it’s useful)
- Listens for HTTP requests
- Looks up an employee row in MySQL by `id`
- Transforms the database row into clean JSON using DataWeave
- Returns `200 OK` with the JSON if found, or `404 Not Found` if there’s no match

It’s a simple pattern you’ll use a lot: receive a request, hit a system of record, shape the data, send it back.

## How it works (in plain English)
- An HTTP Listener accepts `GET /employees/{id}`
- The Database connector runs `SELECT id, name, department, salary FROM employees WHERE id = :id`
- DataWeave turns the first row into a JSON object
- If no row comes back, it returns a small JSON error with a 404 status

The flow is defined in `src/main/mule/employee-info-api.xml`.

## Run it locally in Anypoint Studio
1. Clone or copy this folder into your workspace and open it in Anypoint Studio as a Mule project.
2. Start MySQL locally and create the schema/table with the provided script:
   - File: `src/main/resources/sql/employees.sql`
   - Run it with your favorite MySQL client.
3. Configure database connection properties in `src/main/resources/db-config.properties`:
   ```properties
db.host=localhost
db.port=3306
db.user=root
db.password=your_password
db.database=employees_db
   ```
4. Optionally adjust the HTTP listener in `src/main/resources/http.properties` (defaults to `0.0.0.0:8081`).
5. Make sure the Database connector has access to a MySQL JDBC driver:
   - In Studio, open the DB config and ensure the MySQL driver is available. If not, add the MySQL 8+ driver (e.g., `mysql-connector-j`).
6. Run the app in Studio (right-click project → Run As → Mule Application). You should see the app start on port 8081.

## Try it out
- Example request:
  ```bash
curl -i http://localhost:8081/employees/1
  ```
- Example 200 response:
  ```json
{
  "id": 1,
  "name": "Alice Johnson",
  "department": "Engineering",
  "salary": 105000
}
  ```
- Example 404 response:
  ```json
{
  "message": "Employee not found",
  "id": "999"
}
  ```

## Notes
- This is intentionally beginner-friendly. The idea is to keep it small and clear so you can build on top of it later.
- From here you could add POST/PUT endpoints, validation, error handling with custom error types, or use APIkit/RAML to formalize the API. But for now, it’s just the essentials.