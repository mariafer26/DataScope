# DataScope

DataScope is a web application designed to help users analyze and manage their data efficiently. This guide provides instructions on how to set up, execute, and test the application.

## Prerequisites

Before running the application, ensure you have the following installed:

- **Python 3.13**
- **pip** (Python package manager)
- **Git** (optional, for cloning the repository)

## Installation and Setup

Follow these steps to set up the application:

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/mariafer26/DataScope.git
   cd DataScope
   ```
2. **Create a Virtual Environment**:

   ```bash
   python -m venv venv
   ```
3. **Activate the Virtual Environment**:

   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
4. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```
5. **Apply Migrations**:

   ```bash
   python manage.py migrate
   ```
6. **Run the Development Server**:

   ```bash
   python manage.py runserver
   ```

## Accessing the Application

Once the server is running, open your web browser and navigate to:

```
http://127.0.0.1:8000
```

## Notes

- Ensure that the `db.sqlite3` file is present in the root directory for database functionality. If not, create it by running the migrations step.
- If you encounter issues, check the installed Python version and dependencies.


## Documentation

For the complete user guide, please refer to the [User Manual](./docs/USER_MANUAL.md).

The manual includes detailed explanations of all functionalities, along with 19 verified screenshots covering registration, login, uploads, visualizations, database connections, and AI chat.

