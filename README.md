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


# DataScope - Cloud Deployment

## Introduction

Deploying an application to the cloud allows it to be accessible to end-users over the Internet, providing scalability, availability, and accessibility without relying on local physical infrastructure. In this project, the **DataScope** application has been deployed to a public cloud using **Heroku**, a Platform as a Service (PaaS) solution ideal for Django web applications with PostgreSQL databases.

## Deployment Platform: Heroku

**Heroku** is a cloud platform that enables developers to deploy, manage, and scale applications easily. Key advantages include seamless integration with Git repositories, native support for Python and Django, built-in PostgreSQL support, and automatically managed runtime environments.

## Deployment Process

The **DataScope** application was deployed to Heroku following these steps:

1. **Prepare the Local Project**  
   Install the required packages (`gunicorn`, `dj-database-url`, `psycopg2-binary`, `whitenoise`) and configure the project settings to use PostgreSQL in production. Add a `Procfile` with the command to run the application and configure `settings.py` to include database settings, static files configuration, and Whitenoise for static file management.

2. **Create the Heroku Application**  
   Log in to Heroku and create a new application. Heroku automatically assigns a unique URL for the app.

3. **Configure the Database**  
   Add a PostgreSQL service to the Heroku app to store all application data in a cloud-managed relational database.

4. **Set Environment Variables**  
   Define essential environment variables such as `DJANGO_SECRET_KEY`, `DEBUG`, and `DISABLE_COLLECTSTATIC` to ensure proper operation in the cloud environment.

5. **Run Database Migrations**  
   Execute Django migrations on the Heroku server to create all necessary database tables.

6. **Push the Project to Heroku**  
   Commit all local changes and push the project to Heroku using Git. Heroku automatically detects the Python/Django project, installs dependencies, and starts the application.

## Final Result

After completing these steps, the **DataScope** application is fully deployed on Heroku’s cloud platform, connected to a PostgreSQL database, and accessible from any web browser.

### Live App Link

[[Add your public Heroku app link here]](https://rocky-everglades-82372-1d32dbcc097e.herokuapp.com/)


## Notes

- Ensure that the `db.sqlite3` file is present in the root directory for database functionality. If not, create it by running the migrations step.
- If you encounter issues, check the installed Python version and dependencies.
