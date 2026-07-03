import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

DATABASE_NAME = "project.db"


def get_connection():
    connection = sqlite3.connect(DATABASE_NAME)
    connection.row_factory = sqlite3.Row
    return connection


def create_tables():
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            model_name TEXT NOT NULL,
            input_values TEXT NOT NULL,
            prediction TEXT NOT NULL,
            probability TEXT
        )
    """)

    connection.commit()
    connection.close()


def register_user(username, password):
    connection = get_connection()
    cursor = connection.cursor()

    hashed_password = generate_password_hash(password)

    try:
        cursor.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)",
            (username, hashed_password)
        )

        connection.commit()
        return True

    except sqlite3.IntegrityError:
        return False

    finally:
        connection.close()


def login_user(username, password):
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        "SELECT * FROM users WHERE username = ?",
        (username,)
    )

    user = cursor.fetchone()
    connection.close()

    if user is None:
        return False

    return check_password_hash(user["password"], password)


def save_prediction(username, model_name, input_values, prediction, probability=None):
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO predictions
        (username, model_name, input_values, prediction, probability)
        VALUES (?, ?, ?, ?, ?)
    """, (username, model_name, input_values, str(prediction), str(probability)))

    connection.commit()
    connection.close()


def get_predictions(username):
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
        SELECT * FROM predictions
        WHERE username = ?
        ORDER BY id DESC
    """, (username,))

    predictions = cursor.fetchall()
    connection.close()

    return predictions