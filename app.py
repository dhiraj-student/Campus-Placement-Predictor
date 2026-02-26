import numpy as np
import pickle
import sqlite3
import pandas as pd
import os
from datetime import datetime
from flask import Flask, request, render_template, g, session, redirect, url_for, flash
from functools import wraps
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__, template_folder='templates')
app.jinja_env.globals.update(enumerate=enumerate)
app.secret_key = 'placement_predictor_secret_key_2024'

# ─────────────────────────────────────────────────────────────
# LOGIN CREDENTIALS
# ─────────────────────────────────────────────────────────────
VALID_USERS = {
    'admin'  : 'admin123',
    'dhiraj' : 'dhiraj123',
}

# ─────────────────────────────────────────────────────────────
# LOGIN REQUIRED DECORATOR
# ─────────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

model      = pickle.load(open('model.pkl', 'rb'))
model_name = type(model).__name__

MODEL_DISPLAY = {
    'RandomForestClassifier'    : 'Random Forest',
    'XGBClassifier'             : 'XGBoost',
    'SVC'                       : 'SVM',
    'LogisticRegression'        : 'Logistic Regression',
    'DecisionTreeClassifier'    : 'Decision Tree',
    'KNeighborsClassifier'      : 'KNN',
    'GradientBoostingClassifier': 'Gradient Boosting',
}
MODEL_LABEL = MODEL_DISPLAY.get(model_name, model_name)

DATABASE = 'predictions.db'

# ─────────────────────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────────────────────
def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def init_db():
    with app.app_context():
        db = get_db()
        db.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                age INTEGER, gender TEXT, stream TEXT,
                internships INTEGER, cgpa REAL, projects TEXT,
                backlogs TEXT, result TEXT, probability REAL, timestamp TEXT
            )
        ''')
        db.commit()

init_db()

STREAM_MAP = {
    '1': 'Electronics & Communication', '2': 'Computer Science',
    '3': 'Information Technology',       '4': 'Mechanical',
    '5': 'Electrical',                   '6': 'Civil'
}

# ─────────────────────────────────────────────────────────────
# MODEL COMPARISON
# ─────────────────────────────────────────────────────────────
def get_model_comparison():
    df = pd.read_csv('archive/collegePlace.csv')
    df.dropna(inplace=True)
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    df['Stream'] = df['Stream'].map({
        'Electronics And Communication': 1, 'Computer Science': 2,
        'Information Technology': 3, 'Mechanical': 4,
        'Electrical': 5, 'Civil': 6
    })
    df.dropna(inplace=True)

    X = df[['Age','Gender','Stream','Internships','CGPA','Hostel','HistoryOfBacklogs']]
    y = df['PlacedOrNot']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        'Random Forest'    : RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost'          : XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        'SVM'              : SVC(probability=True, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree'    : DecisionTreeClassifier(random_state=42),
        'KNN'              : KNeighborsClassifier(n_neighbors=5),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    results = []
    for name, m in models.items():
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)
        results.append({
            'name'     : name,
            'accuracy' : round(accuracy_score(y_test, y_pred) * 100, 2),
            'precision': round(precision_score(y_test, y_pred) * 100, 2),
            'recall'   : round(recall_score(y_test, y_pred) * 100, 2),
            'f1'       : round(f1_score(y_test, y_pred) * 100, 2),
            'is_best'  : MODEL_DISPLAY.get(model_name, model_name) == name
        })

    results.sort(key=lambda x: x['accuracy'], reverse=True)
    return results

# ─────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/home')
@login_required
def base():
    return render_template('form.html')

@app.route('/contact')
@login_required
def contact():
    return render_template('contact.html')

@app.route('/predict', methods=['GET'])
@login_required
def predict():
    age        = int(request.args.get('age'))
    gender     = int(request.args.get('gender'))
    stream     = int(request.args.get('stream'))
    internship = int(request.args.get('internship'))
    cgpa       = float(request.args.get('cgpa'))
    projects   = int(request.args.get('projects'))   # replaces hostel
    backlogs   = int(request.args.get('backlogs'))

    # Model still uses position 6 (was Hostel, now Projects — both binary 0/1)
    arr         = np.array([age, gender, stream, internship, cgpa, projects, backlogs])
    output      = model.predict([arr])
    probability = round(model.predict_proba([arr])[0][1] * 100, 2)
    result      = 'High' if output == 1 else 'Low'

    db = get_db()
    db.execute('''INSERT INTO predictions
        (age,gender,stream,internships,cgpa,projects,backlogs,result,probability,timestamp)
        VALUES (?,?,?,?,?,?,?,?,?,?)''', (
        age, 'Male' if gender == 1 else 'Female',
        STREAM_MAP.get(str(stream), 'Unknown'),
        internship, cgpa,
        'Yes' if projects == 1 else 'No',
        'Yes' if backlogs == 1 else 'No',
        result, probability,
        datetime.now().strftime('%d-%m-%Y %H:%M:%S')
    ))
    db.commit()

    return render_template('output.html', output=result, probability=probability, model_label=MODEL_LABEL)

@app.route('/history')
@login_required
def history():
    db   = get_db()
    rows = db.execute('SELECT * FROM predictions ORDER BY id DESC').fetchall()
    return render_template('history.html', predictions=rows)

@app.route('/dashboard')
@login_required
def dashboard():
    db   = get_db()
    rows = db.execute('SELECT * FROM predictions').fetchall()

    total      = len(rows)
    placed     = sum(1 for r in rows if r['result'] == 'High')
    not_placed = total - placed

    avg_cgpa_placed     = round(sum(r['cgpa'] for r in rows if r['result'] == 'High') / placed, 2)     if placed     else 0
    avg_cgpa_not_placed = round(sum(r['cgpa'] for r in rows if r['result'] == 'Low')  / not_placed, 2) if not_placed else 0

    stream_data = {}
    for r in rows:
        stream_data[r['stream']] = stream_data.get(r['stream'], {'High': 0, 'Low': 0})
        stream_data[r['stream']][r['result']] += 1

    intern_data = {}
    for r in rows:
        k = str(r['internships'])
        intern_data[k] = intern_data.get(k, {'High': 0, 'Low': 0})
        intern_data[k][r['result']] += 1

    gender_data = {}
    for r in rows:
        gender_data[r['gender']] = gender_data.get(r['gender'], {'High': 0, 'Low': 0})
        gender_data[r['gender']][r['result']] += 1

    return render_template('dashboard.html',
        total=total, placed=placed, not_placed=not_placed,
        avg_cgpa_placed=avg_cgpa_placed, avg_cgpa_not_placed=avg_cgpa_not_placed,
        stream_data=stream_data, intern_data=intern_data, gender_data=gender_data,
        model_label=MODEL_LABEL
    )

@app.route('/models')
@login_required
def models_page():
    results = get_model_comparison()
    return render_template('models.html', results=results, model_label=MODEL_LABEL)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'username' in session:
        return redirect(url_for('home'))
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        if username in VALID_USERS and VALID_USERS[username] == password:
            session['username'] = username
            return redirect(url_for('home'))
        else:
            error = 'Invalid username or password. Please try again.'
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/clear_history', methods=['POST'])
@login_required
def clear_history():
    db = get_db()
    db.execute('DELETE FROM predictions')
    db.commit()
    return history()



if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(debug=False, host='0.0.0.0', port=port)
