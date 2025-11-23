# create_webapp_files.py
import os

# Define base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, 'src')

# Create directories
os.makedirs(os.path.join(SRC_DIR, 'templates'), exist_ok=True)
os.makedirs(os.path.join(SRC_DIR, 'static', 'css'), exist_ok=True)
os.makedirs(os.path.join(SRC_DIR, 'static', 'js'), exist_ok=True)

print("📁 Creating folder structure...")
print(f"✅ Created: {os.path.join(SRC_DIR, 'templates')}")
print(f"✅ Created: {os.path.join(SRC_DIR, 'static', 'css')}")
print(f"✅ Created: {os.path.join(SRC_DIR, 'static', 'js')}")

# Create index.html
index_html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentiment Analysis App</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
        <div class="container">
            <a class="navbar-brand" href="/"><i class="fas fa-smile"></i> Sentiment Analyzer</a>
            <div class="navbar-nav ms-auto">
                <a class="nav-link active" href="/">Home</a>
                <a class="nav-link" href="/batch">Batch</a>
                <a class="nav-link" href="/metrics">Metrics</a>
            </div>
        </div>
    </nav>

    <div class="container mt-5">
        <div class="row">
            <div class="col-lg-8 mx-auto">
                <div class="text-center mb-5">
                    <h1 class="display-4">😊 Sentiment Analysis</h1>
                    <p class="lead">Analyze the emotional tone of your text using AI</p>
                </div>

                <div class="card shadow">
                    <div class="card-body p-4">
                        <h3 class="mb-4">Enter Your Text</h3>
                        
                        <textarea id="textInput" class="form-control mb-3" rows="6" 
                                  placeholder="Type your text here..."></textarea>
                        
                        <div class="mb-3">
                            <button class="btn btn-success btn-sm me-2" onclick="setExample('happy')">
                                😊 Happy Example
                            </button>
                            <button class="btn btn-danger btn-sm" onclick="setExample('sad')">
                                😢 Sad Example
                            </button>
                        </div>

                        <button id="analyzeBtn" class="btn btn-primary btn-lg w-100" onclick="analyzeSentiment()">
                            Analyze Sentiment
                        </button>

                        <div id="loader" class="text-center my-4" style="display: none;">
                            <div class="spinner-border text-primary"></div>
                            <p class="mt-2">Analyzing...</p>
                        </div>

                        <div id="results" class="mt-4" style="display: none;">
                            <div class="alert" id="resultAlert">
                                <h3 id="sentimentLabel"></h3>
                                <p id="confidenceText"></p>
                                <div class="progress mb-3">
                                    <div id="happyBar" class="progress-bar bg-success"></div>
                                </div>
                                <div class="progress">
                                    <div id="sadBar" class="progress-bar bg-danger"></div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="card shadow mt-4">
                    <div class="card-body">
                        <h5>Model Performance</h5>
                        <p>Accuracy: {{ (metrics.accuracy * 100)|round(1) }}%</p>
                        <p>Precision: {{ (metrics.precision * 100)|round(1) }}%</p>
                        <p>Recall: {{ (metrics.recall * 100)|round(1) }}%</p>
                        <p>F1 Score: {{ (metrics.f1_score * 100)|round(1) }}%</p>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        const examples = {
            happy: "I absolutely love this beautiful day! Everything is wonderful!",
            sad: "I'm feeling really down and disappointed today. Everything is terrible."
        };

        function setExample(type) {
            document.getElementById('textInput').value = examples[type];
        }

        async function analyzeSentiment() {
            const text = document.getElementById('textInput').value.trim();
            if (!text) {
                alert('Please enter some text');
                return;
            }

            document.getElementById('loader').style.display = 'block';
            document.getElementById('results').style.display = 'none';

            try {
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ text: text })
                });

                const data = await response.json();

                if (data.error) {
                    alert('Error: ' + data.error);
                    return;
                }

                displayResults(data);

            } catch (error) {
                alert('Error: ' + error.message);
            } finally {
                document.getElementById('loader').style.display = 'none';
            }
        }

        function displayResults(data) {
            const resultAlert = document.getElementById('resultAlert');
            resultAlert.className = 'alert alert-' + (data.sentiment === 'happy' ? 'success' : 'danger');

            document.getElementById('sentimentLabel').textContent = data.sentiment_label;
            document.getElementById('confidenceText').textContent = 
                'Confidence: ' + data.confidence.toFixed(1) + '%';

            document.getElementById('happyBar').style.width = data.probabilities.happy + '%';
            document.getElementById('happyBar').textContent = data.probabilities.happy.toFixed(1) + '%';

            document.getElementById('sadBar').style.width = data.probabilities.sad + '%';
            document.getElementById('sadBar').textContent = data.probabilities.sad.toFixed(1) + '%';

            document.getElementById('results').style.display = 'block';
        }
    </script>
</body>
</html>'''

# Create batch.html
batch_html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Batch Analysis</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
        <div class="container">
            <a class="navbar-brand" href="/">Sentiment Analyzer</a>
            <div class="navbar-nav ms-auto">
                <a class="nav-link" href="/">Home</a>
                <a class="nav-link active" href="/batch">Batch</a>
                <a class="nav-link" href="/metrics">Metrics</a>
            </div>
        </div>
    </nav>

    <div class="container mt-5">
        <div class="col-lg-8 mx-auto">
            <h1 class="text-center mb-4">Batch Analysis</h1>
            <div class="card shadow">
                <div class="card-body p-4">
                    <h5>Upload CSV File</h5>
                    <input type="file" id="fileInput" class="form-control mb-3" accept=".csv">
                    <button class="btn btn-primary w-100" onclick="analyzeBatch()">Analyze</button>
                    <div id="results" class="mt-4"></div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        async function analyzeBatch() {
            const fileInput = document.getElementById('fileInput');
            if (!fileInput.files[0]) {
                alert('Please select a file');
                return;
            }

            const formData = new FormData();
            formData.append('file', fileInput.files[0]);

            try {
                const response = await fetch('/api/predict-batch', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();
                document.getElementById('results').innerHTML = 
                    '<div class="alert alert-success">Analyzed ' + data.summary.total + ' texts</div>';
            } catch (error) {
                alert('Error: ' + error.message);
            }
        }
    </script>
</body>
</html>'''

# Create metrics.html
metrics_html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Metrics</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
        <div class="container">
            <a class="navbar-brand" href="/">Sentiment Analyzer</a>
            <div class="navbar-nav ms-auto">
                <a class="nav-link" href="/">Home</a>
                <a class="nav-link" href="/batch">Batch</a>
                <a class="nav-link active" href="/metrics">Metrics</a>
            </div>
        </div>
    </nav>

    <div class="container mt-5">
        <h1 class="text-center mb-4">Model Performance Metrics</h1>
        
        <div class="row">
            <div class="col-md-3">
                <div class="card shadow text-center">
                    <div class="card-body">
                        <h6>Accuracy</h6>
                        <h3 class="text-primary">{{ (metrics.accuracy * 100)|round(1) }}%</h3>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card shadow text-center">
                    <div class="card-body">
                        <h6>Precision</h6>
                        <h3 class="text-success">{{ (metrics.precision * 100)|round(1) }}%</h3>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card shadow text-center">
                    <div class="card-body">
                        <h6>Recall</h6>
                        <h3 class="text-info">{{ (metrics.recall * 100)|round(1) }}%</h3>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card shadow text-center">
                    <div class="card-body">
                        <h6>F1 Score</h6>
                        <h3 class="text-warning">{{ (metrics.f1_score * 100)|round(1) }}%</h3>
                    </div>
                </div>
            </div>
        </div>

        {% if has_cm %}
        <div class="row mt-5">
            <div class="col-lg-8 mx-auto">
                <div class="card shadow">
                    <div class="card-body text-center">
                        <h5>Confusion Matrix</h5>
                        <img src="/confusion-matrix" class="img-fluid" alt="Confusion Matrix">
                    </div>
                </div>
            </div>
        </div>
        {% endif %}
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>'''

# Create style.css
style_css = '''body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background-color: #f8f9fa;
}

.card {
    border-radius: 15px;
    border: none;
}

.btn-primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border: none;
}

.progress {
    height: 25px;
    border-radius: 10px;
}
'''

# Create main.js
main_js = '''console.log("Sentiment Analysis App Loaded");'''

# Write files
files_to_create = {
    os.path.join(SRC_DIR, 'templates', 'index.html'): index_html,
    os.path.join(SRC_DIR, 'templates', 'batch.html'): batch_html,
    os.path.join(SRC_DIR, 'templates', 'metrics.html'): metrics_html,
    os.path.join(SRC_DIR, 'static', 'css', 'style.css'): style_css,
    os.path.join(SRC_DIR, 'static', 'js', 'main.js'): main_js,
}

print("\n📝 Creating files...")
for filepath, content in files_to_create.items():
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✅ Created: {filepath}")

print("\n" + "="*60)
print("✅ ALL FILES CREATED SUCCESSFULLY!")
print("="*60)
print("\n🚀 Now run: python src/app.py")
print("🌐 Then visit: http://localhost:5001")
print("="*60)