from flask import Flask, send_from_directory, render_template_string
import os

app = Flask(__name__)

@app.route('/')
def index():
    """Serve halaman dashboard guru"""
    try:
        # Baca file guru.html dari folder yang sama
        with open('guru.html', 'r', encoding='utf-8') as f:
            content = f.read()
        return render_template_string(content)
    except FileNotFoundError:
        return """
        <h1>Error: guru.html tidak ditemukan!</h1>
        <p>Pastikan file guru.html ada di folder client2/</p>
        """, 404

@app.route('/health')
def health():
    """Health check endpoint"""
    return {'status': 'running', 'client': 'guru', 'port': 5002}

if __name__ == '__main__':
    print("="*60)
    print("👨‍🏫 CLIENT 2 - DASHBOARD GURU")
    print("="*60)
    print("📍 URL: http://localhost:5002")
    print("📊 Dashboard monitoring & approval sistem")
    print("🔗 Terhubung ke Central Server: http://localhost:5000")
    print("="*60)
    
    app.run(
        host='0.0.0.0',  # Bisa diakses dari device lain di network
        port=5002,
        debug=True,
        use_reloader=False  # Hindari double load
    )