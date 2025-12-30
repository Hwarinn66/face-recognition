from flask import Flask, send_from_directory, render_template_string
import os

app = Flask(__name__)

@app.route('/')
def index():
    """Serve halaman portal siswa"""
    try:
        # Baca file siswa.html dari folder yang sama
        with open('siswa.html', 'r', encoding='utf-8') as f:
            content = f.read()
        return render_template_string(content)
    except FileNotFoundError:
        return """
        <h1>Error: siswa.html tidak ditemukan!</h1>
        <p>Pastikan file siswa.html ada di folder client3/</p>
        """, 404

@app.route('/health')
def health():
    """Health check endpoint"""
    return {'status': 'running', 'client': 'siswa', 'port': 5003}

if __name__ == '__main__':
    print("="*60)
    print("🎓 CLIENT 3 - PORTAL SISWA")
    print("="*60)
    print("📍 URL: http://localhost:5003")
    print("📱 Portal registrasi mandiri untuk siswa")
    print("🔗 Terhubung ke Central Server: http://localhost:5000")
    print("="*60)
    
    app.run(
        host='0.0.0.0',  # Bisa diakses dari device lain di network
        port=5003,
        debug=True,
        use_reloader=False  # Hindari double load
    )