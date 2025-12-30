from flask import Flask, send_from_directory, render_template_string
import os

app = Flask(__name__)

@app.route('/')
def index():
    """Serve halaman absensi"""
    try:
        # Baca file absensi.html dari folder yang sama
        with open('absensi.html', 'r', encoding='utf-8') as f:
            content = f.read()
        return render_template_string(content)
    except FileNotFoundError:
        return """
        <h1>Error: absensi.html tidak ditemukan!</h1>
        <p>Pastikan file absensi.html ada di folder client1/</p>
        """, 404

@app.route('/health')
def health():
    """Health check endpoint"""
    return {'status': 'running', 'client': 'absensi', 'port': 5001}

if __name__ == '__main__':
    print("="*60)
    print("🚀 CLIENT 1 - ALAT ABSENSI")
    print("="*60)
    print("📍 URL: http://localhost:5001")
    print("📱 Akses dari browser untuk mulai absensi")
    print("🔗 Terhubung ke Central Server: http://localhost:5000")
    print("="*60)
    
    app.run(
        host='0.0.0.0',  # Bisa diakses dari device lain di network
        port=5001,
        debug=True,
        use_reloader=False  # Hindari double load
    )