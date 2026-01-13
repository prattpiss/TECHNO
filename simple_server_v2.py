"""
🎵 Ultra-Simple HTTP Server for Techno Tools
Minimal server with maximum compatibility for Windows UTF-8 paths
"""

from http.server import HTTPServer, SimpleHTTPRequestHandler
import os
import sys

PORT = 8080

class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-cache')
        super().end_headers()
    
    def log_message(self, format, *args):
        sys.stdout.write(f"[{self.log_date_time_string()}] {format % args}\n")
        sys.stdout.flush()

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Try ports until one works
    for port in [8080, 8888, 9000, 3000, 5500]:
        try:
            server = HTTPServer(('', port), Handler)
            PORT = port
            break
        except OSError:
            continue
    
    print("=" * 60)
    print("🎵 TECHNO TOOLS SERVER (Simple Mode)")
    print("=" * 60)
    print(f"\n✅ Server: http://localhost:{PORT}")
    print(f"📂 Directory: {os.getcwd()}\n")
    print(f"🎨 Open: http://localhost:{PORT}/TECHNOTOOLS.html")
    print(f"\n⚠️  Press Ctrl+C to stop\n")
    print("=" * 60)
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped.")
