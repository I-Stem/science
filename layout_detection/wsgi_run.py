from application import app
from werkzeug.middleware.proxy_fix import ProxyFix

app.wsgi_app = ProxyFix(app.wsgi_app)
if __name__ == "__main__":
      app.run()
