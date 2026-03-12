#!/bin/bash
# Setup-Skript fuer Kalorimetrie-App auf Uberspace
# Ausfuehren nach einem frischen Reset:
#
#   git clone https://github.com/JoniBoi1990/kalorimetrie.git ~/projects/kalorimetrie
#   bash ~/projects/kalorimetrie/setup_uberspace.sh

set -e

APP_DIR="$HOME/projects/kalorimetrie"
PORT=8501

echo "==> Virtualenv anlegen..."
python3 -m venv "$APP_DIR/venv"

echo "==> Dependencies installieren..."
"$APP_DIR/venv/bin/pip" install --upgrade pip -q
"$APP_DIR/venv/bin/pip" install -r "$APP_DIR/requirements.txt" -q

echo "==> supervisord-Config schreiben..."
mkdir -p "$HOME/etc/services.d"
cat > "$HOME/etc/services.d/kalorimetrie.ini" << EOF
[program:kalorimetrie]
directory=$APP_DIR
command=$APP_DIR/venv/bin/streamlit run Waerme2.py --server.address 0.0.0.0 --server.port $PORT
autostart=yes
autorestart=yes
EOF

echo "==> supervisord neu laden..."
supervisorctl reread
supervisorctl update

echo "==> Web-Backend konfigurieren..."
uberspace web backend set / http:$PORT --remove-prefix

echo ""
echo "Fertig! App laeuft unter https://$(hostname -s).uber.space"
echo ""
echo "Nuetzliche Befehle:"
echo "  supervisorctl status kalorimetrie   # Status"
echo "  supervisorctl stop kalorimetrie     # pausieren"
echo "  supervisorctl start kalorimetrie    # starten"
