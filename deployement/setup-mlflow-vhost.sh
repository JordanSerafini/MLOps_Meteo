#!/usr/bin/env bash
# Setup vhost mlflow.jordan-s.org + durcissement default_server 443
# Idempotent — relançable sans dommage.
set -euo pipefail

DOMAIN="mlflow.jordan-s.org"
BACKEND="http://192.168.1.36:5000"
HTPASSWD="/etc/nginx/.htpasswd-mlflow"
AVAIL="/etc/nginx/sites-available"
ENABLED="/etc/nginx/sites-enabled"
STAMP="$(date +%Y%m%d-%H%M%S)"
BACKUP="/root/nginx-backup-$STAMP"

echo "### 0. Sauvegarde de la conf nginx -> $BACKUP"
mkdir -p "$BACKUP"
cp -a /etc/nginx/sites-available "$BACKUP/" 2>/dev/null || true

rollback() {
  echo "!!! ECHEC — restauration de la conf depuis $BACKUP"
  rm -f "$ENABLED/$DOMAIN"
  [ -f "$BACKUP/sites-available/00-default-deny" ] \
    && cp -a "$BACKUP/sites-available/00-default-deny" "$AVAIL/00-default-deny"
  nginx -t && systemctl reload nginx
  exit 1
}

# ─────────────────────────────────────────────────────────────
echo "### 1. Verification du fichier basic-auth"
if [ ! -f "$HTPASSWD" ]; then
  echo "!!! $HTPASSWD absent — abandon (MLflow n'a pas d'auth native, on ne l'expose pas nu)"
  exit 1
fi
# Le piege n2 du README : les workers nginx (www-data) doivent pouvoir le lire
chown root:www-data "$HTPASSWD"
chmod 640 "$HTPASSWD"
echo "    ok — $(ls -l "$HTPASSWD")"

# ─────────────────────────────────────────────────────────────
echo "### 2. Ecriture du vhost $DOMAIN (HTTP ; certbot ajoutera le SSL)"
cat > "$AVAIL/$DOMAIN" <<EOF
server {
    listen 80;
    listen [::]:80;
    server_name $DOMAIN;

    location / {
        auth_basic "MLflow restreint";
        auth_basic_user_file $HTPASSWD;

        proxy_pass $BACKEND;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_http_version 1.1;
        client_max_body_size 200m;
    }
}
EOF
ln -sf "$AVAIL/$DOMAIN" "$ENABLED/$DOMAIN"
echo "    ok — vhost ecrit et active"

echo "### 3. Test config + reload"
nginx -t || rollback
systemctl reload nginx
echo "    ok"

# ─────────────────────────────────────────────────────────────
echo "### 4. Certificat Let's Encrypt (HTTP-01) pour $DOMAIN"
if [ -d "/etc/letsencrypt/live/$DOMAIN" ]; then
  echo "    certificat deja present — on saute certbot"
else
  certbot --nginx -d "$DOMAIN" \
      --non-interactive --agree-tos \
      -m jordan@solution-logique.fr \
      --redirect || rollback
  echo "    ok — certificat obtenu et vhost passe en HTTPS"
fi

# ─────────────────────────────────────────────────────────────
echo "### 5. Durcissement : default_server sur 443 (le trou identifie)"
# Sans ca, tout nom DNS pointant vers l'IP tombe sur le 1er vhost 443
# par ordre alphabetique (= airflow.jordan-s.org aujourd'hui).
CRT="/etc/ssl/certs/nginx-default.crt"
KEY="/etc/ssl/private/nginx-default.key"

if [ ! -f "$CRT" ]; then
  echo "    generation d'un certificat auto-signe (jamais presente a un vrai client)"
  openssl req -x509 -nodes -days 3650 -newkey rsa:2048 \
    -keyout "$KEY" -out "$CRT" -subj "/CN=invalid" 2>/dev/null
  chmod 600 "$KEY"
fi

if grep -q "default_server" "$AVAIL/00-default-deny" && grep -q "443" "$AVAIL/00-default-deny"; then
  echo "    default_server 443 deja present — on saute"
else
  cat >> "$AVAIL/00-default-deny" <<EOF

# Ajoute le $STAMP — ferme le SNI par defaut en HTTPS.
# Sans ce bloc, un nom DNS non prevu pointant vers cette IP tombe sur le
# premier vhost 443 par ordre alphabetique et expose son UI.
server {
    listen 443 ssl default_server;
    listen [::]:443 ssl default_server;
    server_name _;

    ssl_certificate     $CRT;
    ssl_certificate_key $KEY;

    return 444;
}
EOF
  echo "    bloc ajoute"
fi

echo "### 6. Test config final + reload"
nginx -t || rollback
systemctl reload nginx
echo "    ok"

# ─────────────────────────────────────────────────────────────
echo
echo "=== TERMINE ==="
echo "Backup de la conf precedente : $BACKUP"
echo "Vhosts actifs :"
ls "$ENABLED"
