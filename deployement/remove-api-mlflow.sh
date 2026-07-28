#!/usr/bin/env bash
# Retrait de l'ancien vhost api.mlflow.jordan-s.org (remplace par mlflow.jordan-s.org)
set -euo pipefail

OLD="api.mlflow.jordan-s.org"
NEW="mlflow.jordan-s.org"
AVAIL="/etc/nginx/sites-available"
ENABLED="/etc/nginx/sites-enabled"
STAMP="$(date +%Y%m%d-%H%M%S)"
BACKUP="/root/nginx-backup-$STAMP"

echo "### 0. Garde-fou : le nouveau vhost doit repondre avant qu'on supprime l'ancien"
if [ ! -f "$ENABLED/$NEW" ] && [ ! -L "$ENABLED/$NEW" ]; then
  echo "!!! $NEW n'est pas active — abandon"
  exit 1
fi
code=$(curl -s -o /dev/null -w "%{http_code}" "https://$NEW" || echo 000)
if [ "$code" != "401" ]; then
  echo "!!! https://$NEW renvoie $code au lieu de 401 — abandon, on ne supprime rien"
  exit 1
fi
echo "    ok — $NEW repond 401 (basic-auth active)"

echo "### 1. Sauvegarde -> $BACKUP"
mkdir -p "$BACKUP"
cp -a "$AVAIL/$OLD" "$BACKUP/" 2>/dev/null || true
echo "    ok"

echo "### 2. Desactivation + suppression du vhost $OLD"
rm -f "$ENABLED/$OLD"
rm -f "$AVAIL/$OLD"
echo "    ok"

echo "### 3. Test config + reload"
if ! nginx -t; then
  echo "!!! ECHEC — restauration"
  cp -a "$BACKUP/$OLD" "$AVAIL/$OLD"
  ln -sf "$AVAIL/$OLD" "$ENABLED/$OLD"
  nginx -t && systemctl reload nginx
  exit 1
fi
systemctl reload nginx
echo "    ok"

echo "### 4. Suppression du certificat Let's Encrypt de $OLD"
if [ -d "/etc/letsencrypt/live/$OLD" ]; then
  certbot delete --cert-name "$OLD" --non-interactive
  echo "    ok — certificat supprime (plus de renouvellement inutile)"
else
  echo "    aucun certificat a supprimer"
fi

echo
echo "=== TERMINE ==="
echo "Backup : $BACKUP"
echo "Certificats encore geres par certbot :"
certbot certificates 2>/dev/null | grep -E "Certificate Name|Expiry" | sed 's/^/  /'
