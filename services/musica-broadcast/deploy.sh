#!/usr/bin/env bash
# Deploys the Musica settings-broadcast service (ADR-182) as a 2nd-gen Cloud
# Function alongside the ADR-179 Lyria broker.
#
# Public-invokable on purpose: the function does its own Cognitum JWT
# verification, so the gate — not a network ACL — is the boundary.
set -euo pipefail

PROJECT="${MUSICA_BROADCAST_PROJECT:-cognitum-20260110}"
REGION="${MUSICA_BROADCAST_REGION:-us-central1}"
NAME="${MUSICA_BROADCAST_NAME:-musicaBroadcast}"

# The HMAC key that turns an OAuth subject into an opaque public broadcast id.
# Create once, then never rotate casually: rotating it renames every existing
# broadcast.
#   printf '%s' "$(openssl rand -base64 32)" | \
#     gcloud secrets create BROADCAST_ID_SECRET --data-file=- --project "$PROJECT"
gcloud functions deploy "$NAME" \
  --project "$PROJECT" \
  --region "$REGION" \
  --gen2 \
  --runtime nodejs20 \
  --source . \
  --entry-point musicaBroadcast \
  --trigger-http \
  --allow-unauthenticated \
  --set-secrets "BROADCAST_ID_SECRET=BROADCAST_ID_SECRET:latest" \
  --set-env-vars "COGNITUM_ISSUER=https://auth.cognitum.one"

cat <<'NOTE'

Deployed. Two follow-up steps are NOT automated:

1. Firestore TTL policy — reclaims presence documents:
     gcloud firestore fields ttls update expiresAtTimestamp \
       --collection-group=listeners --enable-ttl --project "$PROJECT"
   Leaderboard correctness does not depend on this; the count query filters on
   `seenAt` directly, because TTL deletion can lag by hours. This is storage
   hygiene only.

2. Single-field index on the `listeners` subcollection for `seenAt`, and on
   `musica_broadcasts.listenerCount` (descending) for the directory query.
   Firestore will print the exact `gcloud` command on the first query that
   needs one.
NOTE
