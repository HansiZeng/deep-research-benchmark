#!/bin/bash
# check_servers.sh — test all retrieval servers
# Usage: bash check_servers.sh

QUERY="What is the capital of France?"

# Format: "label host port"
SERVERS=(
  "E5_trqa-wiki          gpu018          8001"
  "E5_trec-rag           gpu022          8041"
  "E5_trqa-ecommerce     gpu016          8011"
  "E5_browsecomp-plus    gpu021          8021"
  "E5_wiki-18            gypsum-gpu176   8031"
  "BM25_trqa-wiki        gypsum-gpu126   8002"
  "BM25_trqa-ecommerce   gypsum-gpu126   8012"
  "BM25_browsecomp       gypsum-gpu127   8022"
  "BM25_wiki-18          cpu068          8032"
  "BM25_trec-rag         gypsum-gpu144   8042"
  "Proxy_trqa-wiki       gypsum-gpu116   8003"
  "Proxy_trqa-ecommerce  gypsum-gpu094   8013"
  "Proxy_browsecomp      gypsum-gpu124   8023"
  "Proxy_wiki-18         gypsum-gpu144   8033"
  "Proxy_trec-rag        gypsum-gpu145   8043"
)

PENDING_SERVERS=(
  "E5_trec-rag  ???  8041"
)

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

check_retrieve() {
  local label=$1 host=$2 port=$3
  local url="http://${host}:${port}/retrieve"
  local payload="{\"queries\": [\"${QUERY}\"], \"topk\": 3}"
  local response
  response=$(curl -sf --max-time 30 -X POST "$url" \
    -H "Content-Type: application/json" \
    -d "$payload" 2>&1)
  if [ $? -eq 0 ] && echo "$response" | python3 -c "import sys,json; d=json.load(sys.stdin); assert len(d['result'][0])>0" 2>/dev/null; then
    local first_id
    first_id=$(echo "$response" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['result'][0][0].get('id','?'))" 2>/dev/null)
    echo -e "${GREEN}[OK]${NC}  ${label} (${host}:${port}) — top doc id: ${first_id}"
  else
    echo -e "${RED}[FAIL]${NC} ${label} (${host}:${port})"
    echo "       $(echo "$response" | head -c 200)"
  fi
}

echo "============================================"
echo " Deep Research Benchmark — Server Status"
echo "============================================"
echo ""
for entry in "${SERVERS[@]}"; do
  read -r label host port <<< "$entry"
  check_retrieve "$label" "$host" "$port"
done

echo ""
echo "--- PENDING ---"
for entry in "${PENDING_SERVERS[@]}"; do
  read -r label host port <<< "$entry"
  echo -e "${YELLOW}[PEND]${NC} ${label} (port ${port})"
done
