#!/bin/bash
set -e
CMD_ID="${1:-}"
INSTANCE="${2:-}"
REGION="${AWS_REGION:-us-east-1}"

if [[ -z "$CMD_ID" || -z "$INSTANCE" ]]; then
    echo "Usage: $0 <command-id> <instance-id>"
    exit 1
fi

echo "=== STATUS ==="
aws ssm get-command-invocation --command-id "$CMD_ID" --instance-id "$INSTANCE" --region "$REGION" --query "Status" --output text

echo
echo "=== STDOUT (last 4KB) ==="
aws ssm get-command-invocation --command-id "$CMD_ID" --instance-id "$INSTANCE" --region "$REGION" --query "StandardOutputContent" --output text | tr -d '\r' | tail -c 4096

echo
echo "=== STDERR ==="
aws ssm get-command-invocation --command-id "$CMD_ID" --instance-id "$INSTANCE" --region "$REGION" --query "StandardErrorContent" --output text
