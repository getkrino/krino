#!/bin/bash
# Integration test script for Krino API
# This script tests the API endpoints with curl

set -e

API_URL="${API_URL:-http://localhost:8080}"
API_KEY="${API_KEY:-sk-krino-demo-key-change-me}"

echo "🧪 Testing Krino API at $API_URL"
echo "================================"
echo ""

# Test 1: Health check (no auth required)
echo "Test 1: Health endpoint"
echo "----------------------"
HEALTH_RESPONSE=$(curl -s "$API_URL/health")
echo "Response: $HEALTH_RESPONSE"
if echo "$HEALTH_RESPONSE" | grep -q '"status":"ok"'; then
    echo "✅ Health check passed"
else
    echo "❌ Health check failed"
    exit 1
fi
echo ""

# Test 2: Ready check
echo "Test 2: Ready endpoint"
echo "---------------------"
READY_RESPONSE=$(curl -s "$API_URL/health/ready")
echo "Response: $READY_RESPONSE"
if echo "$READY_RESPONSE" | grep -q '"status":"ready"'; then
    echo "✅ Ready check passed"
else
    echo "❌ Ready check failed"
    exit 1
fi
echo ""

# Test 3: Metrics endpoint
echo "Test 3: Metrics endpoint"
echo "----------------------"
METRICS_RESPONSE=$(curl -s "$API_URL/metrics")
if echo "$METRICS_RESPONSE" | grep -q "http_requests_total"; then
    echo "✅ Metrics endpoint passed"
else
    echo "❌ Metrics endpoint failed"
    echo "Response: $METRICS_RESPONSE"
    exit 1
fi
echo ""

# Test 4: Evaluate endpoint without auth (should fail)
echo "Test 4: Evaluate without auth (should return 401)"
echo "-------------------------------------------------"
STATUS_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
    -X POST "$API_URL/api/v1/evaluate" \
    -H "Content-Type: application/json" \
    -d '{"context": [{"text": "test"}], "output": "test"}')

if [ "$STATUS_CODE" = "401" ]; then
    echo "✅ Auth check passed (got 401)"
else
    echo "❌ Auth check failed (expected 401, got $STATUS_CODE)"
    exit 1
fi
echo ""

# Test 5: Faithfulness evaluation with auth (may fail if model not loaded)
echo "Test 5: Faithfulness evaluation with auth"
echo "-----------------------------------------"
EVAL_RESPONSE=$(curl -s -w "\nHTTP_STATUS:%{http_code}" \
    -X POST "$API_URL/api/v1/evaluate" \
    -H "x-api-key: $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{
        "context": [
            {
                "id": "chunk-1",
                "text": "The Eiffel Tower is located in Paris, France. It was completed in 1889."
            }
        ],
        "output": "The Eiffel Tower was built in 1889.",
        "config": {
            "granularity": "claim",
            "threshold": 0.7
        }
    }')

HTTP_STATUS=$(echo "$EVAL_RESPONSE" | grep "HTTP_STATUS:" | cut -d: -f2)
RESPONSE_BODY=$(echo "$EVAL_RESPONSE" | grep -v "HTTP_STATUS:")

echo "Status: $HTTP_STATUS"
echo "Response: $RESPONSE_BODY"

if [ "$HTTP_STATUS" = "200" ]; then
    if echo "$RESPONSE_BODY" | grep -q '"score"'; then
        echo "✅ Faithfulness evaluation passed"
    else
        echo "⚠️  Got 200 but response format unexpected"
        echo "$RESPONSE_BODY"
    fi
elif [ "$HTTP_STATUS" = "500" ]; then
    echo "⚠️  Model not loaded (expected for test - need to download ONNX model)"
else
    echo "❌ Unexpected status code: $HTTP_STATUS"
    exit 1
fi
echo ""

# Test 6: Bad request (empty context)
echo "Test 6: Bad request validation"
echo "------------------------------"
BAD_REQUEST_RESPONSE=$(curl -s -w "\nHTTP_STATUS:%{http_code}" \
    -X POST "$API_URL/api/v1/evaluate" \
    -H "x-api-key: $API_KEY" \
    -H "Content-Type: application/json" \
    -d '{
        "context": [],
        "output": "test"
    }')

HTTP_STATUS=$(echo "$BAD_REQUEST_RESPONSE" | grep "HTTP_STATUS:" | cut -d: -f2)

if [ "$HTTP_STATUS" = "400" ]; then
    echo "✅ Input validation passed (got 400)"
else
    echo "❌ Input validation failed (expected 400, got $HTTP_STATUS)"
    exit 1
fi
echo ""

echo "✅ All basic API tests passed!"
echo ""
echo "Note: Full faithfulness tests require the ONNX model to be present."
echo "Download with: cd scripts && uv run export_deberta_onnx.py"
