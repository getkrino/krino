#!/bin/bash
aws s3 ls s3://krino-models-bucket/ --region us-east-1
echo "==="
aws s3 ls s3://krino-models-bucket/deberta-nli-onnx/ --region us-east-1
echo "==="
aws s3 ls s3://krino-models-bucket/distilbert-nli-onnx-quantized/ --region us-east-1
