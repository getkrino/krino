# Krino API Deployment Guide

This guide covers deploying the Krino API service to production.

## Prerequisites

- Rust 1.93+ (for building)
- Docker (for containerized deployment)
- ONNX model files (download with scripts)
- AWS account (for cloud deployment)

## Local Development

### Building from Source

```bash
# Build the workspace
cargo build --release --workspace

# Run the API server
./target/release/krino-api

# Server will start on http://localhost:8080
```

### Configuration

The API loads configuration from `krino-api.toml` and environment variables with the `KRINO_` prefix.

**Example: Override port via environment**
```bash
export KRINO_SERVER__PORT=9000
export KRINO_LOGGING__LEVEL=debug
./target/release/krino-api
```

### Testing

```bash
# Run unit tests
cargo test --package krino-api

# Run integration tests (requires running server)
./scripts/test_api.sh

# Test with curl
curl http://localhost:8080/health
```

## Docker Deployment

### Building the Docker Image

```bash
# Build image
docker build -t krino-api:latest .

# Tag for registry
docker tag krino-api:latest your-registry/krino-api:v0.8.2
```

### Running with Docker

```bash
# Run with default config
docker run -p 8080:8080 krino-api:latest

# Run with environment overrides
docker run -p 8080:8080 \
  -e KRINO_SERVER__PORT=8080 \
  -e KRINO_LOGGING__FORMAT=json \
  -e KRINO_AUTH__API_KEYS__0__KEY=your-production-key \
  krino-api:latest

# Run with volume-mounted config
docker run -p 8080:8080 \
  -v $(pwd)/krino-api.prod.toml:/opt/krino/krino-api.toml:ro \
  -v $(pwd)/models:/opt/krino/models:ro \
  krino-api:latest
```

### Docker Compose

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## AWS Deployment

### Option 1: Single EC2 Instance

**Instance Type:** `c7a.xlarge` (4 vCPU, 8 GiB RAM, AVX-512 support)

**1. Launch Instance**

```bash
aws ec2 run-instances \
    --image-id ami-0c02fb55956c7d316 \
    --instance-type c7a.xlarge \
    --key-name krino-prod \
    --security-group-ids sg-xxxxxxxxx \
    --subnet-id subnet-xxxxxxxxx \
    --iam-instance-profile Name=krino-ec2-role \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=krino-api-prod}]'
```

**2. Security Group Rules**

- Allow inbound TCP 8080 from ALB security group
- Allow inbound TCP 22 from your IP (for SSH access)
- Allow outbound all

**3. Install Docker and Deploy**

```bash
# SSH into instance
ssh -i krino-prod.pem ec2-user@<instance-ip>

# Install Docker
sudo yum update -y
sudo yum install -y docker
sudo systemctl start docker
sudo usermod -aG docker ec2-user

# Login to ECR
aws ecr get-login-password --region us-east-1 | \
    docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com

# Pull and run
docker pull <account>.dkr.ecr.us-east-1.amazonaws.com/krino-api:latest
docker run -d \
    --name krino-api \
    -p 8080:8080 \
    --restart unless-stopped \
    -e KRINO_AUTH__API_KEYS__0__KEY=$PROD_API_KEY \
    <account>.dkr.ecr.us-east-1.amazonaws.com/krino-api:latest
```

**4. Application Load Balancer**

```bash
# Create target group
aws elbv2 create-target-group \
    --name krino-api-tg \
    --protocol HTTP \
    --port 8080 \
    --vpc-id vpc-xxxxxxxxx \
    --health-check-path /health \
    --health-check-interval-seconds 30 \
    --health-check-timeout-seconds 5 \
    --healthy-threshold-count 2 \
    --unhealthy-threshold-count 3 \
    --target-type instance

# Register instance
aws elbv2 register-targets \
    --target-group-arn arn:aws:elasticloadbalancing:... \
    --targets Id=i-xxxxxxxxx

# Create HTTPS listener (assuming ALB exists)
aws elbv2 create-listener \
    --load-balancer-arn arn:aws:elasticloadbalancing:... \
    --protocol HTTPS \
    --port 443 \
    --certificates CertificateArn=arn:aws:acm:... \
    --default-actions Type=forward,TargetGroupArn=arn:aws:elasticloadbalancing:...
```

### Option 2: ECS Fargate

**Task Definition:**

```json
{
  "family": "krino-api",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "containerDefinitions": [
    {
      "name": "krino-api",
      "image": "<account>.dkr.ecr.us-east-1.amazonaws.com/krino-api:latest",
      "portMappings": [
        {
          "containerPort": 8080,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "KRINO_LOGGING__FORMAT",
          "value": "json"
        },
        {
          "name": "KRINO_LOGGING__LEVEL",
          "value": "info"
        }
      ],
      "secrets": [
        {
          "name": "KRINO_AUTH__API_KEYS__0__KEY",
          "valueFrom": "arn:aws:secretsmanager:us-east-1:...:secret:krino-api-key"
        }
      ],
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8080/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      },
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/krino-api",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

**Create Service:**

```bash
aws ecs create-service \
    --cluster krino-production \
    --service-name krino-api \
    --task-definition krino-api:1 \
    --desired-count 2 \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx,subnet-yyy],securityGroups=[sg-xxx],assignPublicIp=DISABLED}" \
    --load-balancers "targetGroupArn=arn:aws:elasticloadbalancing:...,containerName=krino-api,containerPort=8080"
```

## CI/CD Pipeline

### GitHub Actions

Create `.github/workflows/deploy.yml`:

```yaml
name: Build and Deploy Krino API

on:
  push:
    branches: [main]
    paths:
      - 'krino-api/**'
      - 'krino-engine/**'
      - 'Cargo.toml'
      - 'Dockerfile'

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    permissions:
      id-token: write
      contents: read

    steps:
      - uses: actions/checkout@v4

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::${{ secrets.AWS_ACCOUNT_ID }}:role/github-actions-deploy
          aws-region: us-east-1

      - name: Login to Amazon ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v2

      - name: Build, tag, and push image
        env:
          ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
          ECR_REPOSITORY: krino-api
          IMAGE_TAG: ${{ github.sha }}
        run: |
          docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG .
          docker tag $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG $ECR_REGISTRY/$ECR_REPOSITORY:latest
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:latest

      - name: Deploy to EC2 (via SSM)
        env:
          INSTANCE_ID: ${{ secrets.EC2_INSTANCE_ID }}
          ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
        run: |
          aws ssm send-command \
            --instance-ids $INSTANCE_ID \
            --document-name "AWS-RunShellScript" \
            --parameters 'commands=[
              "aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin '$ECR_REGISTRY'",
              "docker pull '$ECR_REGISTRY'/krino-api:latest",
              "docker stop krino-api || true",
              "docker rm krino-api || true",
              "docker run -d --name krino-api -p 8080:8080 --restart unless-stopped '$ECR_REGISTRY'/krino-api:latest"
            ]'
```

## Monitoring

### CloudWatch Logs

```bash
# Create log group
aws logs create-log-group --log-group-name /aws/krino-api

# Stream logs
aws logs tail /aws/krino-api --follow
```

### Prometheus Metrics

The `/metrics` endpoint exposes Prometheus-compatible metrics:

```bash
# Scrape metrics
curl http://localhost:8080/metrics

# Sample Prometheus config
scrape_configs:
  - job_name: 'krino-api'
    static_configs:
      - targets: ['krino-api.internal:8080']
    metrics_path: '/metrics'
```

**Key Metrics:**
- `http_requests_total` - Total requests by endpoint and status
- `http_request_duration_seconds` - Latency histogram
- `evaluations_total` - Total evaluations by type
- `evaluation_duration_seconds` - Evaluation latency
- `evaluation_claims` - Number of claims per evaluation

## Security Best Practices

### API Keys

1. **Production Keys:** Generate strong keys with sufficient entropy
   ```bash
   openssl rand -hex 32
   ```

2. **Key Rotation:** Store keys in AWS Secrets Manager and rotate regularly
   ```bash
   aws secretsmanager create-secret \
       --name krino-api-keys \
       --secret-string '{"prod_key": "sk-krino-..."}'
   ```

3. **Environment Variables:** Never commit keys to version control

### Network Security

1. **HTTPS Only:** Terminate TLS at the ALB with ACM certificates
2. **VPC:** Deploy in private subnets with NAT gateway for outbound
3. **Security Groups:** Restrict access to ALB only
4. **WAF:** Add AWS WAF rules for DDoS protection

### CORS Configuration

For production, tighten CORS in `server.rs`:

```rust
.layer(CorsLayer::new()
    .allow_origin("https://your-domain.com".parse::<HeaderValue>().unwrap())
    .allow_methods([Method::GET, Method::POST])
    .allow_headers([AUTHORIZATION, CONTENT_TYPE])
)
```

## Performance Tuning

### Tokio Runtime

Adjust blocking threads for your instance size:

```toml
[server]
blocking_threads = 4  # Set to number of physical cores
```

### Model Optimization

1. **INT8 Quantization:** Use quantized ONNX models (50% size reduction)
2. **AVX-512:** Deploy on c7a instances for 2-3× speedup
3. **Pre-filtering:** Use CandleEmbeddingBackend for 180× fewer NLI calls

### Scaling

**Vertical Scaling:**
- c7a.xlarge: ~130 evals/min
- c7a.2xlarge: ~260 evals/min
- c7a.4xlarge: ~520 evals/min

**Horizontal Scaling:**
- Deploy multiple instances behind ALB
- Use round-robin or least-connections routing
- Future: Implement model pooling for concurrent inference

## Troubleshooting

### Model Not Found

```
Error: Models not found at path: models/deberta-nli-onnx
```

**Solution:** Download ONNX model
```bash
cd scripts && uv run export_deberta_onnx.py
```

### Out of Memory

```
Error: ONNX Runtime: BFC arena allocation failed
```

**Solution:** Reduce `blocking_threads` or increase instance RAM

### Slow Inference

Check CPU capabilities in logs:
```
INFO CPU capabilities avx512f=false
```

**Solution:** Deploy on AVX-512 capable instance (c7a family)

### Permission Denied

```
Error: Permission denied (os error 13)
```

**Solution:** Ensure non-root user in Docker has correct permissions
```dockerfile
RUN chown -R krino:krino /opt/krino
USER krino
```

## Support

- **Issues:** https://github.com/smithjustinm/krino/issues
- **Documentation:** See `krino-api/README.md`
- **Contact:** justin@krino.dev
