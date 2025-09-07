# Google Cloud Setup Guide

This guide walks you through setting up the Food Analyzer application on Google Cloud Run with Cloud Build for CI/CD.

## Prerequisites

1. **Google Cloud Account**: You need a Google Cloud account with billing enabled
2. **Google Cloud SDK**: Install from https://cloud.google.com/sdk/docs/install
3. **Docker**: Install from https://docs.docker.com/get-docker/ (for local testing)

## Step-by-Step Setup

### 1. Google Cloud Project Setup

```bash
# Create a new project (optional)
gcloud projects create YOUR_PROJECT_ID --name="Food Analyzer"

# Set the project
gcloud config set project YOUR_PROJECT_ID

# Enable billing (required for Cloud Run)
# This must be done through the Cloud Console at:
# https://console.cloud.google.com/billing
```

### 2. Enable Required APIs

```bash
# Enable necessary Google Cloud APIs
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
```

### 3. Set Up Authentication

```bash
# Login to Google Cloud
gcloud auth login

# Configure Docker for Container Registry
gcloud auth configure-docker
```

### 4. Clone and Deploy

```bash
# Clone the repository
git clone YOUR_REPO_URL
cd food_analyze

# Option 1: Use the automated deployment script
chmod +x deploy.sh
./deploy.sh

# Option 2: Manual deployment
gcloud builds submit --config cloudbuild.yaml .
```

### 5. Configure Environment Variables (Optional)

For enhanced functionality with Hugging Face models:

```bash
# Set environment variables for the deployed service
gcloud run services update food-analyzer \
  --region us-central1 \
  --update-env-vars HUGGINGFACE_TOKEN=your_token_here
```

### 6. Custom Domain (Optional)

```bash
# Verify domain ownership in Cloud Console first
# Then map the domain
gcloud run domain-mappings create \
  --service food-analyzer \
  --domain your-domain.com \
  --region us-central1
```

## Configuration Options

### Cloud Build Configuration

The `cloudbuild.yaml` file can be customized:

```yaml
# Change deployment region
--region us-central1  # or us-east1, europe-west1, etc.

# Adjust memory allocation
--memory 2Gi  # or 1Gi, 4Gi, etc.

# Change CPU allocation  
--cpu 1  # or 2, 4, etc.

# Modify concurrency
--concurrency 80  # requests per instance

# Set timeout
--timeout 300  # seconds
```

### Environment Variables

Available environment variables:

| Variable | Purpose | Default |
|----------|---------|---------|
| `HUGGINGFACE_TOKEN` | Enhanced model access | None |
| `LOG_LEVEL` | Logging verbosity | INFO |
| `PORT` | Application port | 8080 |

### Security Configuration

```bash
# Remove public access (require authentication)
gcloud run services remove-iam-policy-binding food-analyzer \
  --region us-central1 \
  --member="allUsers" \
  --role="roles/run.invoker"

# Add specific users
gcloud run services add-iam-policy-binding food-analyzer \
  --region us-central1 \
  --member="user:email@example.com" \
  --role="roles/run.invoker"
```

## Monitoring and Maintenance

### View Logs

```bash
# View application logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=food-analyzer" --limit 50

# Stream logs in real-time
gcloud logging tail "resource.type=cloud_run_revision AND resource.labels.service_name=food-analyzer"
```

### Monitor Performance

1. Go to [Cloud Console](https://console.cloud.google.com)
2. Navigate to Cloud Run > food-analyzer
3. Click on "Metrics" tab for performance data

### Update Deployment

```bash
# Redeploy with latest code
gcloud builds submit --config cloudbuild.yaml .

# Or use the deployment script
./deploy.sh
```

### Scale Configuration

```bash
# Update scaling settings
gcloud run services update food-analyzer \
  --region us-central1 \
  --min-instances 0 \
  --max-instances 100 \
  --concurrency 80
```

## Costs

Typical costs for Cloud Run:

- **Free Tier**: 2 million requests/month, 400,000 GB-seconds/month
- **Paid Usage**: ~$0.40 per million requests + compute time
- **Memory**: ~$0.0000009 per GB-second
- **CPU**: ~$0.0000024 per vCPU-second

## Troubleshooting

### Common Issues

1. **Build Fails**
   ```bash
   # Check build logs
   gcloud builds list
   gcloud builds log BUILD_ID
   ```

2. **Service Won't Start**
   ```bash
   # Check service logs
   gcloud logging read "resource.type=cloud_run_revision" --limit 10
   ```

3. **Memory Issues**
   ```bash
   # Increase memory allocation
   gcloud run services update food-analyzer \
     --region us-central1 \
     --memory 4Gi
   ```

4. **Timeout Issues**
   ```bash
   # Increase timeout
   gcloud run services update food-analyzer \
     --region us-central1 \
     --timeout 900
   ```

### Local Testing

```bash
# Test Docker build locally
docker build -t food-analyzer-local .
docker run -p 8080:8080 food-analyzer-local

# Test with Docker Compose
docker-compose up --build
```

## Support

- **Google Cloud Documentation**: https://cloud.google.com/run/docs
- **Cloud Build Documentation**: https://cloud.google.com/build/docs
- **GitHub Issues**: Create an issue in this repository for application-specific problems