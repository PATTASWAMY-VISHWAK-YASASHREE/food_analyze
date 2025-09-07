#!/bin/bash

# Food Analyzer - Google Cloud Deployment Script
# This script helps deploy the Food Analyzer to Google Cloud Run

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if required tools are installed
check_requirements() {
    print_status "Checking requirements..."
    
    if ! command -v gcloud &> /dev/null; then
        print_error "Google Cloud SDK is not installed. Please install it from: https://cloud.google.com/sdk/docs/install"
        exit 1
    fi
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install it from: https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    print_status "All requirements satisfied."
}

# Set up Google Cloud project
setup_gcloud() {
    print_status "Setting up Google Cloud..."
    
    # Check if user is logged in
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
        print_warning "Not logged in to Google Cloud. Running 'gcloud auth login'..."
        gcloud auth login
    fi
    
    # Get current project
    PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
    
    if [ -z "$PROJECT_ID" ]; then
        print_warning "No project set. Please set your project ID:"
        read -p "Enter your Google Cloud Project ID: " PROJECT_ID
        gcloud config set project $PROJECT_ID
    fi
    
    print_status "Using project: $PROJECT_ID"
    
    # Enable required APIs
    print_status "Enabling required APIs..."
    gcloud services enable cloudbuild.googleapis.com
    gcloud services enable run.googleapis.com
    gcloud services enable containerregistry.googleapis.com
}

# Build and deploy
deploy() {
    print_status "Starting deployment..."
    
    # Submit build to Cloud Build
    print_status "Submitting build to Cloud Build..."
    gcloud builds submit --config cloudbuild.yaml .
    
    print_status "Deployment completed successfully!"
    
    # Get the service URL
    SERVICE_URL=$(gcloud run services describe food-analyzer --region=us-central1 --format="value(status.url)")
    print_status "Your Food Analyzer API is available at: $SERVICE_URL"
    print_status "API Documentation: $SERVICE_URL/docs"
    print_status "Health Check: $SERVICE_URL/health"
}

# Test local Docker build
test_local() {
    print_status "Testing local Docker build..."
    
    # Build the Docker image
    docker build -t food-analyzer-local .
    
    print_status "Docker image built successfully!"
    print_status "To run locally: docker run -p 8080:8080 food-analyzer-local"
}

# Main function
main() {
    echo "==================================="
    echo "   Food Analyzer Deployment Tool   "
    echo "==================================="
    echo
    
    case "${1:-deploy}" in
        "check")
            check_requirements
            ;;
        "setup")
            check_requirements
            setup_gcloud
            ;;
        "test")
            check_requirements
            test_local
            ;;
        "deploy")
            check_requirements
            setup_gcloud
            deploy
            ;;
        *)
            echo "Usage: $0 {check|setup|test|deploy}"
            echo
            echo "Commands:"
            echo "  check  - Check if all requirements are installed"
            echo "  setup  - Set up Google Cloud project and enable APIs"
            echo "  test   - Test local Docker build"
            echo "  deploy - Deploy to Google Cloud Run (default)"
            echo
            exit 1
            ;;
    esac
}

main "$@"