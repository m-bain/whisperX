#!/bin/bash

# WhisperX Docker Deployment Script
# This script helps you easily deploy WhisperX with Docker

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    print_success "Docker and Docker Compose are installed"
}

# Function to check GPU support
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        print_success "NVIDIA GPU detected"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
        return 0
    else
        print_warning "No NVIDIA GPU detected or nvidia-smi not available"
        return 1
    fi
}

# Function to setup directories
setup_directories() {
    print_info "Setting up directories..."
    mkdir -p input output models
    print_success "Directories created: input/, output/, models/"
}

# Function to setup environment
setup_env() {
    if [ ! -f .env ]; then
        print_info "Creating .env file from template..."
        cp .env.template .env
        print_warning "Please edit .env file to add your Hugging Face token for speaker diarization"
        print_info "Get your token from: https://huggingface.co/settings/tokens"
    else
        print_info ".env file already exists"
    fi
}

# Function to display menu
show_menu() {
    echo ""
    echo "==================================="
    echo "    WhisperX Docker Deployment"
    echo "==================================="
    echo "1. Deploy with GPU (Recommended)"
    echo "2. Deploy with CPU only"
    echo "3. Deploy Web API with GPU"
    echo "4. Deploy Web API with CPU"
    echo "5. Stop all containers"
    echo "6. View logs"
    echo "7. Transcribe audio file"
    echo "8. Clean up everything"
    echo "9. Exit"
    echo "==================================="
}

# Function to deploy with GPU
deploy_gpu() {
    print_info "Deploying WhisperX with GPU support..."
    docker-compose --profile gpu up -d
    print_success "WhisperX GPU container is running!"
    print_info "Use: docker exec whisperx-gpu python -m whisperx /input/audio.wav --output_dir /output"
}

# Function to deploy with CPU
deploy_cpu() {
    print_info "Deploying WhisperX with CPU only..."
    docker-compose --profile cpu up -d
    print_success "WhisperX CPU container is running!"
    print_info "Use: docker exec whisperx-cpu python -m whisperx /input/audio.wav --output_dir /output --device cpu --compute_type int8"
}

# Function to deploy API with GPU
deploy_api_gpu() {
    print_info "Deploying WhisperX Web API with GPU support..."
    docker-compose --profile api-gpu up -d
    print_success "WhisperX Web API (GPU) is running!"
    print_info "Access the web interface at: http://localhost:8000"
}

# Function to deploy API with CPU
deploy_api_cpu() {
    print_info "Deploying WhisperX Web API with CPU only..."
    docker-compose --profile api-cpu up -d
    print_success "WhisperX Web API (CPU) is running!"
    print_info "Access the web interface at: http://localhost:8000"
}

# Function to stop containers
stop_containers() {
    print_info "Stopping all WhisperX containers..."
    docker-compose down
    print_success "All containers stopped"
}

# Function to view logs
view_logs() {
    echo "Select container to view logs:"
    echo "1. GPU container"
    echo "2. CPU container"
    echo "3. API GPU container"
    echo "4. API CPU container"
    read -p "Enter choice (1-4): " log_choice
    
    case $log_choice in
        1) docker-compose logs -f whisperx-gpu ;;
        2) docker-compose logs -f whisperx-cpu ;;
        3) docker-compose logs -f whisperx-api-gpu ;;
        4) docker-compose logs -f whisperx-api-cpu ;;
        *) print_error "Invalid choice" ;;
    esac
}

# Function to transcribe audio
transcribe_audio() {
    # List audio files in input directory
    print_info "Audio files in input/ directory:"
    if ls input/*.{wav,mp3,m4a,flac,ogg} 1> /dev/null 2>&1; then
        ls input/*.{wav,mp3,m4a,flac,ogg} 2>/dev/null | sed 's|input/||'
    else
        print_warning "No audio files found in input/ directory"
        print_info "Please place audio files (.wav, .mp3, .m4a, .flac, .ogg) in the input/ directory"
        return
    fi
    
    echo ""
    read -p "Enter audio filename (from input/ directory): " audio_file
    
    if [ ! -f "input/$audio_file" ]; then
        print_error "File input/$audio_file not found"
        return
    fi
    
    echo "Select transcription mode:"
    echo "1. GPU (fast)"
    echo "2. CPU (slower)"
    read -p "Enter choice (1-2): " mode_choice
    
    echo "Additional options:"
    read -p "Enable speaker diarization? (y/n): " diarize
    read -p "Model (tiny/base/small/medium/large-v2): " model
    
    # Set default model
    if [ -z "$model" ]; then
        model="large-v2"
    fi
    
    # Build command
    cmd="python -m whisperx /input/$audio_file --output_dir /output --model $model"
    
    if [ "$diarize" = "y" ] || [ "$diarize" = "Y" ]; then
        cmd="$cmd --diarize"
    fi
    
    case $mode_choice in
        1)
            container="whisperx-gpu"
            ;;
        2)
            container="whisperx-cpu"
            cmd="$cmd --device cpu --compute_type int8"
            ;;
        *)
            print_error "Invalid choice"
            return
            ;;
    esac
    
    print_info "Running transcription..."
    print_info "Command: docker exec $container $cmd"
    
    if docker exec $container $cmd; then
        print_success "Transcription completed! Check output/ directory for results."
    else
        print_error "Transcription failed. Check container logs for details."
    fi
}

# Function to clean up
cleanup() {
    print_warning "This will remove all containers, images, and cached models!"
    read -p "Are you sure? (y/N): " confirm
    
    if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
        print_info "Stopping containers..."
        docker-compose down
        
        print_info "Removing images..."
        docker rmi $(docker images | grep whisperx | awk '{print $3}') 2>/dev/null || true
        
        print_info "Cleaning up model cache..."
        rm -rf models/*
        
        print_success "Cleanup completed"
    else
        print_info "Cleanup cancelled"
    fi
}

# Main script
main() {
    # Check prerequisites
    check_docker
    
    # Setup directories and environment
    setup_directories
    setup_env
    
    # Check for GPU
    if check_gpu; then
        print_info "GPU mode available"
        GPU_AVAILABLE=true
    else
        print_info "Only CPU mode available"
        GPU_AVAILABLE=false
    fi
    
    # Main menu loop
    while true; do
        show_menu
        read -p "Enter your choice (1-9): " choice
        
        case $choice in
            1)
                if [ "$GPU_AVAILABLE" = true ]; then
                    deploy_gpu
                else
                    print_error "GPU not available. Please use CPU mode."
                fi
                ;;
            2) deploy_cpu ;;
            3)
                if [ "$GPU_AVAILABLE" = true ]; then
                    deploy_api_gpu
                else
                    print_error "GPU not available. Please use CPU mode."
                fi
                ;;
            4) deploy_api_cpu ;;
            5) stop_containers ;;
            6) view_logs ;;
            7) transcribe_audio ;;
            8) cleanup ;;
            9)
                print_info "Goodbye!"
                exit 0
                ;;
            *)
                print_error "Invalid choice. Please try again."
                ;;
        esac
        
        echo ""
        read -p "Press Enter to continue..."
    done
}

# Run main function
main