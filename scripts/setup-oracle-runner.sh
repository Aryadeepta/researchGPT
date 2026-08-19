#!/bin/bash
set -Eeo pipefail

echo "Setting up Oracle Cloud Runner..."

# Update and install dependencies
sudo apt-get update
sudo apt-get install -y git rsync zstd jq docker.io docker-compose-plugin

# Ensure user is in docker group
sudo usermod -aG docker $USER
echo "Added $USER to docker group. Please log out and back in for changes to take effect."

# Setup directories
mkdir -p $HOME/research-agent-data $HOME/gemini-code-tasks $HOME/gemini-worktrees

echo "Remaining manual steps:"
echo "1. Log out and back in to apply docker group membership."
echo "2. Follow GitHub documentation to register the self-hosted runner."
echo "3. For optional legacy Oracle execution, add explicit labels such as 'oracle', 'arm64', and 'research-control'. Normal runs do not use this runner."
echo "4. Configure GEMINI_API_KEY in the runner environment or secrets."
