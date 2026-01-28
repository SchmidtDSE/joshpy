#!/bin/bash

set -e

echo "Fetching Josh fat jars..."
echo ""

mkdir -p jar

# Function to get hash from jar
get_hash() {
    local jar_path=$1
    if [ -f "$jar_path" ]; then
        sha256sum "$jar_path" 2>/dev/null | cut -d' ' -f1 || echo "unknown"
    else
        echo "not installed"
    fi
}

# Function to get version from jar (for display purposes)
get_version() {
    local jar_path=$1
    if [ -f "$jar_path" ]; then
        java -jar "$jar_path" --version 2>&1 | head -1 | tr -d '\n' || echo "unknown"
    else
        echo "not installed"
    fi
}

# Function to download and report changes
download_jar() {
    local name=$1
    local url=$2
    local jar_path=$3

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Downloading: $name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Get old hash and version
    old_hash=$(get_hash "$jar_path")
    old_version=$(get_version "$jar_path")

    if [ "$old_hash" != "not installed" ]; then
        echo "Current version: $old_version"
        echo "Current hash: ${old_hash:0:16}..."
    else
        echo "Current: Not installed"
    fi

    # Download
    if wget -q -O "$jar_path" "$url"; then
        # Get new hash and version
        new_hash=$(get_hash "$jar_path")
        new_version=$(get_version "$jar_path")
        echo "New version: $new_version"
        echo "New hash: ${new_hash:0:16}..."

        # Report status based on hash comparison
        if [ "$old_hash" = "not installed" ]; then
            echo "Status: Installed $name (v$new_version)"
        elif [ "$old_hash" = "$new_hash" ]; then
            echo "Status: Already up to date (no changes detected)"
        else
            echo "Status: Updated jar (hash changed)"
            if [ "$old_version" != "$new_version" ]; then
                echo "         Version changed: v$old_version -> v$new_version"
            else
                echo "         Version unchanged (v$new_version), but jar contents have been updated."
            fi
        fi
    else
        echo "Status: Download failed"
        exit 1
    fi
    echo ""
}

# Download both jars
download_jar "Development jar" "https://joshsim.org/dist/dev/joshsim-fat.jar" "jar/joshsim-fat-dev.jar"
download_jar "Production jar" "https://joshsim.org/dist/main/joshsim-fat.jar" "jar/joshsim-fat-prod.jar"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "All jars downloaded successfully"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
