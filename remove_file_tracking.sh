#!/bin/bash

# Check if the file path argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 path/to/your/file"
    exit 1
fi

FILE_PATH=$1

# Check if git-filter-repo is installed
if ! command -v git-filter-repo &> /dev/null; then
    echo "git-filter-repo is not installed. Please install it first."
    exit 1
fi

# Remove the file from the repository history
git filter-repo --path "$FILE_PATH" --invert-paths

# Add the file to .gitignore
echo "$FILE_PATH" >> .gitignore

# Remove the file from the index
git rm --cached "$FILE_PATH"

# Commit the changes
git add .gitignore
git commit -m "Stop tracking $FILE_PATH and add to .gitignore"

# Force push to the remote repository
git push origin --force --all
git push origin --force --tags

echo "Successfully removed $FILE_PATH from Git history and stopped tracking it."
