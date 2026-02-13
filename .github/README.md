# GitHub Workflows

## create-main-branch.yml

This workflow creates the `main` branch for the repository.

### Usage

1. Go to the Actions tab in the GitHub repository
2. Select "Create Main Branch" from the workflows list
3. Click "Run workflow"
4. (Optional) Specify a base branch, or leave empty to use the default branch
5. The workflow will create and push the `main` branch

### After Running

Once the main branch is created, you can set it as the default branch:

1. Go to **Settings** → **Branches**
2. Under "Default branch", click the switch icon
3. Select `main` from the dropdown
4. Click "Update" and confirm

### Note

This workflow will skip execution if the `main` branch already exists on the remote, preventing accidental overwrites.
