# GitHub Repo Setup

Manual steps if GitHub CLI is unavailable:

1. open repository settings
2. rename `OmniVision` to `Retina`
3. update the local remote:

```bash
git remote set-url origin https://github.com/sushildalavi/Retina.git
```

4. verify:

```bash
git remote -v
git ls-remote origin HEAD
```

