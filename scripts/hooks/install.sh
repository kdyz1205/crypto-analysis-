#!/bin/sh
# Install git hooks from scripts/hooks/ into .git/hooks/.
# .git/hooks/ is not tracked by git, so the hook itself lives here and
# every developer runs this once to activate it.
#
# Usage:
#   sh scripts/hooks/install.sh

set -e
cd "$(git rev-parse --show-toplevel)"

for hook in scripts/hooks/pre-commit; do
    name="$(basename "$hook")"
    dest=".git/hooks/$name"
    cp "$hook" "$dest"
    chmod +x "$dest"
    echo "installed: $dest"
done

echo
echo "Done. Hooks active:"
ls -1 .git/hooks/ | grep -v '\.sample$'
