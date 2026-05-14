#!/usr/bin/env bash
set -euo pipefail

OWNER="guderianXu"
REPO="PlanetaryFeatureMatch"
REMOTE_URL="https://github.com/${OWNER}/${REPO}.git"
API_URL="https://api.github.com/user/repos"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "error: current directory is not a git repository" >&2
    exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
    echo "error: curl is required" >&2
    exit 1
fi

if ! command -v base64 >/dev/null 2>&1; then
    echo "error: base64 is required" >&2
    exit 1
fi

current_branch="$(git branch --show-current)"
if [[ "${current_branch}" != "main" ]]; then
    echo "error: current branch is '${current_branch}', expected 'main'" >&2
    exit 1
fi

if ! git rev-parse --verify HEAD >/dev/null 2>&1; then
    echo "error: no commit exists yet" >&2
    exit 1
fi

read -rsp "GitHub token: " GH_TOKEN
printf '\n'

cleanup() {
    unset GH_TOKEN
}
trap cleanup EXIT

create_body="$(mktemp)"
create_code="$(curl -sS -o "${create_body}" -w '%{http_code}' \
    -X POST "${API_URL}" \
    -H "Authorization: Bearer ${GH_TOKEN}" \
    -H "Accept: application/vnd.github+json" \
    -H "Content-Type: application/json" \
    -d "{\"name\":\"${REPO}\",\"private\":false}")"

if [[ "${create_code}" == "201" ]]; then
    echo "created repository: https://github.com/${OWNER}/${REPO}"
elif [[ "${create_code}" == "422" ]] && grep -q "name already exists" "${create_body}"; then
    echo "repository already exists: https://github.com/${OWNER}/${REPO}"
else
    echo "error: failed to create repository, HTTP ${create_code}" >&2
    cat "${create_body}" >&2
    rm -f "${create_body}"
    exit 1
fi
rm -f "${create_body}"

git remote set-url origin "${REMOTE_URL}"

auth_header="$(printf 'x-access-token:%s' "${GH_TOKEN}" | base64 | tr -d '\n')"
git -c "http.https://github.com/.extraheader=AUTHORIZATION: basic ${auth_header}" push -u origin main

echo "pushed main to https://github.com/${OWNER}/${REPO}"
