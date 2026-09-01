#!/usr/bin/env bash

set -euo pipefail

# Caminho real do repositório na máquina nova.
# Pode ser sobrescrito antes de executar o script:
#
#   SAE_REPO=/outro/caminho/SAE-TCAV run_robustness_env.sh
#
: "${SAE_REPO:=$HOME/projects/SAE-TCAV}"

# Caminho absoluto que aparece nos manifests antigos.
SAE_COMPAT_REPO="/home/joaomarcostomaz/SAE-TCAV-extension/SAE-TCAV"

# ----------------------------------------------------------------------
# Validações
# ----------------------------------------------------------------------

if ! command -v bwrap >/dev/null 2>&1; then
    echo "Erro: bwrap não está instalado ou não está no PATH." >&2
    exit 1
fi

if [[ ! -d "$SAE_REPO" ]]; then
    echo "Erro: repositório não encontrado em:" >&2
    echo "  $SAE_REPO" >&2
    echo >&2
    echo "Defina SAE_REPO antes de executar o script, por exemplo:" >&2
    echo "  export SAE_REPO=/caminho/para/SAE-TCAV" >&2
    exit 1
fi

# Se nenhum comando for passado, abre um shell interativo.
if [[ $# -eq 0 ]]; then
    COMMAND=(/bin/bash)
else
    COMMAND=("$@")
fi

# ----------------------------------------------------------------------
# Sandbox
# ----------------------------------------------------------------------

exec bwrap \
    --ro-bind / / \
    --dev-bind /dev /dev \
    --proc /proc \
    --tmpfs /tmp \
    --tmpfs /home \
    --dir "$HOME" \
    --bind "$HOME" "$HOME" \
    --dir /home/joaomarcostomaz \
    --dir /home/joaomarcostomaz/SAE-TCAV-extension \
    --dir "$SAE_COMPAT_REPO" \
    --bind "$SAE_REPO" "$SAE_COMPAT_REPO" \
    --chdir "$SAE_REPO" \
    --setenv SAE_REPO "$SAE_REPO" \
    --setenv SAE_COMPAT_REPO "$SAE_COMPAT_REPO" \
    "${COMMAND[@]}"
