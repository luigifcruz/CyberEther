#!/usr/bin/env bash

set -euo pipefail

format="${INPUT_FORMAT:?INPUT_FORMAT is required}"
targets_input="${INPUT_TARGETS:?INPUT_TARGETS is required}"
allowlist_input="${INPUT_ALLOWLIST:?INPUT_ALLOWLIST is required}"
tool_path="${INPUT_TOOL_PATH:-}"

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

log() {
    printf '%s\n' "$*"
}

die() {
    printf '%s\n' "$*" >&2
    exit 1
}

trim_line() {
    printf '%s' "$1" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//'
}

normalize_dep() {
    local dep framework_path framework_name

    dep="$(trim_line "$1")"
    [ -n "$dep" ] || return 0

    case "$format" in
        elf)
            printf '%s\n' "$dep"
            ;;
        pe)
            printf '%s\n' "$dep" | tr '[:upper:]' '[:lower:]'
            ;;
        macho)
            case "$dep" in
                *.framework/*)
                    framework_path="${dep%%.framework/*}.framework"
                    framework_name="${framework_path##*/}"
                    printf '%s/%s\n' "$framework_name" "${dep##*/}"
                    ;;
                *)
                    printf '%s\n' "${dep##*/}"
                    ;;
            esac
            ;;
        *)
            die "Unsupported format '$format'. Expected elf, macho, or pe."
            ;;
    esac
}

resolve_tool() {
    case "$format" in
        elf)
            inspect_tool="${tool_path:-readelf}"
            ;;
        macho)
            inspect_tool="${tool_path:-otool}"
            ;;
        pe)
            inspect_tool="${tool_path:-objdump}"
            ;;
        *)
            die "Unsupported format '$format'. Expected elf, macho, or pe."
            ;;
    esac

    if printf '%s' "$inspect_tool" | grep -q '/'; then
        [ -x "$inspect_tool" ] || die "Inspection tool '$inspect_tool' is not executable."
    else
        command -v "$inspect_tool" >/dev/null 2>&1 || die "Inspection tool '$inspect_tool' was not found in PATH."
    fi
}

extract_deps() {
    local target="$1"
    local inspect_tool_name

    case "$format" in
        elf)
            "$inspect_tool" -d "$target" | awk -F'[][]' '/NEEDED/ { print $2 }'
            ;;
        pe)
            inspect_tool_name="$(basename "$inspect_tool" | tr '[:upper:]' '[:lower:]')"
            case "$inspect_tool_name" in
                dumpbin|dumpbin.exe)
                    "$inspect_tool" //DEPENDENTS "$target" | awk '
                        /Image has the following dependencies:/ { in_deps = 1; next }
                        in_deps && NF == 0 { if (seen) exit; next }
                        in_deps {
                            dep = $0
                            sub(/^[[:space:]]+/, "", dep)
                            sub(/[[:space:]]+$/, "", dep)
                            if (dep ~ /^[A-Za-z0-9_.-]+\.dll$/) {
                                print dep
                                seen = 1
                            } else if (seen) {
                                exit
                            }
                        }
                    '
                    ;;
                *)
                    "$inspect_tool" -p "$target" | sed -n 's/^[[:space:]]*DLL Name: //p'
                    ;;
            esac
            ;;
        macho)
            "$inspect_tool" -L "$target" | awk -v expected="$target:" '
                NR == 1 {
                    if ($0 != expected) {
                        exit 1
                    }
                    next
                }
                {
                    sub(/^[[:space:]]+/, "", $0)
                    sub(/ \(.*/, "", $0)
                    print
                }
            '
            ;;
    esac
}

prepare_allowlist() {
    local line

    allowlist_normalized="$tmpdir/allowlist.txt"
    : > "$allowlist_normalized"

    while IFS= read -r line || [ -n "$line" ]; do
        line="${line%%#*}"
        line="$(trim_line "$line")"
        [ -n "$line" ] || continue
        normalize_dep "$line" >> "$allowlist_normalized"
    done <<< "$allowlist_input"

    sort -u "$allowlist_normalized" -o "$allowlist_normalized"
    [ -s "$allowlist_normalized" ] || die 'Allowlist input did not produce any entries.'
}

expand_targets() {
    local pattern match matched_any

    targets_file="$tmpdir/targets.txt"
    : > "$targets_file"

    while IFS= read -r pattern || [ -n "$pattern" ]; do
        pattern="$(trim_line "$pattern")"
        [ -n "$pattern" ] || continue

        matched_any=0
        while IFS= read -r match || [ -n "$match" ]; do
            [ -n "$match" ] || continue
            printf '%s\n' "$match" >> "$targets_file"
            matched_any=1
        done < <(compgen -G "$pattern" || true)

        if [ "$matched_any" -eq 0 ]; then
            log "No targets matched pattern: $pattern"
        fi
    done <<< "$targets_input"

    if [ -s "$targets_file" ]; then
        sort -u "$targets_file" -o "$targets_file"
        return 0
    fi

    die 'No targets matched the provided paths or globs.'
}

check_target() {
    local target="$1"
    local err_file="$tmpdir/inspect.err"
    local raw_deps_file="$tmpdir/raw-deps.txt"
    local deps_file="$tmpdir/deps.txt"
    local bad_file="$tmpdir/bad.txt"
    local dep
    local self_dep=''

    : > "$err_file"
    : > "$raw_deps_file"
    : > "$deps_file"
    : > "$bad_file"

    if [ "$format" = 'macho' ]; then
        self_dep="$(normalize_dep "$target")"
    fi

    if ! extract_deps "$target" > "$raw_deps_file" 2> "$err_file"; then
        printf 'Failed to inspect dynamic dependencies for %s\n' "$target" >&2
        cat "$err_file" >&2
        return 1
    fi

    if [ -s "$err_file" ]; then
        printf 'Failed to inspect dynamic dependencies for %s\n' "$target" >&2
        cat "$err_file" >&2
        return 1
    fi

    while IFS= read -r dep || [ -n "$dep" ]; do
        dep="$(normalize_dep "$dep")"
        [ -n "$dep" ] || continue
        if [ -n "$self_dep" ] && [ "$dep" = "$self_dep" ]; then
            continue
        fi
        printf '%s\n' "$dep" >> "$deps_file"
    done < "$raw_deps_file"

    if [ -s "$deps_file" ]; then
        sort -u "$deps_file" -o "$deps_file"
    fi

    log "::group::Dynamic dependencies for $target"
    if [ -s "$deps_file" ]; then
        sed 's/^/  /' "$deps_file"
    else
        log '  <none>'
    fi
    log '::endgroup::'

    while IFS= read -r dep || [ -n "$dep" ]; do
        [ -n "$dep" ] || continue
        if ! grep -Fqx -- "$dep" "$allowlist_normalized"; then
            printf '%s\n' "$dep" >> "$bad_file"
        fi
    done < "$deps_file"

    if [ -s "$bad_file" ]; then
        printf 'Unexpected dependencies for %s:\n' "$target" >&2
        sed 's/^/  /' "$bad_file" >&2
        return 1
    fi

    return 0
}

main() {
    local target
    local status=0

    resolve_tool
    prepare_allowlist
    expand_targets

    while IFS= read -r target || [ -n "$target" ]; do
        [ -n "$target" ] || continue
        if ! check_target "$target"; then
            status=1
        fi
    done < "$targets_file"

    return "$status"
}

main "$@"
