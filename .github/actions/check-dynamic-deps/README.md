# `check-dynamic-deps`

Local composite action that checks a binary's direct dynamic dependencies against an allowlist.

Supported formats:
- `elf`
- `macho`
- `pe`

Inputs:
- `format`: target binary format
- `targets`: newline-separated target paths or globs
- `allowlist`: newline-separated allowlist entries, usually passed from a workflow env var
- `tool-path`: optional override for `readelf`, `otool`, or `objdump`

Allowlist rules:
- blank lines are ignored
- lines starting with `#` are ignored
- matching is exact after normalization

Normalization:
- `elf`: keeps exact `DT_NEEDED` names like `libc.so.6`
- `pe`: compares lowercased DLL names like `kernel32.dll`
- `macho`: normalizes dylibs to their leaf name like `libSystem.B.dylib`
- `macho`: normalizes frameworks to `Framework.framework/Binary` like `Metal.framework/Metal`
- `macho`: ignores a checked dylib's own install name so only linked dependencies are compared

Example usage:

```yaml
- uses: ./.github/actions/check-dynamic-deps
  with:
    format: elf
    targets: |
      /opt/cyberether/bin/cyberether
    allowlist: ${{ env.SYSTEM_SHARED_LIBS }}
```

```yaml
- uses: ./.github/actions/check-dynamic-deps
  with:
    format: macho
    targets: |
      build/cyberether
    allowlist: ${{ env.SYSTEM_MACOS_DYLIBS }}
```

```yaml
- uses: ./.github/actions/check-dynamic-deps
  with:
    format: pe
    targets: |
      build/cyberether.exe
    allowlist: ${{ env.SYSTEM_WINDOWS_DLLS }}
    tool-path: /usr/src/mxe/usr/bin/x86_64-w64-mingw32.static.posix-objdump
```
