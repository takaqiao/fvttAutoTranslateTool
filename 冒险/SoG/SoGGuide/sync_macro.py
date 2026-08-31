"""Inject suppress_haunt_macro.js into the exported Macro JSON's `command` field.

Editing the script as a .js file beats hand-escaping a multi-line string inside JSON:
you get syntax highlighting, and `node --check suppress_haunt_macro.js` actually works.
Only `command` is touched; every other field keeps its original value and key order.
"""
import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'suppress_haunt_macro.js')
MACRO = os.path.join(HERE, 'fvtt-Macro-suppress-haunt-9AtRaJhc9zz2MNND.json')

# Strings the PF2e item's rule elements and the inline check depend on.
REQUIRED = [
    'action:suppress-haunt',   # matches the item's Note predicate
    'traits:haunt,hazard',
    '@Check[',
]


def main():
    command = open(SRC, encoding='utf-8').read().rstrip('\n')

    for token in REQUIRED:
        if token not in command:
            raise SystemExit(f'refusing to sync: {token!r} missing from the script')

    try:
        subprocess.run(['node', '--check', SRC], check=True,
                       capture_output=True, text=True)
        checked = 'node --check passed'
    except FileNotFoundError:
        checked = 'node not found, syntax unchecked'
    except subprocess.CalledProcessError as e:
        raise SystemExit(f'syntax error in {SRC}:\n{e.stderr}')

    with open(MACRO, encoding='utf-8') as f:
        doc = json.load(f)
    doc['command'] = command
    with open(MACRO, 'w', encoding='utf-8') as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
        f.write('\n')

    print(f'synced {len(command.splitlines())} lines into {os.path.basename(MACRO)} ({checked})')


if __name__ == '__main__':
    main()
