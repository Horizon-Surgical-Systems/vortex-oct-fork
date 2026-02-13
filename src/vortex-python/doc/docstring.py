from pathlib import Path
import re
import json
from dataclasses import dataclass

entry_regex = re.compile(r'^(?P<indent> *)\.\. (?P<type>[a-z]+):: (?P<name>[\w\.]+)(\((?P<args>.*)\))?\s*$')
entry_types = ['class', 'method', 'function', 'property', 'module', 'exception', 'data', 'attribute', 'staticmethod', 'classmethod', 'decorator', 'currentmodule']

skip_regex = re.compile(r'^(?P<indent> *)\.\. (?P<type>[a-z]+)::')
skip_types = ['plot'] #, 'math']

line_regex = re.compile(r'^(?P<indent> *)(?P<rest>.*?)\s*$')

bases_regex = re.compile(r'^Bases?: (?P<bases>:[a-z]+:`~[\w\.]+`,?\s*)+\n')
base_regex = re.compile(r':(?P<type>[a-z]+):`~(?P<base>[\w\.]+)`')

param_regex = re.compile(r'^:(param(eter)?|arg(ument)|keyword|kwarg|kwparam) (?P<type>.*) (?P<name>\w+):\s*$')
returns_regex = re.compile(r'^:returns? (?P<name>\w+):$')
raises_regex = re.compile(r'^:(raises?|except(ion)?) (?P<name>\w+):$')
cleanup_regex = re.compile(r'(?::(?:class|exception|method|data|func|attribute):)?`(?:~|`)?(.+?)``?')
admonition_regex = re.compile(r'.. (caution|warning|seealso|note|attention|danger)::')

def docstr_name(python_name) -> str:
    parts = python_name.split('.')
    return '__'.join(parts)

def docstr_name_args(args) -> str:
    raise NotImplementedError

cpp = str.maketrans({
    '\n': r'\n',
    '"':  r'\"',
    '\\': r'\\'
})

@dataclass
class DocStr:
    name: str
    args: str
    lines: list
    path: str
    line_number: int

def process_file(docstrs: dict, name_counts: dict, path: Path):

    scope = []

    def update_scope(indent, name):
        while scope and scope[-1][0] >= indent:
            del scope[-1]

        scope.append((indent, name))

    def current_scope():
        return [s[1] for s in scope]

    def python_name():
        return '.'.join(current_scope())

    for (i, line) in enumerate(path.read_text().splitlines()):
        match = entry_regex.match(line)
        if match:
            (indent, type_, name, _, args) = match.groups()
            indent = len(indent)

            if type_ in entry_types:

                update_scope(indent, name)
                active_name = python_name()

                # if args:
                #     print(active_name, '(', args, ')')
                # else:
                #     print(active_name)

                if (active_name, args) in docstrs:
                    raise RuntimeError(f'duplicate docstring for {active_name}({args})')

                docstrs[(active_name, args)] = DocStr(active_name, args, [], path, i)
                name_counts[active_name] = name_counts.get(active_name, 0) + 1

                continue

        match = skip_regex.match(line)
        if match:
            (indent, type_) = match.groups()

            if type_ in skip_types:
                update_scope(len(indent), None)

        if scope:

            (scope_indent, scope_name) = scope[-1]
            if scope_name:

                (indent, rest) = line_regex.match(line).groups()
                indent = len(indent)

                if indent > scope_indent or not rest:
                    docstrs[(active_name, args)].lines.append((indent, rest))
                    # print(indent, scope_indent, line)

def export(docstrs: dict, name_counts: dict, output_cpp, output_json=None):

    inherits = {}

    print('''\
// Auto-generated docstrings extracted from RST files

#include <unordered_map>
#include <string_view>
''', file=output_cpp)

    pairs = {}
    mapping = []

    for docstr in docstrs.values(): # sorted(docstrs.values(), key=lambda o: (o.name, o.args)):
        key = docstr_name(docstr.name)

        if name_counts[docstr.name] > 1:
            key += docstr_name_args(docstr.args)

        min_indent = min([l[0] for l in docstr.lines if l[1]] or [0])

        value = '\n'.join([' ' * (l[0] - min_indent) + l[1].rstrip() for l in docstr.lines]).strip()

        lines = value.split('\n')
        i = 0
        emitted = {}
        while i < len(lines):
            match = param_regex.match(lines[i])
            if match:
                header = 'Parameters'
                if not emitted.get(header):
                    lines = lines[:i] + [header, '-' * len(header)] + lines[i:]
                    i += 2
                emitted[header] = True

                *_, type_, name = match.groups()
                lines[i] = f'``{name}`` : ``{type_}``'
                lines.insert(i, '')
                i += 1

            else:
                for (regex, header) in [(returns_regex, 'Returns'), (raises_regex, 'Raises')]:
                    match = regex.match(lines[i])
                    if match:
                        if not emitted.get(header):
                            lines = lines[:i] + [header, '-' * len(header)] + lines[i:]
                            i += 2
                        emitted[header] = True

                        *_, type_ = match.groups()
                        lines[i] = f'``{type_}``'
                        lines.insert(i, '')
                        i += 1

            # lines[i] = cleanup_regex.sub(r'`\1`', lines[i])
            lines[i] = admonition_regex.sub(lambda m: f'__{m.group(1).upper()}__', lines[i])

            i += 1

        value = '\n'.join(lines)

        # check for inheritance
        match = bases_regex.match(value)
        if match:
            bases, = match.groups()

            for match in base_regex.finditer(bases):
                _, base = match.groups()
                inherits.setdefault(docstr.name, []).append(base)

        print(f'// {docstr.name} from {docstr.path}:{docstr.line_number}', file=output_cpp)
        print(f'constexpr const char* {key} = "{value.translate(cpp)}";', file=output_cpp)

        pairs[docstr.name] = key
        mapping.append((docstr.name, value))

    # add inheritance entries, loop since there may multiple levels of inheritance
    while True:
        inherited_pairs = {}
        for (child, bases) in inherits.items():
            for base in bases:
                for (name, key) in pairs.items():
                    if not name.startswith(base) or name == base:
                        continue

                    suffix = name[len(base):]
                    new_name = child + suffix

                    if new_name in pairs:
                        # overridden
                        continue

                    inherited_pairs[new_name] = key

        pairs.update(inherited_pairs)
        if not inherited_pairs:
            break

    print('''\

static const std::unordered_map<std::string_view, const char*> _docstring_lookup = {''', file=output_cpp)
    for (key, value) in sorted(pairs.items(), key=lambda x: x[0]):
        print(f'    {{ "{key}", {value} }},', file=output_cpp)
    print(r'''};

const char* doc(const std::string_view& key) {
    auto it = _docstring_lookup.find(key);
    if(it != _docstring_lookup.end()) {
        return it->second;
    } else {
        return "";
    }
}''', file=output_cpp)

    if output_json:
        json.dump(dict(mapping), output_json, indent=4)

if __name__ == '__main__':
    import sys

    docstrs = {}
    name_counts = {}

    root = Path.cwd() / Path(sys.argv[1])

    try:
        output_path = Path.cwd() / Path(sys.argv[2])
    except IndexError:
        output_cpp = sys.stdout
    else:
        output_path.parent.mkdir(exist_ok=True)
        output_cpp = output_path.open('w')

    for path in root.glob("**/*.rst"):
        process_file(docstrs, name_counts, path)

    export(docstrs, name_counts, output_cpp)
