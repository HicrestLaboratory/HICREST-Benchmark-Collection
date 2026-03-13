class NodeListParseError(Exception):
    pass

def expand_slurm_nodelist(input: str) -> list[str]:
    """Expand SLURM-style node list expressions."""
    main = input.split()[0] if input.split() else input

    items = _split_preserving_brackets(main)

    result = []
    for item in items:
        result.extend(_expand_item(item))

    return result


def _expand_item(s: str) -> list[str]:
    """Expand one item that may contain multiple bracket groups."""
    start = s.find('[')
    if start != -1:
        end_offset = s[start:].find(']')
        if end_offset == -1:
            raise NodeListParseError(f"missing closing ']' in '{s}'")
        end = end_offset + start

        prefix = s[:start]
        inside = s[start + 1:end]
        suffix = s[end + 1:]

        if not inside:
            raise NodeListParseError(f"empty bracket expression in '{s}'")

        results = []
        for part in inside.split(','):
            expanded = _expand_part(part)
            for val in expanded:
                new_string = f"{prefix}{val}{suffix}"
                results.extend(_expand_item(new_string))

        return results
    else:
        return [s]


def _expand_part(part: str) -> list[str]:
    """Expand a single range element (e.g. '01-04' or '05')."""
    dash = part.find('-')
    if dash != -1:
        start_str = part[:dash]
        end_str = part[dash + 1:]

        if not start_str or not end_str:
            raise NodeListParseError(f"invalid range '{part}'")

        try:
            start = int(start_str)
        except ValueError:
            raise NodeListParseError(f"invalid number '{start_str}'")

        try:
            end = int(end_str)
        except ValueError:
            raise NodeListParseError(f"invalid number '{end_str}'")

        if start > end:
            raise NodeListParseError(f"range start greater than end in '{part}'")

        width = len(start_str)
        return [str(v).zfill(width) for v in range(start, end + 1)]
    else:
        if not part.isdigit():
            raise NodeListParseError(f"invalid element '{part}'")
        return [part]


def _split_preserving_brackets(s: str) -> list[str]:
    """Split comma-separated items but keep bracket groups intact."""
    result = []
    current = []
    depth = 0

    for ch in s:
        if ch == '[':
            depth += 1
            current.append(ch)
        elif ch == ']':
            if depth == 0:
                raise NodeListParseError("closing ']' without matching '['")
            depth -= 1
            current.append(ch)
        elif ch == ',' and depth == 0:
            if current:
                result.append(''.join(current))
                current.clear()
        else:
            current.append(ch)

    if depth != 0:
        raise NodeListParseError("missing closing ']'")

    if current:
        result.append(''.join(current))

    return result
