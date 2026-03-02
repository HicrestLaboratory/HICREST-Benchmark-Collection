def dict_get(d, key):
    r = d.get(key)
    if r is None:
        raise KeyError(f"{key} not found")
    return r


def raise_none(v, msg):
    if v is None:
        raise ValueError(f"{msg} not found")
    return v
