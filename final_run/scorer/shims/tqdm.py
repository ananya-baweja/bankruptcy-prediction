"""Fallback used only when the real tqdm is not installed: progress bars become no-ops."""


def tqdm(iterable=None, *args, **kwargs):
    return iterable if iterable is not None else []
