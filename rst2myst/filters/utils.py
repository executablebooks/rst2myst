"""
Apply function from ipypublish project
"""

import panflute as pf
from six import string_types
from panflute import Doc  # noqa: F401

def apply_filter(
    in_object,
    filter_func=None,
    out_format="panflute",
    in_format="markdown",
    strip_meta=False,
    strip_blank_lines=False,
    replace_api_version=True,
    dry_run=False,
    **kwargs
):
    # type: (list[str], FunctionType) -> str
    """convenience function to apply a panflute filter(s)
    to a string, list of string lines, pandoc AST or panflute.Doc

    Parameters
    ----------
    in_object: str or list[str] or dict
        can also be panflute.Doc
    filter_func:
        the filter function or a list of filter functions
    out_format: str
        for use by pandoc or, if 'panflute', return the panflute.Doc
    in_format="markdown": str
    strip_meta=False: bool
        strip the document metadata before final conversion
    strip_blank_lines: bool
    strip_ends: bool
        strip any blank lines or space from the start and end
    replace_api_version: bool
        for dict input only, if True,
        find the api_version of the available pandoc and
        reformat the json as appropriate
    dry_run: bool
        If True, return the Doc object, before applying the filter
    kwargs:
        to parse to filter func

    Returns
    -------
    str

    """
    if isinstance(in_object, pf.Doc):
        pass
    elif isinstance(in_object, dict):
        if not in_format == "json":
            raise AssertionError(
                "the in_format for a dict should be json, " "not {}".format(in_format)
            )
        if "meta" not in in_object:
            raise ValueError("the in_object does contain a 'meta' key")
        if "blocks" not in in_object:
            raise ValueError("the in_object does contain a 'blocks' key")
        if "pandoc-api-version" not in in_object:
            raise ValueError("the in_object does contain a 'pandoc-api-version' key")
        if replace_api_version:
            # run pandoc on a null object, to get the correct api version
            null_raw = pf.run_pandoc("", args=["-t", "json"])
            null_stream = io.StringIO(null_raw)
            api_version = pf.load(null_stream).api_version

            # see panflute.load, w.r.t to legacy version
            if api_version is None:
                in_object = [{"unMeta": in_object["meta"]}, in_object["blocks"]]
            else:
                ans = OrderedDict()
                ans["pandoc-api-version"] = api_version
                ans["meta"] = in_object["meta"]
                ans["blocks"] = in_object["blocks"]
                in_object = ans
        in_str = json.dumps(in_object)
    elif isinstance(in_object, (list, tuple)):
        in_str = "\n".join(in_object)
    elif isinstance(in_object, string_types):
        in_str = in_object
    else:
        raise TypeError("object not accepted: {}".format(in_object))

    if not isinstance(in_object, pf.Doc):
        doc = pf.convert_text(in_str, input_format=in_format, standalone=True)
        # f = io.StringIO(in_json)
        # doc = pf.load(f)
    else:
        doc = in_object

    doc.format = out_format

    if dry_run:
        return doc

    if not isinstance(filter_func, (list, tuple, set)):
        filter_func = [filter_func]

    out_doc = doc
    for func in filter_func:
        out_doc = func(out_doc, **kwargs)  # type: Doc

    # post-process Doc
    if strip_meta:
        out_doc.metadata = {}
    if out_format == "panflute":
        return out_doc

    # create out str
    # with io.StringIO() as f:
    #     pf.dump(doc, f)
    #     jsonstr = f.getvalue()
    # jsonstr = json.dumps(out_doc.to_json()
    out_str = pf.convert_text(
        out_doc, input_format="panflute", output_format=out_format
    )

    # post-process final str
    if strip_blank_lines:
        out_str = out_str.replace("\n\n", "\n")

    return out_str