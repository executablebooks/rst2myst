"""
Pandoc filters used in converting RST to MyST format

A list of resources by topic

RST:
- [reStructuredText Primer](http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)
- [reStructuredText Directives](http://docutils.sourceforge.net/docs/ref/rst/directives.html#figure)

Myst:
- [MyST Syntax Guide](https://myst-parser.readthedocs.io/en/latest/using/syntax.html)

Pandoc:
- [Pandoc User Guide](https://pandoc.org/MANUAL.html#citations)
- [List of Pandoc Elements](https://metacpan.org/pod/Pandoc::Elements)
"""  # noqa: E501

import panflute as pf

# TODO: Import Filters

def pandoc_filters():
    """ run a set of rst2myst pandoc filters directly on the pandoc AST,
    via ``pandoc --filter rst2myst``
    """
    doc = pf.load()
    meta = pf.tools.meta2builtin(doc.metadata)

    apply_filters = doc.get_metadata(IPUB_META_ROUTE + ".apply_filters", default=True)
    convert_raw = doc.get_metadata(IPUB_META_ROUTE + ".convert_raw", default=True)

    filters = [
                # Filters
            ]

    out_doc = doc
    for func in filters:
        out_doc = func(out_doc)  # type: pf.Doc

    pf.dump(doc)