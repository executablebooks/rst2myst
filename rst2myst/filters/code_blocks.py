"""
Pandoc Filter for Parsing Code Blocks into Myst(Markdown)
"""

import panflute as pf
import pdb

def prepare(doc):
    pass


def action(elem, doc):
    if isinstance(elem, pf.CodeBlock):
        import pdb; pdb.set_trace()
    if isinstance(elem, pf.Emph):
        return pf.Strong(*elem.content)

def finalize(doc):
    pass


def main(doc=None):
    return pf.run_filter(action,
                         prepare=prepare,
                         finalize=finalize,
                         doc=doc) 


if __name__ == '__main__':
    main()