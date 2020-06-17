"""
Misc Pandoc Filter
"""

import panflute as pf
import pdb

def remove_div_html(elem, doc):
    for blck in elem.content:
        if isinstance(blck, pf.RawBlock) and blck.format == 'html':
            soup = bs4.BeautifulSoup(pf.stringify(blck))
            content = soup.prettify()
            blck.text = content
            import pdb; pdb.set_trace()
            

def main(doc=None):
    return pf.run_filter(action, 
        prepare=prepare,
        finalize=finalize,
        doc=doc)

def prepare(doc):
    pass

def action(elem, doc):
    # Remove `.. content::`
    if isinstance(elem, pf.Div):
        import pdb; pdb.set_trace()
        if 'contents' in elem.classes:
            return []

def finalize(doc):
    pass

if __name__ == '__main__':
    main()