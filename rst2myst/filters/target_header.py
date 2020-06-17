"""
Pandoc Filter for Parsing Target Headers into Myst(Markdown)
"""

import panflute as pf

def main(doc=None):
    return pf.run_filter(action, 
        prepare=prepare,
        finalize=finalize,
        doc=doc)

def prepare(doc):
    """Add target headers if specified. auto_identifiers
    has been disabled.
    """
    final_blocks = []

    for block in doc.content:
        # If header
        if isinstance(block, pf.Header):
            # If prev is Div with identifier update prev identifier and replace prev block
            if isinstance(block.prev, pf.Div) and len(block.prev.identifier) > 0:
                target_header = pf.convert_text('({})='.format(block.prev.identifier))[0]
                block.prev.identifier = ''
                final_blocks = final_blocks[:-1] + [block.prev] 
            else:
                # If header identifier is empty, add block and continue
                if len(block.identifier) == 0:
                    final_blocks.append(block)
                    continue
                else:
                    # otherwise, update target header
                    target_header = pf.convert_text('({})='.format(block.identifier))[0]
            block.identifier = ''
            final_blocks = final_blocks + [target_header, block]
        else:
            final_blocks.append(block)
    doc.content = final_blocks

def action(elem, doc):
    pass

def finalize(doc):
    pass

if __name__ == '__main__':
    main()