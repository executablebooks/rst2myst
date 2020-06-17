from utils import apply_filter
from code_blocks import main as code_block
from target_header import main as target_header
from misc import main as misc

if __name__ == '__main__':
    #-Test-#
    apply_filter("[a link](https://pandoc.org/filters.html)", [], "rst")
    #-CodeBlocks-#
    s1 = """
    ::
        This is a block quote
    """
    s2 = """
    .. code-block:: python3
       :linenos:
       :emphasize-lines: 1
       :name: test-block

       import pandas as pd
    """
    out = apply_filter(s2,filter_func=code_block, out_format="markdown", in_format="rst")

    #-TargetHeaders-#
    fp = open('./lecture.rst')
    out = apply_filter(fp.read(), filter_func=target_header,
        out_format="markdown-raw_attribute", in_format="rst-auto_identifiers")