"""
RST Chunk Parser to Coordinate Conversion to Myst Markdown

This parser uses pandoc for the majority of conversions and
handles block migration such as directives.

"""

class RSTChunkParser(object):

    def __init__(self, inputstring):
        

