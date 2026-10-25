TEXT_LABELS = frozenset(
    {
        "Text",
        "SectionHeader",
        "PageHeader",
        "PageFooter",
        "Caption",
        "Footnote",
        "Code",
        "Bibliography",
    }
)
LAYOUT_PRED_RELABEL = {
    "Caption": "Caption",
    "Footnote": "Footnote",
    "Equation-Block": "Equation",
    "List-Group": "ListGroup",
    "Page-Header": "PageHeader",
    "Page-Footer": "PageFooter",
    "Image": "Picture",
    "Section-Header": "SectionHeader",
    "Table": "Table",
    "Text": "Text",
    "Complex-Block": "Figure",
    "Code-Block": "Code",
    "Form": "Form",
    "Table-Of-Contents": "TableOfContents",
    "Figure": "Figure",
    "Chemical-Block": "ChemicalBlock",
    "Diagram": "Diagram",
    "Bibliography": "Bibliography",
    "Blank-Page": "BlankPage",
}
