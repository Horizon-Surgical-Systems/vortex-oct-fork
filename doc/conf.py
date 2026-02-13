# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
from pathlib import Path
PROJECT_SOURCE_DIR = Path(os.environ.get('PROJECT_SOURCE_DIR', '.'), Path(__file__).parent.parent)
PROJECT_BINARY_DIR = Path(os.environ.get('PROJECT_BINARY_DIR', 'build'))
VORTEX_VERSION_STRING = os.environ.get('VORTEX_VERSION_STRING', '0.0.0')

import sys
sys.path.insert(0, (PROJECT_SOURCE_DIR / 'doc').as_posix())

# -- Project information -----------------------------------------------------

project = 'vortex'

from datetime import datetime
copyright = f'2020 - {datetime.now().year}, Mark Draelos'
author = 'Mark Draelos'

# The full version, including alpha/beta/rc tags
release = f'v{VORTEX_VERSION_STRING}'

# -- General configuration ---------------------------------------------------

# do not show class, function, etc. entries in sidebar table of contents
toc_object_entries = False

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'breathe',
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx_copybutton',
    'sphinx_inline_tabs',
    'plot_directive',
    'tikz_directive'
]

from sphinx import addnodes
from sphinx.environment import BuildEnvironment
from sphinx.domains.python import PyObject, PyTypedField, PyGroupedField, PyField, PyXrefMixin, _
from sphinx.util.docfields import GroupedField
from docutils import nodes
from docutils.nodes import Node
from docutils.parsers.rst.states import Inliner
from typing import cast, List, Tuple, Dict

class MyPyGroupedField(PyXrefMixin, GroupedField):
    def make_field(self, types: Dict[str, List[Node]], domain: str,
                   items: Tuple, env: BuildEnvironment = None,
                   inliner: Inliner = None, location: Node = None) -> nodes.field:
        fieldname = nodes.field_name('', self.label)
        listnode = self.list_type()
        for fieldarg, content in items:
            par = nodes.paragraph()
            par.extend(self.make_xrefs(self.rolename, domain, fieldarg,
                                       addnodes.literal_emphasis,
                                       env=env, inliner=inliner, location=location))
            par += nodes.Text(' -- ')
            par += content
            listnode += nodes.list_item('', par)

        if len(items) == 1 and self.can_collapse:
            list_item = cast(nodes.list_item, listnode[0])
            fieldbody = nodes.field_body('', list_item[0])
            return nodes.field('', fieldname, fieldbody)

        fieldbody = nodes.field_body('', listnode)
        return nodes.field('', fieldname, fieldbody)

PyObject.doc_field_types = [
    PyTypedField('parameter', label=_('Parameters'),
                    names=('param', 'parameter', 'arg', 'argument',
                        'keyword', 'kwarg', 'kwparam'),
                    typerolename='class', typenames=('paramtype', 'type'),
                    can_collapse=True),
    PyTypedField('variable', label=_('Variables'),
                    names=('var', 'ivar', 'cvar'),
                    typerolename='class', typenames=('vartype',),
                    can_collapse=True),
    MyPyGroupedField('exceptions', label=_('Raises'), rolename='exc',
                    names=('raises', 'raise', 'exception', 'except'),
                    can_collapse=True),
    MyPyGroupedField('returnvalue', label=_('Returns'), rolename='class',
                    names=('returns', 'return'), can_collapse=True),
    PyField('returntype', label=_('Return type'), has_arg=False,
            names=('rtype',), bodyrolename='class'),
]

# Add any paths that contain templates here, relative to this directory.
templates_path = []

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []
if tags.has('draft'):
    exclude_patterns.append('api')


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'furo'
# NOTE: insert a zero width space so the development version breaks at a better spot
html_title = 'Vortex ' + release.replace('+', "\u200B+")
html_logo = (PROJECT_SOURCE_DIR / 'doc' / 'vortex-v0.svg').as_posix()
html_show_sourcelink = False

html_theme_options = {
    'light_css_variables': {
        'admonition-font-size': '1rem',
        # 'admonition-title-font-size': '1rem',
    },
}
html_css_files = [
    'theme-overrides.css',
    'version-selector.css'
]
html_js_files = [
    'version-selector.js',
]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named 'default.css' will overwrite the builtin 'default.css'.
html_static_path = [(PROJECT_SOURCE_DIR / 'doc' / '_static').as_posix()]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'cupy': ('https://docs.cupy.dev/en/stable', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
}

breathe_default_project = 'vortex'
breathe_projects = {
    'vortex': (PROJECT_BINARY_DIR / 'doc' / 'xml').as_posix()
}

from matplotlib import pyplot as plt
from cycler import cycler

plot_formats = ['svg', 'pdf']

plot_rcparams = {
    'font.family': 'Roboto',
    'font.size': 12,

    'lines.linewidth': 1,

    'axes.titlesize': 12,
    'axes.prop_cycle': cycler(color=['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf','#999999']),

    'savefig.transparent': True
}
plot_rcparams_light = {
    'lines.color': '#000000',
    'patch.edgecolor': '#000000',
    'text.color': '#000000',

    'axes.facecolor': '#ffffff',
    'axes.edgecolor': '#000000',
    'axes.labelcolor': '#000000',

    'xtick.color': '#000000',
    'ytick.color': '#000000',

    'grid.color': '#000000',

    'figure.edgecolor': '#ffffff',
}
plot_rcparams_dark = {
    'lines.color': '#cccccc',
    'patch.edgecolor': '#cccccc',
    'text.color': '#cccccc',

    'axes.facecolor': '#131416',
    'axes.edgecolor': '#cccccc',
    'axes.labelcolor': '#cccccc',

    'xtick.color': '#cccccc',
    'ytick.color': '#cccccc',

    'grid.color': '#cccccc',

    'figure.edgecolor': '#131416',
}
plot_variants = [
    (plot_rcparams_light, 'only-light'),
    (plot_rcparams_dark, 'only-dark'),
]

plot_template = """
.. only:: html

   {% for img in images %}
   .. figure:: {{ build_dir }}/{{ img.basename }}.{{ default_fmt }}
      {% for option in options -%}
      {{ option }}
      {% endfor %}

      {{ caption }}  {# appropriate leading whitespace added beforehand #}
      {% if html_show_formats and multi_image -%}
        (
        {%- if src_name -%}
        :download:`code <{{ build_dir }}/{{ src_name }}>`
        {%- endif -%}
        {%- for fmt in img.formats -%}
        {%- if not loop.first or src_name -%}, {% endif -%}
        :download:`{{ fmt }} <{{ build_dir }}/{{ img.basename }}.{{ fmt }}>`
        {%- endfor -%}
        )
      {%- endif -%}
   {% endfor %}

.. only:: not html

   {% for img in images %}
   .. figure:: {{ build_dir }}/{{ img.basename }}.*
      {% for option in options -%}
      {{ option }}
      {% endfor -%}

      {{ caption }}  {# appropriate leading whitespace added beforehand #}
   {% endfor %}

"""

# ref: https://tex.stackexchange.com/questions/302923/every-draw-style
tikz_latex_preamble = r'''
\documentclass[tikz, border=6pt]{standalone}
\usepackage{fontspec}
\setmainfont{roboto}
\setsansfont{roboto}
\usepackage{pagecolor}
\usetikzlibrary{positioning,arrows,calc,fit,chains}
'''

# ref: https://tex.stackexchange.com/questions/82498/change-background-colour-for-entire-document
tikz_body_template_light =  r'''
\definecolor{forecolor}{HTML}{000000}
\definecolor{backcolor}{HTML}{FFFFFF}
\begin{document}
\color{forecolor}
\nopagecolor
\fontsize{12}{14}\selectfont
%s
\end{document}
'''
tikz_body_template_dark =  r'''
\definecolor{forecolor}{HTML}{CCCCCC}
\definecolor{backcolor}{HTML}{131416}
\begin{document}
\color{forecolor}
\nopagecolor
\fontsize{12}{14}\selectfont
%s
\end{document}
'''
tikz_variants = [
    (tikz_body_template_light, 'only-light'),
    (tikz_body_template_dark, 'only-dark'),
]
tikz_proc_suite = 'dvisvgm'
latex_engine = 'lualatex'

pygments_style = 'sphinx'
pygments_dark_style = 'monokai'

copybutton_prompt_text = r">>> |\.\.\. |\$ |> |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

rst_epilog = f'''
.. _download-source: https://www.vortex-oct.dev/rel/v{VORTEX_VERSION_STRING}/src/vortex-{VORTEX_VERSION_STRING}-src.zip
'''
