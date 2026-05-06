import os
import sys

from inspect import getsourcefile

# Point Sphinx to the source code so it can read docstrings
sys.path.insert(0, os.path.abspath('../'))

project = 'PowerGridSynth'
copyright = '2025-2026, PowerGridSynth Developers'
author = 'PowerGridSynth Developers'
release = '0.1.1'


# -- General configuration ---------------------------------------------------

# Extensions
extensions = [
    'autoapi.extension',
    'sphinx.ext.autodoc',      # Generates docs from code docstrings
    'sphinx.ext.duration',
    'sphinx.ext.napoleon',     # Parses Google-style docstrings
    'sphinx.ext.viewcode',     # Adds links to source code
    'sphinx_math_dollar',
    'sphinx.ext.mathjax',
    'sphinx.ext.todo',
    'nbsphinx',
    'nbsphinx_link',
    'sphinxcontrib.bibtex',
    'myst_parser',             # Allows using Markdown (.md) instead of .rst
]

suppress_warnings = [
    'config.cache',
]

# Support for Markdown files
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}


# autoapi
autoapi_dirs = ["../powergrid_synth"]
autodoc_typehints = 'description'
autodoc_typehints_format = 'short'
autoapi_add_toctree_entry = True
autoapi_keep_files = True
autoapi_root = 'autoapi'
autoapi_python_class_content = "class"  # we handle __init__ and __new__ below
autoapi_member_order = "groupwise"
autoapi_options = [
    "members",
    # "undoc-members",
    "private-members",
    "special-members",
    "imported-members",
    "show-inheritance",
]

# Only skip certain special members if they have an empty docstring.
privileged_special_members = ["__init__", "__new__", "__call__"]
def never_skip_init_or_new(app, what, name, obj, would_skip, options):
    if any(psm in name for psm in privileged_special_members):
        return not bool(obj._docstring)  # skip only if the docstring is empty
    return would_skip


# Add any paths that contain templates here, relative to this directory.
# The templates are as in https://stackoverflow.com/a/62613202
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']


# -- Options for HTML output -------------------------------------------------

# The theme to match ReadTheDocs.org
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

html_theme_options = {
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'display_version': True,
}

html_context = {
    'display_github': True,
    'github_user': 'cookbook-ms',
    'github_repo': 'Chung_Lu_Chain-synthesizer',
    'github_version': 'main',
    'conf_py_path': '/docs/',
}

html_css_files = ['custom.css']

# furo theme
# html_theme = "furo"
# html_theme_options = {
#     "light_css_variables": {
#         "color-brand-primary": "red",
#         "color-brand-content": "#CC3333",
#         "color-admonition-background": "orange",
#     },
#     "navigation_with_keys": True,
#     "sidebar_hide_name": True,
#     "top_of_page_buttons": ["view", "edit"],
# }



# For sphinx_math_dollar
mathjax3_config = {
  "tex": {
    "inlineMath": [['\\(', '\\)']],
    "displayMath": [["\\[", "\\]"]],
  }
}


# -- nbsphinx ----------------------------------------------------------------
nbsphinx_execute = 'never'

# Don't load require.js because it conflicts with bootstrap.min.js being
# loaded after it. If you need require.js, load it manually by adding it
# to the html_js_files list.
nbsphinx_requirejs_path = ""

# -- pandoc (required by nbsphinx) -------------------------------------------
# Get path to directory containing this file, conf.py.
DOCS_DIRECTORY = os.path.dirname(os.path.abspath(getsourcefile(lambda: 0)))

def ensure_pandoc_installed(_):
    import pypandoc

    # Download pandoc if necessary. If pandoc is already installed and on
    # the PATH, the installed version will be used. Otherwise, we will
    # download a copy of pandoc into docs/bin/ and add that to our PATH.
    pandoc_dir = os.path.join(DOCS_DIRECTORY, "bin")
    # Add dir containing pandoc binary to the PATH environment variable
    if pandoc_dir not in os.environ["PATH"].split(os.pathsep):
        os.environ["PATH"] += os.pathsep + pandoc_dir
    pypandoc.ensure_pandoc_installed(
        targetfolder=pandoc_dir,
        delete_installer=True,
    )
    
# -- Bibtex ------------------------------------------------------------------

bibtex_bibfiles = ['references.bib']
bibtex_reference_style = "author_year"

import pybtex.plugin
from pybtex.style.formatting.plain import Style as UnsrtStyle
from pybtex.style.template import field, sentence

class GKUnsrtStyle(UnsrtStyle):

    def format_title(self, e, which_field, as_sentence=True):
        formatted_title = field(which_field)  # Leave the field exactly as is.
        if as_sentence:
            return sentence [ formatted_title ]
        else:
            return formatted_title

pybtex.plugin.register_plugin('pybtex.style.formatting', 'gkunsrt', GKUnsrtStyle)

# -- Connect stuff -----------------------------------------------------------

def setup(sphinx):
    sphinx.connect("builder-inited", ensure_pandoc_installed)
    # sphinx.connect("autoapi-skip-member", never_skip_init_or_new)