"""Configuration for LLM validation pipeline."""

MODEL_NAME = "Qwen/Qwen3.5-9B"

# MASIVE-style verification prompt, translated to Romanian.
# System message establishes expertise; user message provides scale
# definitions, one in-context example, then the candidate text.

SYSTEM_MESSAGE = "Ești expert în emoții și sentimente umane."

USER_TEMPLATE = """\
Stare afectivă se referă la orice termen pe care oamenii îl folosesc pentru a descrie experiențele lor de simțire, inclusiv emoții, dispoziții și expresii figurative ale sentimentelor (de ex. „a vedea negru" ca expresie a disperării, nu a culorii). Termenul dintre <span> și </span> reflectă o stare afectivă? Răspunde doar cu un singur caracter dintre următoarele: 0, 1, 2 sau 3.

0 înseamnă Nu este o stare afectivă: termenul nu se referă la o emoție, sentiment sau stare interioară.
1 înseamnă Improbabil o stare afectivă: termenul se referă la altceva decât o emoție.
2 înseamnă Probabil o stare afectivă: termenul pare să se refere la o emoție, sentiment sau stare interioară.
3 înseamnă Categoric o stare afectivă: termenul este definitiv o emoție, sentiment sau stare interioară.

Nu explica și nu preface răspunsul.

Text: Am fost la munte weekendul trecut și am <span>încredere</span> că vom merge din nou.
Răspuns: 0

Text: Mă simt <span>fericit</span> și recunoscător pentru tot ce am primit.
Răspuns: 3

Text: Sunt <span>sigur</span> că vine mâine, am vorbit cu el la telefon.
Răspuns: 0

Text: Mi-e <span>dor</span> de casa părinților, nu am mai fost de un an.
Răspuns: 3

Text: Eu nu am <span>încredere</span> în acest produs, pare de calitate slabă.
Răspuns: 1

Text: Sunt <span>confuz</span> de tot ce se întâmplă în jurul meu, nu înțeleg nimic.
Răspuns: 2

Text: Mă simt <span>tulburată</span>, sau poate că nu neapărat asta e cuvântul potrivit, doar că nu știu eu să dau un nume la ce simt.
Răspuns: 3

Text: {context_with_span}
Răspuns:"""

# Defaults
DEFAULT_BATCH_SIZE = 500
DEFAULT_MAX_CANDIDATES = 0  # 0 = all
MAX_CONTEXT_CHARS = 5000
CONTEXT_WINDOW = 2400  # chars before/after match for truncation
