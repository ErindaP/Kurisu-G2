import pdfplumber
import re

def is_code_like(line):
    """
    Détecte si une ligne ressemble à du code SQL ou à une définition syntaxique
    Basé sur indentation, majuscules, symboles typiques ([, ], |, etc.)
    """
    line = line.rstrip()
    return (
        line.startswith(" ") or  # indentation
        bool(re.search(r'\b(ALTER|CREATE|DROP|SELECT|INSERT|DELETE|UPDATE|SET|TABLE|INDEX|VIEW|GRANT|REVOKE)\b', line)) or
        bool(re.search(r'[|[\]{}()]', line)) or
        line.isupper()
    )

def extract_sections(pdf_path):
    sections = []
    with pdfplumber.open(pdf_path) as pdf:
        full_lines = []
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_lines.extend(text.splitlines())

    current_section = []
    in_code_block = False

    for line in full_lines:
        stripped = line.strip()

        if not stripped:
            # Fin de paragraphe sauf si dans bloc de code
            if current_section:
                if in_code_block:
                    sections.append("\n".join(current_section))
                else:
                    sections.append(" ".join(current_section))
                current_section = []
                in_code_block = False
            continue

        if is_code_like(line):
            if not in_code_block and current_section:
                # Fin de bloc normal, début de code
                sections.append(" ".join(current_section))
                current_section = []
            in_code_block = True
            current_section.append(line)
        else:
            if in_code_block:
                # Fin de code, début texte
                sections.append("\n".join(current_section))
                current_section = []
                in_code_block = False
            current_section.append(stripped)

    # Dernier bloc
    if current_section:
        sections.append("\n".join(current_section) if in_code_block else " ".join(current_section))

    return sections

# Exemple d'utilisation
pdf_path = "pg18.pdf"  # Ton fichier PDF PostgreSQL
sections = extract_sections(pdf_path)

# ✅ Sauvegarde dans un fichier
with open("pg18_parsed.txt", "w", encoding="utf-8") as f:
    for i, section in enumerate(sections):
        f.write(f"\n\n--- Section {i+1} ---\n{section}")

print(f"{len(sections)} sections extraites.")
