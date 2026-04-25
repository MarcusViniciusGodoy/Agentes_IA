# Aula 08 - Agente de Busca Inteligente
# Texto livre -> recomendação de página real da Wikipédia
# Python 3.10+
#
# Requer:
# pip install requests matplotlib

import re
import requests
import webbrowser
from dataclasses import dataclass
from typing import List
from urllib.parse import quote

import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button, RadioButtons, Slider


# =========================================================
# Configurações
# =========================================================

LANGUAGES = {
    "Português": "pt",
    "Inglês": "en",
    "Espanhol": "es",
}

STOPWORDS = {
    "a", "o", "as", "os", "um", "uma", "uns", "umas",
    "de", "da", "do", "das", "dos", "em", "no", "na", "nos", "nas",
    "para", "por", "com", "sem", "sobre", "entre",
    "que", "qual", "quais", "como", "quando", "onde", "porque", "porquê",
    "eu", "voce", "você", "ele", "ela", "eles", "elas",
    "quero", "preciso", "gostaria", "entender", "saber", "explicar",
    "me", "ajude", "sobre", "coisa", "tema", "assunto",
    "the", "a", "an", "of", "to", "in", "on", "for", "with",
    "what", "how", "why", "when", "where", "about", "explain",
}


@dataclass
class SearchResult:
    title: str
    description: str
    url: str
    score: float


# =========================================================
# Processamento de linguagem simples
# =========================================================

def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.replace("ç", "c")
    text = text.replace("á", "a").replace(
        "à", "a").replace("ã", "a").replace("â", "a")
    text = text.replace("é", "e").replace("ê", "e")
    text = text.replace("í", "i")
    text = text.replace("ó", "o").replace("õ", "o").replace("ô", "o")
    text = text.replace("ú", "u")
    return text


def extract_keywords(text: str, max_keywords: int = 6) -> List[str]:
    clean = normalize_text(text)
    words = re.findall(r"[a-zA-Z0-9]+", clean)

    keywords = []

    for word in words:
        if len(word) > 3 and word not in STOPWORDS:
            keywords.append(word)

    # remove repetidas mantendo ordem
    unique = []
    for word in keywords:
        if word not in unique:
            unique.append(word)

    return unique[:max_keywords]


def build_query(user_text: str) -> str:
    keywords = extract_keywords(user_text)

    if not keywords:
        return user_text.strip()

    return " ".join(keywords)


# =========================================================
# Busca na Wikipédia
# =========================================================

def search_wikipedia(query: str, lang: str = "pt", limit: int = 8) -> List[SearchResult]:
    url = f"https://{lang}.wikipedia.org/w/rest.php/v1/search/page"

    headers = {
        "User-Agent": "Aula08-AgenteBusca/1.0 (educational demo)"
    }

    params = {
        "q": query,
        "limit": limit
    }

    response = requests.get(url, headers=headers, params=params, timeout=10)
    response.raise_for_status()

    data = response.json()
    pages = data.get("pages", [])

    results = []

    for i, page in enumerate(pages):
        title = page.get("title", "Sem título")
        description = page.get("description") or "Sem descrição"

        page_url = f"https://{lang}.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"

        # Score didático:
        # posição no ranking + presença de palavras da query no título/descrição
        base_score = 1 - (i / max(1, limit))

        text_to_score = normalize_text(title + " " + description)
        terms = query.split()

        matches = sum(1 for term in terms if normalize_text(
            term) in text_to_score)
        match_bonus = matches / max(1, len(terms))

        score = 0.7 * base_score + 0.3 * match_bonus

        results.append(
            SearchResult(
                title=title,
                description=description,
                url=page_url,
                score=score
            )
        )

    results.sort(key=lambda r: r.score, reverse=True)

    return results


# =========================================================
# Interface visual
# =========================================================

class SearchAgentUI:
    def __init__(self):
        self.results: List[SearchResult] = []
        self.best_url = ""

        self.fig = plt.figure(figsize=(15, 8))
        self.fig.canvas.manager.set_window_title(
            "Aula 08 - Agente de Busca Inteligente")

        self.ax_title = self.fig.add_axes([0.05, 0.88, 0.90, 0.08])
        self.ax_input = self.fig.add_axes([0.08, 0.78, 0.62, 0.06])
        self.ax_button_search = self.fig.add_axes([0.74, 0.78, 0.18, 0.06])

        self.ax_language = self.fig.add_axes([0.08, 0.54, 0.18, 0.16])
        self.ax_slider = self.fig.add_axes([0.08, 0.46, 0.30, 0.04])
        self.ax_button_open = self.fig.add_axes([0.08, 0.35, 0.18, 0.06])

        self.ax_bar = self.fig.add_axes([0.35, 0.32, 0.57, 0.38])
        self.ax_text = self.fig.add_axes([0.08, 0.05, 0.84, 0.23])

        self.ax_title.axis("off")
        self.ax_text.axis("off")

        self.input_box = TextBox(
            self.ax_input,
            "Texto:",
            initial="quero entender como computadores aprendem com dados"
        )

        self.button_search = Button(
            self.ax_button_search,
            "Buscar"
        )

        self.language_selector = RadioButtons(
            self.ax_language,
            list(LANGUAGES.keys())
        )
        self.ax_language.set_title("Idioma", fontsize=10, fontweight="bold")

        self.slider = Slider(
            self.ax_slider,
            "Resultados",
            valmin=3,
            valmax=12,
            valinit=8,
            valstep=1
        )

        self.button_open = Button(
            self.ax_button_open,
            "Abrir melhor página"
        )

        self.button_search.on_clicked(self.run_search)
        self.button_open.on_clicked(self.open_best_page)

        self.draw_initial()

    def draw_initial(self):
        self.ax_title.clear()
        self.ax_title.axis("off")
        self.ax_title.text(
            0.5,
            0.5,
            "Agente de Busca Inteligente",
            ha="center",
            va="center",
            fontsize=17,
            fontweight="bold"
        )

        self.ax_bar.clear()
        self.ax_bar.set_title("Ranking de páginas recomendadas")
        self.ax_bar.set_xlim(0, 1)

        self.ax_text.clear()
        self.ax_text.axis("off")
        self.ax_text.text(
            0,
            1,
            "Digite uma pergunta ou descrição do que você quer aprender.\n\n"
            "Exemplos:\n"
            "• Quero entender como computadores aprendem com dados\n"
            "• Preciso estudar redes neurais artificiais\n"
            "• Quero saber como funciona criptografia\n"
            "• Explique banco de dados relacional\n",
            va="top",
            fontsize=10
        )

        self.fig.canvas.draw_idle()

    def run_search(self, event):
        user_text = self.input_box.text.strip()

        if not user_text:
            self.show_message("Digite um texto antes de buscar.")
            return

        query = build_query(user_text)
        lang = LANGUAGES[self.language_selector.value_selected]
        limit = int(self.slider.val)

        try:
            self.results = search_wikipedia(query, lang=lang, limit=limit)

            if not self.results:
                self.show_message("Nenhum resultado encontrado.")
                return

            self.best_url = self.results[0].url
            self.draw_results(user_text, query, self.results)

        except Exception as e:
            self.show_message(f"Erro durante a busca: {e}")

    def draw_results(self, user_text: str, query: str, results: List[SearchResult]):
        self.ax_bar.clear()

        labels = [
            r.title[:35] + "..." if len(r.title) > 35 else r.title
            for r in results
        ]

        scores = [r.score for r in results]

        self.ax_bar.barh(labels[::-1], scores[::-1])
        self.ax_bar.set_xlim(0, 1)
        self.ax_bar.set_xlabel("Score de relevância")
        self.ax_bar.set_title("Ranking de páginas recomendadas")

        best = results[0]

        self.ax_text.clear()
        self.ax_text.axis("off")

        info = (
            f"TEXTO ORIGINAL:\n{user_text}\n\n"
            f"CONSULTA GERADA PELO AGENTE:\n{query}\n\n"
            f"MELHOR PÁGINA RECOMENDADA:\n"
            f"{best.title}\n"
            f"{best.description}\n\n"
            f"LINK:\n{best.url}\n\n"
            f"Discussão: o agente não 'entende' como uma pessoa; ele extrai evidências textuais, "
            f"consulta uma base real e ranqueia alternativas por relevância."
        )

        self.ax_text.text(
            0,
            1,
            info,
            va="top",
            fontsize=9
        )

        self.fig.canvas.draw_idle()

    def show_message(self, message: str):
        self.ax_text.clear()
        self.ax_text.axis("off")
        self.ax_text.text(
            0,
            1,
            message,
            va="top",
            fontsize=10
        )

        self.fig.canvas.draw_idle()

    def open_best_page(self, event):
        if self.best_url:
            webbrowser.open(self.best_url)
        else:
            self.show_message(
                "Nenhuma página selecionada ainda. Faça uma busca primeiro.")

    def run(self):
        plt.show()


# =========================================================
# Execução
# =========================================================

if __name__ == "__main__":
    app = SearchAgentUI()
    app.run()
