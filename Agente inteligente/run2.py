# Aula 08 - Recomendador Real com Busca Online
# Palavras-chave -> recomendação de música ou filme
# Python 3.10+
#
# Requer:
# pip install requests matplotlib

import requests
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button, RadioButtons, Slider
from dataclasses import dataclass
from typing import List
from urllib.parse import quote


COUNTRIES = {
    "Brasil": "BR",
    "Estados Unidos": "US",
    "Reino Unido": "GB",
    "Japão": "JP",
    "França": "FR",
    "Alemanha": "DE",
}


@dataclass
class Recommendation:
    title: str
    creator: str
    genre: str
    year: str
    kind: str
    url: str
    score: float


def search_itunes(term: str, media: str, country: str = "BR", limit: int = 10) -> List[Recommendation]:
    encoded_term = quote(term)

    url = (
        "https://itunes.apple.com/search"
        f"?term={encoded_term}"
        f"&media={media}"
        f"&country={country}"
        f"&limit={limit}"
    )

    response = requests.get(url, timeout=10)
    response.raise_for_status()

    data = response.json()
    results = []

    for i, item in enumerate(data.get("results", [])):
        title = item.get("trackName") or item.get("collectionName") or "Sem título"
        creator = item.get("artistName", "Desconhecido")
        genre = item.get("primaryGenreName", "Sem gênero")
        release_date = item.get("releaseDate", "")
        year = release_date[:4] if release_date else "----"
        kind = item.get("kind", media)
        link = item.get("trackViewUrl") or item.get("collectionViewUrl") or ""

        score = 1 - (i / max(1, limit))

        results.append(
            Recommendation(
                title=title,
                creator=creator,
                genre=genre,
                year=year,
                kind=kind,
                url=link,
                score=score,
            )
        )

    return results


class RecommenderApp:
    def __init__(self):
        self.fig = plt.figure(figsize=(14, 8))
        self.fig.canvas.manager.set_window_title("Aula 08 - Recomendador Real com IA")

        self.ax_title = self.fig.add_axes([0.05, 0.88, 0.9, 0.08])
        self.ax_input = self.fig.add_axes([0.08, 0.78, 0.42, 0.06])
        self.ax_button = self.fig.add_axes([0.78, 0.78, 0.15, 0.06])

        self.ax_media = self.fig.add_axes([0.08, 0.56, 0.18, 0.15])
        self.ax_country = self.fig.add_axes([0.08, 0.30, 0.22, 0.20])
        self.ax_slider = self.fig.add_axes([0.08, 0.22, 0.35, 0.04])

        self.ax_bar = self.fig.add_axes([0.38, 0.28, 0.55, 0.42])
        self.ax_text = self.fig.add_axes([0.08, 0.05, 0.85, 0.14])

        self.ax_title.axis("off")
        self.ax_text.axis("off")

        self.input_box = TextBox(
            self.ax_input,
            "Palavras-chave:",
            initial="ficção científica",
        )

        self.media_selector = RadioButtons(
            self.ax_media,
            ("music", "movie"),
        )
        self.ax_media.set_title("Mídia", fontsize=10, fontweight="bold")

        self.country_selector = RadioButtons(
            self.ax_country,
            list(COUNTRIES.keys()),
        )
        self.ax_country.set_title("País", fontsize=10, fontweight="bold")

        self.slider = Slider(
            self.ax_slider,
            "Resultados",
            valmin=3,
            valmax=15,
            valinit=8,
            valstep=1,
        )

        self.button = Button(
            self.ax_button,
            "Recomendar",
        )

        self.button.on_clicked(self.on_search)

        self.draw_initial()

    def draw_initial(self):
        self.ax_title.clear()
        self.ax_title.axis("off")
        self.ax_title.text(
            0.5,
            0.5,
            "Recomendador Real de Filmes e Músicas",
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
        )

        self.ax_bar.clear()
        self.ax_bar.set_title("Ranking das recomendações")
        self.ax_bar.set_ylabel("Score didático")
        self.ax_bar.set_ylim(0, 1)

        self.ax_text.clear()
        self.ax_text.axis("off")
        self.ax_text.text(
            0,
            1,
            "Digite palavras-chave, escolha mídia e país, depois clique em Recomendar.\n"
            "Exemplos: aventura, inteligência artificial, comédia, rock, romance, ação, suspense.",
            va="top",
            fontsize=11,
        )

        self.fig.canvas.draw_idle()

    def on_search(self, event):
        term = self.input_box.text.strip()
        media = self.media_selector.value_selected
        country_name = self.country_selector.value_selected
        country = COUNTRIES[country_name]
        limit = int(self.slider.val)

        if not term:
            self.show_message("Digite uma palavra-chave antes de buscar.")
            return

        try:
            results = search_itunes(term, media, country, limit)

            if not results:
                self.show_message(f"Nenhum resultado encontrado para: {term}")
                return

            self.draw_results(term, media, country_name, results)

        except Exception as e:
            self.show_message(f"Erro na busca: {e}")

    def draw_results(self, term: str, media: str, country_name: str, results: List[Recommendation]):
        self.ax_bar.clear()

        labels = [
            f"{r.title[:18]}..." if len(r.title) > 18 else r.title
            for r in results
        ]

        scores = [r.score for r in results]

        self.ax_bar.barh(labels[::-1], scores[::-1])
        self.ax_bar.set_xlim(0, 1)
        self.ax_bar.set_xlabel("Score didático")
        self.ax_bar.set_title(f"Resultados para: {term} | {media} | {country_name}")

        best = results[0]

        self.ax_text.clear()
        self.ax_text.axis("off")

        info = (
            f"RECOMENDAÇÃO PRINCIPAL\n\n"
            f"Título: {best.title}\n"
            f"Artista/Diretor/Autor: {best.creator}\n"
            f"Gênero: {best.genre}\n"
            f"Ano: {best.year}\n"
            f"Tipo: {best.kind}\n\n"
            f"Discussão: a recomendação depende das palavras-chave, do país escolhido "
            f"e do ranking retornado pela API."
        )

        self.ax_text.text(
            0,
            1,
            info,
            va="top",
            fontsize=10,
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
            fontsize=11,
        )
        self.fig.canvas.draw_idle()

    def run(self):
        plt.show()


if __name__ == "__main__":
    app = RecommenderApp()
    app.run()