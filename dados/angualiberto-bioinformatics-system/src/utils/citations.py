# Citation management module for the AnguaLiberto Bioinformatics System

class Citation:
    def __init__(self, id: int, text: str, author: str, category: str, date: str, context: str):
        self.id = id
        self.text = text
        self.author = author
        self.category = category
        self.date = date
        self.context = context

class CitationManager:
    def __init__(self):
        self.citations = self.initialize_citations()

    def initialize_citations(self):
        return [
            Citation(1, "A ciência é a poesia da realidade, e a bioinformática é sua linguagem universal.", "AnguaLiberto", "Ciência", "2024", "Filosofia da Bioinformática"),
            Citation(2, "Cada sequência de DNA conta uma história; nossa missão é aprender a lê-la.", "AnguaLiberto", "Genética", "2024", "Análise de Sequências"),
            Citation(3, "O código aberto não é apenas uma licença, é uma filosofia de colaboração científica.", "AnguaLiberto", "Open Source", "2024", "Desenvolvimento Colaborativo"),
            Citation(4, "Na era da IA, o cientista não é substituído, mas potencializado.", "AnguaLiberto", "Inteligência Artificial", "2024", "IA na Ciência"),
        ]

    def get_random_citation(self):
        import random
        return random.choice(self.citations)