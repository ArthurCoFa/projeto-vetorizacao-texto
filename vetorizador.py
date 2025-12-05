import string
import math
from collections import defaultdict
from collections import Counter

class CEUBVetorizadorTFIDF:
    
    def __init__(self):
        """
        Inicializa a classe TextVetorizer e os atributos necessários 
        para o cálculo de TF-IDF.
        """

        # Mapeia palavra -> índice
        self.vocabulario = {}

        # Mapeia palavra -> valor
        self.idf = {}

        self.num_documentos = 0
    
    def processar_texto(self, texto: str) -> list: 
        """
        Método auxiliar privado para limpar e "tokenizar" o texto.
        """

        # Tabela que possui as pontuações para serem retiradas
        tabela = str.maketrans('', '', string.punctuation)
        
        # Retirando pontuações
        texto = texto.translate(tabela)

        # Convertendo para minúsculo
        texto = texto.lower()

        # Divide o texto em palavras
        return texto.split()
    
    def fit(self, corpus: list[str]):
        """
        Aprende o vocabulário e calcula os pesos IDF.
        Input:
        corpus: Uma lista de strings, onde cada string é um documento.
        """
        
        # Definir número de documentos
        self.num_documentos = len(corpus)

        # Dicionário para em quantos documentos cada palavra aparece
        frequencia_documento = defaultdict(int)

        # Índice para novas palavras
        indice = 0

        # Construir vocabulário e contar frequência de documentos
        for documento in corpus:

            # Processa e tokeniza o documento
            tokens = self.processar_texto(documento)

            # Palavras únicas para contar frequência
            palavras_unicas = set(tokens)

            for palavra in palavras_unicas:
                
                # Incrementa contagem de documentos por palavra
                frequencia_documento[palavra] += 1

                # Adiciona palavras para self.vocabulario com 1 índice
                if palavra not in self.vocabulario:
                    self.vocabulario[palavra] = indice
                    indice += 1

        total_documentos = self.num_documentos

        # Calcular IDF
        for palavra, fd in frequencia_documento.items():

            # fd = contagem_documentos[palavra]
            valor_idf = math.log((total_documentos + 1) / (fd + 1)) + 1

            self.idf[palavra] = valor_idf
    
    def transform(self, corpus:list[str]) -> list[list[float]]:
        """
        Transforma o corpus em uma matriz TF-IDF, usando o vocabulário
        e os IDFs aprendidos no método 'fit'.

        Input:
        corpus: Uma lista de strings (documentos) a serem transformadas.

        Output:
        matriz_tfidf: Uma lista de listas (ou numpy array), onde 
        matriz_tfidf[i][j] é o score TF-IDF da j-ésima
        palavra do vocabulário no i-ésimo documento.
        """

        # O tamanho do vocabulário define a largura do vetor
        tamanho_vocabulario = len(self.vocabulario)

        # Inicializando matriz TF-IDF zerada
        # número de documentos x tamanho do vocabulário
        matriz_tfidf = []

        # Iterar documentos do corpus
        for doc_index, documento in enumerate(corpus):

            # Inicializa linha com 0
            vetor_doc = [0.0] * tamanho_vocabulario

            # Processar documento para obter os tokens
            tokens = self.processar_texto(documento)

            # Ignora documentos vazios
            if not tokens:
                matriz_tfidf.append(vetor_doc)
                continue

            # Calcular Frequência e TF    
            
            term_counts = Counter(tokens)
            total_tokens = len(tokens)

            # Loop para contagem e calcular TF e TF-IDF
            for palavra, contagem in term_counts.items():

                # Calcular TF = contagem da palavra no documento / Total de Palavras
                tf_score = contagem / total_tokens

                # Verifica se palavra está no dicionário
                if palavra in self.vocabulario:

                    # Índice da palavra
                    j = self.vocabulario[palavra]

                    # Obtem o IDF da palavra, se não tem palavra IDF = 0
                    idf_score = self.idf.get(palavra, 0.0)

                    # Calcular TF-IDF
                    tfidf_score = tf_score * idf_score

                    # Adiciona valor no vetor
                    vetor_doc[j] = tfidf_score
            
            # Adiciona TF-IDF do documento na matriz
            matriz_tfidf.append(vetor_doc)
        
        return matriz_tfidf
    
    def fit_transform(self, corpus: list[str]) -> list[list[float]]:
        """
        Executa fit() e transform() em sequência no mesmo corpus.
        
        Input:
        corpus: Uma lista de strings (documentos) para aprendizado e transformação.
        
        Output:
        matriz_tfidf: A matriz TF-IDF resultante.
        """

        # Aprende os vocabulários e calcula IDF das palavras
        self.fit(corpus)

        # Transforma o corpus usando aprendizado de fit
        return self.transform(corpus)