
# Archivo para las funciones y clases.py

import numpy as np
from variables import *
import random

# Definición de la clase Tablero
class Tablero:
    def __init__(self, jugador_id):
        self.jugador_id = jugador_id
        self.dimension = 10
        self.barcos = []  # Lista de barcos colocados
        self.tablero = np.zeros((self.dimension, self.dimension), dtype=int)  # Tablero con barcos
        self.impactos = np.zeros((self.dimension, self.dimension), dtype=int)  # Tablero de disparos
        self.inicializar_barcos()

    def inicializar_barcos(self):
        """Coloca los barcos aleatoriamente en el tablero."""
        for barco, eslora in BARCOS.items():
            for _ in range(5 - eslora):  # Según la cantidad de barcos
                colocado = False
                while not colocado:
                    orientacion = random.choice(["H", "V"])
                    fila, columna = random.randint(0, self.dimension - 1), random.randint(0, self.dimension - 1)
                    if self.validar_posicion(fila, columna, eslora, orientacion):
                        self.colocar_barco(fila, columna, eslora, orientacion)
                        colocado = True

    def validar_posicion(self, fila, columna, eslora, orientacion):
        """Valida que el barco se pueda colocar en la posición indicada."""
        if orientacion == "H":
            if columna + eslora > self.dimension:
                return False
            if np.any(self.tablero[fila, columna:columna + eslora] != 0):
                return False
        else:  # Vertical
            if fila + eslora > self.dimension:
                return False
            if np.any(self.tablero[fila:fila + eslora, columna] != 0):
                return False
        return True

    def colocar_barco(self, fila, columna, eslora, orientacion):
        """Coloca un barco en el tablero."""
        if orientacion == "H":
            self.tablero[fila, columna:columna + eslora] = 1
        else:
            self.tablero[fila:fila + eslora, columna] = 1
    
    def disparar(self, fila, columna):
        """Ejecuta un disparo al tablero."""
        if self.tablero[fila, columna] == 1:  # Impacto
            self.impactos[fila, columna] = 2
            self.tablero[fila, columna] = 0  # Elimina esa parte del barco
            return True
        else:  # Agua
            self.impactos[fila, columna] = -1
            return False

    def quedan_barcos(self):
        """Verifica si quedan barcos en el tablero."""
        return np.any(self.tablero == 1)

# Función para obtener coordenadas del jugador
def obtener_coordenadas():
    """Solicita coordenadas al usuario."""
    while True:
        try:
            x, y = map(int, input("Introduce las coordenadas (X, Y): ").split(","))
            if 0 <= x < 10 and 0 <= y < 10:
                return x, y
            else:
                print("Las coordenadas deben estar entre 0 y 9.")
        except ValueError:
            print("Entrada inválida. Usa el formato X, Y.")

# Mapeo de símbolos para impresión
symbol_mapping = {
    0: AGUA,  # AGUA
    1: BARCO,  # BARCO
    2: IMPACTO,  # IMPACTO
    -1: FALLO,  # FALLO
}

# Función para imprimir el tablero
def imprimir_tablero(tablero):
    """Muestra el tablero en pantalla con símbolos."""
    print("\n".join(" ".join(symbol_mapping.get(cell, "?") for cell in row) for row in tablero))