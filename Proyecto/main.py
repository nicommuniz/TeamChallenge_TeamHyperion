# Archivo main.py

from variables import *
from clasesyfunciones import *
import numpy as np

def main():
    print("¡Bienvenido a Hundir la Flota!")
    print("Instrucciones: ...")  # Agrega instrucciones si es necesario

    # Inicializar tableros
    jugador = Tablero("Jugador")
    maquina = Tablero("Máquina")

    # Bucle principal
    turno = "Jugador"
    while jugador.quedan_barcos() and maquina.quedan_barcos():
        if turno == "Jugador":
            print("\nTu tablero:")
            imprimir_tablero(jugador.impactos)
            print("\nTablero enemigo:")
            imprimir_tablero(maquina.impactos)

            x, y = obtener_coordenadas()
            if maquina.disparar(x, y):
                print("¡Impacto!")
            else:
                print("Agua.")
                turno = "Máquina"
        else:
            print("\nTurno de la máquina...")
            x, y = np.random.randint(0, TABLERO_DIMENSION, size=2)
            if jugador.disparar(x, y):
                print(f"La máquina ha impactado en ({x}, {y}).")
            else:
                print(f"La máquina disparó a ({x}, {y}) y falló.")
                turno = "Jugador"

    # Fin del juego
    if jugador.quedan_barcos():
        print("¡Felicidades! Ganaste.")
    else:
        print("La máquina ha ganado. Mejor suerte la próxima vez.")

if __name__ == "__main__":
    main()
